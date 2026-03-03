"""
TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics

Implements the TOPReward method from Chen et al. (2026).
Uses VLM token probabilities (specifically P("True")) as a reward signal
for estimating robotic task progress, bypassing text generation entirely.

Backend-agnostic: pass any VLMBackend from backends.py.
"""

import math
import numpy as np

def extract_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """Extract num_frames uniformly spaced frames from a video file."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Could not read video: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def build_prompt(instruction: str) -> str:
    """Build the TOPReward completion query (Section 3.1 of paper).

    The model is asked to judge whether the observed trajectory completes
    the instruction. We then read off log P("True") from its token logits
    rather than parsing generated text.
    """
    return (
        "The above images show a robot manipulation trajectory that completes "
        f"the following task: {instruction}. "
        "Decide whether the above statement is True or not. The answer is:"
    )


# ** TOPREWARD: ALWAYS REMOVE CHAT TEMPLATE (use_chat_template=False) **
# ** Chat template degrades TOPReward VOC by ~47% per paper §5.4.      **

def compute_topreward(
    video_path: str,
    instruction: str,
    num_frames: int = 10,
    backend=None,
    # Convenience params — used to create a backend when none is supplied
    backend_name: str = "qwen",
    model: str | None = None,
    verbose: bool = True,
) -> dict:
    """Compute TOPReward progress estimates for a video trajectory.

    Implements prefix sampling (Section 3.2): for each of K uniformly spaced
    prefix lengths, the video frames 1..k are fed to the VLM and
    log P("True") is extracted from the first generated token's logits.

    Args:
        video_path:   Path to the video file.
        instruction:  Task instruction (e.g. "Pick up the cube").
        num_frames:   Number of prefix endpoints K (default 10).
        backend:      A VLMBackend instance. If None, one is created from
                      backend_name / model.
        backend_name: "qwen" or "openai" (used when backend is None).
        model:        Model name override for the backend.
        verbose:      Print per-frame progress.

    Returns:
        {
            raw_log_probs:       list[float]  — log P("True") per prefix
            normalized_progress: list[float]  — min-max normalised to [0, 1]
            dense_rewards:       list[float]  — per-step dense rewards (Eq. 3)
            frame_indices:       list[int]    — 0-based prefix endpoint indices
        }
    """
    if backend is None:
        from backends import make_backend
        backend = make_backend(
            backend_name,
            model=model,
            use_chat_template=False,  # always off for TOPReward (see comment above)
        )

    frames = extract_frames(video_path, num_frames)
    prompt_text = build_prompt(instruction)

    # Prefix sampling: evaluate on prefixes [1..1], [1..2], …, [1..K]
    raw_log_probs = []
    for k in range(1, len(frames) + 1):
        lp = backend.log_prob_true(frames[:k], prompt_text)
        raw_log_probs.append(lp)
        if verbose:
            print(f"  Frame {k}/{len(frames)}: log P(True) = {lp:.4f}")


    # Min-max normalisation (Eq. 2)
    raw = np.array(raw_log_probs)
    eps = 1e-8
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + eps)

    # Dense per-step rewards (Eq. 3): tau=2.0, delta_max=2.0
    tau, delta_max = 2.0, 2.0
    dense = [1.0]  # first step has no previous frame to compare
    for k in range(1, len(normalized)):
        diff = normalized[k] - normalized[k - 1]
        dense.append(float(np.clip(tau * math.exp(diff), 0.0, delta_max)))

    return {
        "raw_log_probs": raw_log_probs,
        "normalized_progress": normalized.tolist(),
        "dense_rewards": dense,
        "frame_indices": list(range(len(frames))),
    }
