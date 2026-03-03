"""
BruteforceVLM: Direct progress estimation via text generation.

Baseline that combines prefix sampling (like TOPReward) with direct numeric
output (like GVL): for each prefix length k, the cumulative frames 1..k are
sent to the VLM with a prompt asking it to output a single progress number
from 0.0 (not started) to 1.0 (fully completed).

Backend-agnostic: pass any VLMBackend from backends.py.
"""

import re
import numpy as np
from scipy.stats import spearmanr

from reward_functions.topreward import extract_frames


def build_bruteforce_prompt(instruction: str) -> str:
    """Build the progress estimation prompt for a single prefix."""
    return (
        f'A robot is performing the following task: "{instruction}". '
        "The images above show the robot trajectory so far. "
        "Estimate how much progress the robot has made in completing the task "
        "as a number from 0.0 (not started) to 1.0 (fully completed). "
        "Reply with only the number and nothing else."
    )


# ** BruteforceVLM: ALWAYS USE CHAT TEMPLATE (use_chat_template=True) **
# ** Relies on text generation; raw next-token mode is not appropriate.  **

def compute_bruteforce_vlm(
    video_path: str,
    instruction: str,
    num_frames: int = 10,
    backend=None,
    # Convenience params — used to create a backend when none is supplied
    backend_name: str = "qwen",
    model: str | None = None,
    verbose: bool = True,
) -> dict:
    """Compute BruteforceVLM progress estimates for a video trajectory.

    For each prefix length k (1..num_frames), sends the cumulative frames
    1..k to the VLM and asks it to output a single progress number in [0, 1].

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
            progress_scores: list[float]  — progress estimate per prefix (0–1)
            dense_rewards:   list[float]  — per-step dense rewards
            voc:             float        — Value-Order Correlation (Spearman)
            raw_responses:   list[str]    — raw model text per prefix
            frame_indices:   list[int]    — 0-based prefix endpoint indices
        }
    """
    if backend is None:
        from backends import make_backend
        backend = make_backend(
            backend_name,
            model=model,
            use_chat_template=True,  # always on for text generation
        )

    frames = extract_frames(video_path, num_frames)
    prompt_text = build_bruteforce_prompt(instruction)

    # Prefix sampling: evaluate on prefixes [1..1], [1..2], …, [1..K]
    progress_scores = []
    raw_responses = []
    for k in range(1, len(frames) + 1):
        raw_text = backend.generate(frames[:k], prompt_text, max_tokens=16)
        score = _parse_score(raw_text)
        progress_scores.append(score)
        raw_responses.append(raw_text)
        if verbose:
            print(f"  Frame {k}/{len(frames)}: response={raw_text!r}  score={score:.4f}")

    # Dense per-step rewards: clipped finite difference
    dense = [progress_scores[0]]
    for k in range(1, len(progress_scores)):
        diff = progress_scores[k] - progress_scores[k - 1]
        dense.append(float(np.clip(diff, 0.0, 1.0)))

    voc = _compute_voc(progress_scores)

    return {
        "progress_scores": progress_scores,
        "dense_rewards": dense,
        "voc": float(voc),
        "raw_responses": raw_responses,
        "frame_indices": list(range(len(frames))),
    }


def _parse_score(text: str) -> float:
    """Extract a float in [0, 1] from the model's response."""
    # Match 1.0 or 0.xx or bare integers 0/1
    m = re.search(r"\b(1\.0+|0\.\d+|[01])\b", text.strip())
    if m:
        return float(np.clip(float(m.group()), 0.0, 1.0))
    return 0.0


def _compute_voc(predicted: list[float]) -> float:
    """Value-Order Correlation: Spearman rank correlation with chronological order."""
    n = len(predicted)
    if len(set(predicted)) <= 1:
        return 0.0
    rho, _ = spearmanr(list(range(n)), predicted)
    return float(rho)
