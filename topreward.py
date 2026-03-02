"""
TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics

Implements the TOPReward method from Chen et al. (2026).
Uses VLM token probabilities (specifically P("True")) as a reward signal
for estimating robotic task progress, bypassing text generation entirely.
"""

import math
import numpy as np
from google import genai
from google.genai import types


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


def frame_to_part(frame: np.ndarray) -> types.Part:
    """Convert an OpenCV BGR frame to a Gemini inline image Part."""
    import cv2

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return types.Part(
        inline_data=types.Blob(data=buf.tobytes(), mime_type="image/jpeg")
    )


def build_prompt(instruction: str) -> str:
    """Build the TOPReward prompt (Section 3.1 of paper).

    Prompt structure:
        "The above images show a robot manipulation trajectory that completes
         the following task: {INSTRUCTION}. Decide whether the above statement
         is True or not. The answer is:"

    The model's log-probability of generating "True" as the next token
    becomes the reward signal.
    """
    return (
        "The above images show a robot manipulation trajectory that completes "
        f"the following task: {instruction}. "
        "Decide whether the above statement is True or not. The answer is:"
    )


def compute_log_prob_true(
    client: genai.Client,
    model: str,
    frame_parts: list[types.Part],
    prompt_text: str,
) -> float:
    """Query Gemini and extract log P("True") from the first generated token.

    Returns the log probability of the "True" token. If "True" is not among
    the top candidates, returns a large negative value as fallback.
    """
    content = types.Content(
        parts=frame_parts + [types.Part(text=prompt_text)]
    )

    response = client.models.generate_content(
        model=model,
        contents=content,
        config=types.GenerateContentConfig(
            max_output_tokens=1,
            temperature=0.0,
            response_logprobs=True,
            logprobs=20,
        ),
    )

    # Search for "True" in top candidates at the first token position
    logprobs_result = response.candidates[0].logprobs_result
    if logprobs_result and logprobs_result.top_candidates:
        for candidate in logprobs_result.top_candidates[0].candidates:
            if candidate.token.strip().lower() == "true":
                return candidate.log_probability

    # Fallback: "True" not in top-k, use a very negative log prob
    return -20.0


def compute_topreward(
    video_path: str,
    instruction: str,
    num_frames: int = 10,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    verbose: bool = True,
) -> dict:
    """Compute TOPReward progress estimates for a video trajectory.

    Implements prefix sampling (Section 3.2): for each prefix length k,
    we feed frames 1..k to the VLM and measure log P("True").

    Args:
        video_path: Path to the video file.
        instruction: Task instruction (e.g. "Pick up the cube").
        num_frames: Number of uniformly sampled prefix endpoints (K).
        model: Gemini model ID.
        api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
        verbose: Print progress during computation.

    Returns:
        Dictionary with keys:
            raw_log_probs: list of raw log P("True") per prefix
            normalized_progress: list of min-max normalized progress [0,1]
            dense_rewards: list of per-step dense rewards (Eq. 3)
            frame_indices: list of prefix endpoint indices (0-based)
    """
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()

    frames = extract_frames(video_path, num_frames)
    all_parts = [frame_to_part(f) for f in frames]
    prompt_text = build_prompt(instruction)

    # Prefix sampling: evaluate on prefixes [1..1], [1..2], ..., [1..K]
    raw_log_probs = []
    for k in range(1, len(frames) + 1):
        prefix_parts = all_parts[:k]
        lp = compute_log_prob_true(client, model, prefix_parts, prompt_text)
        raw_log_probs.append(lp)
        if verbose:
            print(f"  Frame {k}/{len(frames)}: log P(True) = {lp:.4f}")

    # Min-max normalization (Eq. 2)
    raw = np.array(raw_log_probs)
    eps = 1e-8
    r_min, r_max = raw.min(), raw.max()
    normalized = (raw - r_min) / (r_max - r_min + eps)

    # Dense rewards (Eq. 3) with tau=2.0, delta_max=2.0
    tau, delta_max = 2.0, 2.0
    dense = []
    for k in range(len(normalized)):
        if k == 0:
            dense.append(1.0)  # No previous step to compare
        else:
            diff = normalized[k] - normalized[k - 1]
            reward = np.clip(tau * math.exp(diff), 0.0, delta_max)
            dense.append(float(reward))

    return {
        "raw_log_probs": raw_log_probs,
        "normalized_progress": normalized.tolist(),
        "dense_rewards": dense,
        "frame_indices": list(range(len(frames))),
    }
