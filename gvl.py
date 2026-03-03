"""
GVL: Generative Value Learning (Ma et al., 2024)

Baseline implementation. GVL casts progress estimation as visual
question-answering: given shuffled trajectory frames, the VLM is prompted
to assign per-frame progress scores from 0 to 1.

Backend-agnostic: pass any VLMBackend from backends.py.
"""

import json
import random
import numpy as np
from scipy.stats import spearmanr

from topreward import extract_frames


def build_gvl_prompt(instruction: str, num_frames: int) -> str:
    """Build the GVL scoring prompt.

    Presents numbered (shuffled) images and asks the VLM to assign a
    progress score (0.0–1.0) to each.
    """
    frame_labels = ", ".join(f"Image {i + 1}" for i in range(num_frames))
    return (
        f"You are given {num_frames} images from a robot manipulation trajectory "
        f'for the task: "{instruction}". '
        f"The images are presented in a SHUFFLED order (not chronological). "
        f"The images are labeled: {frame_labels}.\n\n"
        f"For each image, estimate how much progress the robot has made toward "
        f"completing the task, as a number from 0.0 (not started) to 1.0 (fully completed).\n\n"
        f"Return your answer as a JSON object mapping image labels to scores, e.g.:\n"
        f'{{"Image 1": 0.3, "Image 2": 0.8, ...}}\n\n'
        f"Return ONLY the JSON object, no other text."
    )


def compute_gvl(
    video_path: str,
    instruction: str,
    num_frames: int = 10,
    backend=None,
    # Convenience params — used to create a backend when none is supplied
    backend_name: str = "gemini",
    model: str | None = None,
    api_key: str | None = None,
    use_chat_template: bool = False,
    verbose: bool = True,
) -> dict:
    """Compute GVL progress estimates for a video trajectory.

    Shuffles frames, sends them with a scoring prompt to the VLM, parses the
    JSON scores, and unshuffles to get chronological progress values.

    Args:
        video_path:        Path to the video file.
        instruction:       Task instruction.
        num_frames:        Number of frames to sample.
        backend:           A VLMBackend instance. If None, one is created from
                           backend_name / model / api_key / use_chat_template.
        backend_name:      "gemini" or "qwen" (used when backend is None).
        model:             Model name override.
        api_key:           API key for Gemini backend.
        use_chat_template: For Qwen backend only.
        verbose:           Print intermediate output.

    Returns:
        {
            progress_scores: list[float]  — chronological progress (0–1)
            voc:             float        — Value-Order Correlation (Spearman)
            raw_response:    str          — raw model text output
        }
    """
    if backend is None:
        from backends import make_backend
        backend = make_backend(
            backend_name,
            model=model,
            api_key=api_key,
            use_chat_template=use_chat_template,
        )

    frames = extract_frames(video_path, num_frames)

    # Shuffle frame ordering
    shuffled_indices = list(range(num_frames))
    random.shuffle(shuffled_indices)

    # Build a single interleaved list: "Image N:" label then the frame
    # The backend.generate() call sends all of them at once.
    labeled_frames = []
    for display_pos, orig_idx in enumerate(shuffled_indices):
        labeled_frames.append((f"Image {display_pos + 1}:", frames[orig_idx]))

    prompt_text = build_gvl_prompt(instruction, num_frames)

    if verbose:
        print("  Querying VLM for GVL progress scores…")

    # Flatten to (label, frame, label, frame, …, prompt) — backends receive
    # frames as a list; we encode the labels into the prompt instead.
    all_frames = [f for _, f in labeled_frames]
    label_prefix = "\n".join(f"{label} (see image {i+1})" for i, (label, _) in enumerate(labeled_frames))
    full_prompt = label_prefix + "\n\n" + prompt_text

    raw_text = backend.generate(all_frames, full_prompt)

    if verbose:
        print(f"  Raw response: {raw_text[:300]}")

    scores_by_shuffled = _parse_scores(raw_text, num_frames)

    # Unshuffle: map scores back to chronological order
    chronological_scores = [0.0] * num_frames
    for shuffled_pos, orig_idx in enumerate(shuffled_indices):
        chronological_scores[orig_idx] = scores_by_shuffled[shuffled_pos]

    voc = compute_voc(chronological_scores)

    return {
        "progress_scores": chronological_scores,
        "voc": float(voc),
        "raw_response": raw_text,
    }


def _parse_scores(text: str, num_frames: int) -> list[float]:
    """Parse a GVL JSON response into a list of scores (in shuffled order)."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        return [float(data.get(f"Image {i + 1}", 0.0)) for i in range(num_frames)]
    except (json.JSONDecodeError, ValueError):
        import re
        numbers = re.findall(r"\d+\.?\d*", text)
        scores = [float(n) for n in numbers[:num_frames]]
        while len(scores) < num_frames:
            scores.append(0.0)
        return scores


def compute_voc(predicted: list[float], ground_truth_order: list[int] | None = None) -> float:
    """Compute Value-Order Correlation (Eq. 4).

    Spearman rank correlation between chronological order and predicted values.
    """
    n = len(predicted)
    if ground_truth_order is None:
        ground_truth_order = list(range(n))
    if len(set(predicted)) <= 1:
        return 0.0
    rho, _ = spearmanr(ground_truth_order, predicted)
    return float(rho)
