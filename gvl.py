"""
GVL: Generative Value Learning (Ma et al., 2024)

Baseline implementation. GVL casts progress estimation as visual
question-answering: given shuffled trajectory frames, the VLM is prompted
to assign per-frame progress scores from 0 to 1.
"""

import json
import random
import numpy as np
from scipy.stats import spearmanr
from google import genai
from google.genai import types

from topreward import extract_frames, frame_to_part


def build_gvl_prompt(instruction: str, num_frames: int, shuffled_order: list[int]) -> str:
    """Build the GVL prompt asking the model to score each shuffled frame.

    The prompt presents numbered images in a shuffled order and asks the VLM
    to assign a progress score (0.0 to 1.0) to each, based on how close
    the frame is to completing the task.
    """
    frame_labels = ", ".join(f"Image {i + 1}" for i in range(num_frames))
    return (
        f"You are given {num_frames} images from a robot manipulation trajectory "
        f"for the task: \"{instruction}\". "
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
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    verbose: bool = True,
) -> dict:
    """Compute GVL progress estimates for a video trajectory.

    Shuffles frames, sends them to the VLM with a prompt asking for
    per-frame progress scores, then unshuffles to get temporal ordering.

    Args:
        video_path: Path to the video file.
        instruction: Task instruction.
        num_frames: Number of frames to sample.
        model: Gemini model ID.
        api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
        verbose: Print progress during computation.

    Returns:
        Dictionary with keys:
            progress_scores: list of progress scores in chronological order
            voc: Value-Order Correlation (Spearman's rho)
            raw_response: the raw model text response
    """
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()

    frames = extract_frames(video_path, num_frames)
    all_parts = [frame_to_part(f) for f in frames]

    # Create shuffled ordering
    original_indices = list(range(num_frames))
    shuffled_indices = original_indices.copy()
    random.shuffle(shuffled_indices)

    # Build content with shuffled frames
    shuffled_parts = []
    for i, shuf_idx in enumerate(shuffled_indices):
        shuffled_parts.append(types.Part(text=f"Image {i + 1}:"))
        shuffled_parts.append(all_parts[shuf_idx])

    prompt_text = build_gvl_prompt(instruction, num_frames, shuffled_indices)
    shuffled_parts.append(types.Part(text=prompt_text))

    content = types.Content(parts=shuffled_parts)

    if verbose:
        print("  Querying VLM for GVL progress scores...")

    response = client.models.generate_content(
        model=model,
        contents=content,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
        ),
    )

    raw_text = response.text.strip()
    if verbose:
        print(f"  Raw response: {raw_text[:200]}")

    # Parse JSON response
    scores_by_shuffled = _parse_scores(raw_text, num_frames)

    # Unshuffle: map scores back to chronological order
    chronological_scores = [0.0] * num_frames
    for shuffled_pos, original_idx in enumerate(shuffled_indices):
        chronological_scores[original_idx] = scores_by_shuffled[shuffled_pos]

    # Compute VOC (Eq. 4): Spearman rank correlation between
    # chronological order and predicted values
    if len(set(chronological_scores)) > 1:
        voc, _ = spearmanr(
            list(range(num_frames)),
            chronological_scores,
        )
    else:
        voc = 0.0

    return {
        "progress_scores": chronological_scores,
        "voc": float(voc),
        "raw_response": raw_text,
    }


def _parse_scores(text: str, num_frames: int) -> list[float]:
    """Parse GVL JSON response into a list of scores (in shuffled order)."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        scores = []
        for i in range(num_frames):
            key = f"Image {i + 1}"
            scores.append(float(data.get(key, 0.0)))
        return scores
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract numbers in order
        import re
        numbers = re.findall(r"(\d+\.?\d*)", text)
        scores = [float(n) for n in numbers[:num_frames]]
        while len(scores) < num_frames:
            scores.append(0.0)
        return scores


def compute_voc(predicted: list[float], ground_truth_order: list[int] | None = None) -> float:
    """Compute Value-Order Correlation (Eq. 4).

    VOC = Spearman rank correlation between chronological order
    and predicted progress values.
    """
    n = len(predicted)
    if ground_truth_order is None:
        ground_truth_order = list(range(n))
    if len(set(predicted)) <= 1:
        return 0.0
    rho, _ = spearmanr(ground_truth_order, predicted)
    return float(rho)
