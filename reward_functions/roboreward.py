"""
RoboReward: General-purpose vision-language reward model for robotics.

Uses teetone/RoboReward-8B (fine-tuned Qwen3-VL) to predict discrete
end-of-episode progress scores (1–5) from robot rollout video prefixes.

Score conversion: progress_probability = (score - 1) * 0.25
  1 → 0.00  (No Success)
  2 → 0.25  (Minimal Progress)
  3 → 0.50  (Partial Completion)
  4 → 0.75  (Near Completion)
  5 → 1.00  (Perfect Completion)
"""

import re
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

ROBOREWARD_MODEL = "teetone/RoboReward-8B"
# Note: use teetone/RoboReward-4B for faster inference.

ROBOREWARD_PROMPT_TEMPLATE = (
    "Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video "
    "in the format: ANSWER: <score>\n"
    "Rubric for end-of-episode progress (judge only the final state without time limits):\n"
    "1 - No Success: Final state shows no goal-relevant change for the command.\n"
    "2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.\n"
    "3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.\n"
    "4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.\n"
    "5 - Perfect Completion: Final state satisfies all requirements.\n\n"
    "Task: {instruction}"
)


class RoboRewardModel:
    """Loads RoboReward-8B and scores frame sequences via video messages.

    Uses qwen_vl_utils.process_vision_info with frames saved as JPEG files,
    which is required for correct output from this model.
    """

    def __init__(self, model_name: str = ROBOREWARD_MODEL, max_new_tokens: int = 128):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        print(f"Loading {model_name} …")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            do_sample_frames=False,
            fps=1,
        )
        print(f"  Loaded. Device: {self._model.device}")

    def score_frames(self, frames: list[np.ndarray], instruction: str) -> tuple[float, str]:
        """Score a frame sequence, returning (discrete_score 1–5, raw_output_text)."""
        import torch
        import cv2
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        # Qwen3-VL requires at least 2 frames for video input
        if len(pil_frames) == 1:
            pil_frames = [pil_frames[0], pil_frames[0]]

        prompt = ROBOREWARD_PROMPT_TEMPLATE.format(instruction=instruction)

        # Save frames as JPEG files — required by qwen_vl_utils video processing
        tmpdir = tempfile.mkdtemp()
        try:
            uid = uuid.uuid4().hex
            frame_paths = []
            for i, img in enumerate(pil_frames):
                path = Path(tmpdir) / f"rr_{uid}_{i:04d}.jpg"
                img.save(path, "JPEG", quality=85)
                frame_paths.append(f"file://{path}")

            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frame_paths, "sample_fps": 1.0},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [message],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            if video_inputs is not None:
                videos_tuple, video_metadatas_tuple = zip(*video_inputs)
                videos: list | None = list(videos_tuple)
                video_metadatas: list | None = list(video_metadatas_tuple)
            else:
                videos, video_metadatas = None, None

            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
            output_text = self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return _parse_score(output_text), output_text


def _parse_score(text: str) -> float:
    """Parse 'ANSWER: <score>' (1–5) from model output. Returns 1.0 on failure."""
    match = re.search(r'ANSWER\s*:\s*([1-5])', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Fallback: last standalone digit 1-5 in the output
    matches = re.findall(r'\b([1-5])\b', text)
    if matches:
        return float(matches[-1])
    return 1.0


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


def compute_roboreward(
    video_path: str,
    instruction: str,
    num_frames: int = 10,
    model: RoboRewardModel | None = None,
    verbose: bool = True,
) -> dict:
    """Compute RoboReward progress estimates for a video trajectory.

    Uses prefix sampling: for each of K uniformly spaced prefix lengths,
    the video frames 1..k are scored and converted to a progress probability
    via (score - 1) * 0.25.

    Args:
        video_path:   Path to the video file.
        instruction:  Task instruction (e.g. "Pick up the cube").
        num_frames:   Number of prefix endpoints K (default 10).
        model:        A RoboRewardModel instance. If None, one is created.
        verbose:      Print per-frame progress.

    Returns:
        {
            raw_scores:      list[float]  — discrete scores (1–5) per prefix
            progress_scores: list[float]  — progress probabilities in [0, 1]
            frame_indices:   list[int]    — 0-based prefix endpoint indices
        }
    """
    if model is None:
        model = RoboRewardModel()

    frames = extract_frames(video_path, num_frames)

    raw_scores = []
    progress_scores = []

    for k in range(1, len(frames) + 1):
        score, raw_output = model.score_frames(frames[:k], instruction)
        progress = (score - 1) * 0.25
        raw_scores.append(score)
        progress_scores.append(progress)

        if verbose:
            print(f"  Frame {k}/{len(frames)}: score={score:.0f}  progress={progress:.2f}  raw={raw_output!r}")

    return {
        "raw_scores": raw_scores,
        "progress_scores": progress_scores,
        "frame_indices": list(range(len(frames))),
    }
