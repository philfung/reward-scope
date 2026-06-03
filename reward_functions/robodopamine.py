"""
Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation

Implements the general process reward model (GRM) from Tan et al. (CVPR 2026).
Supports multi-perspective progress fusion by combining Incremental-Mode, 
Forward-Mode, and Backward-Mode predictions.
"""

import re
import cv2
import numpy as np
from PIL import Image

SYSTEM_PROMPT = (
    "You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.\n\n"
    "<Task>\n"
    "`{task}`\n\n"
    "REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):\n"
    "- REFERENCE START — Robot Front Image (task just starting): <image>\n"
    "- REFERENCE END — Robot Front Image (task fully completed): <image>\n"
    "</Task>\n\n"
    "BEFORE Robot Front Image: <image>\n"
    "BEFORE Robot Left Wrist Image: <image>\n"
    "BEFORE Robot Right Wrist Image: <image>\n\n"
    "AFTER Robot Front Image: <image>\n"
    "AFTER Robot Left Wrist Image: <image>\n"
    "AFTER Robot Right Wrist Image: <image>\n\n"
    "Goal\n"
    "Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.\n\n"
    "Progress Estimation (no formulas)\n"
    "1) Calibrate using the references:\n"
    "   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”\n"
    "   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.\n"
    "2) Direction:\n"
    "   - AFTER better than BEFORE → positive score.\n"
    "   - AFTER worse than BEFORE → negative score.\n"
    "   - Essentially the same → 0.\n"
    "3) Normalize to an integer percentage in [-100%, +100%]:\n"
    "   - For improvements, scale the improvement relative to what remained from BEFORE to END.\n"
    "   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.\n"
    "   - Clip to [-100%, +100%] and round to the nearest integer percent.\n\n"
    "Evaluation Criteria (apply across all three views)\n"
    "1) Task Alignment: Evidence directly tied to `{task}`.\n"
    "2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.\n"
    "3) View-Specific Evidence & Consistency:\n"
    "   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.\n"
    "   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).\n"
    "   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.\n"
    "   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.\n"
    "4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.\n"
    "5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.\n\n"
    "Output Format (STRICT)\n"
    "Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:\n"
    "<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>"
)


class RoboDopamineModel:
    """Loads Robo-Dopamine-GRM-2.0-8B-Preview and performs relative process scoring."""

    def __init__(self, model_name: str = "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview", max_new_tokens: int = 128):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        print(f"Loading {model_name} …")
        
        # Support Apple Silicon (MPS) robustly
        try:
            self._model = ModelClass.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            print(f"  [Warning] Failed loading with device_map='auto': {e}")
            if torch.backends.mps.is_available():
                print("  Trying to load on MPS in float16...")
                self._model = ModelClass.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                ).to("mps")
            else:
                print("  Trying to load on CPU in float32...")
                self._model = ModelClass.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                ).to("cpu")

        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if hasattr(self._processor, 'image_processor'):
            self._processor.image_processor.max_pixels = 76800
            self._processor.image_processor.min_pixels = 12544
        print(f"  Loaded. Device: {self._model.device}")

    def score_step(
        self,
        instruction: str,
        ref_start_img: Image.Image,
        ref_end_img: Image.Image,
        before_imgs: list[Image.Image],
        after_imgs: list[Image.Image],
    ) -> tuple[float, str]:
        """Run inference on the step, comparing BEFORE vs AFTER with reference anchors."""
        import torch

        prompt_parts = SYSTEM_PROMPT.format(task=instruction).split("<image>")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_parts[0]},
                {"type": "image", "image": ref_start_img},
                {"type": "text", "text": prompt_parts[1]},
                {"type": "image", "image": ref_end_img},
                {"type": "text", "text": prompt_parts[2]},
                {"type": "image", "image": before_imgs[0]},
                {"type": "text", "text": prompt_parts[3]},
                {"type": "image", "image": before_imgs[1]},
                {"type": "text", "text": prompt_parts[4]},
                {"type": "image", "image": before_imgs[2]},
                {"type": "text", "text": prompt_parts[5]},
                {"type": "image", "image": after_imgs[0]},
                {"type": "text", "text": prompt_parts[6]},
                {"type": "image", "image": after_imgs[1]},
                {"type": "text", "text": prompt_parts[7]},
                {"type": "image", "image": after_imgs[2]},
                {"type": "text", "text": prompt_parts[8]},
            ]
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[
                ref_start_img,
                ref_end_img,
                before_imgs[0],
                before_imgs[1],
                before_imgs[2],
                after_imgs[0],
                after_imgs[1],
                after_imgs[2]
            ],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        output_text = self._processor.decode(
            trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        score = _parse_score(output_text)
        return score, output_text


def _parse_score(text: str) -> float:
    """Parse score wrapped in <score> tags from the output string."""
    try:
        match = re.search(r"<score>(.*?)</score>", text, re.IGNORECASE)
        if match:
            score_str = match.group(1).replace("%", "").strip()
            val = float(score_str)
        else:
            matches = re.findall(r"([+-]?\d+)\s*%", text)
            val = float(matches[-1]) if matches else 0.0
        val = min(100.0, max(-100.0, val))
        return val / 100.0
    except Exception:
        return 0.0


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


def _to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def compute_robodopamine(
    video_path: str,
    instruction: str,
    num_frames: int = 10,
    model: RoboDopamineModel | None = None,
    verbose: bool = True,
) -> dict:
    """Compute Robo-Dopamine spatiotemporal progress estimates for a video trajectory using fusion."""
    if model is None:
        model = RoboDopamineModel()

    frames = extract_frames(video_path, num_frames)

    # Initial frame progress is always 0.0
    progress_forward = [0.0]
    progress_incremental = [0.0]
    progress_backward = [0.0]
    
    reasoning_forward = [""]
    reasoning_incremental = [""]
    reasoning_backward = [""]

    ref_start_img = _to_pil(frames[0])
    ref_end_img = _to_pil(frames[-1])

    for t in range(1, len(frames)):
        after_imgs = [_to_pil(frames[t])] * 3  # replicate single-view

        # 1. Forward Mode (Start -> Current)
        before_fwd = [ref_start_img] * 3
        score_fwd, text_fwd = model.score_step(instruction, ref_start_img, ref_end_img, before_fwd, after_imgs)
        progress_fwd = max(0.0, min(1.0, score_fwd))
        progress_forward.append(progress_fwd)
        reasoning_forward.append(text_fwd)

        # 2. Incremental Mode (Previous -> Current)
        before_inc = [_to_pil(frames[t - 1])] * 3
        score_inc, text_inc = model.score_step(instruction, ref_start_img, ref_end_img, before_inc, after_imgs)
        prev_inc = progress_incremental[-1]
        if score_inc >= 0:
            progress_inc = prev_inc + (1.0 - prev_inc) * score_inc
        else:
            progress_inc = prev_inc + prev_inc * score_inc
        progress_inc = max(0.0, min(1.0, progress_inc))
        progress_incremental.append(progress_inc)
        reasoning_incremental.append(text_inc)

        # 3. Backward Mode (Goal -> Current)
        before_bwd = [ref_end_img] * 3
        score_bwd, text_bwd = model.score_step(instruction, ref_start_img, ref_end_img, before_bwd, after_imgs)
        progress_bwd = max(0.0, min(1.0, 1.0 + score_bwd))
        progress_backward.append(progress_bwd)
        reasoning_backward.append(text_bwd)

        if verbose:
            fused_score = (progress_fwd + progress_inc + progress_bwd) / 3.0
            print(f"  Frame {t}/{len(frames)-1}: progress: fused={fused_score:.3f} (fwd={progress_fwd:.2f}, inc={progress_inc:.2f}, bwd={progress_bwd:.2f})")

    # Compute fused final progress
    progress_scores = []
    for fwd, inc, bwd in zip(progress_forward, progress_incremental, progress_backward):
        progress_scores.append((fwd + inc + bwd) / 3.0)

    return {
        "progress_scores": progress_scores,
        "progress_forward": progress_forward,
        "progress_incremental": progress_incremental,
        "progress_backward": progress_backward,
        "reasoning_forward": reasoning_forward,
        "reasoning_incremental": reasoning_incremental,
        "reasoning_backward": reasoning_backward,
        "frame_indices": list(range(len(frames))),
    }
