"""
VLM backends for RewardScope.

Two backends are supported:
  qwen    — Qwen2.5-VL / Qwen3-VL running locally via HuggingFace transformers.
              Best TOPReward results per the paper (Table 1/2), requires GPU
              with ~16 GB VRAM.
  openai  — OpenAI Chat Completions API with vision and logprobs support.

Both expose the same two methods used by all reward functions:
  log_prob_true(frames, prompt_text) -> float   # log P("True")
  generate(frames, prompt_text, max_tokens) -> str
"""

import io
import math
import time
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image

# Seconds to sleep after each OpenAI API call (to avoid rate limits)
OPENAI_CALL_SLEEP_S = 45.0

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB image."""
    import cv2
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    """Encode an OpenCV frame to JPEG bytes."""
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _debug_dump_frames(frames: list[np.ndarray], prefix: str = "frame", out_dir: str = "debug_frames") -> None:
    """Save frames to out_dir/<prefix>_<i>.jpg for visual inspection."""
    import os, cv2
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.jpg")
        cv2.imwrite(path, frame)
    print(f"  [debug] wrote {len(frames)} frame(s) to {out_dir}/")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VLMBackend(ABC):
    """Common interface for all VLM backends."""

    @abstractmethod
    def log_prob_true(self, frames: list[np.ndarray], prompt_text: str) -> float:
        """Return log P("True") given video frames and a completion prompt.

        The prompt should end with 'The answer is:' so the model's first
        generated token is the affirmative/negative answer.
        """

    @abstractmethod
    def generate(self, frames: list[np.ndarray], prompt_text: str, max_tokens: int = 512) -> str:
        """Generate free-form text given video frames and a prompt."""


# ---------------------------------------------------------------------------
# Qwen-VL backend
# ---------------------------------------------------------------------------

class QwenVLBackend(VLMBackend):
    """Local Qwen3-VL / Qwen2.5-VL backend via HuggingFace transformers.

    Per the paper (Section 3.1 and ablation 5.4), NOT using a chat template
    gives the best TOPReward results — 0.947 VOC vs ~0.500 with chat template.
    This backend defaults to use_chat_template=False to match the paper.

    In raw mode the prompt is prefixed with one vision-token placeholder per
    frame and fed directly to the model as a next-token-prediction task,
    bypassing all role/system markers.

    Requirements:
        pip install transformers torch torchvision accelerate qwen-vl-utils
    Recommended hardware: GPU with ≥16 GB VRAM (fp16/bf16).
    """

    # Qwen3-VL / Qwen2.5-VL raw vision placeholder (one per image in the text string)
    _IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        use_chat_template: bool = False,
        torch_dtype: str = "auto",
    ):
        import torch
        from transformers import AutoProcessor

        if "Qwen3" in model_name:
            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

        self.model_name = model_name
        self.use_chat_template = use_chat_template

        print(f"Loading {model_name} …")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = ModelClass.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self._model.eval()

        # Pre-compute the token ID for " True" (with leading space).
        # The prompt ends with "The answer is:" so the model generates " True"
        # (ĠTrue, token ID 3007) — not bare "True" (ID 2514).  Using the
        # wrong ID silently returns near-zero probability for every query.
        true_ids = self._processor.tokenizer.encode(" True", add_special_tokens=False)
        assert len(true_ids) == 1, f"' True' should be a single token but got {true_ids}"
        self._true_token_id = true_ids[0]
        print(f"  Loaded. ' True' → token ID {self._true_token_id}")

    def _build_inputs(self, frames: list[np.ndarray], prompt_text: str, use_chat_template: bool | None = None):
        """Build tokenised model inputs with or without a chat template.

        Args:
            use_chat_template: Override self.use_chat_template if provided.
                               generate() always passes True since the
                               instruction-tuned model requires the generation
                               prompt to produce coherent output.
        """
        import torch

        if use_chat_template is None:
            use_chat_template = self.use_chat_template

        pil_images = [_frame_to_pil(f) for f in frames]

        if use_chat_template:
            # Standard instruction-tuned chat format.
            content = [{"type": "image", "image": img} for img in pil_images]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Raw mode: no role markers — matches the paper's best configuration.
            # Each frame gets one vision-token placeholder; the prompt follows directly.
            image_tokens = self._IMAGE_PLACEHOLDER * len(pil_images)
            text = image_tokens + "\n" + prompt_text

        inputs = self._processor(
            text=[text],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def log_prob_true(self, frames: list[np.ndarray], prompt_text: str) -> float:
        import torch

        inputs = self._build_inputs(frames, prompt_text)
        with torch.no_grad():
            # Direct forward pass — equivalent to the paper's log p_θ(a | context).
            # Avoids generation-mode logit processors that could perturb the scores.
            outputs = self._model(**inputs)
        # logits[:, -1, :] are the next-token logits at the last input position
        logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[self._true_token_id].item()

    def generate(self, frames: list[np.ndarray], prompt_text: str, max_tokens: int = 512) -> str:
        import torch

        # GVL free-form generation always needs the chat template so the
        # instruction-tuned model produces a proper assistant response.
        # (use_chat_template=False is only correct for log_prob_true's
        # next-token prediction path, per paper §5.4.)
        inputs = self._build_inputs(frames, prompt_text, use_chat_template=True)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        # Decode only the newly generated tokens (skip the input prompt)
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_len:]
        return self._processor.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIBackend(VLMBackend):
    """OpenAI Chat Completions backend with vision and logprobs support.

    Uses the `logprobs=True` + `top_logprobs=20` parameters to extract
    log P("True") from the first generated token, matching the TOPReward
    formulation exactly.

    Requires: pip install openai
    Set OPENAI_API_KEY env var or pass api_key.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def _image_content(self, frame: np.ndarray) -> dict:
        import base64
        b64 = base64.b64encode(_frame_to_jpeg_bytes(frame)).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}

    def log_prob_true(self, frames: list[np.ndarray], prompt_text: str) -> float:
        shapes = [f.shape for f in frames]
        print(f"  [OpenAI] frames={len(frames)} shapes={shapes}")
        print(f"  [OpenAI] prompt: {prompt_text!r}")
        _debug_dump_frames(frames, prefix=f"logprob_k{len(frames)}")
        content = [self._image_content(f) for f in frames]
        content.append({"type": "text", "text": prompt_text})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=2,  # only token_logprobs[0] is used; 2 avoids a "can't finish" 400 with max=1
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )

        u = response.usage
        if u:
            print(f"  [OpenAI] tokens: {u.prompt_tokens} prompt + {u.completion_tokens} completion = {u.total_tokens}")

        token_logprobs = response.choices[0].logprobs.content
        if token_logprobs:
            candidates = token_logprobs[0].top_logprobs
            print(f"  [OpenAI] top-{len(candidates)} first-token candidates:")
            for lp in candidates:
                marker = " <-- THIS" if lp.token.strip().lower() == "true" else ""
                print(f"           {lp.token!r:12s}  logprob={lp.logprob:.4f}{marker}")
            for lp in candidates:
                if lp.token.strip().lower() == "true":
                    print(f"  [OpenAI] sleeping {OPENAI_CALL_SLEEP_S}s to prevent rate limiting …")
                    time.sleep(OPENAI_CALL_SLEEP_S)
                    return lp.logprob
        print(f"  [OpenAI] WARNING: 'True' not in top-5; returning -20.0")
        print(f"  [OpenAI] sleeping {OPENAI_CALL_SLEEP_S}s …")
        time.sleep(OPENAI_CALL_SLEEP_S)
        return -20.0

    def generate(self, frames: list[np.ndarray], prompt_text: str, max_tokens: int = 512) -> str:
        content = [self._image_content(f) for f in frames]
        content.append({"type": "text", "text": prompt_text})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_tokens,
            temperature=0.0,
        )
        u = response.usage
        if u:
            print(f"  [OpenAI] generate | model={self.model} | frames={len(frames)} | tokens={u.prompt_tokens}p + {u.completion_tokens}c = {u.total_tokens}")
        result = (response.choices[0].message.content or "").strip()
        print(f"  [OpenAI] sleeping {OPENAI_CALL_SLEEP_S}s to prevent rate limiting …")
        time.sleep(OPENAI_CALL_SLEEP_S)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_backend(
    backend: str,
    model: str | None = None,
    api_key: str | None = None,
    openai_api_key: str | None = None,
    use_chat_template: bool = False,
) -> VLMBackend:
    """Create a VLMBackend by name.

    Args:
        backend: "openai" or "qwen".
        model:   Model name/ID override.
                 OpenAI default:  "gpt-4o-mini"
                 Qwen default:    "Qwen/Qwen3-VL-8B-Instruct"
        openai_api_key: OpenAI API key (OpenAI only).
        use_chat_template: Qwen only. Default False matches paper's best results.
    """
    if backend == "openai":
        return OpenAIBackend(model=model or "gpt-4o-mini", api_key=openai_api_key)
    elif backend == "qwen":
        return QwenVLBackend(
            model_name=model or "Qwen/Qwen3-VL-8B-Instruct",
            use_chat_template=use_chat_template,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'openai' or 'qwen'.")
