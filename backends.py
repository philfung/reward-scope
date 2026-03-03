"""
VLM backends for TOPReward and GVL.

Two backends are supported:
  gemini  — Google Gemini API via google-genai SDK.
              Easy setup, requires an API key.
  qwen    — Qwen2.5-VL / Qwen3-VL running locally via HuggingFace transformers.
              Best TOPReward results per the paper (Table 1/2), requires GPU
              with ~16 GB VRAM.

Both expose the same two methods used by topreward.py and gvl.py:
  log_prob_true(frames, prompt_text) -> float   # log P("True")
  generate(frames, prompt_text, max_tokens) -> str
"""

import io
import math
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image


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
# Gemini backend
# ---------------------------------------------------------------------------

class GeminiBackend(VLMBackend):
    """Google Gemini API backend (google-genai SDK).

    Uses response_logprobs=True to extract log P("True") at the first
    generated token position.

    Note: Gemini enforces a chat template on all requests, which the paper
    (Section 5.4) shows reduces TOPReward VOC. GVL actually performs better
    on Gemini than open-source models (Table 1).
    """

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        from google import genai
        self.model = model
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _parts(self, frames: list[np.ndarray]):
        from google.genai import types
        return [
            types.Part(inline_data=types.Blob(data=_frame_to_jpeg_bytes(f), mime_type="image/jpeg"))
            for f in frames
        ]

    def log_prob_true(self, frames: list[np.ndarray], prompt_text: str) -> float:
        from google.genai import types

        content = types.Content(parts=self._parts(frames) + [types.Part(text=prompt_text)])
        response = self._client.models.generate_content(
            model=self.model,
            contents=content,
            config=types.GenerateContentConfig(
                max_output_tokens=1,
                temperature=0.0,
                response_logprobs=True,
                logprobs=20,
            ),
        )
        result = response.candidates[0].logprobs_result
        if result and result.top_candidates:
            for cand in result.top_candidates[0].candidates:
                if cand.token.strip().lower() == "true":
                    return cand.log_probability
        return -20.0  # "True" not in top-k

    def generate(self, frames: list[np.ndarray], prompt_text: str, max_tokens: int = 512) -> str:
        from google.genai import types

        content = types.Content(parts=self._parts(frames) + [types.Part(text=prompt_text)])
        response = self._client.models.generate_content(
            model=self.model,
            contents=content,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.0),
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# Qwen-VL backend
# ---------------------------------------------------------------------------

class QwenVLBackend(VLMBackend):
    """Local Qwen2.5-VL / Qwen3-VL backend via HuggingFace transformers.

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

    # Qwen2.5-VL raw vision placeholder (one per image in the text string)
    _IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        use_chat_template: bool = False,
        torch_dtype: str = "auto",
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_name = model_name
        self.use_chat_template = use_chat_template

        print(f"Loading {model_name} …")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self._model.eval()

        # Pre-compute the token ID for "True" (single token in Qwen vocab)
        true_ids = self._processor.tokenizer.encode("True", add_special_tokens=False)
        self._true_token_id = true_ids[0]
        print(f"  Loaded. 'True' → token ID {self._true_token_id}")

    def _build_inputs(self, frames: list[np.ndarray], prompt_text: str):
        """Build tokenised model inputs with or without a chat template."""
        import torch

        pil_images = [_frame_to_pil(f) for f in frames]

        if self.use_chat_template:
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
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # scores[0]: logits for the first generated token, shape (1, vocab_size)
        logits = outputs.scores[0][0]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[self._true_token_id].item()

    def generate(self, frames: list[np.ndarray], prompt_text: str, max_tokens: int = 512) -> str:
        import torch

        inputs = self._build_inputs(frames, prompt_text)
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
                    return lp.logprob
        print(f"  [OpenAI] WARNING: 'True' not in top-5; returning -20.0")
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
        return response.choices[0].message.content.strip()


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
        backend: "gemini", "openai", or "qwen".
        model:   Model name/ID override.
                 Gemini default:  "gemini-2.5-flash"
                 OpenAI default:  "gpt-4o-mini"
                 Qwen default:    "Qwen/Qwen2.5-VL-7B-Instruct"
        api_key:        Google API key (Gemini only).
        openai_api_key: OpenAI API key (OpenAI only).
        use_chat_template: Qwen only. Default False matches paper's best results.
    """
    if backend == "gemini":
        return GeminiBackend(model=model or "gemini-2.5-flash", api_key=api_key)
    elif backend == "openai":
        return OpenAIBackend(model=model or "gpt-4o-mini", api_key=openai_api_key)
    elif backend == "qwen":
        return QwenVLBackend(
            model_name=model or "Qwen/Qwen2.5-VL-7B-Instruct",
            use_chat_template=use_chat_template,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'gemini', 'openai', or 'qwen'.")
