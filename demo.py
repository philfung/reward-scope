#!/usr/bin/env python3
"""
Demo: Run TOPReward and GVL on a video and save results for the viewer.

Usage — Gemini backend (default):
    export GOOGLE_API_KEY="your-key"
    python demo.py --video robot.mp4 --instruction "Pick up the cube"

Usage — Qwen backend (local, best results per paper):
    python demo.py --video robot.mp4 --instruction "Fold the towel" \\
                   --backend qwen

Common flags:
    --num-frames   N          frames to sample (default 10)
    --method       both|topreward|gvl
    --model        override model name/ID for either backend
    --save-json    path.json  save results for viewer.html

Gemini-specific:
    --api-key      KEY        or set GOOGLE_API_KEY env var

Qwen-specific:
    --model        Qwen/Qwen3-VL-8B-Instruct  (default)
    --use-chat-template       add chat template (hurts TOPReward per paper §5.4)
"""

import argparse
import os
import sys
import time

from constants import TOPREWARD_LOGPROB_TIMEOUT_SECS, NUM_FRAMES_DEFAULT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(args):
    """Instantiate the requested VLMBackend from CLI args."""
    from backends import make_backend
    return make_backend(
        backend=args.backend,
        model=args.model or None,
        api_key=getattr(args, "api_key", None),
        openai_api_key=getattr(args, "openai_api_key", None),
        use_chat_template=getattr(args, "use_chat_template", False),
    )


def _backend_label(args) -> str:
    """Short label for plot titles / summary."""
    if args.backend == "gemini":
        return f"Gemini ({args.model or 'gemini-2.5-flash'})"
    if args.backend == "openai":
        return f"OpenAI ({args.model or 'gpt-4o-mini'})"
    model_short = (args.model or "Qwen3-VL-8B").split("/")[-1]
    tmpl = "+chat" if getattr(args, "use_chat_template", False) else "no-chat"
    return f"Qwen ({model_short}, {tmpl})"


# ---------------------------------------------------------------------------
# Run methods
# ---------------------------------------------------------------------------

def run_topreward(args, backend):
    from topreward import compute_topreward

    print(f"\n{'='*60}")
    print(f"TOPReward  [{_backend_label(args)}]")
    print(f"{'='*60}")
    print(f"  Video:       {args.video}")
    print(f'  Instruction: "{args.instruction}"')
    print(f"  Frames:      {args.num_frames}")
    print()

    start = time.time()
    result = compute_topreward(
        video_path=args.video,
        instruction=args.instruction,
        num_frames=args.num_frames,
        backend=backend,
        verbose=True,
    )
    elapsed = time.time() - start

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Progress:     {[f'{v:.3f}' for v in result['normalized_progress']]}")
    print(f"  Dense rewards:{[f'{v:.3f}' for v in result['dense_rewards']]}")
    return result


def run_gvl(args, backend):
    from gvl import compute_gvl

    print(f"\n{'='*60}")
    print(f"GVL  [{_backend_label(args)}]")
    print(f"{'='*60}")
    print(f"  Video:       {args.video}")
    print(f'  Instruction: "{args.instruction}"')
    print(f"  Frames:      {args.num_frames}")
    print()

    start = time.time()
    result = compute_gvl(
        video_path=args.video,
        instruction=args.instruction,
        num_frames=args.num_frames,
        backend=backend,
        verbose=True,
    )
    elapsed = time.time() - start

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Progress: {[f'{v:.3f}' for v in result['progress_scores']]}")
    print(f"  VOC:      {result['voc']:.4f}")
    return result



def save_json(top_result, gvl_result, args, backend_label, path):
    """Save results as JSON for the viewer.html visualiser."""
    import json
    data = {
        "instruction": args.instruction,
        "backend": backend_label,
        "num_frames": args.num_frames,
    }
    if top_result:
        data["topreward"] = {
            "raw_log_probs": top_result["raw_log_probs"],
            "normalized_progress": top_result["normalized_progress"],
            "dense_rewards": top_result["dense_rewards"],
        }
    if gvl_result:
        data["gvl"] = {
            "progress_scores": gvl_result["progress_scores"],
            "voc": gvl_result["voc"],
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TOPReward & GVL Demo — estimate task progress from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", required=True,
                        help="Path to video file (mp4, avi, …)")
    parser.add_argument("--instruction", required=True,
                        help='Task instruction, e.g. "Pick up the cube"')
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT,
                        help="Frames to sample (default: %d)" % NUM_FRAMES_DEFAULT)
    parser.add_argument("--method", choices=["both", "topreward", "gvl"], default="both",
                        help="Which method(s) to run (default: both)")
    parser.add_argument("--save-json", default=None,
                        help="Save results as JSON (for viewer.html)")

    # Backend selection
    backend_group = parser.add_argument_group("backend")
    backend_group.add_argument(
        "--backend", choices=["gemini", "openai", "qwen"], default="gemini",
        help='VLM backend: "gemini" (default), "openai", or "qwen" (local GPU)',
    )
    backend_group.add_argument(
        "--model", default=None,
        help=(
            "Model override. "
            "Gemini default: gemini-2.5-flash  "
            "Qwen default: Qwen/Qwen3-VL-8B-Instruct"
        ),
    )

    # Gemini-specific
    gemini_group = parser.add_argument_group("Gemini options")
    gemini_group.add_argument("--api-key", default=None,
                               help="Google API key (or set GOOGLE_API_KEY)")

    # OpenAI-specific
    openai_group = parser.add_argument_group("OpenAI options")
    openai_group.add_argument("--openai-api-key", default=None,
                               help="OpenAI API key (or set OPENAI_API_KEY)")

    # Qwen-specific
    qwen_group = parser.add_argument_group("Qwen options")
    qwen_group.add_argument(
        "--use-chat-template", action="store_true",
        help=(
            "Wrap prompt in chat template. "
            "Default: off (raw mode). "
            "Note: chat template degrades TOPReward VOC by ~47%% per paper §5.4."
        ),
    )

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.backend == "gemini":
        if not args.api_key and not os.environ.get("GOOGLE_API_KEY"):
            print(
                "Error: Gemini backend requires --api-key or GOOGLE_API_KEY env var.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.backend == "openai":
        if not args.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            print(
                "Error: OpenAI backend requires --openai-api-key or OPENAI_API_KEY env var.",
                file=sys.stderr,
            )
            sys.exit(1)

    backend = _make_backend(args)
    backend_label = _backend_label(args)

    top_result = None
    gvl_result = None

    if args.method in ("both", "topreward"):
        top_result = run_topreward(args, backend)

    if args.method in ("both", "gvl"):
        gvl_result = run_gvl(args, backend)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary  [{backend_label}]")
    print(f"{'='*60}")
    if top_result:
        from gvl import compute_voc
        print(f"  TOPReward VOC: {compute_voc(top_result['normalized_progress']):.4f}")
    if gvl_result:
        print(f"  GVL VOC:       {gvl_result['voc']:.4f}")

    # JSON export
    if args.save_json:
        save_json(top_result, gvl_result, args, backend_label, args.save_json)


if __name__ == "__main__":
    main()
