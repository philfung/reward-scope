#!/usr/bin/env python3
"""
Demo: Run TOPReward and GVL on a video and save results for the viewer.

Usage — Qwen backend (local, best results per paper):
    python demo.py --video robot.mp4 --instruction "Pick up the cube"

Usage — OpenAI backend:
    export OPENAI_API_KEY="your-key"
    python demo.py --video robot.mp4 --instruction "Pick up the cube" \\
                   --backend openai

Common flags:
    --num-frames   N          frames to sample (default 10)
    --method       topreward,gvl  comma-separated list of methods (default: all)
    --model        override model name/ID for either backend
    --save-json    path.json  save results for viewer.html
"""

import argparse
import os
import sys
import time



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(args):
    """Instantiate the requested VLMBackend from CLI args."""
    from backends import make_backend
    return make_backend(
        backend=args.backend,
        model=args.model or None,
        openai_api_key=getattr(args, "openai_api_key", None),
    )


def _backend_label(args) -> str:
    """Short label for plot titles / summary."""
    if args.backend == "openai":
        return f"OpenAI ({args.model or 'gpt-4o-mini'})"
    model_short = (args.model or "Qwen3-VL-8B").split("/")[-1]
    return f"Qwen ({model_short})"


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
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Frames to sample (default: 10)")
    parser.add_argument("--method", default="topreward,gvl",
                        help="Comma-separated list of methods to run: topreward,gvl (default: all)")
    parser.add_argument("--save-json", default=None, nargs="?", const="auto",
                        help="Save results as JSON (for viewer.html). Defaults to viewer_files/<video>.json")

    # Backend selection
    backend_group = parser.add_argument_group("backend")
    backend_group.add_argument(
        "--backend", choices=["openai", "qwen"], default="qwen",
        help='VLM backend: "qwen" (default, local GPU) or "openai"',
    )
    backend_group.add_argument(
        "--model", default=None,
        help=(
            "Model override. "
            "Qwen default: Qwen/Qwen3-VL-8B-Instruct  "
            "OpenAI default: gpt-4o-mini"
        ),
    )

    # OpenAI-specific
    openai_group = parser.add_argument_group("OpenAI options")
    openai_group.add_argument("--openai-api-key", default=None,
                               help="OpenAI API key (or set OPENAI_API_KEY)")

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.backend == "openai":
        if not args.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            print(
                "Error: OpenAI backend requires --openai-api-key or OPENAI_API_KEY env var.",
                file=sys.stderr,
            )
            sys.exit(1)

    methods = [m.strip() for m in args.method.split(",")]
    valid_methods = {"topreward", "gvl"}
    unknown = set(methods) - valid_methods
    if unknown:
        print(f"Error: unknown method(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(sorted(valid_methods))}", file=sys.stderr)
        sys.exit(1)

    backend = _make_backend(args)
    backend_label = _backend_label(args)

    top_result = None
    gvl_result = None

    if "topreward" in methods:
        top_result = run_topreward(args, backend)

    if "gvl" in methods:
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
    import shutil
    os.makedirs("viewer_files", exist_ok=True)
    json_path = args.save_json
    if json_path is None or json_path == "auto":
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        json_path = os.path.join("viewer_files", video_stem + ".json")
    save_json(top_result, gvl_result, args, backend_label, json_path)
    print(f"\nJSON saved to: {json_path}")

    # Copy video into viewer_files/
    viewer_dir = os.path.dirname(json_path)
    video_dest = os.path.join(viewer_dir, os.path.basename(args.video))
    if os.path.abspath(args.video) != os.path.abspath(video_dest):
        shutil.copy2(args.video, video_dest)
        print(f"Video copied to: {video_dest}")


if __name__ == "__main__":
    main()

# To view results, run: ./run_viewer.sh
