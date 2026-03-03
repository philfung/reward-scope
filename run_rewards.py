#!/usr/bin/env python3
"""
RewardScope: Run and compare VLM-based robot reward functions on a video.

Backends are fixed per method:
    TOPReward     → Qwen (local GPU)
    GVL           → OpenAI
    BruteforceVLM → OpenAI

Usage:
    export OPENAI_API_KEY="your-key"
    python run_rewards.py --video robot.mp4 --instruction "Pick up the cube"

Common flags:
    --num-frames   N                              frames to sample (default 10)
    --method       topreward,gvl,bruteforce_vlm   comma-separated list of methods (default: all)
    --save-json    path.json                      save results for viewer.html
"""

import argparse
import os
import sys
import time


# ---------------------------------------------------------------------------
# Run methods
# ---------------------------------------------------------------------------

def run_topreward(args):
    from backends import make_backend
    from reward_functions.topreward import compute_topreward

    backend = make_backend("qwen", use_chat_template=False)

    print(f"\n{'='*60}")
    print(f"TOPReward  [Qwen]")
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
    result["model"] = backend.model_name
    return result


def run_gvl(args):
    from backends import make_backend
    from reward_functions.gvl import compute_gvl

    backend = make_backend("openai", openai_api_key=getattr(args, "openai_api_key", None))

    print(f"\n{'='*60}")
    print(f"GVL  [OpenAI]")
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
    result["model"] = backend.model
    return result


def run_bruteforce_vlm(args):
    from backends import make_backend
    from reward_functions.bruteforce_vlm import compute_bruteforce_vlm

    backend = make_backend("openai", openai_api_key=getattr(args, "openai_api_key", None))

    print(f"\n{'='*60}")
    print(f"BruteforceVLM  [OpenAI]")
    print(f"{'='*60}")
    print(f"  Video:       {args.video}")
    print(f'  Instruction: "{args.instruction}"')
    print(f"  Frames:      {args.num_frames}")
    print()

    start = time.time()
    result = compute_bruteforce_vlm(
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
    result["model"] = backend.model
    return result


def save_json(top_result, gvl_result, bruteforce_result, args, path):
    """Save results as JSON for the viewer.html visualiser."""
    import json
    data = {
        "instruction": args.instruction,
        "backend": "TOPReward=Qwen | GVL=OpenAI | BruteforceVLM=OpenAI",
        "num_frames": args.num_frames,
    }
    if top_result:
        data["topreward"] = {
            "model": top_result.get("model"),
            "raw_log_probs": top_result["raw_log_probs"],
            "normalized_progress": top_result["normalized_progress"],
            "dense_rewards": top_result["dense_rewards"],
        }
    if gvl_result:
        data["gvl"] = {
            "model": gvl_result.get("model"),
            "progress_scores": gvl_result["progress_scores"],
            "voc": gvl_result["voc"],
        }
    if bruteforce_result:
        data["bruteforce_vlm"] = {
            "model": bruteforce_result.get("model"),
            "progress_scores": bruteforce_result["progress_scores"],
            "dense_rewards": bruteforce_result["dense_rewards"],
            "voc": bruteforce_result["voc"],
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RewardScope — compare VLM-based robot reward functions on a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", required=True,
                        help="Path to video file (mp4, avi, …)")
    parser.add_argument("--instruction", required=True,
                        help='Task instruction, e.g. "Pick up the cube"')
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Frames to sample (default: 10)")
    parser.add_argument("--method", default="topreward,gvl,bruteforce_vlm",
                        help="Comma-separated list of methods to run: topreward,gvl,bruteforce_vlm (default: all)")
    parser.add_argument("--save-json", default=None, nargs="?", const="auto",
                        help="Save results as JSON (for viewer.html). Defaults to viewer_files/<video>.json")

    openai_group = parser.add_argument_group("OpenAI options")
    openai_group.add_argument("--openai-api-key", default=None,
                               help="OpenAI API key (or set OPENAI_API_KEY)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    methods = [m.strip() for m in args.method.split(",")]
    valid_methods = {"topreward", "gvl", "bruteforce_vlm"}
    unknown = set(methods) - valid_methods
    if unknown:
        print(f"Error: unknown method(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(sorted(valid_methods))}", file=sys.stderr)
        sys.exit(1)

    openai_methods = {"gvl", "bruteforce_vlm"} & set(methods)
    if openai_methods and not args.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        print(
            f"Error: {', '.join(sorted(openai_methods))} require --openai-api-key or OPENAI_API_KEY env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    top_result = None
    gvl_result = None
    bruteforce_result = None

    if "topreward" in methods:
        top_result = run_topreward(args)

    if "gvl" in methods:
        gvl_result = run_gvl(args)

    if "bruteforce_vlm" in methods:
        bruteforce_result = run_bruteforce_vlm(args)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    if top_result:
        from reward_functions.gvl import compute_voc
        print(f"  TOPReward VOC:      {compute_voc(top_result['normalized_progress']):.4f}")
    if gvl_result:
        print(f"  GVL VOC:            {gvl_result['voc']:.4f}")
    if bruteforce_result:
        print(f"  BruteforceVLM VOC:  {bruteforce_result['voc']:.4f}")

    # JSON export
    import shutil
    os.makedirs("viewer_files", exist_ok=True)
    json_path = args.save_json
    if json_path is None or json_path == "auto":
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        json_path = os.path.join("viewer_files", video_stem + ".json")
    save_json(top_result, gvl_result, bruteforce_result, args, json_path)
    print(f"\nJSON saved to: {json_path}")

    # Update manifest.json
    import json
    manifest_path = os.path.join("viewer_files", "manifest.json")
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []
    if video_stem not in manifest:
        manifest.append(video_stem)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"manifest.json updated: added '{video_stem}'")

    # Copy video into viewer_files/
    viewer_dir = os.path.dirname(json_path)
    video_dest = os.path.join(viewer_dir, os.path.basename(args.video))
    if os.path.abspath(args.video) != os.path.abspath(video_dest):
        shutil.copy2(args.video, video_dest)
        print(f"Video copied to: {video_dest}")

    print("\nRun ./run_viewer.sh to view results in your browser.")


if __name__ == "__main__":
    main()

# To view results, run: ./run_viewer.sh
