#!/usr/bin/env python3
"""
Demo: Run TOPReward and GVL on a video and visualize progress curves.

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
    --combined-plot           overlay both methods on one axes
    --save-plot    path.png   save instead of showing

Gemini-specific:
    --api-key      KEY        or set GOOGLE_API_KEY env var

Qwen-specific:
    --model        Qwen/Qwen2.5-VL-7B-Instruct  (default)
    --use-chat-template       add chat template (hurts TOPReward per paper §5.4)
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

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
    model_short = (args.model or "Qwen2.5-VL-7B").split("/")[-1]
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(top_result, gvl_result, args, backend_label, save_path=None):
    """Side-by-side plot for TOPReward and/or GVL."""
    has_top = top_result is not None
    has_gvl = gvl_result is not None
    ncols = 2 if (has_top and has_gvl) else 1

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    x = np.linspace(0, 1, args.num_frames)
    ax_idx = 0

    if has_top:
        ax = axes[ax_idx]
        p = top_result["normalized_progress"]
        ax.plot(x, p, "o-", color="#E8734A", linewidth=2, markersize=6, label="TOPReward")
        ax.fill_between(x, 0, p, alpha=0.15, color="#E8734A")
        ax.set_title("TOPReward", fontsize=14, fontweight="bold")
        _style_ax(ax, x, p)
        ax_idx += 1

    if has_gvl:
        ax = axes[ax_idx]
        s = gvl_result["progress_scores"]
        label = f"GVL  (VOC={gvl_result['voc']:.3f})"
        ax.plot(x, s, "s-", color="#4A90D9", linewidth=2, markersize=6, label=label)
        ax.fill_between(x, 0, s, alpha=0.15, color="#4A90D9")
        ax.set_title("GVL", fontsize=14, fontweight="bold")
        _style_ax(ax, x, s)

    fig.suptitle(
        f'"{args.instruction}"  —  {backend_label}',
        fontsize=12, style="italic", y=1.01,
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_combined(top_result, gvl_result, args, backend_label, save_path=None):
    """Overlay both methods on a single axes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, args.num_frames)

    if top_result:
        p = top_result["normalized_progress"]
        ax.plot(x, p, "o-", color="#E8734A", linewidth=2.5, markersize=7, label="TOPReward")

    if gvl_result:
        s = gvl_result["progress_scores"]
        ax.plot(x, s, "s--", color="#4A90D9", linewidth=2, markersize=6,
                label=f"GVL  (VOC={gvl_result['voc']:.3f})")

    ax.plot(x, x, ":", color="gray", alpha=0.5, label="Ideal (linear)")
    ax.set_title(
        f'"{args.instruction}"  —  {backend_label}',
        fontsize=13, fontweight="bold",
    )
    _style_ax(ax, x, [])
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _style_ax(ax, x, values):
    ax.set_xlabel("Normalised Time", fontsize=12)
    ax.set_ylabel("Progress (0–1)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    for i, (xi, yi) in enumerate(zip(x, values)):
        if i % max(1, len(x) // 5) == 0 or i == len(x) - 1:
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


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
    parser.add_argument("--combined-plot", action="store_true",
                        help="Overlay both methods on one axes")
    parser.add_argument("--save-plot", default=None,
                        help="Save plot to this path instead of displaying")
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
            "Qwen default: Qwen/Qwen2.5-VL-7B-Instruct"
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

    # Plot
    if args.combined_plot and top_result and gvl_result:
        plot_combined(top_result, gvl_result, args, backend_label, save_path=args.save_plot)
    else:
        plot_results(top_result, gvl_result, args, backend_label, save_path=args.save_plot)


if __name__ == "__main__":
    main()
