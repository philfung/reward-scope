#!/usr/bin/env python3
"""
Demo: Run TOPReward and GVL on a video and visualize progress curves.

Usage:
    python demo.py --video path/to/video.mp4 --instruction "Pick up the cube"
    python demo.py --video path/to/video.mp4 --instruction "Fold the towel" --num-frames 12
    python demo.py --video path/to/video.mp4 --instruction "Put cube in cup" --method topreward
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def run_topreward(args):
    from topreward import compute_topreward

    print(f"\n{'='*60}")
    print("Running TOPReward")
    print(f"{'='*60}")
    print(f"  Video: {args.video}")
    print(f"  Instruction: \"{args.instruction}\"")
    print(f"  Frames: {args.num_frames}")
    print()

    start = time.time()
    result = compute_topreward(
        video_path=args.video,
        instruction=args.instruction,
        num_frames=args.num_frames,
        model=args.model,
        api_key=args.api_key,
        verbose=True,
    )
    elapsed = time.time() - start

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Progress: {[f'{v:.3f}' for v in result['normalized_progress']]}")
    print(f"  Dense rewards: {[f'{v:.3f}' for v in result['dense_rewards']]}")

    return result


def run_gvl(args):
    from gvl import compute_gvl

    print(f"\n{'='*60}")
    print("Running GVL (Generative Value Learning)")
    print(f"{'='*60}")
    print(f"  Video: {args.video}")
    print(f"  Instruction: \"{args.instruction}\"")
    print(f"  Frames: {args.num_frames}")
    print()

    start = time.time()
    result = compute_gvl(
        video_path=args.video,
        instruction=args.instruction,
        num_frames=args.num_frames,
        model=args.model,
        api_key=args.api_key,
        verbose=True,
    )
    elapsed = time.time() - start

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Progress: {[f'{v:.3f}' for v in result['progress_scores']]}")
    print(f"  VOC: {result['voc']:.4f}")

    return result


def plot_results(top_result, gvl_result, args, save_path=None):
    """Plot progress curves for TOPReward and/or GVL side by side."""
    has_top = top_result is not None
    has_gvl = gvl_result is not None

    fig, axes = plt.subplots(1, 2 if (has_top and has_gvl) else 1, figsize=(14 if (has_top and has_gvl) else 8, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    x = np.linspace(0, 1, args.num_frames)
    ax_idx = 0

    if has_top:
        ax = axes[ax_idx]
        progress = top_result["normalized_progress"]
        ax.plot(x, progress, "o-", color="#E8734A", linewidth=2, markersize=6, label="TOPReward")
        ax.fill_between(x, 0, progress, alpha=0.15, color="#E8734A")
        ax.set_xlabel("Normalized Time", fontsize=12)
        ax.set_ylabel("Progress (0-1)", fontsize=12)
        ax.set_title("TOPReward Progress", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Annotate values
        for i, (xi, yi) in enumerate(zip(x, progress)):
            if i % max(1, len(x) // 5) == 0 or i == len(x) - 1:
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8)
        ax_idx += 1

    if has_gvl:
        ax = axes[ax_idx]
        scores = gvl_result["progress_scores"]
        ax.plot(x, scores, "s-", color="#4A90D9", linewidth=2, markersize=6, label=f"GVL (VOC={gvl_result['voc']:.3f})")
        ax.fill_between(x, 0, scores, alpha=0.15, color="#4A90D9")
        ax.set_xlabel("Normalized Time", fontsize=12)
        ax.set_ylabel("Progress (0-1)", fontsize=12)
        ax.set_title("GVL Progress", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        for i, (xi, yi) in enumerate(zip(x, scores)):
            if i % max(1, len(x) // 5) == 0 or i == len(x) - 1:
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8)

    fig.suptitle(f'Instruction: "{args.instruction}"', fontsize=13, style="italic", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def plot_combined(top_result, gvl_result, args, save_path=None):
    """Plot both methods overlaid on a single axis for direct comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, args.num_frames)

    if top_result:
        progress = top_result["normalized_progress"]
        ax.plot(x, progress, "o-", color="#E8734A", linewidth=2.5, markersize=7, label="TOPReward")

    if gvl_result:
        scores = gvl_result["progress_scores"]
        ax.plot(x, scores, "s--", color="#4A90D9", linewidth=2, markersize=6,
                label=f"GVL (VOC={gvl_result['voc']:.3f})")

    # Ideal progress line
    ax.plot(x, x, ":", color="gray", alpha=0.5, label="Ideal (linear)")

    ax.set_xlabel("Normalized Time", fontsize=12)
    ax.set_ylabel("Progress (0-1)", fontsize=12)
    ax.set_title(f'Progress Comparison — "{args.instruction}"', fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nCombined plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="TOPReward & GVL Demo: Estimate task progress from video"
    )
    parser.add_argument("--video", required=True, help="Path to video file (mp4, avi, etc.)")
    parser.add_argument("--instruction", required=True, help="Task instruction (e.g. 'Pick up the cube')")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to sample (default: 10)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model ID (default: gemini-2.5-flash)")
    parser.add_argument("--api-key", default=None, help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--method", choices=["both", "topreward", "gvl"], default="both",
                        help="Which method(s) to run (default: both)")
    parser.add_argument("--save-plot", default=None, help="Save plot to file instead of showing")
    parser.add_argument("--combined-plot", action="store_true", help="Overlay both methods on one plot")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not args.api_key and not os.environ.get("GOOGLE_API_KEY"):
        print("Error: Set GOOGLE_API_KEY env var or pass --api-key")
        sys.exit(1)

    top_result = None
    gvl_result = None

    if args.method in ("both", "topreward"):
        top_result = run_topreward(args)

    if args.method in ("both", "gvl"):
        gvl_result = run_gvl(args)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if top_result:
        from gvl import compute_voc
        top_voc = compute_voc(top_result["normalized_progress"])
        print(f"  TOPReward VOC: {top_voc:.4f}")
    if gvl_result:
        print(f"  GVL VOC:       {gvl_result['voc']:.4f}")

    # Plot
    if args.combined_plot and top_result and gvl_result:
        plot_combined(top_result, gvl_result, args, save_path=args.save_plot)
    else:
        plot_results(top_result, gvl_result, args, save_path=args.save_plot)


if __name__ == "__main__":
    main()
