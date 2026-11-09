"""Plot a Needle-in-a-Haystack grid as a heatmap, one panel per variant.

Reads the CSV written by benchmarks/run_needle.py and draws context length
across the x-axis against needle depth down the y-axis, coloured by the
fraction of repeats that recovered the needle.

Run:  python scripts/plot_needle.py [results/needle_b1024_L16384.csv]
Writes <input>.png next to the CSV.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Red at 0 through amber to green at 1, so a lost needle reads as a warning
# rather than as "low but fine".
CMAP = LinearSegmentedColormap.from_list(
    "needle", ["#C0392B", "#E8A33D", "#D8C13A", "#4FA96B", "#2E9E6B"])

LABEL = {"k0": "no retrieval (evict only)", "hold": "held retrieval (CPU tier on)"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/needle_b1024_L16384.csv"
    df = pd.read_csv(path)
    variants = [v for v in ("k0", "hold") if v in set(df["variant"])]
    variants += [v for v in sorted(set(df["variant"])) if v not in variants]

    lengths = sorted(df["length"].unique())
    depths = sorted(df["depth"].unique())
    budget = os.path.basename(path).split("_b")[-1].split("_")[0]

    fig, axes = plt.subplots(1, len(variants), figsize=(7.0 * len(variants), 4.2),
                             squeeze=False)
    for ax, variant in zip(axes[0], variants):
        sub = df[df["variant"] == variant]
        grid = (sub.pivot_table(index="depth", columns="length", values="hit", aggfunc="mean")
                   .reindex(index=depths, columns=lengths))
        im = ax.imshow(grid.values, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f"{x:,}" for x in lengths], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{d:g}" for d in depths], fontsize=8)
        ax.set_xlabel("Context length (tokens)", fontsize=9)
        ax.set_ylabel("Needle depth (%)", fontsize=9)
        ax.set_title(LABEL.get(variant, variant), fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(lengths), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
        ax.grid(which="minor", color="#888888", linewidth=0.4)
        ax.tick_params(which="minor", length=0)

    fig.colorbar(im, ax=axes[0].tolist(), label="Needle recovered", fraction=0.025, pad=0.02)
    fig.suptitle(f"Needle-in-a-Haystack, Llama-3.1-8B, hot/cold at KV budget {budget}",
                 fontsize=11)

    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)

    print("\nMean score by variant and length:")
    print(df.groupby(["variant", "length"])["hit"].mean().unstack().round(3).to_string())
    if {"k0", "hold"} <= set(df["variant"]):
        k0 = df[df.variant == "k0"].set_index(["length", "depth", "repeat"])["hit"]
        hd = df[df.variant == "hold"].set_index(["length", "depth", "repeat"])["hit"]
        both = pd.concat({"k0": k0, "hold": hd}, axis=1).dropna()
        won = int(((both.k0 == 0) & (both.hold == 1)).sum())
        lost = int(((both.k0 == 1) & (both.hold == 0)).sum())
        print(f"\nretrieval won back {won} needles, lost {lost}, out of {len(both)} cells")


if __name__ == "__main__":
    main()
