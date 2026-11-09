"""Plot a Needle-in-a-Haystack grid as a heatmap, one panel per variant.

Reads the CSV written by benchmarks/run_needle.py and draws context length
across the x-axis against needle depth down the y-axis, coloured by the
fraction of repeats that recovered the needle.

The styling follows the convention the needle-in-a-haystack literature settled
on: red for a lost needle through amber to green for a recovered one, white
cell borders, one shared colorbar.

Run:  python scripts/plot_needle.py [results/needle/needle_b1024_L16384.csv]
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
    "needle", ["#C8384B", "#E8743B", "#EBB839", "#9CC95C", "#1FA97B"])

LABEL = {"k0": "No retrieval\nevict only",
         "hold": "Scheduled retrieval\nevery 16 decode steps",
         "prefill": "Retrieval from prefill\nsynchronous, before token 1"}
ORDER = ("k0", "hold", "prefill")

INK = "#2B2B2B"
MUTED = "#6E6E6E"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#D5D5D5",
        "axes.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/needle/needle_b1024_L16384.csv"
    df = pd.read_csv(path)
    style()

    variants = [v for v in ORDER if v in set(df["variant"])]
    variants += [v for v in sorted(set(df["variant"])) if v not in variants]

    lengths = sorted(df["length"].unique())
    depths = sorted(df["depth"].unique())
    budget = os.path.basename(path).split("_b")[-1].split("_")[0]
    repeats = int(df.groupby(["variant", "length", "depth"]).size().max())

    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.8, 4.9), squeeze=False)
    fig.subplots_adjust(wspace=0.22, top=0.74, bottom=0.17, left=0.055, right=0.90)

    for col, (ax, variant) in enumerate(zip(axes[0], variants)):
        sub = df[df["variant"] == variant]
        grid = (sub.pivot_table(index="depth", columns="length", values="hit", aggfunc="mean")
                   .reindex(index=depths, columns=lengths))
        im = ax.imshow(grid.values, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")

        # White borders between cells, drawn as minor gridlines so they sit on
        # the cell edges rather than through their centres.
        ax.set_xticks(np.arange(-0.5, len(lengths), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", length=0)
        ax.set_axisbelow(False)

        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f"{x//1024}k" if x >= 1024 else str(x) for x in lengths],
                           fontsize=9)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{d:g}%" for d in depths], fontsize=8.5)
        ax.tick_params(axis="both", which="major", length=0, pad=5)
        for s in ax.spines.values():
            s.set_visible(False)

        ax.set_xlabel("Context length", fontsize=9.5, labelpad=8)
        if col == 0:
            ax.set_ylabel("Needle depth in document", fontsize=9.5, labelpad=8)

        head, _, sub_t = LABEL.get(variant, variant).partition("\n")
        ax.set_title(head, fontsize=12, color=INK, fontweight="600", pad=26)
        ax.text(0.5, 1.065, sub_t, transform=ax.transAxes, ha="center",
                fontsize=8.5, color=MUTED)
        ax.text(0.5, 1.015, f"mean recovery {sub['hit'].mean():.2f}",
                transform=ax.transAxes, ha="center", fontsize=9,
                color=INK, fontweight="600")

    cax = fig.add_axes([0.915, 0.17, 0.011, 0.57])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["lost", "50%", "found"], fontsize=8.5, color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    fig.suptitle("Needle-in-a-Haystack: what the CPU tier recovers",
                 fontsize=16, fontweight="600", color=INK, x=0.055, y=0.955,
                 ha="left")
    fig.text(0.055, 0.885,
             f"Llama-3.1-8B  ·  KV budget {budget} tokens  ·  {repeats} repeats per cell  "
             f"·  colour is the fraction recovered",
             fontsize=9.5, color=MUTED, ha="left")

    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.3)
    print("wrote", out)

    print("\nMean score by variant and length:")
    print(df.groupby(["variant", "length"])["hit"].mean().unstack().round(3).to_string())
    base = df[df.variant == "k0"].set_index(["length", "depth", "repeat"])["hit"]
    for other in ("hold", "prefill"):
        if other not in set(df["variant"]) or "k0" not in set(df["variant"]):
            continue
        o = df[df.variant == other].set_index(["length", "depth", "repeat"])["hit"]
        both = pd.concat({"k0": base, "o": o}, axis=1).dropna()
        won = int(((both.k0 == 0) & (both.o == 1)).sum())
        lost = int(((both.k0 == 1) & (both.o == 0)).sum())
        print(f"{other:8} vs no retrieval: won back {won}, lost {lost}, of {len(both)} cells")


if __name__ == "__main__":
    main()
