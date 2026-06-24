from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_multicriteria(results_csv: Path, out_png: Path):
    df = pd.read_csv(results_csv, index_col=0)
    cols = ["Conflicts", "Congestion score", "Risk score", "Mean route offset, km"]
    normed = df[cols] / df[cols].max(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=600)
    x = range(len(cols))
    for method in normed.index:
        ax.plot(x, normed.loc[method].values, marker="o", linewidth=1.8, label=method)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Conflicts", "Congestion", "Risk", "Mean offset"], fontsize=8)
    ax.set_ylabel("Normalized value (maximum = 1)", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    ax.set_title("Multi-criteria performance comparison", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="results/CANOPY_TD_results.csv")
    parser.add_argument("--out_png", default="figures/Figure_8_demo.png")
    args = parser.parse_args()
    plot_multicriteria(Path(args.results_csv), Path(args.out_png))

if __name__ == "__main__":
    main()
