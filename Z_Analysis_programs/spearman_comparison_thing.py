"""
Cross-split REPORTED-ORDER consistency (Spearman's rank)
===========================================================
Compares the ORDER themes were reported in (theme_order.csv) between every
pair of deductive granularity splits — independent of how prevalent each
theme actually was. This is distinct from spearman_crosssplit_heatmap.png
(which compares prevalence *counts* between splits) and from
order_vs_prevalence.csv (which compares a split's own order against its
own prevalence). This script answers a third question:

    "Did split CC and split CS report the 7 themes in the same order
     as each other?"

rho close to +1  -> the two splits foregrounded themes in essentially the
                     same order, regardless of how often each theme appeared
rho near 0 / neg -> the splits disagreed on which themes to lead with

INPUT (same folder as before)
-------------------------------
theme_order.csv
    Columns: code, rank_1, rank_2, ... rank_7 (rank_1 = reported first)
    Theme-name spelling is normalized (case/whitespace-insensitive) before
    matching between splits, so minor inconsistencies (e.g. "Consistent "
    vs "Consistent") don't break the comparison. Anything that still can't
    be reconciled between two splits is reported as a warning and those
    themes are simply excluded from that pair's correlation.

OUTPUT (written to --out, default spearman_output/)
------------------------------------------------------
spearman_crosssplit_order_rho.csv
spearman_crosssplit_order_pvalues.csv
spearman_crosssplit_order_heatmap.png
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


DEFAULT_DIR = "spearmans_rank"
DEFAULT_CSV = "themes_order.csv"
DEFAULT_OUT = "spearman_output"


def normalize_theme(s: str) -> str:
    """Whitespace-collapsed, casefolded form used only for MATCHING themes
    across splits — does not change what gets displayed or saved."""
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def load_reported_order(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "code" not in df.columns:
        sys.exit(f"{csv_path} must have a 'code' column")
    rank_cols = [c for c in df.columns if c.lower().startswith("rank_")]
    if not rank_cols:
        sys.exit(f"{csv_path} must have columns named rank_1, rank_2, ... rank_N")
    rank_cols_sorted = sorted(rank_cols, key=lambda x: int(re.search(r"\d+", x).group()))

    order = {}         # code -> {normalized_theme: reported_position}
    display = {}       # normalized_theme -> first-seen raw spelling (for readable warnings)
    for _, row in df.iterrows():
        code = str(row["code"]).strip().upper()
        theme_rank = {}
        for i, col in enumerate(rank_cols_sorted, start=1):
            val = row[col]
            if pd.isna(val):
                continue
            raw = str(val).strip()
            norm = normalize_theme(raw)
            theme_rank[norm] = i
            display.setdefault(norm, raw)
        order[code] = theme_rank
    return order, display, len(rank_cols_sorted)


def cross_split_order_spearman(order: dict, display: dict):
    codes = sorted(order.keys())
    n = len(codes)
    rho_mat = np.full((n, n), np.nan)
    p_mat = np.full((n, n), np.nan)

    for i, c1 in enumerate(codes):
        for j, c2 in enumerate(codes):
            themes1, themes2 = order[c1], order[c2]
            common = sorted(set(themes1) & set(themes2))
            missing = (set(themes1) ^ set(themes2))
            if missing and i < j:  # only report each pair once
                readable = [display.get(m, m) for m in missing]
                print(f"  [warn] {c1} vs {c2}: themes not present in both after normalization: {readable}")
            if len(common) < 3:
                print(f"  [warn] {c1} vs {c2}: only {len(common)} common themes, skipping (need >=3)")
                continue
            ranks1 = [themes1[t] for t in common]
            ranks2 = [themes2[t] for t in common]
            rho, p = spearmanr(ranks1, ranks2)
            rho_mat[i, j] = rho
            p_mat[i, j] = p

    rho_df = pd.DataFrame(rho_mat, index=codes, columns=codes)
    p_df = pd.DataFrame(p_mat, index=codes, columns=codes)
    return rho_df, p_df


def plot_rho_heatmap(rho_df, title, out_path):
    plt.figure(figsize=(max(6, len(rho_df) * 0.9), max(5, len(rho_df) * 0.8)))
    sns.heatmap(
        rho_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        square=True, cbar_kws={"label": "Spearman's rho"},
    )
    plt.title(title)
    plt.xlabel("Split")
    plt.ylabel("Split")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-split reported-order consistency (Spearman)")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Folder containing theme_order.csv")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output subfolder")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    csv_path = base_dir / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    out_dir = base_dir / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"Could not find {csv_path}")

    print(f"Loading reported order from {csv_path}...")
    order, display, n_ranks = load_reported_order(csv_path)
    print(f"  {len(order)} splits loaded: {sorted(order.keys())}")

    print("\nComputing cross-split reported-order Spearman correlations...")
    rho_df, p_df = cross_split_order_spearman(order, display)

    rho_path = out_dir / "spearman_crosssplit_order_rho.csv"
    p_path = out_dir / "spearman_crosssplit_order_pvalues.csv"
    rho_df.to_csv(rho_path)
    p_df.to_csv(p_path)
    print(f"  wrote {rho_path}")
    print(f"  wrote {p_path}")

    plot_rho_heatmap(
        rho_df,
        "Cross-split reported-order consistency (Spearman's rho)",
        out_dir / "spearman_crosssplit_order_heatmap.png",
    )

    print("\nDone. rho matrix:")
    print(rho_df.round(2).to_string())


if __name__ == "__main__":
    main()