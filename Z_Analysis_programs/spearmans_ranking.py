"""
Deductive theme prevalence & reported-order analysis (Spearman's rank)
========================================================================

Works on the 7 deductive intermediate-stage JSON coding files (one per
granularity split), plus a manually-authored CSV giving the order the
7 themes were reported in each final report.

Produces two analyses:

  1. CROSS-SPLIT CONSISTENCY
     For each split, count how many coded entries fall under each of the
     7 themes (prevalence). Compute pairwise Spearman's rho between every
     pair of splits' prevalence vectors -> are the themes ranked the same
     way regardless of how the corpus was chunked?
     Output: theme_prevalence_table.csv, spearman_crosssplit_heatmap.png,
             spearman_crosssplit_rho.csv, spearman_crosssplit_pvalues.csv

  2. REPORTED ORDER vs. PREVALENCE
     For each split, does the order themes were *reported* in (1st, 2nd...)
     track raw prevalence, or something else (e.g. perceived impact)?
     rho close to +1  -> report order just follows frequency.
     rho near 0/neg   -> report order reflects something beyond frequency.
     Output: order_vs_prevalence.csv

INPUT LAYOUT EXPECTED (all in one folder, passed via --dir)
--------------------------------------------------------------
your_folder/
├── <anything>_deductive_<CODE>.json   (7 files; CODE = C, D, S, CD, etc.
│                                        - matched via "deductive_<CODE>.json")
│     Each file: a JSON list of objects, each with (at minimum) a "theme" key,
│     e.g. {"code": "...", "quote": "...", "theme": "Tangible and transparent", ...}
└── theme_order.csv
      Columns: code, rank_1, rank_2, rank_3, rank_4, rank_5, rank_6, rank_7
      - code must match the <CODE> extracted from the JSON filenames
      - rank_1..rank_7 hold the CANONICAL theme strings (must match the
        "theme" values inside the JSON exactly), in the order the themes
        were reported in that split's final report (rank_1 = reported first)

Outputs are written to <dir>/spearman_output/ by default.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


DEFAULT_DIR = "spearmans_rank"
DEFAULT_CSV = "themes_order.csv"
DEFAULT_OUT = "spearman_output"

JSON_NAME_RE = re.compile(r"deductive_([A-Za-z]+)\.json$", re.IGNORECASE)


# ----------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------
def load_json_theme_counts(json_dir: Path):
    """Returns {code: Counter(theme -> count)} and sorted list of all themes seen."""
    results = {}
    all_themes = set()
    files_found = sorted(json_dir.glob("*.json"))
    matched_any = False
    for f in files_found:
        m = JSON_NAME_RE.search(f.name)
        if not m:
            continue
        matched_any = True
        code = m.group(1).upper()
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            sys.exit(f"Could not parse {f.name} as JSON: {e}")
        themes = [item.get("theme", "").strip() for item in data if item.get("theme")]
        if not themes:
            print(f"  [warn] {f.name}: no 'theme' values found")
        counts = Counter(themes)
        results[code] = counts
        all_themes.update(counts.keys())
        print(f"  loaded {f.name} -> split '{code}', {len(data)} entries, {len(counts)} distinct themes")

    if not matched_any:
        sys.exit(
            f"No files matching '*deductive_<CODE>.json' found in {json_dir}.\n"
            f"Files present: {[f.name for f in files_found]}\n"
            f"Adjust JSON_NAME_RE in the script if your naming differs."
        )
    return results, sorted(all_themes)


def build_prevalence_table(theme_counts: dict, all_themes: list) -> pd.DataFrame:
    codes = sorted(theme_counts.keys())
    table = pd.DataFrame(0, index=all_themes, columns=codes)
    for code, counts in theme_counts.items():
        for theme, c in counts.items():
            table.loc[theme, code] = c
    return table


def load_reported_order(csv_path: Path) -> dict:
    """Returns {code: {theme_name: reported_position}} where position 1 = reported first."""
    df = pd.read_csv(csv_path)
    if "code" not in df.columns:
        sys.exit(f"{csv_path} must have a 'code' column")
    rank_cols = [c for c in df.columns if c.lower().startswith("rank_")]
    if not rank_cols:
        sys.exit(f"{csv_path} must have columns named rank_1, rank_2, ... rank_N")
    rank_cols_sorted = sorted(rank_cols, key=lambda x: int(re.search(r"\d+", x).group()))

    order = {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip().upper()
        theme_position = {}
        for i, col in enumerate(rank_cols_sorted, start=1):
            val = row[col]
            if pd.isna(val):
                continue
            theme_position[str(val).strip()] = i
        order[code] = theme_position
    return order, len(rank_cols_sorted)


# ----------------------------------------------------------------------
# ANALYSIS 1: CROSS-SPLIT CONSISTENCY
# ----------------------------------------------------------------------
def cross_split_spearman(prevalence_table: pd.DataFrame):
    codes = prevalence_table.columns.tolist()
    n = len(codes)
    rho_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    for i, c1 in enumerate(codes):
        for j, c2 in enumerate(codes):
            rho, p = spearmanr(prevalence_table[c1], prevalence_table[c2])
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
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# ANALYSIS 2: REPORTED ORDER vs. PREVALENCE
# ----------------------------------------------------------------------
def order_vs_prevalence(prevalence_table: pd.DataFrame, reported_order: dict, n_ranks: int) -> pd.DataFrame:
    records = []
    for code in prevalence_table.columns:
        if code not in reported_order:
            print(f"  [warn] no reported order row for split '{code}' in CSV — skipping")
            continue
        theme_position = reported_order[code]  # theme -> reported position (1 = first)

        themes_in_csv = set(theme_position.keys())
        themes_in_json = set(prevalence_table.index)
        missing = themes_in_csv - themes_in_json
        if missing:
            print(f"  [warn] split '{code}': CSV theme names not found in JSON (check exact spelling): {missing}")

        common = sorted(themes_in_csv & themes_in_json)
        if len(common) < 3:
            print(f"  [warn] split '{code}': only {len(common)} matching themes, skipping (need >=3)")
            continue

        # Prominence: reported 1st -> highest score, reported last -> lowest.
        # This makes rho intuitive: positive rho = prominent themes are also prevalent.
        prominence = [(n_ranks + 1) - theme_position[t] for t in common]
        prevalence = [prevalence_table.loc[t, code] for t in common]

        rho, p = spearmanr(prominence, prevalence)
        records.append({
            "code": code,
            "n_themes_matched": len(common),
            "spearman_rho": round(rho, 3),
            "p_value": round(p, 4),
            "interpretation": (
                "order tracks prevalence" if rho > 0.5 and p < 0.05 else
                "order does NOT clearly track prevalence" if p >= 0.05 else
                "order inversely related to prevalence" if rho < -0.5 else
                "weak/mixed relationship"
            ),
        })
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Spearman analysis of deductive theme prevalence and report order")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Folder containing the 7 JSON files and the CSV")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Filename of the manually-created theme-order CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output subfolder (relative to --dir unless absolute)")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    csv_path = base_dir / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    out_dir = base_dir / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading deductive JSON files from {base_dir}...")
    theme_counts, all_themes = load_json_theme_counts(base_dir)
    print(f"  {len(theme_counts)} splits found: {sorted(theme_counts.keys())}")
    print(f"  {len(all_themes)} distinct themes found across all splits")
    if len(all_themes) != 7:
        print(f"  [note] expected 7 themes, found {len(all_themes)} — check for naming inconsistencies across splits: {all_themes}")

    prevalence_table = build_prevalence_table(theme_counts, all_themes)
    prevalence_path = out_dir / "theme_prevalence_table.csv"
    prevalence_table.to_csv(prevalence_path)
    print(f"  wrote {prevalence_path}")

    print("\nComputing cross-split Spearman correlations...")
    rho_df, p_df = cross_split_spearman(prevalence_table)
    rho_df.to_csv(out_dir / "spearman_crosssplit_rho.csv")
    p_df.to_csv(out_dir / "spearman_crosssplit_pvalues.csv")
    plot_rho_heatmap(
        rho_df,
        "Cross-split theme-prevalence consistency (Spearman's rho)",
        out_dir / "spearman_crosssplit_heatmap.png",
    )

    if not csv_path.exists():
        print(f"\n[skip] Reported-order CSV not found at {csv_path} — "
              f"only cross-split analysis was run. Add the CSV and re-run for the order-vs-prevalence analysis.")
        return

    print(f"\nLoading reported theme order from {csv_path}...")
    reported_order, n_ranks = load_reported_order(csv_path)
    print(f"  {len(reported_order)} splits with reported order, {n_ranks} rank columns")

    print("\nComputing reported-order vs. prevalence Spearman correlations...")
    order_table = order_vs_prevalence(prevalence_table, reported_order, n_ranks)
    order_path = out_dir / "order_vs_prevalence.csv"
    order_table.to_csv(order_path, index=False)
    print(f"  wrote {order_path}")

    print("\nDone. Order vs. prevalence results:")
    print(order_table.to_string(index=False))


if __name__ == "__main__":
    main()