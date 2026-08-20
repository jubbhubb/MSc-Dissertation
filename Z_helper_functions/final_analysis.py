"""
Interview corpus vs. summary report analysis
==============================================

Produces:
  1. tfidf_retention_table.csv / .md
       - For each of the 14 reports: how many of the transcript corpus's
         top-N TF-IDF terms are "retained" (present) in that report's own
         top-N TF-IDF terms, expressed as a count and a percentage.
  2. heatmap_inductive_intragroup.png   (7x7 cosine similarity)
  3. heatmap_deductive_intragroup.png   (7x7 cosine similarity)
  4. heatmap_intergroup.png             (7x7 inductive-vs-deductive cosine similarity)

FOLDER STRUCTURE EXPECTED
--------------------------
project/
├── transcripts/
│     *.txt            <- 14 files, any filenames. First 6 lines of each
│                          are stripped automatically (keyword header).
└── reports/
      report_inductive_XX.md   <- XX = granularity code, e.g. C, D, S, CC, CD...
      report_deductive_XX.md
      (14 files total: 7 inductive_*, 7 deductive_*)

Only the "report_(inductive|deductive)_<CODE>.md" naming pattern is required;
CODE can be any run of letters (C/D/S or combinations like CC, DS, etc.)

CONFIGURATION
--------------
Edit the CONFIG block below before running, or pass a config via
command-line: `python analyze_reports.py --transcripts DIR --reports DIR --out DIR --topn 25`
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_TRANSCRIPTS_DIR = "source_files"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_OUT_DIR = "analysis_output"
DEFAULT_TOP_N = 50          # how many top TF-IDF terms define "retention"
HEADER_LINES_TO_STRIP = 6   # transcript keyword header lines to drop

REPORT_NAME_RE = re.compile(
    r"^report_(inductive|deductive)_([A-Za-z]+)(\.\w+)?$", re.IGNORECASE
)

def load_transcript(path: Path, strip_lines: int = HEADER_LINES_TO_STRIP) -> str:
    """Read a transcript .txt file and drop the first N header lines."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[strip_lines:])


def clean_markdown(text: str) -> str:
    """Strip common markdown syntax so it doesn't pollute TF-IDF vocabulary."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    
    text = re.sub(r"`([^`]*)`", r"\1", text)

    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    
    text = re.sub(r"^\s{0,3}([-*_])\1{2,}\s*$", " ", text, flags=re.MULTILINE)
   
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
   
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"[#>*_`~]", " ", text)
    return text


def build_vectorizer(top_n: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # words only, min length 2, drops punctuation/numbers
        max_features=None,
    )


def top_terms_from_matrix(matrix, feature_names, row_index, top_n) -> set:
    """Top-N terms for a single row of a TF-IDF matrix, by weight."""
    row = matrix[row_index].toarray().ravel()
    top_idx = np.argsort(row)[::-1][:top_n]
    return {feature_names[i] for i in top_idx if row[i] > 0}


def top_terms_corpus_mean(matrix, feature_names, top_n) -> set:
    """Top-N terms across an entire corpus, ranked by mean TF-IDF weight."""
    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    top_idx = np.argsort(mean_scores)[::-1][:top_n]
    return {feature_names[i] for i in top_idx if mean_scores[i] > 0}

def load_transcripts(transcripts_dir: Path) -> dict:
    files = sorted(transcripts_dir.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {transcripts_dir}")
    return {f.name: load_transcript(f) for f in files}


def load_reports(reports_dir: Path) -> pd.DataFrame:
    """Returns a DataFrame: filename, group (inductive/deductive), code, raw_text, clean_text"""
    rows = []
    for f in sorted(reports_dir.glob("*.md")):
        m = REPORT_NAME_RE.match(f.name)
        if not m:
            print(f"  [skip] {f.name} does not match report_<inductive|deductive>_<CODE>.md")
            continue
        group, code = m.group(1).lower(), m.group(2).upper()
        raw = f.read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "filename": f.name,
            "group": group,
            "code": code,
            "raw_text": raw,
            "clean_text": clean_markdown(raw),
        })
    if not rows:
        sys.exit(f"No matching report_*.md files found in {reports_dir}")
    df = pd.DataFrame(rows).sort_values(["group", "code"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# ANALYSIS 1: TF-IDF RETENTION TABLE
# ----------------------------------------------------------------------
def compute_retention_table(transcripts: dict, reports_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    t_vectorizer = build_vectorizer(top_n)
    t_matrix = t_vectorizer.fit_transform(transcripts.values())
    t_features = t_vectorizer.get_feature_names_out()
    transcript_top_terms = top_terms_corpus_mean(t_matrix, t_features, top_n)
    r_vectorizer = build_vectorizer(top_n)
    r_matrix = r_vectorizer.fit_transform(reports_df["clean_text"])
    r_features = r_vectorizer.get_feature_names_out()

    records = []
    for i, row in reports_df.iterrows():
        report_top_terms = top_terms_from_matrix(r_matrix, r_features, i, top_n)
        retained = report_top_terms & transcript_top_terms
        records.append({
            "filename": row["filename"],
            "group": row["group"],
            "code": row["code"],
            "top_n": top_n,
            "retained_count": len(retained),
            "retention_pct": round(100 * len(retained) / top_n, 1),
            "retained_terms": ", ".join(sorted(retained)),
        })

    table = pd.DataFrame(records)
    return table, r_matrix, r_features


# ----------------------------------------------------------------------
# ANALYSIS 2: COSINE SIMILARITY HEATMAPS
# ----------------------------------------------------------------------
def plot_heatmap(sim_matrix, row_labels, col_labels, title, out_path):
    print(sim_matrix)
    plt.figure(figsize=(max(6, len(col_labels) * 0.9), max(5, len(row_labels) * 0.8)))
    sns.heatmap(
        sim_matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Cosine similarity"},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


def run_similarity_analysis(reports_df: pd.DataFrame, r_matrix, out_dir: Path):
    sim_full = cosine_similarity(r_matrix)

    ind_idx = reports_df.index[reports_df["group"] == "inductive"].tolist()
    ded_idx = reports_df.index[reports_df["group"] == "deductive"].tolist()

    ind_labels = reports_df.loc[ind_idx, "code"].tolist()
    ded_labels = reports_df.loc[ded_idx, "code"].tolist()

    # Intragroup: inductive vs inductive
    ind_sim = sim_full[np.ix_(ind_idx, ind_idx)]
    plot_heatmap(
        ind_sim, ind_labels, ind_labels,
        "Inductive reports — intragroup cosine similarity",
        out_dir / "heatmap_inductive_intragroup.png",
    )

    # Intragroup: deductive vs deductive
    ded_sim = sim_full[np.ix_(ded_idx, ded_idx)]
    plot_heatmap(
        ded_sim, ded_labels, ded_labels,
        "Deductive reports — intragroup cosine similarity",
        out_dir / "heatmap_deductive_intragroup.png",
    )

    # Intergroup: inductive (rows) vs deductive (cols)
    inter_sim = sim_full[np.ix_(ind_idx, ded_idx)]
    plot_heatmap(
        inter_sim, ind_labels, ded_labels,
        "Inductive vs deductive — intergroup cosine similarity",
        out_dir / "heatmap_intergroup.png",
    )

    return ind_sim, ded_sim, inter_sim

def main():
    parser = argparse.ArgumentParser(description="TF-IDF retention + cosine similarity analysis")
    parser.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS_DIR)
    parser.add_argument("--reports", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--topn", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--strip-lines", type=int, default=HEADER_LINES_TO_STRIP)
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts)
    reports_dir = Path(args.reports)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading transcripts from {transcripts_dir} (stripping first {args.strip_lines} lines each)...")
    transcripts = {
        f.name: load_transcript(f, args.strip_lines)
        for f in sorted(transcripts_dir.glob("*.txt"))
    }
    if not transcripts:
        sys.exit(f"No .txt files found in {transcripts_dir}")
    print(f"  loaded {len(transcripts)} transcripts")

    print(f"Loading reports from {reports_dir}...")
    reports_df = load_reports(reports_dir)
    print(f"  loaded {len(reports_df)} reports "
          f"({(reports_df['group'] == 'inductive').sum()} inductive, "
          f"{(reports_df['group'] == 'deductive').sum()} deductive)")

    print(f"Computing TF-IDF retention (top {args.topn} terms)...")
    retention_table, r_matrix, r_features = compute_retention_table(transcripts, reports_df, args.topn)

    csv_path = out_dir / "tfidf_retention_table.csv"
    md_path = out_dir / "tfidf_retention_table.md"
    retention_table.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(retention_table.drop(columns=["retained_terms"]).to_markdown(index=False))
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")

    print("Computing cosine similarity heatmaps...")
    run_similarity_analysis(reports_df, r_matrix, out_dir)

    print("\nDone. Retention table preview:")
    print(retention_table.drop(columns=["retained_terms"]).to_string(index=False))


if __name__ == "__main__":
    main()