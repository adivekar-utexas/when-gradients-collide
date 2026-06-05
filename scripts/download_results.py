"""
Download pre-generated experiment results from HuggingFace.

Usage:
    python scripts/download_results.py            # download to ./results/
    python scripts/download_results.py --dir /tmp/results

The HuggingFace dataset contains:
  - Full per-step run logs for all 20 TextGrad-SummEval runs (4 modes x 2 val x 3 seeds)
  - Gradient specificity and feedback adherence evaluation parquets
  - Cherry-pick evaluation parquets (6 variants)
  - Aggregated CSV/plots for paper figures and tables

For the HuggingFace dataset ID, see README.md.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download


HF_REPO_ID: str = "adivekar/when-gradients-collide-results"  # placeholder; update after upload


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pre-generated results from HuggingFace.")
    parser.add_argument(
        "--dir",
        default="results",
        help="Local directory to download results into (default: ./results/).",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help="HuggingFace dataset repo ID (default: %(default)s).",
    )
    parser.add_argument(
        "--subset",
        default=None,
        choices=["full", "aggregates"],
        help="If 'aggregates', download only the small aggregated CSV/parquet "
        "files needed to reproduce paper tables/figures (fast, <1 MB). "
        "If 'full', download all raw per-step run logs (slow, ~1.9 GB). "
        "Default: full.",
    )
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    print(f"Downloading from {args.repo_id} into {args.dir} ...")
    allow_patterns = None
    if args.subset == "aggregates":
        allow_patterns = [
            "*.csv",
            "*.md",
            "*.html",
            "rq3_results/*.parquet",
            "*cherrypick*/*.parquet",
            "run_summary.json",
        ]

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.dir,
        allow_patterns=allow_patterns,
    )
    print(f"Results downloaded to: {path}")


if __name__ == "__main__":
    main()
