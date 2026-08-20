"""Run the documented forestry analysis with an explicitly supplied CSV file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.classifier import CropClassifier
from src.config import DEFAULT_CROP_COLUMN, DEFAULT_FEATURES, RESULTS_DIR
from src.data_loader import DataLoader
from src.predictor import CropPredictor
from src.stat_analyzer import StatAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse the reproducible, file-system-independent command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run exploratory statistics plus regression and suitability "
            "classification for one forestry crop column."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file containing the original or already-English column names.",
    )
    parser.add_argument(
        "--crop",
        default=DEFAULT_CROP_COLUMN,
        help=f"Production target column after mapping (default: {DEFAULT_CROP_COLUMN}).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="Candidate soil/climate feature columns after mapping.",
    )
    parser.add_argument(
        "--encoding",
        default="cp949",
        help="Preferred input encoding; UTF-8 fallbacks are attempted automatically.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for CSV analysis tables and the JSON metric summary.",
    )
    return parser.parse_args()


def numeric_metrics(result: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    """Keep only JSON-safe numeric metrics from a model result dictionary."""
    return {key: float(result[key]) for key in keys if key in result}


def main() -> None:
    """Load input, run the maintained workflow, and save inspectable outputs."""
    args = parse_args()
    input_path = args.input_csv.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input CSV was not found: {input_path}")

    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data = DataLoader(input_path, encoding=args.encoding).load_and_transform()
    if args.crop not in data.columns:
        available = ", ".join(map(str, data.columns))
        raise SystemExit(
            f"Crop column '{args.crop}' was not found after mapping. "
            f"Available columns: {available}"
        )

    data[args.crop] = pd.to_numeric(data[args.crop], errors="coerce")
    data = data.dropna(subset=[args.crop]).copy()
    if data.empty:
        raise SystemExit(f"Crop column '{args.crop}' has no numeric observations.")

    features = [feature for feature in args.features if feature in data.columns]
    if not features:
        raise SystemExit(
            "None of --features exist after mapping. Supply headers from your input CSV."
        )

    analyzer = StatAnalyzer(data)
    chi_square, anova, spearman = analyzer.perform_analysis(args.crop)
    chi_square.to_csv(results_dir / f"{args.crop}_chi_square.csv")
    anova.to_csv(results_dir / f"{args.crop}_anova.csv")
    spearman.to_csv(results_dir / f"{args.crop}_spearman.csv")

    regression = CropPredictor(data).train_regression(args.crop, features)
    classification = CropClassifier(data).train_svm(args.crop, features)

    summary = {
        "input_csv": str(input_path),
        "crop_column": args.crop,
        "features_used": features,
        "statistics": {
            "chi_square_rows": int(len(chi_square)),
            "anova_rows": int(len(anova)),
            "spearman_rows": int(len(spearman)),
        },
        "regression": numeric_metrics(
            regression, ("r2", "mse", "cv_r2_mean", "n_rows", "n_features")
        ),
        "classification": numeric_metrics(
            classification, ("auc", "cv_auc_mean", "n_rows", "n_features")
        ),
    }
    summary_path = results_dir / f"{args.crop}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Saved analysis tables and model summary to %s", results_dir)


if __name__ == "__main__":
    main()
