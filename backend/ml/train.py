"""Standalone training CLI used by the Jenkins retrain pipeline.

Usage (from the ``backend`` folder):
    python -m ml.train --reviewed-csv reviewed.csv --out-dir artifacts/models --version v8
"""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

from app.ml_service import TARGETS, MLService, base_dataset

COLUMNS = ["text", "sentiment", "topic", "priority"]
VERSION_PATTERN = re.compile(r"^v\d+$")


def _load_reviewed(csv_path: Path) -> pd.DataFrame:
    # keep_default_na=False so feedback such as "None" or "NA" stays text.
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Reviewed CSV is missing column(s): {', '.join(missing)}")
    frame = frame[COLUMNS]
    if frame.isna().any().any() or (frame["text"].str.strip() == "").any():
        raise ValueError("Reviewed CSV contains rows with empty text or labels")
    return frame


def train_candidate(reviewed_csv: Path | None, out_dir: Path, version: str) -> dict:
    """Train on base data (+ reviewed rows), write joblib files and metrics.json."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"Invalid version '{version}' (expected like v8)")
    frame = base_dataset()
    if reviewed_csv is not None:
        frame = pd.concat([frame, _load_reviewed(Path(reviewed_csv))], ignore_index=True)
    trained, metrics = MLService().train(frame)
    folder = Path(out_dir) / version
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in trained.items():
        joblib.dump(model, folder / f"{name}.joblib")
    result = {
        "version": version,
        "training_samples": int(len(frame)),
        "aggregate_f1": float(sum(metrics[t]["f1_score"] for t in TARGETS) / len(TARGETS)),
        "metrics": metrics,
    }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a candidate model version")
    parser.add_argument("--reviewed-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    try:
        result = train_candidate(args.reviewed_csv, args.out_dir, args.version)
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}")
        return 1
    print(f"Trained {result['version']}: {result['training_samples']} samples, "
          f"aggregate F1 {result['aggregate_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
