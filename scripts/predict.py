"""
scripts/predict.py
==================
Run inference on arbitrary sensor data using a trained CRHNN checkpoint.

Input: a CSV file with 9 sensor columns (AccX, AccY, AccZ, GyrX, GyrY, GyrZ,
EulerX, EulerY, EulerZ), one sample per row.

Output: the same CSV with two extra columns appended:
    • ``predicted_class``  — integer class index
    • ``predicted_label``  — human-readable class name

Run
---
.. code-block:: bash

    python scripts/predict.py --input path/to/sensor.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from kfall.config import FEATURE_COLUMNS, MODELS_DIR, PROCESSED_CSV
from kfall.data.preprocessor import DataPreprocessor
from kfall.models.crhnn import build_crhnn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kfall.predict")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fall-detection inference on sensor CSV data."
    )
    parser.add_argument("--input", required=True, help="Path to input sensor CSV.")
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to save predictions (default: predictions.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_path = MODELS_DIR / "model.h5"
    if not model_path.exists():
        logger.error("No model checkpoint at %s — run train.py first.", model_path)
        sys.exit(1)

    if not PROCESSED_CSV.exists():
        logger.error(
            "Processed CSV not found at %s — run train.py once to build it.", PROCESSED_CSV
        )
        sys.exit(1)

    # ── Build class map from processed data ──────────────────────────────
    ref_df = pd.read_csv(PROCESSED_CSV)
    n_classes = int(ref_df["Class"].nunique())
    class_names = (
        ref_df.drop_duplicates("Class")
        .sort_values("Class")["Description"]
        .tolist()
    )

    # ── Fit scaler on processed data (needed to transform input) ─────────
    prep = DataPreprocessor()
    prep.fit_transform(ref_df, n_classes)   # side-effect: fits scaler

    # ── Load model ────────────────────────────────────────────────────────
    input_shape = (1, len(FEATURE_COLUMNS))
    model = build_crhnn(input_shape=input_shape, n_classes=n_classes)
    model.load_weights(str(model_path))
    logger.info("Loaded model from %s", model_path)

    # ── Load & preprocess input ───────────────────────────────────────────
    input_df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLUMNS if c not in input_df.columns]
    if missing:
        logger.error("Input CSV missing columns: %s", missing)
        sys.exit(1)

    X_raw = input_df[FEATURE_COLUMNS].values.astype(np.float32)
    X = prep.transform(X_raw)

    # ── Predict ───────────────────────────────────────────────────────────
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int)
    labels = [class_names[i] if i < len(class_names) else str(i) for i in preds]

    input_df["predicted_class"] = preds
    input_df["predicted_label"] = labels
    input_df.to_csv(args.output, index=False)
    logger.info("Predictions saved → %s  (%d rows)", args.output, len(input_df))


if __name__ == "__main__":
    main()
