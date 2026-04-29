from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def _flatten_batches(dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    genders = []
    ages = []
    for _, labels in dataset:
        genders.append(labels["gender_output"].numpy().reshape(-1))
        ages.append(labels["age_output"].numpy().reshape(-1))
    return np.concatenate(genders), np.concatenate(ages)


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denominator = (2 * tp) + fp + fn
    if denominator == 0:
        return 0.0
    return float((2 * tp) / denominator)


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    output_path: Path | None = None,
) -> dict[str, float]:
    keras_metrics = model.evaluate(dataset, verbose=0, return_dict=True)
    raw_predictions = model.predict(dataset, verbose=0)
    if isinstance(raw_predictions, dict):
        predictions = raw_predictions
    else:
        predictions = dict(zip(model.output_names, raw_predictions))

    true_gender, true_age = _flatten_batches(dataset)
    pred_gender = (predictions["gender_output"].reshape(-1) >= 0.5).astype(np.float32)
    pred_age = predictions["age_output"].reshape(-1)

    summary = {
        **{key: float(value) for key, value in keras_metrics.items()},
        "gender_accuracy": float(np.mean(true_gender == pred_gender)),
        "gender_f1": _binary_f1(true_gender, pred_gender),
        "age_mae": float(np.mean(np.abs(true_age - pred_age))),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
