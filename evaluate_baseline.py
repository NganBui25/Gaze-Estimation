from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from agender.data import load_splits, make_dataset, scan_utkface_dataset, save_splits, split_dataframe
from agender.evaluate import evaluate_model
from agender.metrics import BinaryF1Score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained age/gender model on UTKFace.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def ensure_splits(args: argparse.Namespace):
    existing = load_splits(args.split_dir)
    if existing is not None:
        return existing

    data = scan_utkface_dataset(args.dataset_dir)
    splits = split_dataframe(
        data=data,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    save_splits(args.split_dir, *splits)
    return splits


def main():
    args = parse_args()
    _, _, test_df = ensure_splits(args)
    test_ds = make_dataset(
        frame=test_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=False,
        shuffle_buffer=2048,
        seed=args.seed,
    )

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"BinaryF1Score": BinaryF1Score},
    )
    metrics = evaluate_model(model, test_ds)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
