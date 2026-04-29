from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import ExperimentConfig
from .data import (
    describe_split,
    load_splits,
    make_dataset,
    save_splits,
    scan_utkface_dataset,
    split_dataframe,
)
from .evaluate import evaluate_model
from .models import build_multitask_model, compile_model, configure_fine_tuning


def _build_callbacks(
    run_dir: Path,
    phase_name: str,
    patience: int,
    reduce_lr_patience: int,
    min_lr: float,
):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / f"{phase_name}_best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / f"{phase_name}_history.csv")),
    ]


def _ensure_splits(config: ExperimentConfig):
    split_dir = config.output_dir / "splits"
    existing = load_splits(split_dir)
    if existing is not None:
        return existing

    data = scan_utkface_dataset(config.dataset_dir)
    train_df, val_df, test_df = split_dataframe(
        data=data,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
    )
    save_splits(split_dir, train_df, val_df, test_df)
    return train_df, val_df, test_df


def run_experiment(backbone_name: str, config: ExperimentConfig) -> dict[str, object]:
    tf.keras.utils.set_random_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = _ensure_splits(config)

    train_ds = make_dataset(
        frame=train_df,
        image_size=config.image_size,
        batch_size=config.batch_size,
        training=True,
        shuffle_buffer=config.train_shuffle_buffer,
        seed=config.seed,
    )
    val_ds = make_dataset(
        frame=val_df,
        image_size=config.image_size,
        batch_size=config.batch_size,
        training=False,
        shuffle_buffer=config.train_shuffle_buffer,
        seed=config.seed,
    )
    test_ds = make_dataset(
        frame=test_df,
        image_size=config.image_size,
        batch_size=config.batch_size,
        training=False,
        shuffle_buffer=config.train_shuffle_buffer,
        seed=config.seed,
    )

    run_name = backbone_name.lower()
    run_dir = config.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {
        "train": describe_split(train_df),
        "val": describe_split(val_df),
        "test": describe_split(test_df),
    }
    (run_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2),
        encoding="utf-8",
    )

    model, backbone = build_multitask_model(
        backbone_name=backbone_name,
        image_size=config.image_size,
    )

    compile_model(
        model=model,
        learning_rate=config.head_learning_rate,
        gender_loss_weight=config.gender_loss_weight,
        age_loss_weight=config.age_loss_weight,
    )
    phase1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.phase1_epochs,
        callbacks=_build_callbacks(
            run_dir=run_dir,
            phase_name="phase1",
            patience=config.early_stopping_patience,
            reduce_lr_patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
        ),
        verbose=1,
    )

    configure_fine_tuning(
        backbone=backbone,
        fine_tune_layers=config.fine_tune_layers,
        freeze_batch_norm=config.freeze_batch_norm,
    )
    compile_model(
        model=model,
        learning_rate=config.fine_tune_learning_rate,
        gender_loss_weight=config.gender_loss_weight,
        age_loss_weight=config.age_loss_weight,
    )
    phase2_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.phase2_epochs,
        callbacks=_build_callbacks(
            run_dir=run_dir,
            phase_name="phase2",
            patience=config.early_stopping_patience,
            reduce_lr_patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
        ),
        verbose=1,
    )

    model.save(run_dir / "final_model.keras")

    val_metrics = evaluate_model(model, val_ds, output_path=run_dir / "val_metrics.json")
    test_metrics = evaluate_model(model, test_ds, output_path=run_dir / "test_metrics.json")

    histories = {
        "phase1": pd.DataFrame(phase1_history.history),
        "phase2": pd.DataFrame(phase2_history.history),
    }
    for phase_name, history_df in histories.items():
        history_df.to_csv(run_dir / f"{phase_name}_history_full.csv", index=False)

    combined_summary = {
        "backbone": backbone_name,
        "artifacts_dir": str(run_dir.resolve()),
        "validation": val_metrics,
        "test": test_metrics,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(combined_summary, indent=2),
        encoding="utf-8",
    )

    return combined_summary
