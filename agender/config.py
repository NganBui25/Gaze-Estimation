from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    weights: str = "imagenet"
    pooling: str = "avg"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_dir: Path
    output_dir: Path = Path("artifacts")
    image_size: int = 128
    batch_size: int = 64
    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    max_age: int = 116
    head_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 5e-5
    phase1_epochs: int = 20
    phase2_epochs: int = 10
    fine_tune_layers: int = 40
    freeze_batch_norm: bool = True
    gender_loss_weight: float = 1.0
    age_loss_weight: float = 0.3
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    min_learning_rate: float = 1e-6
    train_shuffle_buffer: int = 2048
    num_parallel_calls: int = field(default=-1)
