"""Baseline pipeline for age and gender prediction on UTKFace."""

from .config import BackboneConfig, ExperimentConfig
from .evaluate import evaluate_model
from .models import AVAILABLE_BACKBONES, build_multitask_model
from .train import run_experiment

__all__ = [
    "AVAILABLE_BACKBONES",
    "BackboneConfig",
    "ExperimentConfig",
    "build_multitask_model",
    "evaluate_model",
    "run_experiment",
]
