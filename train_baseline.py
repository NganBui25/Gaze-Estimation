from __future__ import annotations

import argparse
import json
from pathlib import Path

from agender.config import ExperimentConfig
from agender.models import AVAILABLE_BACKBONES
from agender.train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a two-phase multi-task age/gender baseline on UTKFace."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to UTKFace images.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["efficientnetb0", "resnet50", "mobilenetv3large"],
        help="Backbones to train. Use names from the available registry.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    requested_backbones = [name.lower() for name in args.backbones]
    unknown = sorted(set(requested_backbones) - set(AVAILABLE_BACKBONES))
    if unknown:
        supported = ", ".join(sorted(AVAILABLE_BACKBONES))
        raise ValueError(f"Unsupported backbones: {', '.join(unknown)}. Available: {supported}")

    config = ExperimentConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        fine_tune_layers=args.fine_tune_layers,
        seed=args.seed,
    )

    summaries = []
    for backbone_name in requested_backbones:
        summary = run_experiment(backbone_name=backbone_name, config=config)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    summary_path = config.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
