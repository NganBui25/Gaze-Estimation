# Baseline Guide

## Files

- `agender/preprocessing.py`: image decode and augmentation
- `agender/data.py`: UTKFace scan, reproducible split, `tf.data` pipeline
- `agender/models.py`: backbone registry and multi-task model factory
- `agender/train.py`: phase 1 freeze + phase 2 fine-tuning
- `agender/evaluate.py`: summary metrics for gender accuracy, gender F1, age MAE
- `train_baseline.py`: train one or many backbones
- `evaluate_baseline.py`: evaluate a saved model on the shared test split

## Train

```powershell
python train_baseline.py `
  --dataset-dir F:\path\to\UTKFace `
  --output-dir artifacts `
  --backbones efficientnetb0 resnet50 mobilenetv3large `
  --image-size 128 `
  --batch-size 64 `
  --phase1-epochs 20 `
  --phase2-epochs 10 `
  --fine-tune-layers 40
```

## Evaluate

```powershell
python evaluate_baseline.py `
  --model-path artifacts\efficientnetb0\final_model.keras `
  --dataset-dir F:\path\to\UTKFace `
  --split-dir artifacts\splits `
  --image-size 128 `
  --batch-size 64
```

## Notes

- The same split in `artifacts/splits/` is reused for all backbones.
- Backbone-specific preprocessing stays inside the model, so the data pipeline remains the same.
- Phase 1 trains only the heads. Phase 2 unfreezes the last `N` backbone layers.
- BatchNorm layers remain frozen in fine-tuning by default for more stable transfer learning.
