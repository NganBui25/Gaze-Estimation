from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .preprocessing import build_augmentation, decode_image

AUTOTUNE = tf.data.AUTOTUNE


def _is_valid_utkface_name(path: Path) -> bool:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return False
    try:
        int(parts[0])
        gender = int(parts[1])
    except ValueError:
        return False
    return gender in (0, 1)


def scan_utkface_dataset(dataset_dir: Path) -> pd.DataFrame:
    image_paths = sorted(
        path for path in dataset_dir.glob("*.jpg") if _is_valid_utkface_name(path)
    )
    rows = []
    for path in image_paths:
        parts = path.stem.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        age_bin = min(age // 10, 11)
        rows.append(
            {
                "filepath": str(path.resolve()),
                "age": age,
                "gender": gender,
                "age_bin": age_bin,
                "stratify_key": f"{gender}_{age_bin}",
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No valid UTKFace .jpg files were found in {dataset_dir!s}."
        )
    return pd.DataFrame(rows)


def _safe_stratify_labels(frame: pd.DataFrame) -> pd.Series | None:
    counts = frame["stratify_key"].value_counts()
    if counts.empty or counts.min() < 2:
        return None
    return frame["stratify_key"]


def split_dataframe(
    data: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout_size = val_size + test_size
    stratify_all = _safe_stratify_labels(data)

    train_df, holdout_df = train_test_split(
        data,
        test_size=holdout_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_all,
    )

    relative_test_size = test_size / holdout_size
    stratify_holdout = _safe_stratify_labels(holdout_df)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_holdout,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    split_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)


def load_splits(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    if not all(path.exists() for path in (train_path, val_path, test_path)):
        return None
    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def make_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
    shuffle_buffer: int,
    seed: int,
) -> tf.data.Dataset:
    paths = frame["filepath"].astype(str).to_numpy()
    genders = frame["gender"].astype("float32").to_numpy()
    ages = frame["age"].astype("float32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, genders, ages))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(shuffle_buffer, len(frame)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _load(path, gender, age):
        image = decode_image(path, image_size=image_size)
        label = {
            "gender_output": tf.expand_dims(gender, axis=-1),
            "age_output": tf.expand_dims(age, axis=-1),
        }
        return image, label

    dataset = dataset.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        augmentation = build_augmentation()
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def describe_split(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "samples": int(len(frame)),
        "male_ratio": float((frame["gender"] == 0).mean()),
        "female_ratio": float((frame["gender"] == 1).mean()),
        "age_mean": float(frame["age"].mean()),
        "age_std": float(frame["age"].std(ddof=0)),
        "age_min": int(frame["age"].min()),
        "age_max": int(frame["age"].max()),
    }
