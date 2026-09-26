"""
Dataset Splitting Module for WasteKit.

Provides reproducible, stratified splitting of waste datasets into training,
validation, and test sets without copying physical image files.
"""

from dataclasses import dataclass, field
import json
import random
from pathlib import Path
from typing import Dict, List, Union, Optional

from wastekit.loader import DatasetInfo, load_dataset


@dataclass
class DatasetSplit:
    """
    Holds file path mappings for train, validation, and test splits.
    """
    train: Dict[str, List[str]]
    validation: Dict[str, List[str]]
    test: Dict[str, List[str]]
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    @property
    def total_train(self) -> int:
        return sum(len(files) for files in self.train.values())

    @property
    def total_val(self) -> int:
        return sum(len(files) for files in self.validation.values())

    @property
    def total_test(self) -> int:
        return sum(len(files) for files in self.test.values())

    @property
    def total_samples(self) -> int:
        return self.total_train + self.total_val + self.total_test

    def save_manifest(self, output_path: Union[str, Path]) -> Path:
        """
        Save the split index as a JSON manifest file.

        Args:
            output_path: Destination JSON file path.

        Returns:
            Path object to saved manifest.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "train_ratio": self.train_ratio,
                "validation_ratio": self.validation_ratio,
                "test_ratio": self.test_ratio,
                "total_train": self.total_train,
                "total_val": self.total_val,
                "total_test": self.total_test,
                "total_samples": self.total_samples
            },
            "train": self.train,
            "validation": self.validation,
            "test": self.test
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return out_p

    def __str__(self) -> str:
        lines = [
            "==================================================",
            "             WASTEKIT DATASET SPLIT               ",
            "==================================================",
            f"Configured Ratios: Train={self.train_ratio:.0%}, Val={self.validation_ratio:.0%}, Test={self.test_ratio:.0%}",
            f"Total Samples    : {self.total_samples}",
            f"Train Set        : {self.total_train} images",
            f"Validation Set   : {self.total_val} images",
            f"Test Set         : {self.total_test} images",
            "--------------------------------------------------",
            "Per-Class Distribution (Train / Val / Test):"
        ]

        all_classes = sorted(set(self.train.keys()) | set(self.validation.keys()) | set(self.test.keys()))
        for c in all_classes:
            tr = len(self.train.get(c, []))
            va = len(self.validation.get(c, []))
            te = len(self.test.get(c, []))
            tot = tr + va + te
            lines.append(f"  - {c:<15}: Train={tr:>5} | Val={va:>5} | Test={te:>5} (Total={tot})")

        lines.append("==================================================")
        return "\n".join(lines)


def split_dataset(
    dataset_input: Union[str, Path, DatasetInfo],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True
) -> DatasetSplit:
    """
    Split dataset into train, validation, and test subsets.

    Args:
        dataset_input: Directory path OR pre-loaded DatasetInfo object.
        train_ratio: Proportion for training set (0.0 to 1.0).
        validation_ratio: Proportion for validation set (0.0 to 1.0).
        test_ratio: Proportion for test set (0.0 to 1.0).
        seed: Random seed for reproducible shuffling.
        stratify: If True, preserves class distribution ratios within each split.

    Returns:
        DatasetSplit object containing train, validation, and test mappings.

    Raises:
        ValueError: If split ratios do not sum to 1.0 or are negative.
    """
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={validation_ratio}, test={test_ratio} (sum={total_ratio:.4f})"
        )

    if min(train_ratio, validation_ratio, test_ratio) < 0.0:
        raise ValueError("Split ratios cannot be negative.")

    if isinstance(dataset_input, DatasetInfo):
        info = dataset_input
    else:
        info = load_dataset(dataset_input)

    train_dict: Dict[str, List[str]] = {c: [] for c in info.classes}
    val_dict: Dict[str, List[str]] = {c: [] for c in info.classes}
    test_dict: Dict[str, List[str]] = {c: [] for c in info.classes}

    rng = random.Random(seed)

    if stratify:
        for c in info.classes:
            file_paths = [str(p) for p in info.class_to_files.get(c, [])]
            rng.shuffle(file_paths)

            n_total = len(file_paths)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * validation_ratio))

            # Handle edge case where rounding exceeds or undershoots total
            if train_ratio > 0 and n_train == 0 and n_total > 0:
                n_train = 1

            train_dict[c] = file_paths[:n_train]
            val_dict[c] = file_paths[n_train:n_train + n_val]
            test_dict[c] = file_paths[n_train + n_val:]
    else:
        # Non-stratified shuffle across entire dataset
        all_samples = []
        for c, files in info.class_to_files.items():
            for fpath in files:
                all_samples.append((c, str(fpath)))

        rng.shuffle(all_samples)
        n_total = len(all_samples)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * validation_ratio))

        for c, fp in all_samples[:n_train]:
            train_dict[c].append(fp)
        for c, fp in all_samples[n_train:n_train + n_val]:
            val_dict[c].append(fp)
        for c, fp in all_samples[n_train + n_val:]:
            test_dict[c].append(fp)

    return DatasetSplit(
        train=train_dict,
        validation=val_dict,
        test=test_dict,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio
    )
