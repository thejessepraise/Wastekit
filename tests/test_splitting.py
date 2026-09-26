"""Tests for wastekit.splitting module."""

import json
from pathlib import Path
import pytest
from PIL import Image

from wastekit.splitting import split_dataset, DatasetSplit


@pytest.fixture
def mock_dataset_for_split(tmp_path):
    dataset_dir = tmp_path / "split_dataset"
    dataset_dir.mkdir()

    for cls_name, count in [("cardboard", 20), ("metal", 10)]:
        cls_dir = dataset_dir / cls_name
        cls_dir.mkdir()
        for i in range(count):
            img = Image.new("RGB", (20, 20), color="white")
            img.save(cls_dir / f"img_{i}.jpg")

    return dataset_dir


def test_split_dataset_stratified(mock_dataset_for_split):
    split = split_dataset(
        mock_dataset_for_split,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        stratify=True
    )

    assert isinstance(split, DatasetSplit)
    assert split.total_samples == 30
    assert split.total_train == 21  # 14 cardboard + 7 metal
    assert split.total_val == 5    # 3 cardboard + 2 metal
    assert split.total_test == 4   # 3 cardboard + 1 metal

    assert len(split.train["cardboard"]) == 14
    assert len(split.train["metal"]) == 7

    split_str = str(split)
    assert "WASTEKIT DATASET SPLIT" in split_str
    assert "cardboard" in split_str


def test_split_dataset_invalid_ratios(mock_dataset_for_split):
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        split_dataset(mock_dataset_for_split, train_ratio=0.8, validation_ratio=0.3, test_ratio=0.1)


def test_save_manifest(mock_dataset_for_split, tmp_path):
    split = split_dataset(mock_dataset_for_split, seed=42)
    manifest_path = tmp_path / "split_manifest.json"

    saved_path = split.save_manifest(manifest_path)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert data["metadata"]["total_samples"] == 30
    assert "train" in data
    assert "validation" in data
    assert "test" in data
