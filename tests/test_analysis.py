"""Tests for wastekit.analysis module."""

from pathlib import Path
import pytest
from PIL import Image

from wastekit.loader import load_dataset
from wastekit.analysis import dataset_summary, SummaryReport


@pytest.fixture
def mock_imbalanced_dataset(tmp_path):
    dataset_dir = tmp_path / "imbalanced_dataset"
    dataset_dir.mkdir()

    class_a = dataset_dir / "organic"
    class_a.mkdir()
    class_b = dataset_dir / "hazardous"
    class_b.mkdir()
    class_c = dataset_dir / "paper"
    class_c.mkdir()  # empty class

    # Add 10 images to organic, 2 to hazardous
    for i in range(10):
        img = Image.new("RGB", (100 + i * 10, 100), color="green")
        img.save(class_a / f"img_{i}.jpg")

    for i in range(2):
        img = Image.new("L", (80, 80), color=128)
        img.save(class_b / f"img_{i}.png")

    return dataset_dir


def test_dataset_summary_imbalanced(mock_imbalanced_dataset):
    summary = dataset_summary(mock_imbalanced_dataset)

    assert isinstance(summary, SummaryReport)
    assert summary.total_images == 12
    assert summary.class_counts["organic"] == 10
    assert summary.class_counts["hazardous"] == 2
    assert summary.class_counts["paper"] == 0
    assert "paper" in summary.empty_classes
    assert summary.is_imbalanced is True

    summary_str = str(summary)
    assert "WASTEKIT DATASET SUMMARY" in summary_str
    assert "organic" in summary_str
    assert "paper" in summary_str


def test_dataset_summary_from_dataset_info(mock_imbalanced_dataset):
    info = load_dataset(mock_imbalanced_dataset)
    summary = dataset_summary(info)

    assert summary.total_images == 12
    assert summary.image_dimension_stats is not None
    assert summary.image_dimension_stats["min_size"] == (80, 80)
