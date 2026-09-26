"""Tests for wastekit.loader module."""

from pathlib import Path
import pytest
from PIL import Image

from wastekit.loader import load_dataset, DatasetInfo


@pytest.fixture
def mock_dataset(tmp_path):
    """Creates a temporary dataset structure with mock images and unsupported files."""
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir()

    metal_dir = dataset_dir / "metal"
    metal_dir.mkdir()
    plastic_dir = dataset_dir / "plastic"
    plastic_dir.mkdir()
    sub_plastic = plastic_dir / "PET"
    sub_plastic.mkdir()

    # Create dummy images
    img = Image.new("RGB", (50, 50), color="blue")
    img.save(metal_dir / "can1.jpg")
    img.save(metal_dir / "can2.png")
    img.save(sub_plastic / "bottle1.jpeg")

    # Create unsupported non-image file
    (metal_dir / "manifest.json").write_text('{"info": "test"}', encoding="utf-8")

    return dataset_dir


def test_load_dataset_success(mock_dataset):
    info = load_dataset(mock_dataset, recursive=True)

    assert isinstance(info, DatasetInfo)
    assert info.classes == ["metal", "plastic"]
    assert info.total_images == 3
    assert info.get_class_count("metal") == 2
    assert info.get_class_count("plastic") == 1
    assert len(info.unsupported_files) == 1
    assert info.unsupported_files[0].name == "manifest.json"


def test_load_dataset_non_recursive(mock_dataset):
    info = load_dataset(mock_dataset, recursive=False)

    assert info.get_class_count("metal") == 2
    # Sub-folder bottle1.jpeg shouldn't be loaded in non-recursive mode
    assert info.get_class_count("plastic") == 0


def test_load_dataset_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_folder_path_12345")


def test_load_dataset_not_a_directory(tmp_path):
    file_path = tmp_path / "some_file.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_dataset(file_path)


def test_load_dataset_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No class subdirectories found"):
        load_dataset(empty_dir)
