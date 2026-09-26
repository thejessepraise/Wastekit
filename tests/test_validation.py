"""Tests for wastekit.validation module."""

from pathlib import Path
import pytest
from PIL import Image

from wastekit.validation import validate_image, validate_dataset, ValidationReport


@pytest.fixture
def mock_validation_dataset(tmp_path):
    dataset_dir = tmp_path / "valid_test_dataset"
    dataset_dir.mkdir()

    glass_dir = dataset_dir / "glass"
    glass_dir.mkdir()

    # Valid image
    img = Image.new("RGB", (60, 60), color="red")
    img.save(glass_dir / "good.jpg")

    # Corrupt image file (truncated/garbage content)
    corrupt_file = glass_dir / "corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_A_REAL_IMAGE_DATA_1234567890")

    # Unsupported file
    (glass_dir / "readme.txt").write_text("ignore me", encoding="utf-8")

    return dataset_dir


def test_validate_image_good(tmp_path):
    good_img_path = tmp_path / "good.png"
    Image.new("RGB", (30, 30)).save(good_img_path)

    is_valid, err = validate_image(good_img_path)
    assert is_valid is True
    assert err is None


def test_validate_image_corrupt(tmp_path):
    bad_img_path = tmp_path / "bad.png"
    bad_img_path.write_bytes(b"corrupt header")

    is_valid, err = validate_image(bad_img_path)
    assert is_valid is False
    assert err is not None


def test_validate_image_nonexistent():
    is_valid, err = validate_image("non_existent_file.png")
    assert is_valid is False
    assert "does not exist" in err


def test_validate_dataset(mock_validation_dataset):
    report = validate_dataset(mock_validation_dataset)

    assert isinstance(report, ValidationReport)
    assert report.valid_count == 1
    assert len(report.corrupt_files) == 1
    assert report.corrupt_files[0][0].name == "corrupt.jpg"
    assert len(report.unsupported_files) == 1
    assert report.is_clean is False

    report_str = str(report)
    assert "WASTEKIT VALIDATION REPORT" in report_str
    assert "corrupt.jpg" in report_str
