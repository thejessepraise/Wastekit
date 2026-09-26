"""Tests for wastekit.preprocessing module."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from wastekit.preprocessing import (
    preprocess_image,
    preprocess_batch,
    save_preprocessed_image
)


def test_preprocess_image_from_file(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (400, 300), color=(255, 128, 0))
    img.save(img_path)

    arr = preprocess_image(img_path, size=(224, 224), mode="RGB", normalize=True)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (224, 224, 3)  # height, width, channels
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_preprocess_image_grayscale(tmp_path):
    img_path = tmp_path / "gray.png"
    Image.new("RGB", (100, 100), color=(200, 200, 200)).save(img_path)

    arr = preprocess_image(img_path, size=(128, 128), mode="L", normalize=False)

    assert arr.shape == (128, 128)
    assert arr.dtype == np.float32
    assert arr.max() > 1.0  # Non-normalized values [0, 255]


def test_preprocess_batch(tmp_path):
    img1 = Image.new("RGB", (50, 50), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")

    batch = preprocess_batch([img1, img2], size=(224, 224), mode="RGB", normalize=True)

    assert batch.shape == (2, 224, 224, 3)
    assert batch.dtype == np.float32


def test_save_preprocessed_image(tmp_path):
    arr = np.random.rand(100, 100, 3).astype(np.float32)
    out_file = tmp_path / "saved.png"

    saved_path = save_preprocessed_image(arr, out_file, denormalize=True)

    assert saved_path.exists()
    with Image.open(saved_path) as loaded:
        assert loaded.size == (100, 100)


def test_preprocess_image_invalid_inputs():
    with pytest.raises(ValueError, match="Target size must be a tuple"):
        preprocess_image(Image.new("RGB", (10, 10)), size=(0, -5))

    with pytest.raises(ValueError, match="Unsupported color mode"):
        preprocess_image(Image.new("RGB", (10, 10)), mode="INVALID_MODE")

    with pytest.raises(FileNotFoundError):
        preprocess_image("non_existent_image_file.jpg")
