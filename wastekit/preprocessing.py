"""
Image Preprocessing Module for WasteKit.

Provides reusable utilities for image resizing, color conversion, float normalization,
and batch creation for machine learning input pipelines.
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
from PIL import Image


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    size: Tuple[int, int] = (224, 224),
    mode: str = "RGB",
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single image for machine learning workflows.

    Args:

        image_input: File path, PIL Image instance, or NumPy array.
        size: Target size tuple (width, height).
        mode: Target color mode ("RGB", "L", or "RGBA").
        normalize: If True, scales pixel values from [0, 255] to [0.0, 1.0].

    Returns:
        NumPy float32 array representing the preprocessed image.

    Raises:
        ValueError: If size dimensions are invalid or color mode is unsupported.
        FileNotFoundError: If image file path does not exist.
    """
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Target size must be a tuple of 2 positive integers (width, height), got {size}")

    mode = mode.upper()
    if mode not in ("RGB", "L", "RGBA"):
        raise ValueError(f"Unsupported color mode '{mode}'. Use 'RGB', 'L', or 'RGBA'.")

    # Load / convert input to PIL Image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {img_path}")
        img = Image.open(img_path)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        # Convert numpy array to PIL Image
        arr = image_input
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # Convert color mode if needed
    if img.mode != mode:
        img = img.convert(mode)

    # Resize using high quality Lanczos resampling
    resample_method = getattr(Image, "Resampling", Image).LANCZOS
    img_resized = img.resize(size, resample=resample_method)

    # Convert to numpy array
    img_array = np.asarray(img_resized, dtype=np.float32)

    # Normalize pixel values
    if normalize:
        img_array = img_array / 255.0

    return img_array


def preprocess_batch(
    image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
    size: Tuple[int, int] = (224, 224),
    mode: str = "RGB",
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a list of images into a single 4D NumPy array batch.

    Args:
        image_inputs: List of file paths, PIL Images, or arrays.
        size: Target size tuple (width, height).
        mode: Target color mode ("RGB", "L", or "RGBA").
        normalize: If True, scales pixel values from [0, 255] to [0.0, 1.0].

    Returns:
        4D NumPy float32 array of shape (batch_size, height, width, channels).
    """
    if not image_inputs:
        raise ValueError("image_inputs list cannot be empty.")

    processed = [
        preprocess_image(inp, size=size, mode=mode, normalize=normalize)
        for inp in image_inputs
    ]
    return np.stack(processed, axis=0)


def save_preprocessed_image(
    array: np.ndarray,
    output_path: Union[str, Path],
    denormalize: bool = True
) -> Path:
    """
    Save a preprocessed NumPy array back to an image file on disk.

    Args:

        array: NumPy array representing preprocessed image.
        output_path: Destination path.
        denormalize: If True, multiplies array by 255.0 before saving.

    Returns:
        Path object of saved image.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    arr = array.copy()
    if denormalize and arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(out_p)
    return out_p
