"""
WasteKit: A Python library for waste image dataset management and ML preprocessing.
"""

from wastekit.loader import load_dataset, DatasetInfo, DEFAULT_IMAGE_EXTENSIONS
from wastekit.analysis import dataset_summary, SummaryReport
from wastekit.validation import validate_dataset, validate_image, ValidationReport
from wastekit.preprocessing import preprocess_image, preprocess_batch, save_preprocessed_image
from wastekit.splitting import split_dataset, DatasetSplit

__version__ = "0.1.0"

__all__ = [
    "load_dataset",
    "DatasetInfo",
    "DEFAULT_IMAGE_EXTENSIONS",
    "dataset_summary",
    "SummaryReport",
    "validate_dataset",
    "validate_image",
    "ValidationReport",
    "preprocess_image",
    "preprocess_batch",
    "save_preprocessed_image",
    "split_dataset",
    "DatasetSplit",
    "__version__"
]
