"""
Dataset Loading Module for WasteKit.

Provides functionality to inspect, discover, and structure image datasets
organized by class directories.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Set, Union, Optional

DEFAULT_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"
}


@dataclass
class DatasetInfo:
    """
    Structured representation of a loaded dataset.

    Attributes:
        root_path: Absolute path to the dataset root directory.
        classes: Sorted list of detected class names.
        class_to_files: Mapping from class name to list of resolved image Paths.
        total_images: Total count of supported images found.
        unsupported_files: List of non-image or unsupported files found.
    """
    root_path: Path
    classes: List[str] = field(default_factory=list)
    class_to_files: Dict[str, List[Path]] = field(default_factory=dict)
    total_images: int = 0
    unsupported_files: List[Path] = field(default_factory=list)

    def get_class_count(self, class_name: str) -> int:
        """Returns the number of images in a specific class."""
        return len(self.class_to_files.get(class_name, []))

    def all_image_paths(self) -> List[Path]:
        """Returns a flat list of all image paths across all classes."""
        paths = []
        for file_list in self.class_to_files.values():
            paths.extend(file_list)
        return paths

    def __repr__(self) -> str:
        class_summary = ", ".join(f"{c}: {len(self.class_to_files[c])}" for c in self.classes)
        return (
            f"DatasetInfo(root='{self.root_path}', "
            f"total_images={self.total_images}, "
            f"classes=[{class_summary}])"
        )


def load_dataset(
    dataset_path: Union[str, Path],
    recursive: bool = True,
    extensions: Optional[Set[str]] = None
) -> DatasetInfo:
    """
    Load an image dataset organized into class directories.

    Args:

        dataset_path: Path to the dataset directory.
        recursive: Whether to recursively scan nested subdirectories within class folders.
        extensions: Set of lower-case file extensions considered valid images.

    Returns:
        DatasetInfo object containing dataset structure, file mappings, and stats.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        NotADirectoryError: If dataset_path is not a directory.
        ValueError: If no valid class folders are found in dataset_path.
    """
    root = Path(dataset_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    valid_extensions = {ext.lower() for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS)}

    # Top-level directories inside dataset_path are classes
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in dataset path: {root}")

    classes = sorted([d.name for d in class_dirs])
    class_to_files: Dict[str, List[Path]] = {c: [] for c in classes}
    unsupported_files: List[Path] = []
    total_images = 0

    for class_dir in sorted(class_dirs):
        c_name = class_dir.name

        if recursive:
            file_iterator = class_dir.rglob("*")
        else:
            file_iterator = class_dir.glob("*")

        for file_path in file_iterator:
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in valid_extensions:
                    class_to_files[c_name].append(file_path)
                    total_images += 1
                else:
                    unsupported_files.append(file_path)

        # Sort file paths for reproducibility
        class_to_files[c_name].sort()

    return DatasetInfo(
        root_path=root,
        classes=classes,
        class_to_files=class_to_files,
        total_images=total_images,
        unsupported_files=sorted(unsupported_files)
    )
