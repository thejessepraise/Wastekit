"""
Image Validation Module for WasteKit.

Provides functions to check whether image files are valid, uncorrupted, and readable
by image processing pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image

from wastekit.loader import DatasetInfo, load_dataset


def validate_image(file_path: Union[str, Path], verify_pixels: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Check if an individual file is a valid, readable image.

    Args:
        file_path: Path to image file.
        verify_pixels: If True, attempts to decode pixel data to ensure file is not truncated.

    Returns:
        Tuple of (is_valid: bool, error_reason: Optional[str])
    """
    path = Path(file_path)
    if not path.exists():
        return False, "File does not exist"
    if not path.is_file():
        return False, "Path is not a file"

    try:
        with Image.open(path) as img:
            img.verify()

        if verify_pixels:
            # verify() closes or invalidates the image instance, so re-open to load pixel bytes
            with Image.open(path) as img:
                img.load()

        return True, None
    except Exception as exc:
        return False, str(exc)


@dataclass
class ValidationReport:
    """
    Report containing results of dataset validation.
    """
    total_files_checked: int
    valid_count: int
    corrupt_files: List[Tuple[Path, str]]
    unsupported_files: List[Path]

    @property
    def is_clean(self) -> bool:
        """True if no corrupt files and no unsupported files are present."""
        return len(self.corrupt_files) == 0 and len(self.unsupported_files) == 0

    def __str__(self) -> str:
        lines = [
            "==================================================",
            "            WASTEKIT VALIDATION REPORT            ",
            "==================================================",
            f"Total Checked    : {self.total_files_checked}",
            f"Valid Images     : {self.valid_count}",
            f"Corrupt Files    : {len(self.corrupt_files)}",
            f"Unsupported Files: {len(self.unsupported_files)}",
            f"Status           : {'CLEAN' if self.is_clean else 'ISSUES DETECTED'}",
            "--------------------------------------------------"
        ]

        if self.corrupt_files:
            lines.append("Corrupt Files Detailed:")
            for fp, err in self.corrupt_files[:10]:
                lines.append(f"  - {fp.name}: {err}")
            if len(self.corrupt_files) > 10:
                lines.append(f"  ... and {len(self.corrupt_files) - 10} more")
            lines.append("--------------------------------------------------")

        if self.unsupported_files:
            lines.append("Unsupported Files Detailed:")
            for fp in self.unsupported_files[:10]:
                lines.append(f"  - {fp.name}")
            if len(self.unsupported_files) > 10:
                lines.append(f"  ... and {len(self.unsupported_files) - 10} more")
            lines.append("==================================================")

        return "\n".join(lines)


def validate_dataset(
    dataset_input: Union[str, Path, DatasetInfo],
    verify_pixels: bool = True
) -> ValidationReport:
    """
    Validate all images in a dataset and produce a ValidationReport.

    Args:
        dataset_input: Directory path OR pre-loaded DatasetInfo object.
        verify_pixels: Whether to decode full pixel data for each image.

    Returns:
        ValidationReport containing validation statistics and problem files.
    """
    if isinstance(dataset_input, DatasetInfo):
        info = dataset_input
    else:
        info = load_dataset(dataset_input)

    corrupt_files: List[Tuple[Path, str]] = []
    valid_count = 0

    all_images = info.all_image_paths()
    for file_path in all_images:
        is_valid, err_msg = validate_image(file_path, verify_pixels=verify_pixels)
        if is_valid:
            valid_count += 1
        else:
            corrupt_files.append((file_path, err_msg or "Unknown error"))

    total_checked = len(all_images) + len(info.unsupported_files)

    return ValidationReport(
        total_files_checked=total_checked,
        valid_count=valid_count,
        corrupt_files=corrupt_files,
        unsupported_files=info.unsupported_files
    )
