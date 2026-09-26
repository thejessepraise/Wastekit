"""
Dataset Analysis Module for WasteKit.

Provides tools for inspecting image counts, class distributions, detecting
class imbalance and empty classes, and gathering image dimension statistics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Optional
from PIL import Image

from wastekit.loader import DatasetInfo, load_dataset


@dataclass
class SummaryReport:
    """
    Summary report of dataset metrics and class balance.
    """
    root_path: Path
    total_images: int
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    empty_classes: List[str]
    imbalance_ratio: float
    is_imbalanced: bool
    image_dimension_stats: Optional[Dict[str, Union[tuple, str, float]]] = field(default=None)

    def __str__(self) -> str:
        lines = [
            "==================================================",
            "             WASTEKIT DATASET SUMMARY             ",
            "==================================================",
            f"Root Path       : {self.root_path}",
            f"Total Classes   : {len(self.class_counts)}",
            f"Total Images    : {self.total_images}",
            "--------------------------------------------------",
            "Class Breakdown :"
        ]

        for cls, count in self.class_counts.items():
            pct = self.class_percentages.get(cls, 0.0)
            lines.append(f"  - {cls:<15}: {count:>6} images ({pct:>5.1f}%)")

        lines.append("--------------------------------------------------")
        if self.empty_classes:
            lines.append(f"Empty Classes   : {', '.join(self.empty_classes)}")
        else:
            lines.append("Empty Classes   : None")

        ratio_str = "Inf" if self.imbalance_ratio == float('inf') else f"{self.imbalance_ratio:.2f}x"
        status_str = "YES (Attention needed)" if self.is_imbalanced else "NO (Balanced)"
        lines.append(f"Imbalance Ratio : {ratio_str}")
        lines.append(f"Class Imbalanced: {status_str}")

        if self.image_dimension_stats:
            lines.append("--------------------------------------------------")
            lines.append("Image Size Statistics (Sampled):")
            stats = self.image_dimension_stats
            lines.append(f"  - Min Dimension: {stats.get('min_size')}")
            lines.append(f"  - Max Dimension: {stats.get('max_size')}")
            lines.append(f"  - Avg Width x Height: {stats.get('avg_width'):.1f} x {stats.get('avg_height'):.1f}")
            lines.append(f"  - Color Modes  : {', '.join(stats.get('modes', []))}")

        lines.append("==================================================")
        return "\n".join(lines)


def dataset_summary(
    dataset_input: Union[str, Path, DatasetInfo],
    sample_dimensions: bool = True,
    sample_limit: int = 100,
    imbalance_threshold: float = 2.0
) -> SummaryReport:
    """
    Generate a summary report for a waste image dataset.

    Args:
        dataset_input: Dataset directory path OR pre-loaded DatasetInfo object.
        sample_dimensions: If True, samples image dimensions using Pillow.
        sample_limit: Maximum number of images to inspect for dimension stats.
        imbalance_threshold: Imbalance ratio threshold above which dataset is marked imbalanced.

    Returns:
        SummaryReport object containing counts, distribution, and imbalance analysis.
    """
    if isinstance(dataset_input, DatasetInfo):
        info = dataset_input
    else:
        info = load_dataset(dataset_input)

    class_counts = {c: info.get_class_count(c) for c in info.classes}
    total_images = info.total_images

    class_percentages = {}
    for c, count in class_counts.items():
        pct = (count / total_images * 100.0) if total_images > 0 else 0.0
        class_percentages[c] = round(pct, 2)

    empty_classes = [c for c, count in class_counts.items() if count == 0]

    non_empty_counts = [count for count in class_counts.values() if count > 0]
    if not non_empty_counts or empty_classes:
        imbalance_ratio = float('inf') if len(non_empty_counts) < len(class_counts) else 1.0
    else:
        max_count = max(non_empty_counts)
        min_count = min(non_empty_counts)
        imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else float('inf')

    is_imbalanced = imbalance_ratio >= imbalance_threshold or len(empty_classes) > 0

    dimension_stats = None
    if sample_dimensions and total_images > 0:
        all_files = info.all_image_paths()
        step = max(1, len(all_files) // sample_limit)
        sampled_files = all_files[::step][:sample_limit]

        widths = []
        heights = []
        modes = set()

        for fpath in sampled_files:
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    modes.add(img.mode)
            except Exception:
                continue

        if widths and heights:
            min_w, max_w = min(widths), max(widths)
            min_h, max_h = min(heights), max(heights)
            dimension_stats = {
                "min_size": (min_w, min_h),
                "max_size": (max_w, max_h),
                "avg_width": sum(widths) / len(widths),
                "avg_height": sum(heights) / len(heights),
                "modes": sorted(list(modes))
            }

    return SummaryReport(
        root_path=info.root_path,
        total_images=total_images,
        class_counts=class_counts,
        class_percentages=class_percentages,
        empty_classes=empty_classes,
        imbalance_ratio=imbalance_ratio,
        is_imbalanced=is_imbalanced,
        image_dimension_stats=dimension_stats
    )
