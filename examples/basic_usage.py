"""
Basic Usage Example for WasteKit Python Library.

This example demonstrates how to load, analyze, validate, preprocess,
and split a waste classification dataset using WasteKit.
"""

from pathlib import Path
import os
import sys

# Ensure wastekit is in Python path when running example directly
package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from wastekit import (
    load_dataset,
    dataset_summary,
    validate_dataset,
    preprocess_image,
    preprocess_batch,
    split_dataset
)


def main():
    # Locate dataset directory relative to project root
    project_root = package_root.parent
    dataset_path = project_root / "dataset"

    if not dataset_path.exists():
        print(f"Dataset path '{dataset_path}' not found. Please specify a valid dataset path.")
        return

    print("==================================================")
    print("      WASTEKIT FEATURE DEMONSTRATION & USAGE      ")
    print("==================================================")

    # 1. Load Dataset
    print("\n[1] Loading Dataset...")
    info = load_dataset(dataset_path, recursive=True)
    print(f"Loaded dataset from: {info.root_path}")
    print(f"Detected Classes   : {info.classes}")
    print(f"Total Images Found : {info.total_images}")
    print(f"Unsupported Files  : {len(info.unsupported_files)}")

    # 2. Analyze Dataset
    print("\n[2] Generating Dataset Summary...")
    summary = dataset_summary(info, sample_dimensions=True, sample_limit=50)
    print(summary)

    # 3. Validate Images
    print("\n[3] Validating Dataset Files...")
    val_report = validate_dataset(info, verify_pixels=True)
    print(val_report)

    # 4. Image Preprocessing
    print("\n[4] Preprocessing Sample Images...")
    all_images = info.all_image_paths()
    if all_images:
        sample_path = all_images[0]
        print(f"Preprocessing single image: {sample_path.name}")
        img_array = preprocess_image(sample_path, size=(224, 224), mode="RGB", normalize=True)
        print(f"Output array shape  : {img_array.shape}")
        print(f"Output array dtype  : {img_array.dtype}")
        print(f"Pixel range         : [{img_array.min():.3f}, {img_array.max():.3f}]")

        # Batch Preprocessing Example
        sample_batch_paths = all_images[:4]
        print(f"\nPreprocessing batch of {len(sample_batch_paths)} images...")
        batch_array = preprocess_batch(sample_batch_paths, size=(224, 224), mode="RGB", normalize=True)
        print(f"Batch array shape   : {batch_array.shape}")

    # 5. Dataset Splitting
    print("\n[5] Performing Stratified Split (70% Train, 15% Val, 15% Test)...")
    split = split_dataset(info, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, seed=42)
    print(split)

    # Save split manifest
    manifest_out = package_root / "split_manifest.json"
    saved_p = split.save_manifest(manifest_out)
    print(f"Saved split manifest to: {saved_p}")

    print("\nDemonstration complete successfully!")


if __name__ == "__main__":
    main()
