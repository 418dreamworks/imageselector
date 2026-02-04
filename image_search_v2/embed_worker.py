#!/usr/bin/env python3
"""Stateless embedding worker. Takes images, outputs numpy arrays.

This script has NO access to databases or FAISS indexes.
It simply reads images and produces embeddings as numpy files.
Safe to run on any machine (macOS, Windows, Linux).

Usage:
    python embed_worker.py --input /path/to/batch --output /path/to/results
    python embed_worker.py --input ~/embed_work/batch_001  # uses default output

Input folder structure:
    batch/
    ├── manifest.json      # [[listing_id, image_id], ...]
    ├── listings.json      # {"listing_id": "title. materials", ...}
    └── images/
        ├── 123_456.jpg
        └── ...

Output folder structure:
    results/
    ├── manifest.json           # copy from input (for verification)
    ├── clip_vitb32.npy         # (N, 512) float32
    ├── clip_vitb32_text.npy    # (N, 512) float32
    ├── clip_vitl14.npy         # (N, 768) float32
    ├── clip_vitl14_text.npy    # (N, 768) float32
    ├── dinov2_base.npy         # (N, 768) float32
    ├── dinov2_large.npy        # (N, 1024) float32
    └── dinov3_base.npy         # (N, 768) float32
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import MODELS, get_loader


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_work_dir() -> Path:
    """Get OS-appropriate default work directory."""
    if platform.system() == "Windows":
        return Path("E:/embed_work")
    else:
        return Path.home() / "embed_work"


def load_all_images(manifest: list, images_dir: Path) -> list:
    """Load all images into RAM upfront.

    This is optimal for HDD machines - one sequential read pass,
    then all processing happens from memory.
    """
    print(f"Loading {len(manifest)} images into memory...")
    images = []

    for lid, iid in tqdm(manifest, desc="Loading images"):
        img_path = images_dir / f"{lid}_{iid}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
            # Force load into memory (PIL is lazy by default)
            img.load()
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            # Placeholder for failed images
            images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    print(f"Loaded {len(images)} images into memory")
    return images


def process_model(
    model_key: str,
    manifest: list,
    images: list,
    listings: dict,
    output_dir: Path,
    batch_size: int = 32,
):
    """Process a single model, save numpy output."""
    loader = get_loader()
    model_info = MODELS[model_key]
    is_clip = model_info["library"] == "open_clip"

    print(f"\n{'='*50}")
    print(f"Model: {model_key} (dim={model_info['dim']})")
    print(f"Device: {get_device()}")
    print(f"{'='*50}")

    # Preload model
    loader.get_model(model_key)

    embeddings = []
    text_embeddings = [] if is_clip else None

    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(images), batch_size), desc=model_key, total=num_batches):
        batch_images = images[i:i + batch_size]
        batch_ids = manifest[i:i + batch_size]

        # Embed images (already in memory)
        emb = loader.embed_batch(model_key, batch_images)
        embeddings.append(emb.cpu().numpy())

        # Embed text (CLIP only)
        if is_clip:
            texts = [listings.get(str(lid), "") for lid, _ in batch_ids]
            text_emb = loader.embed_text_batch(model_key, texts)
            text_embeddings.append(text_emb.cpu().numpy())

    # Stack and save
    all_emb = np.vstack(embeddings).astype("float32")
    np.save(output_dir / f"{model_key}.npy", all_emb)
    print(f"Saved {model_key}.npy: {all_emb.shape}")

    if is_clip:
        all_text = np.vstack(text_embeddings).astype("float32")
        np.save(output_dir / f"{model_key}_text.npy", all_text)
        print(f"Saved {model_key}_text.npy: {all_text.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Stateless embedding worker - images to numpy"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input directory with manifest.json, listings.json, images/"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for numpy files (default: <input>_results)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for GPU processing"
    )
    parser.add_argument(
        "--model", default="all",
        choices=list(MODELS.keys()) + ["all"],
        help="Which model(s) to run"
    )
    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        args.output = args.input.parent / f"{args.input.name}_results"

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input directory not found: {args.input}")
        return 1
    if not (args.input / "manifest.json").exists():
        print(f"ERROR: manifest.json not found in {args.input}")
        return 1
    if not (args.input / "listings.json").exists():
        print(f"ERROR: listings.json not found in {args.input}")
        return 1

    # Show device info
    device = get_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load manifest and listings
    manifest = json.loads((args.input / "manifest.json").read_text())
    listings = json.loads((args.input / "listings.json").read_text())
    images_dir = args.input / "images"

    print(f"Manifest: {len(manifest)} images")

    # Load all images into RAM (one-time disk read)
    images = load_all_images(manifest, images_dir)

    # Setup output
    args.output.mkdir(parents=True, exist_ok=True)

    # Copy manifest to output (for verification during import)
    shutil.copy(args.input / "manifest.json", args.output / "manifest.json")

    # Determine models to run
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    # Process each model
    for model_key in models_to_run:
        process_model(
            model_key, manifest, images, listings,
            args.output, args.batch_size
        )

    print(f"\n{'='*50}")
    print(f"Done! Results in {args.output}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    exit(main())
