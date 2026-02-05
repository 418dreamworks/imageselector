#!/usr/bin/env python3
"""Stateless worker: bg removal + embeddings.

This script has NO access to databases or FAISS indexes.
It removes backgrounds, then produces embeddings as numpy files.
Safe to run on any machine (macOS, Windows, Linux).

Usage:
    python embed_worker.py --input /path/to/batch
    python embed_worker.py --input ~/embed_work/batch_001

Input folder structure:
    batch/
    ├── manifest.json      # [[listing_id, image_id], ...]
    ├── listings.json      # {"listing_id": "title. materials", ...}
    └── images/
        ├── 123_456.jpg    # original images
        └── ...

Output (in same folder):
    batch/
    ├── manifest.json           # unchanged
    ├── listings.json           # unchanged
    ├── images/
    │   ├── 123_456.jpg         # bg-removed (overwritten)
    │   └── ...
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
# Enable MPS fallback for unsupported ops (upsample_bicubic2d in DINOv2)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import MODELS, get_loader

# Global rembg session
_rembg_session = None


_force_cpu = False

def get_device() -> str:
    """Auto-detect best available device."""
    if _force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_rembg():
    """Initialize rembg with GPU acceleration."""
    global _rembg_session
    from rembg import new_session

    device = get_device()
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "mps":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        _rembg_session = new_session("u2net", providers=providers)
        print(f"rembg initialized with providers: {providers}")
    except Exception as e:
        print(f"GPU provider failed ({e}), falling back to CPU")
        _rembg_session = new_session("u2net")


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background from image, return RGB with white background."""
    global _rembg_session
    from rembg import remove

    # Remove background (returns RGBA)
    img_nobg = remove(img, session=_rembg_session)

    # Convert to RGB with white background
    rgb_img = Image.new("RGB", img_nobg.size, (255, 255, 255))
    rgb_img.paste(img_nobg, mask=img_nobg.split()[3])

    return rgb_img


def load_images_only(manifest: list, images_dir: Path) -> list:
    """Load images without bg removal (for testing/already processed)."""
    print(f"Loading {len(manifest)} images (no bg removal)...")
    images = []
    for lid, iid in tqdm(manifest, desc="Loading"):
        img_path = images_dir / f"{lid}_{iid}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
            img.load()
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
    return images


def load_and_process_images(manifest: list, images_dir: Path) -> list:
    """Load images, remove backgrounds, save back to disk, keep in memory.

    Flow:
    1. Load original image
    2. Remove background (GPU accelerated)
    3. Save bg-removed image back to disk (overwrite)
    4. Keep in memory for embedding
    """
    print(f"Processing {len(manifest)} images (bg removal + load)...")
    init_rembg()

    images = []

    for lid, iid in tqdm(manifest, desc="BG removal"):
        img_path = images_dir / f"{lid}_{iid}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
            img.load()

            # Remove background
            img_nobg = remove_background(img)

            # Save back to disk (overwrite original)
            img_nobg.save(img_path, "JPEG", quality=90)

            images.append(img_nobg)
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            # Placeholder for failed images
            images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    print(f"Processed {len(images)} images")
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
        description="Worker: bg removal + embeddings"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Batch directory with manifest.json, listings.json, images/"
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
    parser.add_argument(
        "--skip-bg", action="store_true",
        help="Skip bg removal (images already processed)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU (for testing consistency)"
    )
    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu:
        global _force_cpu
        _force_cpu = True

    # Output is same directory as input
    batch_dir = args.input

    # Validate input
    if not batch_dir.exists():
        print(f"ERROR: Batch directory not found: {batch_dir}")
        return 1
    if not (batch_dir / "manifest.json").exists():
        print(f"ERROR: manifest.json not found in {batch_dir}")
        return 1
    if not (batch_dir / "listings.json").exists():
        print(f"ERROR: listings.json not found in {batch_dir}")
        return 1

    # Show device info
    device = get_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load manifest and listings
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    listings = json.loads((batch_dir / "listings.json").read_text())
    images_dir = batch_dir / "images"

    print(f"Manifest: {len(manifest)} images")

    # Load images (with or without bg removal)
    if args.skip_bg:
        images = load_images_only(manifest, images_dir)
    else:
        images = load_and_process_images(manifest, images_dir)

    # Determine models to run
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    # Process each model (save .npy to batch dir)
    for model_key in models_to_run:
        process_model(
            model_key, manifest, images, listings,
            batch_dir, args.batch_size
        )

    print(f"\n{'='*50}")
    print(f"Done! Results in {batch_dir}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    exit(main())
