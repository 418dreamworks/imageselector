"""Generate embeddings for images using multiple models.

Outputs separate embedding files per model in dev/embeddings_v2/.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import MODELS, get_loader

DEV_DIR = Path(__file__).parent.parent / "dev"
IMAGES_DIR = DEV_DIR / "images_v2_nobg"  # v2 uses nobg images (preferred)
IMAGES_DIR_NOBG = DEV_DIR / "images_nobg"  # Fallback to v1 nobg images
IMAGES_DIR_LEGACY = DEV_DIR / "images"  # Last resort: original images
EMBEDDINGS_DIR = DEV_DIR / "embeddings_v2"
LISTING_IDS_FILE = DEV_DIR / "listing_ids_v2.json"


def get_image_files() -> tuple[list[Path], list[int]]:
    """Get all jpg images and extract listing IDs.

    Priority order:
    1. images_v2_nobg (multi-image nobg - future)
    2. images_nobg (v1 nobg - preferred for Stage A)
    3. images (original with background - last resort)

    v2 format: {listing_id}_{image_id}.jpg
    v1 format: {listing_id}.jpg
    """
    files = []

    # Check v2 nobg folder first (future multi-image)
    if IMAGES_DIR.exists():
        files.extend(sorted(IMAGES_DIR.glob("*.jpg")))

    # Fall back to v1 nobg folder (current best option)
    if not files and IMAGES_DIR_NOBG.exists():
        files = sorted(IMAGES_DIR_NOBG.glob("*.jpg"))
        print(f"Using nobg images from {IMAGES_DIR_NOBG}")

    # Last resort: original images with background
    if not files and IMAGES_DIR_LEGACY.exists():
        files = sorted(IMAGES_DIR_LEGACY.glob("*.jpg"))
        print(f"Using original images from {IMAGES_DIR_LEGACY}")

    listing_ids = []
    valid_files = []

    for f in files:
        # Filename format: {listing_id}.jpg (v1) or {listing_id}_{image_id}.jpg (v2)
        stem = f.stem
        lid = stem.split("_")[0] if "_" in stem else stem
        try:
            listing_ids.append(int(lid))
            valid_files.append(f)
        except ValueError:
            print(f"Skipping {f.name} - can't parse listing ID")

    return valid_files, listing_ids


def generate_embeddings(
    model_key: str,
    image_files: list[Path],
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings for all images using specified model."""
    loader = get_loader()
    embeddings = []

    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Embedding ({model_key})"):
        batch_files = image_files[i : i + batch_size]
        images = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
                # Create placeholder - will be zero vector after normalization issue
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        batch_emb = loader.embed_batch(model_key, images)
        embeddings.append(batch_emb.cpu().numpy())

    return np.vstack(embeddings).astype("float32")


def get_embedding_file(model_key: str) -> Path:
    """Get the embedding file path for a model."""
    return EMBEDDINGS_DIR / f"{model_key}.npy"


def load_existing_embeddings(model_key: str) -> tuple[np.ndarray | None, list[int]]:
    """Load existing embeddings for incremental update."""
    emb_file = get_embedding_file(model_key)

    if emb_file.exists() and LISTING_IDS_FILE.exists():
        embeddings = np.load(emb_file)
        with open(LISTING_IDS_FILE) as f:
            listing_ids = json.load(f)
        return embeddings, listing_ids

    return None, []


def main():
    parser = argparse.ArgumentParser(description="Generate multi-model embeddings")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model(s) to use for embedding",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Regenerate all embeddings from scratch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images (for testing)",
    )
    args = parser.parse_args()

    # Create output directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # Get all image files
    print(f"\nScanning {IMAGES_DIR}...")
    all_files, all_listing_ids = get_image_files()
    print(f"Found {len(all_files)} total images")

    if args.limit > 0:
        all_files = all_files[: args.limit]
        all_listing_ids = all_listing_ids[: args.limit]
        print(f"Limited to {len(all_files)} images")

    # Determine which models to run
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_key}")
        print(f"{'='*60}")

        # Check for existing embeddings (incremental mode)
        if not args.full:
            existing_emb, existing_ids = load_existing_embeddings(model_key)
            if existing_emb is not None:
                existing_set = set(existing_ids)
                new_files = []
                new_listing_ids = []
                for f, lid in zip(all_files, all_listing_ids):
                    if lid not in existing_set:
                        new_files.append(f)
                        new_listing_ids.append(lid)
                print(f"Found {len(existing_ids)} existing embeddings")
                print(f"New images to embed: {len(new_files)}")

                if not new_files:
                    print(f"No new images for {model_key}. Skipping.")
                    continue

                # Generate new embeddings
                new_embeddings = generate_embeddings(model_key, new_files, args.batch_size)

                # Combine
                final_embeddings = np.vstack([existing_emb, new_embeddings])
                final_listing_ids = existing_ids + new_listing_ids
            else:
                # No existing - generate all
                final_embeddings = generate_embeddings(model_key, all_files, args.batch_size)
                final_listing_ids = all_listing_ids
        else:
            # Full regeneration
            final_embeddings = generate_embeddings(model_key, all_files, args.batch_size)
            final_listing_ids = all_listing_ids

        # Save embeddings
        emb_file = get_embedding_file(model_key)
        print(f"\nEmbeddings shape: {final_embeddings.shape}")
        print(f"Saving to {emb_file}...")
        np.save(emb_file, final_embeddings)

        # Save listing IDs after each model (so we can resume if interrupted)
        print(f"Saving listing IDs to {LISTING_IDS_FILE}...")
        with open(LISTING_IDS_FILE, "w") as f:
            json.dump(final_listing_ids, f)

    # Final save of listing IDs
    print(f"\nFinal save: listing IDs to {LISTING_IDS_FILE}...")
    with open(LISTING_IDS_FILE, "w") as f:
        json.dump(all_listing_ids, f)

    print("\nDone!")


if __name__ == "__main__":
    main()
