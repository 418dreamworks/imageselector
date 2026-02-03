"""Generate embeddings for images using multiple models.

Outputs separate embedding files per model in dev/embeddings/.
Tracks embedded_{model} flags per image in metadata.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import MODELS, get_loader

BASE_DIR = Path(__file__).parent.parent

# Paths - adjust for iMac vs local dev
IMAGES_DIR_NOBG = BASE_DIR / "images_nobg"
IMAGES_DIR = BASE_DIR / "images"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# If images dir doesn't exist, try dev/images (local dev)
if not IMAGES_DIR.exists():
    IMAGES_DIR_NOBG = BASE_DIR / "dev" / "images_nobg"
    IMAGES_DIR = BASE_DIR / "dev" / "images"
    EMBEDDINGS_DIR = BASE_DIR / "dev" / "embeddings"

IMAGE_INDEX_FILE = EMBEDDINGS_DIR / "image_index.json"
METADATA_FILE = BASE_DIR / "image_metadata.json"
DB_FILE = BASE_DIR / "etsy_data.db"
KILL_FILE = BASE_DIR / "KILL_EMBED"

# Metadata cache
_metadata = None
_metadata_dirty = False


def check_kill_file() -> bool:
    """Check if kill file exists. If so, remove it and return True."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        print("\nKill file detected. Stopping gracefully...")
        return True
    return False


def load_metadata() -> dict:
    """Load metadata from file."""
    global _metadata
    if _metadata is None:
        if METADATA_FILE.exists():
            with open(METADATA_FILE) as f:
                _metadata = json.load(f)
        else:
            _metadata = {}
    return _metadata


def save_metadata():
    """Save metadata to file if dirty."""
    global _metadata, _metadata_dirty
    if _metadata_dirty and _metadata is not None:
        tmp_file = METADATA_FILE.with_suffix('.json.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(_metadata, f)
        tmp_file.rename(METADATA_FILE)
        _metadata_dirty = False
        print(f"Saved metadata to {METADATA_FILE}")


def mark_embedded(listing_id: str, image_id: int, model_key: str):
    """Mark an image as embedded for a specific model."""
    global _metadata, _metadata_dirty
    metadata = load_metadata()

    if listing_id in metadata and isinstance(metadata[listing_id], dict):
        entry = metadata[listing_id]
        images = entry.get("images", [])
        for img in images:
            if img.get("image_id") == image_id:
                img[f"embedded_{model_key}"] = True
                _metadata_dirty = True
                break


# Listing text cache (title + materials)
_listing_texts = None


def load_listing_texts() -> dict[int, str]:
    """Load listing titles and materials from database.

    Returns dict mapping listing_id -> "title. materials"
    """
    global _listing_texts
    if _listing_texts is not None:
        return _listing_texts

    _listing_texts = {}
    if not DB_FILE.exists():
        print(f"Warning: Database not found at {DB_FILE}")
        return _listing_texts

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT listing_id, title, materials FROM listings")

    for listing_id, title, materials_json in cursor.fetchall():
        parts = []
        if title:
            parts.append(title)
        if materials_json:
            try:
                materials = json.loads(materials_json)
                if materials:
                    parts.append(", ".join(materials))
            except (json.JSONDecodeError, TypeError):
                pass
        _listing_texts[listing_id] = ". ".join(parts) if parts else ""

    conn.close()
    print(f"Loaded {len(_listing_texts)} listing texts from database")
    return _listing_texts


def get_texts_for_batch(listing_ids: list[int]) -> list[str]:
    """Get text descriptions for a batch of listing IDs."""
    texts = load_listing_texts()
    return [texts.get(lid, "") for lid in listing_ids]


def get_image_files() -> tuple[list[Path], list[tuple[int, int]]]:
    """Get all images and extract (listing_id, image_id) tuples.

    Priority: images_nobg (png) > images (jpg)
    Filename format: {listing_id}_{image_id}.png or .jpg
    """
    files = []

    # Prefer nobg images (png format)
    if IMAGES_DIR_NOBG.exists():
        files = sorted(IMAGES_DIR_NOBG.glob("*.png"))
        if files:
            print(f"Using nobg images from {IMAGES_DIR_NOBG}")

    # Fallback to original images (jpg format)
    if not files and IMAGES_DIR.exists():
        files = sorted(IMAGES_DIR.glob("*.jpg"))
        if files:
            print(f"Using original images from {IMAGES_DIR}")

    image_ids = []  # List of (listing_id, image_id) tuples
    valid_files = []

    for f in files:
        # Filename format: {listing_id}_{image_id}.png/jpg
        stem = f.stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                lid = int(parts[0])
                iid = int(parts[1])
                image_ids.append((lid, iid))
                valid_files.append(f)
            except ValueError:
                print(f"Skipping {f.name} - can't parse IDs")
        else:
            print(f"Skipping {f.name} - expected format: listingid_imageid")

    return valid_files, image_ids


def generate_embeddings(
    model_key: str,
    image_files: list[Path],
    image_ids: list[tuple[int, int]],
    batch_size: int = 32,
    save_interval: int = 1000,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate embeddings for all images using specified model.

    For CLIP models, also generates text embeddings from listing titles+materials.
    Returns (image_embeddings, text_embeddings) - text is None for non-CLIP models.
    Also marks each image as embedded in metadata.
    """
    loader = get_loader()
    image_embeddings = []
    text_embeddings = []
    processed = 0
    is_clip = MODELS[model_key]["library"] == "open_clip"

    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Embedding ({model_key})"):
        # Check for kill file
        if check_kill_file():
            save_metadata()
            # Return what we have so far
            if image_embeddings:
                img_arr = np.vstack(image_embeddings).astype("float32")
                text_arr = np.vstack(text_embeddings).astype("float32") if text_embeddings else None
                return img_arr, text_arr
            return np.array([]).reshape(0, MODELS[model_key]["dim"]).astype("float32"), None

        batch_files = image_files[i : i + batch_size]
        batch_ids = image_ids[i : i + batch_size]
        images = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
                # Create placeholder
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        # Always get image embeddings
        img_emb = loader.embed_batch(model_key, images)
        image_embeddings.append(img_emb.cpu().numpy())

        # For CLIP, also get text embeddings
        if is_clip:
            listing_ids = [lid for lid, _ in batch_ids]
            texts = get_texts_for_batch(listing_ids)
            text_emb = loader.embed_text_batch(model_key, texts)
            text_embeddings.append(text_emb.cpu().numpy())

        # Mark each image as embedded in metadata
        for lid, iid in batch_ids:
            mark_embedded(str(lid), iid, model_key)

        processed += len(batch_files)

        # Periodically save metadata
        if processed % save_interval == 0:
            save_metadata()

    # Final save
    save_metadata()

    img_arr = np.vstack(image_embeddings).astype("float32")
    text_arr = np.vstack(text_embeddings).astype("float32") if text_embeddings else None
    return img_arr, text_arr


def get_embedding_file(model_key: str, emb_type: str = "image") -> Path:
    """Get the FAISS index file path for a model.

    emb_type: "image" or "text" (text only for CLIP models)
    """
    if emb_type == "text":
        return EMBEDDINGS_DIR / f"{model_key}_text.faiss"
    return EMBEDDINGS_DIR / f"{model_key}.faiss"


def create_faiss_index(dim: int) -> faiss.Index:
    """Create a FAISS index for cosine similarity.

    Uses IndexFlatIP (inner product) since vectors are normalized.
    """
    return faiss.IndexFlatIP(dim)


def load_faiss_index(path: Path) -> faiss.Index | None:
    """Load a FAISS index from file."""
    if path.exists():
        return faiss.read_index(str(path))
    return None


def save_faiss_index(index: faiss.Index, path: Path):
    """Save a FAISS index to file."""
    faiss.write_index(index, str(path))


def load_existing_index() -> list[tuple[int, int]]:
    """Load existing image index."""
    if IMAGE_INDEX_FILE.exists():
        with open(IMAGE_INDEX_FILE) as f:
            return [tuple(x) for x in json.load(f)]
    return []


def save_image_index(image_ids: list[tuple[int, int]]):
    """Save image index mapping row → (listing_id, image_id)."""
    with open(IMAGE_INDEX_FILE, 'w') as f:
        json.dump(image_ids, f)
    print(f"Saved image index to {IMAGE_INDEX_FILE}")


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
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=20000,
        help="Max new images to embed per run (default: 20000)",
    )
    args = parser.parse_args()

    # Create output directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # Get all image files
    print(f"\nScanning for images...")
    all_files, all_image_ids = get_image_files()
    print(f"Found {len(all_files)} total images")

    if args.limit > 0:
        all_files = all_files[: args.limit]
        all_image_ids = all_image_ids[: args.limit]
        print(f"Limited to {len(all_files)} images")

    if not all_files:
        print("No images found!")
        return

    # Load existing image index (shared across all models)
    existing_index = set(load_existing_index())
    print(f"Existing index has {len(existing_index)} images")

    # Determine which models to run
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_key}")
        print(f"{'='*60}")

        dim = MODELS[model_key]["dim"]
        emb_file = get_embedding_file(model_key)
        is_clip = MODELS[model_key]["library"] == "open_clip"
        text_emb_file = get_embedding_file(model_key, "text") if is_clip else None

        # Check for existing FAISS index (incremental mode)
        if not args.full and emb_file.exists() and existing_index:
            img_index = load_faiss_index(emb_file)
            text_index = load_faiss_index(text_emb_file) if is_clip and text_emb_file else None

            # Find images not yet embedded for this model
            new_files = []
            new_image_ids = []
            for f, (lid, iid) in zip(all_files, all_image_ids):
                if (lid, iid) not in existing_index:
                    new_files.append(f)
                    new_image_ids.append((lid, iid))

            print(f"Existing embeddings: {img_index.ntotal}")
            print(f"New images to embed: {len(new_files)}")

            if not new_files:
                print(f"No new images for {model_key}. Skipping.")
                continue

            # Apply batch limit
            if args.batch_limit > 0 and len(new_files) > args.batch_limit:
                print(f"Limiting to {args.batch_limit} images (batch limit)")
                new_files = new_files[:args.batch_limit]
                new_image_ids = new_image_ids[:args.batch_limit]

            # Generate new embeddings
            new_img_emb, new_text_emb = generate_embeddings(
                model_key, new_files, new_image_ids, args.batch_size
            )

            # Add to existing FAISS indexes
            img_index.add(new_img_emb)
            if is_clip and text_index is not None and new_text_emb is not None:
                text_index.add(new_text_emb)
            elif is_clip and new_text_emb is not None:
                text_index = create_faiss_index(dim)
                text_index.add(new_text_emb)
        else:
            # Full regeneration
            new_img_emb, new_text_emb = generate_embeddings(
                model_key, all_files, all_image_ids, args.batch_size
            )

            # Create new FAISS indexes
            img_index = create_faiss_index(dim)
            img_index.add(new_img_emb)

            if is_clip and new_text_emb is not None:
                text_index = create_faiss_index(dim)
                text_index.add(new_text_emb)
            else:
                text_index = None

        # Save FAISS indexes
        print(f"\nImage index: {img_index.ntotal} vectors, dim={dim}")
        print(f"Saving to {emb_file}...")
        save_faiss_index(img_index, emb_file)

        if is_clip and text_index is not None:
            print(f"Text index: {text_index.ntotal} vectors, dim={dim}")
            print(f"Saving to {text_emb_file}...")
            save_faiss_index(text_index, text_emb_file)

    # Save image index (shared across all models)
    save_image_index(all_image_ids)

    # Final metadata save
    save_metadata()

    print("\nDone!")


if __name__ == "__main__":
    main()
