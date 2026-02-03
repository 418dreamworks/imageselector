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

# NOTE: faiss is imported lazily in functions that need it
# Importing faiss before loading open_clip models on MPS causes segfaults
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


def get_image_files_for_model(model_key: str) -> tuple[list[Path], list[tuple[int, int]]]:
    """Get images that need embedding for a specific model.

    Only returns images that:
    1. Have bg_removed=true
    2. Do NOT have embedded_{model_key}=true

    This allows each model to track its own progress independently.
    """
    metadata = load_metadata()
    image_ids = []  # List of (listing_id, image_id) tuples
    valid_files = []
    embed_flag = f"embedded_{model_key}"

    for lid, entry in metadata.items():
        if not isinstance(entry, dict):
            continue
        for img in entry.get("images", []):
            # Only process images with background removed
            if not img.get("bg_removed"):
                continue
            # Skip if already embedded for this model
            if img.get(embed_flag):
                continue
            image_id = img.get("image_id")
            if not image_id:
                continue
            # Build file path - bg_remover saves back to images/ as jpg
            img_path = IMAGES_DIR / f"{lid}_{image_id}.jpg"
            if img_path.exists():
                image_ids.append((int(lid), image_id))
                valid_files.append(img_path)

    # Sort by listing_id, image_id for consistent ordering
    combined = sorted(zip(valid_files, image_ids), key=lambda x: x[1])
    if combined:
        valid_files, image_ids = zip(*combined)
        valid_files = list(valid_files)
        image_ids = list(image_ids)

    print(f"Found {len(valid_files)} images needing {model_key} embedding")
    return valid_files, image_ids


def generate_embeddings_incremental(
    model_key: str,
    image_files: list[Path],
    image_ids: list[tuple[int, int]],
    img_index,
    text_index,
    emb_file: Path,
    text_emb_file: Path | None,
    batch_size: int = 32,
    save_interval: int = 1000,
    index_save_interval: int = 50000,
) -> tuple[int, list[tuple[int, int]]]:
    """Generate embeddings incrementally and add to FAISS indexes.

    Adds embeddings directly to FAISS indexes instead of accumulating in memory.
    Saves FAISS indexes periodically to avoid data loss.

    Returns (num_embedded, list of actually embedded image_ids).
    """
    loader = get_loader()
    # Preload model before starting (avoids segfault when loading inside tqdm)
    loader.get_model(model_key)

    processed = 0
    is_clip = MODELS[model_key]["library"] == "open_clip"
    actually_embedded = []

    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Embedding ({model_key})"):
        # Check for kill file
        if check_kill_file():
            save_metadata()
            # Save FAISS indexes before exiting
            print(f"\nSaving FAISS index ({img_index.ntotal} vectors)...")
            save_faiss_index(img_index, emb_file)
            if is_clip and text_index is not None:
                save_faiss_index(text_index, text_emb_file)
            return processed, actually_embedded

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

        # Get image embeddings and add to index immediately
        img_emb = loader.embed_batch(model_key, images)
        img_emb_np = img_emb.cpu().numpy().astype("float32")
        img_index.add(img_emb_np)

        # For CLIP, also get text embeddings
        if is_clip and text_index is not None:
            listing_ids = [lid for lid, _ in batch_ids]
            texts = get_texts_for_batch(listing_ids)
            text_emb = loader.embed_text_batch(model_key, texts)
            text_emb_np = text_emb.cpu().numpy().astype("float32")
            text_index.add(text_emb_np)

        # Mark each image as embedded in metadata
        for lid, iid in batch_ids:
            mark_embedded(str(lid), iid, model_key)
            actually_embedded.append((lid, iid))

        processed += len(batch_files)

        # Periodically save metadata
        if processed % save_interval == 0:
            save_metadata()

        # Periodically save FAISS indexes (every 50K images)
        if processed % index_save_interval == 0:
            print(f"\nCheckpoint: saving FAISS index ({img_index.ntotal} vectors)...")
            save_faiss_index(img_index, emb_file)
            if is_clip and text_index is not None:
                save_faiss_index(text_index, text_emb_file)
            save_metadata()

    # Final save
    save_metadata()
    return processed, actually_embedded


def get_embedding_file(model_key: str, emb_type: str = "image") -> Path:
    """Get the FAISS index file path for a model.

    emb_type: "image" or "text" (text only for CLIP models)
    """
    if emb_type == "text":
        return EMBEDDINGS_DIR / f"{model_key}_text.faiss"
    return EMBEDDINGS_DIR / f"{model_key}.faiss"


def create_faiss_index(dim: int):
    """Create a FAISS index for cosine similarity.

    Uses IndexFlatIP (inner product) since vectors are normalized.
    """
    import faiss
    return faiss.IndexFlatIP(dim)


def load_faiss_index(path: Path):
    """Load a FAISS index from file."""
    import faiss
    if path.exists():
        return faiss.read_index(str(path))
    return None


def save_faiss_index(index, path: Path):
    """Save a FAISS index to file."""
    import faiss
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
        default=0,
        help="Max new images to embed per run (0 = unlimited)",
    )
    args = parser.parse_args()

    # Create output directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # Load existing image index (shared across all models for search)
    existing_index_list = load_existing_index()
    existing_index = set(existing_index_list)
    final_index = list(existing_index_list)
    print(f"Existing image index has {len(existing_index)} images")

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

        # Load or create FAISS indexes
        if not args.full and emb_file.exists():
            img_index = load_faiss_index(emb_file)
            text_index = load_faiss_index(text_emb_file) if is_clip and text_emb_file else None
            if text_index is None and is_clip:
                text_index = create_faiss_index(dim)
            print(f"Loaded existing FAISS index: {img_index.ntotal} vectors")
        else:
            img_index = create_faiss_index(dim)
            text_index = create_faiss_index(dim) if is_clip else None
            print(f"Created new FAISS index")

        # Get images that need embedding for THIS model (checks embedded_{model} flag)
        new_files, new_image_ids = get_image_files_for_model(model_key)

        if args.limit > 0:
            new_files = new_files[: args.limit]
            new_image_ids = new_image_ids[: args.limit]
            print(f"Limited to {len(new_files)} images")

        if not new_files:
            print(f"No new images for {model_key}. Skipping.")
            continue

        # Apply batch limit if set
        if args.batch_limit > 0 and len(new_files) > args.batch_limit:
            print(f"Limiting to {args.batch_limit} images (batch limit)")
            new_files = new_files[:args.batch_limit]
            new_image_ids = new_image_ids[:args.batch_limit]

        # Generate embeddings incrementally (adds directly to FAISS, saves periodically)
        num_embedded, actually_embedded = generate_embeddings_incremental(
            model_key, new_files, new_image_ids,
            img_index, text_index,
            emb_file, text_emb_file,
            args.batch_size
        )

        if num_embedded < len(new_image_ids):
            print(f"Partial embedding: {num_embedded}/{len(new_image_ids)} images")

        # Add newly embedded IDs to shared image index (for search)
        for img_id in actually_embedded:
            if img_id not in existing_index:
                final_index.append(img_id)
                existing_index.add(img_id)

        # Save FAISS indexes
        print(f"\nImage index: {img_index.ntotal} vectors, dim={dim}")
        print(f"Saving to {emb_file}...")
        save_faiss_index(img_index, emb_file)

        if is_clip and text_index is not None:
            print(f"Text index: {text_index.ntotal} vectors, dim={dim}")
            print(f"Saving to {text_emb_file}...")
            save_faiss_index(text_index, text_emb_file)

    # Save image index (shared across all models) - only actually embedded images
    save_image_index(final_index)

    # Final metadata save
    save_metadata()

    print("\nDone!")


if __name__ == "__main__":
    main()
