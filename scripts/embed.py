"""Generate embeddings for images using multiple models.

Outputs separate embedding files per model in embeddings/.
Tracks embedded_{model} flags per image in SQLite image_status table.

BATCH-FIRST APPROACH:
- Select a fixed batch of images (e.g., 50K) that need ANY embedding
- Process that SAME batch through ALL 5 models before considering next batch
- Each model only processes images missing its specific flag
- Consistent ORDER BY ensures FAISS row alignment across models
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sqlite3
import sys
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
IMAGES_DIR_NOBG = BASE_DIR / "images" / "imagedownload"
IMAGES_DIR = BASE_DIR / "images" / "imagedownload"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

# If images dir doesn't exist, try dev/images (local dev)
if not IMAGES_DIR.exists():
    IMAGES_DIR_NOBG = BASE_DIR / "dev" / "images_nobg"
    IMAGES_DIR = BASE_DIR / "dev" / "images"
    EMBEDDINGS_DIR = BASE_DIR / "dev" / "embeddings"

IMAGE_INDEX_FILE = EMBEDDINGS_DIR / "image_index.json"
DB_FILE = BASE_DIR / "data" / "db" / "etsy_data.db"
KILL_FILE = BASE_DIR / "KILL_EMBED"

# Import from shared image_db module
sys.path.insert(0, str(BASE_DIR / "bin"))
from image_db import (
    get_connection, get_images_for_embedding, get_images_for_embedding_batch,
    get_embedding_status_for_images, mark_embedded as db_mark_embedded
)

# Global kill flag - set when kill file is detected
_kill_requested = False

# DB connection for batch operations
_db_conn = None


def check_kill_file() -> bool:
    """Check if kill file exists or kill was previously requested.

    Once kill is detected, the flag stays set so all models stop.
    """
    global _kill_requested
    if _kill_requested:
        return True
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        _kill_requested = True
        print("\nKill file detected. Stopping gracefully...")
        return True
    return False


def get_db_conn():
    """Get or create DB connection."""
    global _db_conn
    if _db_conn is None:
        _db_conn = get_connection()
    return _db_conn


def commit_db():
    """Commit any pending DB changes."""
    global _db_conn
    if _db_conn is not None:
        _db_conn.commit()


def mark_embedded(listing_id: int, image_id: int, model_key: str):
    """Mark an image as embedded for a specific model in SQLite."""
    conn = get_db_conn()
    db_mark_embedded(conn, listing_id, image_id, model_key)


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


def get_image_files_for_model(model_key: str, limit: int = 0) -> tuple[list[Path], list[tuple[int, int]]]:
    """Get images that need embedding for a specific model.

    Uses SQLite image_status table:
    - bg_removed=1 AND embed_{model_key}=0

    This allows each model to track its own progress independently.
    """
    conn = get_db_conn()
    # Get large batch from SQL (limit 0 = no limit in query, default to 100k)
    query_limit = limit if limit > 0 else 100000
    images = get_images_for_embedding(conn, model_key, limit=query_limit)

    image_ids = []  # List of (listing_id, image_id) tuples
    valid_files = []

    for img in images:
        lid = img["listing_id"]
        iid = img["image_id"]
        # Build file path - bg_remover saves back to images/ as jpg
        img_path = IMAGES_DIR / f"{lid}_{iid}.jpg"
        if img_path.exists():
            image_ids.append((lid, iid))
            valid_files.append(img_path)

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
            commit_db()
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

        # Mark each image as embedded in SQL (atomic update)
        for lid, iid in batch_ids:
            mark_embedded(lid, iid, model_key)
            actually_embedded.append((lid, iid))

        processed += len(batch_files)

        # Commit after EVERY batch to release DB lock quickly
        commit_db()

        # Periodically save FAISS indexes (every 50K images)
        if processed % index_save_interval == 0:
            print(f"\nCheckpoint: saving FAISS index ({img_index.ntotal} vectors)...")
            save_faiss_index(img_index, emb_file)
            if is_clip and text_index is not None:
                save_faiss_index(text_index, text_emb_file)
            commit_db()

    # Final commit
    commit_db()
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
    """Load a FAISS index from file.

    Returns None if file doesn't exist.
    Raises exception if file exists but is corrupted (don't silently lose data).
    """
    import faiss
    if path.exists():
        try:
            return faiss.read_index(str(path))
        except Exception as e:
            # File exists but corrupted - this is a serious error, don't silently continue
            raise RuntimeError(
                f"FAISS index {path} exists but failed to load: {e}. "
                f"This may indicate corruption. Refusing to create new index to prevent data loss. "
                f"Manually delete the file if you want to start fresh."
            )
    return None


def save_faiss_index(index, path: Path):
    """Save a FAISS index to file atomically.

    Writes to temp file first, then renames to prevent corruption
    if process crashes mid-write.
    """
    import faiss
    import tempfile

    # Write to temp file in same directory (for same filesystem rename)
    temp_fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix='.faiss.tmp')
    os.close(temp_fd)

    try:
        faiss.write_index(index, temp_path)
        # Atomic rename
        os.replace(temp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def load_existing_index() -> list[tuple[int, int]]:
    """Load existing image index."""
    if IMAGE_INDEX_FILE.exists():
        with open(IMAGE_INDEX_FILE) as f:
            return [tuple(x) for x in json.load(f)]
    return []


def save_image_index(image_ids: list[tuple[int, int]]):
    """Save image index mapping row → (listing_id, image_id) atomically."""
    import tempfile

    # Write to temp file first
    temp_fd, temp_path = tempfile.mkstemp(dir=IMAGE_INDEX_FILE.parent, suffix='.json.tmp')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(image_ids, f)
        # Atomic rename
        os.replace(temp_path, str(IMAGE_INDEX_FILE))
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    print(f"Saved image index to {IMAGE_INDEX_FILE}")


def process_batch_all_models(
    batch_image_ids: list[tuple[int, int]],
    models_to_run: list[str],
    batch_size: int = 32,
) -> list[tuple[int, int]]:
    """Process a batch of images through ALL models.

    This ensures the same images are processed in the same order for all models,
    maintaining FAISS row alignment.

    Returns list of image_ids that were successfully processed through all models.
    """
    if not batch_image_ids:
        return []

    # Build file paths and filter to existing files
    valid_files = []
    valid_ids = []
    for lid, iid in batch_image_ids:
        img_path = IMAGES_DIR / f"{lid}_{iid}.jpg"
        if img_path.exists():
            valid_files.append(img_path)
            valid_ids.append((lid, iid))

    if not valid_files:
        print("No valid image files found in batch")
        return []

    print(f"\nProcessing batch of {len(valid_ids)} images through {len(models_to_run)} models")

    # Get current embedding status for all images in batch
    conn = get_db_conn()
    status = get_embedding_status_for_images(conn, valid_ids)

    successfully_processed = set(valid_ids)  # Start assuming all succeed

    for model_key in models_to_run:
        if check_kill_file():
            print(f"Kill requested, stopping before {model_key}")
            return []

        # Filter to images that need this model's embedding
        need_embedding = []
        need_embedding_files = []
        for (lid, iid), fpath in zip(valid_ids, valid_files):
            img_status = status.get((lid, iid), {})
            if not img_status.get(model_key, False):
                need_embedding.append((lid, iid))
                need_embedding_files.append(fpath)

        if not need_embedding:
            print(f"  {model_key}: all {len(valid_ids)} images already done, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing model: {model_key} ({len(need_embedding)}/{len(valid_ids)} images need embedding)")
        print(f"{'='*60}")

        dim = MODELS[model_key]["dim"]
        emb_file = get_embedding_file(model_key)
        is_clip = MODELS[model_key]["library"] == "open_clip"
        text_emb_file = get_embedding_file(model_key, "text") if is_clip else None

        # Load or create FAISS indexes
        if emb_file.exists():
            img_index = load_faiss_index(emb_file)
            text_index = load_faiss_index(text_emb_file) if is_clip and text_emb_file else None
            if text_index is None and is_clip:
                text_index = create_faiss_index(dim)
            print(f"Loaded existing FAISS index: {img_index.ntotal} vectors")
        else:
            img_index = create_faiss_index(dim)
            text_index = create_faiss_index(dim) if is_clip else None
            print(f"Created new FAISS index")

        # Generate embeddings
        num_embedded, actually_embedded = generate_embeddings_incremental(
            model_key, need_embedding_files, need_embedding,
            img_index, text_index,
            emb_file, text_emb_file,
            batch_size
        )

        # Track which images failed
        if num_embedded < len(need_embedding):
            print(f"Partial embedding: {num_embedded}/{len(need_embedding)} images")
            # Remove images that weren't fully processed from success set
            embedded_set = set(actually_embedded)
            for img_id in need_embedding:
                if img_id not in embedded_set:
                    successfully_processed.discard(img_id)

        # Save FAISS indexes
        print(f"Image index: {img_index.ntotal} vectors, dim={dim}")
        print(f"Saving to {emb_file}...")
        save_faiss_index(img_index, emb_file)

        if is_clip and text_index is not None:
            print(f"Text index: {text_index.ntotal} vectors, dim={dim}")
            print(f"Saving to {text_emb_file}...")
            save_faiss_index(text_index, text_emb_file)

        # Update status cache for next model iteration
        for lid, iid in actually_embedded:
            if (lid, iid) in status:
                status[(lid, iid)][model_key] = True

    return [img_id for img_id in valid_ids if img_id in successfully_processed]


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
        help="Batch size for GPU embedding",
    )
    parser.add_argument(
        "--job-size",
        type=int,
        default=50000,
        help="Number of images per job (processed through all models before next job)",
    )
    args = parser.parse_args()

    # Create output directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # Load existing image index
    existing_index_list = load_existing_index()
    existing_index_set = set(existing_index_list)
    final_index = list(existing_index_list)
    print(f"Existing image index has {len(existing_index_set)} images")

    # Determine which models to run
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    print(f"Models to run: {models_to_run}")

    # PHASE 1: Complete existing partial embeddings
    # Images in image_index that don't have all models done
    if existing_index_list and not args.full:
        print(f"\n{'='*60}")
        print("PHASE 1: Completing existing partial embeddings")
        print(f"{'='*60}")

        conn = get_db_conn()
        status = get_embedding_status_for_images(conn, existing_index_list)

        # Find images missing any model
        incomplete = []
        for img_id in existing_index_list:
            img_status = status.get(img_id, {})
            for model_key in models_to_run:
                if not img_status.get(model_key, False):
                    incomplete.append(img_id)
                    break

        if incomplete:
            print(f"Found {len(incomplete)} images needing completion")
            # Process in batches of job_size
            for i in range(0, len(incomplete), args.job_size):
                if check_kill_file():
                    break
                batch = incomplete[i:i + args.job_size]
                print(f"\nProcessing completion batch {i//args.job_size + 1}: {len(batch)} images")
                process_batch_all_models(batch, models_to_run, args.batch_size)
        else:
            print("All existing images have complete embeddings")

    # PHASE 2: Process new batches
    print(f"\n{'='*60}")
    print("PHASE 2: Processing new image batches")
    print(f"{'='*60}")

    while True:
        if check_kill_file():
            print("Kill requested, stopping")
            break

        # Get next batch of images needing ANY embedding
        conn = get_db_conn()
        batch_images = get_images_for_embedding_batch(conn, limit=args.job_size)

        if not batch_images:
            print("No more images to process")
            break

        # Convert to list of tuples
        batch_ids = [(img["listing_id"], img["image_id"]) for img in batch_images]

        # Filter out images already in index (they were handled in Phase 1)
        new_batch_ids = [img_id for img_id in batch_ids if img_id not in existing_index_set]

        if not new_batch_ids:
            print("All queried images already in index, checking for more...")
            # This shouldn't happen normally, but if it does, the next query should get different images
            continue

        print(f"\nNew batch: {len(new_batch_ids)} images")

        # Process through all models
        successfully_processed = process_batch_all_models(new_batch_ids, models_to_run, args.batch_size)

        # Add successfully processed images to index
        for img_id in successfully_processed:
            if img_id not in existing_index_set:
                final_index.append(img_id)
                existing_index_set.add(img_id)

        # Save image index after each batch
        save_image_index(final_index)
        print(f"Image index now has {len(final_index)} images")

        # Only process one batch per run (pipeline_monitor will restart)
        print(f"\nCompleted batch of {len(successfully_processed)} images through all models")
        break

    # Final save
    save_image_index(final_index)
    commit_db()
    print("\nDone!")


if __name__ == "__main__":
    main()
