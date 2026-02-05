#!/usr/bin/env python3
"""Embedding manager. Exports batches, imports results into FAISS/DB.

This is the ONLY script that writes to FAISS indexes and the database.
Workers produce numpy files, manager validates and merges them.

Usage:
    # Export a batch of images for workers
    python embed_manager.py export --batch-id 001 --job-size 50000

    # Import verified results back into FAISS/DB
    python embed_manager.py import --results-dir ~/embed_work/batch_001_results

    # Check status of embeddings
    python embed_manager.py status

Workflow:
    1. export: Creates batch folder with images + manifest + listings
    2. (manually copy to workers, run embed_worker.py)
    3. (manually copy results back)
    4. import: Validates numpy files, appends to FAISS, updates DB
"""
import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Paths
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EXPORTS_DIR = BASE_DIR / "embed_exports"
DB_FILE = BASE_DIR / "etsy_data.db"
IMAGE_INDEX_FILE = EMBEDDINGS_DIR / "image_index.json"

# Add parent to path for imports
sys.path.insert(0, str(BASE_DIR))
from image_db import (
    get_connection, get_images_for_embedding_batch,
    mark_embedded, commit_with_retry
)

MODELS = {
    "clip_vitb32": {"dim": 512, "has_text": True},
    "clip_vitl14": {"dim": 768, "has_text": True},
    "dinov2_base": {"dim": 768, "has_text": False},
    "dinov2_large": {"dim": 1024, "has_text": False},
    "dinov3_base": {"dim": 768, "has_text": False},
}


# ============================================================
# EXPORT
# ============================================================

def get_listing_texts(conn: sqlite3.Connection, listing_ids: list) -> dict:
    """Get title + materials for listings."""
    if not listing_ids:
        return {}

    placeholders = ",".join(["?"] * len(listing_ids))
    cursor = conn.execute(f"""
        SELECT listing_id, title, materials
        FROM listings
        WHERE listing_id IN ({placeholders})
    """, listing_ids)

    texts = {}
    for row in cursor.fetchall():
        lid, title, materials_json = row
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
        texts[str(lid)] = ". ".join(parts) if parts else ""

    return texts


def cmd_export(args):
    """Export a batch of images for worker processing."""
    conn = get_connection()

    # Get images that need embedding
    print(f"Querying for images needing embedding (limit={args.job_size})...")
    images = get_images_for_embedding_batch(conn, limit=args.job_size)

    if not images:
        print("No images need embedding")
        return 0

    # Filter to images that exist on disk
    manifest = []
    for img in images:
        lid, iid = img["listing_id"], img["image_id"]
        img_path = IMAGES_DIR / f"{lid}_{iid}.jpg"
        if img_path.exists():
            manifest.append([lid, iid])

    if not manifest:
        print("No image files found on disk")
        return 1

    print(f"Found {len(manifest)} images to export")

    # Create export directory
    batch_dir = EXPORTS_DIR / f"batch_{args.batch_id}"
    if batch_dir.exists():
        if not args.force:
            print(f"ERROR: {batch_dir} already exists. Use --force to overwrite.")
            return 1
        shutil.rmtree(batch_dir)

    batch_dir.mkdir(parents=True)
    images_out_dir = batch_dir / "images"
    images_out_dir.mkdir()

    # Copy images
    print(f"Copying images to {batch_dir}...")
    for lid, iid in manifest:
        src = IMAGES_DIR / f"{lid}_{iid}.jpg"
        dst = images_out_dir / f"{lid}_{iid}.jpg"
        shutil.copy(src, dst)

    # Save manifest
    (batch_dir / "manifest.json").write_text(json.dumps(manifest))

    # Get and save listing texts
    listing_ids = list(set(lid for lid, _ in manifest))
    texts = get_listing_texts(conn, listing_ids)
    (batch_dir / "listings.json").write_text(json.dumps(texts))

    conn.close()

    print(f"\nExported batch_{args.batch_id}:")
    print(f"  Images: {len(manifest)}")
    print(f"  Location: {batch_dir}")
    print(f"\nNext steps:")
    print(f"  1. Copy {batch_dir} to worker machine")
    print(f"  2. Run: python embed_worker.py --input <batch_dir>")
    print(f"  3. Copy results back")
    print(f"  4. Run: python embed_manager.py import --results-dir <results_dir>")

    return 0


# ============================================================
# IMPORT
# ============================================================

def load_faiss_index(path: Path):
    """Load FAISS index from file."""
    import faiss
    if path.exists():
        return faiss.read_index(str(path))
    return None


def create_faiss_index(dim: int):
    """Create new FAISS index."""
    import faiss
    return faiss.IndexFlatIP(dim)


def save_faiss_index(index, path: Path):
    """Save FAISS index atomically."""
    import faiss
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix='.faiss.tmp')
    import os
    os.close(temp_fd)

    try:
        faiss.write_index(index, temp_path)
        os.replace(temp_path, str(path))
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def validate_results(results_dir: Path, required_models: list = None) -> tuple[list, dict, list]:
    """Validate worker results before importing.

    Args:
        results_dir: Directory containing worker output
        required_models: List of model keys to expect. If None, accepts any models present.

    Returns (manifest, numpy_files, models_found) if valid, raises exception if not.
    """
    # Check manifest exists
    manifest_file = results_dir / "manifest.json"
    if not manifest_file.exists():
        raise ValueError(f"manifest.json not found in {results_dir}")

    manifest = json.loads(manifest_file.read_text())
    if not manifest:
        raise ValueError("Empty manifest")

    expected_count = len(manifest)
    print(f"Manifest: {expected_count} images")

    # Check which models to validate
    if required_models:
        models_to_check = {k: MODELS[k] for k in required_models if k in MODELS}
    else:
        # Auto-detect which models are present
        models_to_check = {}
        for model_key, info in MODELS.items():
            if (results_dir / f"{model_key}.npy").exists():
                models_to_check[model_key] = info

    if not models_to_check:
        raise ValueError("No model .npy files found")

    # Validate numpy files
    numpy_files = {}
    models_found = []

    for model_key, info in models_to_check.items():
        npy_file = results_dir / f"{model_key}.npy"
        if not npy_file.exists():
            if required_models:
                raise ValueError(f"Missing {model_key}.npy")
            continue

        arr = np.load(npy_file)
        if arr.shape[0] != expected_count:
            raise ValueError(
                f"{model_key}.npy has {arr.shape[0]} rows, expected {expected_count}"
            )
        if arr.shape[1] != info["dim"]:
            raise ValueError(
                f"{model_key}.npy has dim={arr.shape[1]}, expected {info['dim']}"
            )

        numpy_files[model_key] = arr
        models_found.append(model_key)

        # Check text embeddings for CLIP
        if info["has_text"]:
            text_file = results_dir / f"{model_key}_text.npy"
            if text_file.exists():
                text_arr = np.load(text_file)
                if text_arr.shape[0] != expected_count:
                    raise ValueError(
                        f"{model_key}_text.npy has {text_arr.shape[0]} rows, expected {expected_count}"
                    )
                numpy_files[f"{model_key}_text"] = text_arr

        print(f"  {model_key}: OK ({arr.shape})")

    return manifest, numpy_files, models_found


def cmd_import(args):
    """Import validated worker results into FAISS and DB."""
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    # Parse models if specified
    required_models = args.models.split(',') if hasattr(args, 'models') and args.models else None

    # Validate results
    print("Validating results...")
    try:
        manifest, numpy_files, models_found = validate_results(results_dir, required_models)
    except ValueError as e:
        print(f"ERROR: Validation failed: {e}")
        return 1

    print(f"Validation passed! Models: {models_found}")

    if args.dry_run:
        print("\n[DRY RUN] Would import:")
        print(f"  Images: {len(manifest)}")
        print(f"  Models: {models_found}")
        return 0

    # Load shared image index
    if IMAGE_INDEX_FILE.exists():
        existing_index = json.loads(IMAGE_INDEX_FILE.read_text())
    else:
        existing_index = []

    existing_set = set(tuple(x) for x in existing_index)

    # Find new images (not in shared index)
    new_indices = [i for i, img in enumerate(manifest) if tuple(img) not in existing_set]

    if not new_indices:
        print("All images already in index, nothing to do")
        return 0

    print(f"\nImporting {len(new_indices)} new images for models: {models_found}")

    # Append to FAISS indexes
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in models_found:
        info = MODELS[model_key]
        faiss_file = EMBEDDINGS_DIR / f"{model_key}.faiss"

        # Load or create index
        index = load_faiss_index(faiss_file)
        if index is None:
            index = create_faiss_index(info["dim"])
            print(f"  {model_key}: created new index")
        else:
            print(f"  {model_key}: loaded existing index ({index.ntotal} vectors)")

        # Add new embeddings
        new_emb = numpy_files[model_key][new_indices]
        index.add(new_emb)

        save_faiss_index(index, faiss_file)
        print(f"  {model_key}: added {len(new_indices)} vectors (total: {index.ntotal})")

        # Handle text embeddings
        if info["has_text"] and f"{model_key}_text" in numpy_files:
            text_file = EMBEDDINGS_DIR / f"{model_key}_text.faiss"
            text_index = load_faiss_index(text_file)
            if text_index is None:
                text_index = create_faiss_index(info["dim"])

            new_text_emb = numpy_files[f"{model_key}_text"][new_indices]
            text_index.add(new_text_emb)
            save_faiss_index(text_index, text_file)

    # Update shared image index
    for i in new_indices:
        existing_index.append(manifest[i])

    # Save index atomically
    import tempfile
    import os
    temp_fd, temp_path = tempfile.mkstemp(dir=EMBEDDINGS_DIR, suffix='.json.tmp')
    os.close(temp_fd)
    try:
        with open(temp_path, 'w') as f:
            json.dump(existing_index, f)
        os.replace(temp_path, str(IMAGE_INDEX_FILE))
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    print(f"  image_index.json: {len(existing_index)} total entries")

    # Update DB flags for each model separately
    print("Updating database flags...")
    conn = get_connection()

    for model_key in models_found:
        index_file = get_model_index_file(model_key)
        if index_file.exists():
            model_index = json.loads(index_file.read_text())
            existing_set = set(tuple(x) for x in model_index)

            # Mark all images that are now in this model's index
            for img in manifest:
                if tuple(img) in existing_set:
                    lid, iid = img
                    mark_embedded(conn, lid, iid, model_key)

    commit_with_retry(conn)
    conn.close()

    print(f"\nImport complete!")
    print(f"  Total new embeddings added: {total_added}")

    return 0


# ============================================================
# STATUS
# ============================================================

def cmd_status(args):
    """Show current embedding status."""
    import faiss

    print("=== Embedding Status ===\n")

    # Image index
    if IMAGE_INDEX_FILE.exists():
        index = json.loads(IMAGE_INDEX_FILE.read_text())
        print(f"image_index.json: {len(index)} entries")
    else:
        print("image_index.json: not found")

    print()

    # FAISS indexes
    for model_key, info in MODELS.items():
        faiss_file = EMBEDDINGS_DIR / f"{model_key}.faiss"
        if faiss_file.exists():
            idx = faiss.read_index(str(faiss_file))
            print(f"{model_key}.faiss: {idx.ntotal} vectors (dim={info['dim']})")
        else:
            print(f"{model_key}.faiss: not found")

    print()

    # DB stats
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE bg_removed = 1")
    bg_done = cursor.fetchone()[0]

    for model_key in MODELS:
        col = f"embed_{model_key}"
        cursor.execute(f"SELECT COUNT(*) FROM image_status WHERE {col} = 1")
        count = cursor.fetchone()[0]
        print(f"DB {col}: {count}")

    conn.close()

    print(f"\nImages ready for embedding (bg_removed=1): {bg_done}")

    # Pending exports
    if EXPORTS_DIR.exists():
        batches = list(EXPORTS_DIR.glob("batch_*"))
        if batches:
            print(f"\nPending exports: {len(batches)}")
            for b in sorted(batches):
                manifest_file = b / "manifest.json"
                if manifest_file.exists():
                    count = len(json.loads(manifest_file.read_text()))
                    print(f"  {b.name}: {count} images")

    return 0


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Embedding manager - export batches, import results"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export batch for workers")
    export_parser.add_argument(
        "--batch-id", required=True,
        help="Batch identifier (e.g., 001, 002)"
    )
    export_parser.add_argument(
        "--job-size", type=int, default=50000,
        help="Number of images per batch"
    )
    export_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing batch directory"
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import worker results")
    import_parser.add_argument(
        "--results-dir", required=True,
        help="Directory containing worker results"
    )
    import_parser.add_argument(
        "--models",
        help="Comma-separated list of models to import (default: auto-detect)"
    )
    import_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate without actually importing"
    )

    # Status command
    subparsers.add_parser("status", help="Show embedding status")

    args = parser.parse_args()

    if args.command == "export":
        return cmd_export(args)
    elif args.command == "import":
        return cmd_import(args)
    elif args.command == "status":
        return cmd_status(args)


if __name__ == "__main__":
    exit(main())
