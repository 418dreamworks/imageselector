"""Shared shard utilities for FAISS index management.

Shards split the monolithic FAISS indexes into 500K-row chunks.
Each shard_NNNN/ directory contains its own image_index.json and 7 FAISS files.
Completed shards (500K rows) are immutable; only the last shard is active.
"""

import json
import time
from pathlib import Path
from typing import List, Tuple

SHARD_SIZE = 500_000
EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / 'data' / 'embeddings'

ALL_FAISS_FILES = [
    'clip_vitb32.faiss', 'clip_vitb32_text.faiss',
    'clip_vitl14.faiss', 'clip_vitl14_text.faiss',
    'dinov2_base.faiss', 'dinov2_large.faiss', 'dinov3_base.faiss',
]


def shard_dir_name(shard_num: int) -> str:
    return f"shard_{shard_num:04d}"


def get_shard_dirs() -> List[Path]:
    """Return sorted list of shard_NNNN/ directories."""
    if not EMBEDDINGS_DIR.exists():
        return []
    dirs = sorted(d for d in EMBEDDINGS_DIR.iterdir()
                  if d.is_dir() and d.name.startswith('shard_'))
    return dirs


def get_active_shard() -> Tuple[Path, int]:
    """Return (active_shard_dir, current_row_count).

    Active shard = last directory. Creates shard_0000 if none exist.
    """
    dirs = get_shard_dirs()
    if not dirs:
        shard_path = EMBEDDINGS_DIR / shard_dir_name(0)
        shard_path.mkdir(parents=True, exist_ok=True)
        return shard_path, 0

    active = dirs[-1]
    idx = load_shard_index(active)
    return active, len(idx)


def get_shard_num(shard_dir: Path) -> int:
    """Extract shard number from directory name."""
    return int(shard_dir.name.split('_')[1])


def get_total_rows() -> int:
    """Sum of image_index lengths across all shards."""
    total = 0
    for shard_dir in get_shard_dirs():
        idx = load_shard_index(shard_dir)
        total += len(idx)
    return total


def load_all_embedded_pairs() -> set:
    """Load non-finalized shards' image_index, return set of (lid, iid).

    Finalized shards (500K rows) are skipped — their images are download_done=3
    so they won't appear in DB queries and don't need dedup tracking.
    """
    pairs = set()
    for shard_dir in get_shard_dirs():
        idx = load_shard_index(shard_dir)
        if len(idx) >= SHARD_SIZE:
            continue  # finalized, skip
        for entry in idx:
            pairs.add((entry[0], entry[1]))
    return pairs


def load_shard_index(shard_dir: Path) -> list:
    """Load one shard's image_index.json. Waits for lock."""
    index_file = shard_dir / 'image_index.json'
    lock_file = shard_dir / 'image_index.json.lock'

    while True:
        if lock_file.exists():
            time.sleep(0.5)
            continue
        if not index_file.exists():
            return []
        try:
            return json.loads(index_file.read_text())
        except (json.JSONDecodeError, ValueError):
            time.sleep(0.5)


def save_shard_index(shard_dir: Path, data: list):
    """Save one shard's image_index.json with lock file."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_file = shard_dir / 'image_index.json'
    lock_file = shard_dir / 'image_index.json.lock'
    try:
        lock_file.touch()
        index_file.write_text(json.dumps(data))
    finally:
        lock_file.unlink(missing_ok=True)


def check_shard_consistency(shard_dir: Path, exclude: set = None) -> Tuple[bool, str]:
    """Verify all FAISS files in shard have same ntotal, matching image_index.

    Args:
        shard_dir: Path to shard directory
        exclude: Set of FAISS filenames to skip (e.g. {'clip_vitl14.faiss'} for
                 known-empty placeholders awaiting re-embed)

    Returns (is_consistent, message).
    """
    import faiss

    exclude = exclude or set()
    idx = load_shard_index(shard_dir)
    index_count = len(idx)
    name = shard_dir.name

    faiss_counts = {}
    for fname in ALL_FAISS_FILES:
        if fname in exclude:
            continue
        fpath = shard_dir / fname
        if fpath.exists():
            index = faiss.read_index(str(fpath))
            faiss_counts[fname] = index.ntotal

    if not faiss_counts and index_count == 0:
        return True, f"{name}: empty"

    if not faiss_counts and index_count > 0:
        return False, f"{name}: image_index has {index_count} but no FAISS files"

    # All FAISS files should have same count
    counts = list(faiss_counts.values())
    if len(set(counts)) > 1:
        return False, f"{name}: FAISS files have different counts: {faiss_counts}"

    faiss_count = counts[0]
    if index_count != faiss_count:
        return False, f"{name}: image_index has {index_count}, FAISS has {faiss_count}"

    skipped = f", excluded: {exclude}" if exclude else ""
    return True, f"{name}: consistent ({index_count} rows{skipped})"


def check_all_shards_consistency(exclude: set = None) -> Tuple[bool, str]:
    """Check consistency of active shard only. Returns (all_ok, combined_message).

    Finalized shards (500K rows) are immutable and already verified, so skipped.

    Args:
        exclude: Set of FAISS filenames to skip (e.g. {'clip_vitl14.faiss'})
    """
    import faiss  # noqa: F401 — ensure available

    dirs = get_shard_dirs()
    if not dirs:
        return True, "No shards exist"

    messages = []
    all_ok = True
    total = 0

    for shard_dir in dirs:
        idx = load_shard_index(shard_dir)
        rows = len(idx)
        total += rows
        if rows >= SHARD_SIZE:
            messages.append(f"{shard_dir.name}: finalized ({rows} rows, skipped)")
            continue
        ok, msg = check_shard_consistency(shard_dir, exclude=exclude)
        messages.append(msg)
        if not ok:
            all_ok = False

    summary = f"Total rows: {total}" if all_ok else "INCONSISTENCIES FOUND"
    return all_ok, f"{summary}\n  " + "\n  ".join(messages)
