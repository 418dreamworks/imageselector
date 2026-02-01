"""Filter images to keep up to 5 best per listing.

Algorithms:
1. Zoom detection: All-pairs similarity + edge density comparison
2. Diversity selection: Bounded farthest-point sampling

Run as Streamlit app to visually inspect filtering decisions:
    streamlit run filter_images.py

Or run batch processing:
    python filter_images.py --batch
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import random
import argparse
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import torch
import requests
import httpx
from PIL import Image
from dotenv import load_dotenv

# For embedding computation
import open_clip

# For background removal (CPU mode for training)
from rembg import remove as rembg_remove

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")
ETSY_API_KEY = os.getenv("ETSY_API_KEY")

DEV_DIR = Path(__file__).parent.parent / "dev"
IMAGES_DIR = DEV_DIR / "images"  # Source: all downloaded images
METADATA_FILE = DEV_DIR.parent / "image_metadata.json"  # On iMac, this is in the project root
TRAINING_DATA_FILE = Path(__file__).parent / "training_data.json"

# Filtering parameters
MAX_IMAGES_TO_KEEP = 5
ZOOM_SIMILARITY_MIN = 0.70  # Pairs with similarity in this range may be zooms
ZOOM_SIMILARITY_MAX = 0.95
DUPLICATE_SIMILARITY = 0.95  # Above this = nearly identical
MIN_SIMILARITY_TO_PRIMARY = 0.40  # Below this = different item
EDGE_RATIO_THRESHOLD = 0.5  # If edge_A < 0.5 * edge_B, A is likely zoom of B

# Global model cache
_model = None
_preprocess = None
_device = None


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def remove_background_cpu(image: Image.Image) -> Image.Image:
    """Remove background from image using CPU-only rembg.

    Used for training data to match production nobg images.
    """
    # rembg.remove() works on PIL images directly
    # By not specifying providers, it defaults to CPU
    nobg = rembg_remove(image)
    # Convert RGBA to RGB with white background for CLIP
    if nobg.mode == 'RGBA':
        background = Image.new('RGB', nobg.size, (255, 255, 255))
        background.paste(nobg, mask=nobg.split()[3])  # Use alpha as mask
        return background
    return nobg.convert('RGB')


def load_clip_model():
    """Load CLIP model for embedding computation."""
    global _model, _preprocess, _device
    if _model is None:
        _device = get_device()
        print(f"Loading CLIP model on {_device}...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=_device
        )
        _model.eval()
    return _model, _preprocess, _device


def compute_embedding(image: Image.Image) -> np.ndarray:
    """Compute normalized CLIP embedding for an image."""
    model, preprocess, device = load_clip_model()
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def compute_edge_density(image: Image.Image) -> float:
    """Compute edge density using Canny edge detection.

    Higher density = more edges = more context = probably not a zoom.
    Lower density = fewer edges = less context = probably a zoom.
    """
    # Convert to grayscale numpy array
    img_gray = np.array(image.convert("L"))

    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Compute ratio of edge pixels
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2))


def detect_zooms(images: list[dict]) -> list[dict]:
    """Detect and mark zoom images using all-pairs comparison.

    Args:
        images: List of dicts with 'path', 'embedding', 'edge_density', 'rank'

    Returns:
        Same list with 'is_zoom' and 'zoom_of' fields added.
    """
    n = len(images)

    # Initialize
    for img in images:
        img['is_zoom'] = False
        img['zoom_of'] = None
        img['is_duplicate'] = False
        img['duplicate_of'] = None

    # All-pairs comparison
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(images[i]['embedding'], images[j]['embedding'])

            # Check for duplicates (nearly identical)
            if sim > DUPLICATE_SIMILARITY:
                # Keep the one with higher rank (lower number = more important)
                if images[i]['rank'] <= images[j]['rank']:
                    images[j]['is_duplicate'] = True
                    images[j]['duplicate_of'] = images[i]['path'].name
                else:
                    images[i]['is_duplicate'] = True
                    images[i]['duplicate_of'] = images[j]['path'].name
                continue

            # Check for zooms (similar but one has less edge density)
            if ZOOM_SIMILARITY_MIN <= sim <= ZOOM_SIMILARITY_MAX:
                edge_i = images[i]['edge_density']
                edge_j = images[j]['edge_density']

                # Image with significantly less edge density is likely a zoom
                if edge_i < EDGE_RATIO_THRESHOLD * edge_j:
                    images[i]['is_zoom'] = True
                    images[i]['zoom_of'] = images[j]['path'].name
                elif edge_j < EDGE_RATIO_THRESHOLD * edge_i:
                    images[j]['is_zoom'] = True
                    images[j]['zoom_of'] = images[i]['path'].name

    return images


def select_diverse_images(images: list[dict], max_count: int = MAX_IMAGES_TO_KEEP) -> list[dict]:
    """Select up to max_count most diverse images using farthest-point sampling.

    Args:
        images: List of image dicts (already filtered for zooms/duplicates)
        max_count: Maximum number of images to keep

    Returns:
        List of selected image dicts with 'selected' and 'selection_reason' fields.
    """
    if not images:
        return []

    # Find primary image (rank 1)
    primary = None
    for img in images:
        if img['rank'] == 1:
            primary = img
            break
    if primary is None:
        primary = images[0]  # Fallback to first image

    primary_emb = primary['embedding']

    # Filter candidates: must be similar enough to primary (same item)
    # but not too similar (would be redundant)
    valid_candidates = []
    for img in images:
        if img is primary:
            continue
        if img['is_zoom'] or img['is_duplicate']:
            continue

        sim_to_primary = cosine_similarity(img['embedding'], primary_emb)
        img['sim_to_primary'] = sim_to_primary

        if MIN_SIMILARITY_TO_PRIMARY < sim_to_primary < DUPLICATE_SIMILARITY:
            valid_candidates.append(img)
        elif sim_to_primary <= MIN_SIMILARITY_TO_PRIMARY:
            img['rejected_reason'] = f'different_item (sim={sim_to_primary:.2f})'

    # Start with primary
    selected = [primary]
    primary['selected'] = True
    primary['selection_reason'] = 'primary (rank 1)'

    # Farthest-point sampling
    while len(selected) < max_count and valid_candidates:
        best_candidate = None
        best_min_dist = -1

        for candidate in valid_candidates:
            # Find minimum distance to any selected image
            min_dist = float('inf')
            for sel in selected:
                dist = 1 - cosine_similarity(candidate['embedding'], sel['embedding'])
                min_dist = min(min_dist, dist)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate

        if best_candidate:
            best_candidate['selected'] = True
            best_candidate['selection_reason'] = f'diverse (min_dist={best_min_dist:.3f})'
            selected.append(best_candidate)
            valid_candidates.remove(best_candidate)

    # Mark remaining as not selected
    for img in images:
        if 'selected' not in img:
            img['selected'] = False
            if 'rejected_reason' not in img:
                if img['is_zoom']:
                    img['rejected_reason'] = f'zoom of {img["zoom_of"]}'
                elif img['is_duplicate']:
                    img['rejected_reason'] = f'duplicate of {img["duplicate_of"]}'
                else:
                    img['rejected_reason'] = 'diversity limit (>5 images)'

    return images


def filter_listing_images(image_paths: list[Path]) -> list[dict]:
    """Run full filtering pipeline on a listing's images.

    Args:
        image_paths: List of paths to images for this listing

    Returns:
        List of dicts with filtering results for each image.
    """
    if not image_paths:
        return []

    # Load images and compute features
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")

            # Parse rank from filename if possible
            # Format: {listing_id}-{image_id}.jpg or {listing_id}.jpg
            stem = path.stem
            if '-' in stem:
                # New format with image_id - we don't have rank info, use index
                rank = len(images) + 1
            else:
                rank = 1  # Legacy single image is always primary

            emb = compute_embedding(img)
            edge = compute_edge_density(img)

            images.append({
                'path': path,
                'image': img,
                'embedding': emb,
                'edge_density': edge,
                'rank': rank,
            })
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not images:
        return []

    # Step 1: Detect zooms and duplicates
    images = detect_zooms(images)

    # Step 2: Select diverse images
    images = select_diverse_images(images)

    return images


def get_listing_images(listing_id: int) -> list[Path]:
    """Get all image paths for a listing.

    Supports multiple naming formats:
    - {listing_id}-{image_id}.jpg (new hyphen format)
    - {listing_id}_{image_id}.jpg (legacy underscore format)
    - {listing_id}.jpg (legacy single image)
    """
    paths = []

    # Check hyphen format
    paths.extend(IMAGES_DIR.glob(f"{listing_id}-*.jpg"))

    # Check underscore format
    paths.extend(IMAGES_DIR.glob(f"{listing_id}_*.jpg"))

    # Check single image format
    single = IMAGES_DIR / f"{listing_id}.jpg"
    if single.exists() and single not in paths:
        paths.append(single)

    return sorted(set(paths))


def get_listings_with_multiple_images() -> list[int]:
    """Get listing IDs that have multiple images."""
    # Group files by listing ID
    listing_counts = {}

    for f in IMAGES_DIR.glob("*.jpg"):
        stem = f.stem
        # Parse listing ID from various formats
        if '-' in stem:
            lid = stem.split('-')[0]
        elif '_' in stem:
            lid = stem.split('_')[0]
        else:
            lid = stem

        try:
            lid = int(lid)
            listing_counts[lid] = listing_counts.get(lid, 0) + 1
        except ValueError:
            pass

    # Return listings with more than 1 image
    return [lid for lid, count in listing_counts.items() if count > 1]


def get_demo_image_groups(group_size: int = 8) -> list[list[Path]]:
    """Create demo groups of random images for testing the algorithm.

    Since we don't have multi-image listings yet, this groups random
    single-image listings together to demonstrate how filtering works.
    """
    all_images = list(IMAGES_DIR.glob("*.jpg"))
    random.shuffle(all_images)

    # Create groups of group_size images
    groups = []
    for i in range(0, len(all_images) - group_size, group_size):
        groups.append(all_images[i:i + group_size])

    return groups


# =============================================================================
# Training data functions
# =============================================================================

def load_training_data() -> dict:
    """Load training data from JSON file."""
    if TRAINING_DATA_FILE.exists():
        with open(TRAINING_DATA_FILE) as f:
            return json.load(f)
    return {"examples": [], "derived_rules": {}}


def save_training_data(data: dict):
    """Save training data to JSON file."""
    with open(TRAINING_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def derive_rules_from_training(data: dict) -> dict:
    """Analyze training examples to derive optimal thresholds."""
    examples = data.get("examples", [])
    if not examples:
        return {}

    # Collect stats by label type
    stats_by_label = {
        "keep": {"sim": [], "edge_ratio": []},
        "zoom": {"sim": [], "edge_ratio": []},
        "duplicate": {"sim": [], "edge_ratio": []},
        "different": {"sim": [], "edge_ratio": []},
        "bad_quality": {"sim": [], "edge_ratio": []},
    }

    for ex in examples:
        for img in ex.get("images", []):
            if img.get("is_primary"):
                continue
            label = img.get("label", "keep" if img.get("user_keep") else "different")
            if label in stats_by_label:
                stats_by_label[label]["sim"].append(img.get("sim_to_primary", 0))
                stats_by_label[label]["edge_ratio"].append(img.get("edge_ratio", 1))

    rules = {"n_examples": len(examples)}

    for label, stats in stats_by_label.items():
        if stats["sim"]:
            rules[f"{label}_count"] = len(stats["sim"])
            rules[f"{label}_sim_min"] = min(stats["sim"])
            rules[f"{label}_sim_max"] = max(stats["sim"])
            rules[f"{label}_sim_mean"] = sum(stats["sim"]) / len(stats["sim"])
        if stats["edge_ratio"]:
            rules[f"{label}_edge_min"] = min(stats["edge_ratio"])
            rules[f"{label}_edge_max"] = max(stats["edge_ratio"])
            rules[f"{label}_edge_mean"] = sum(stats["edge_ratio"]) / len(stats["edge_ratio"])

    return rules


# =============================================================================
# Etsy API functions for fetching listing images
# =============================================================================

def load_listing_ids_from_metadata() -> list[int]:
    """Load all listing IDs from image_metadata.json."""
    if not METADATA_FILE.exists():
        return []
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    return [int(lid) for lid in metadata.keys()]


def fetch_listing_images_from_etsy(listing_id: int) -> list[dict]:
    """Fetch all images for a listing from Etsy API.

    Returns list of dicts with image info:
    [{"url": "...", "rank": 1, "image_id": 123}, ...]
    """
    if not ETSY_API_KEY:
        raise ValueError("ETSY_API_KEY not set in .env")

    url = f"https://openapi.etsy.com/v3/application/listings/{listing_id}/images"
    headers = {"x-api-key": ETSY_API_KEY}

    try:
        response = httpx.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        images = []
        for img in data.get("results", []):
            # Get the full-size URL (url_fullxfull)
            images.append({
                "url": img.get("url_fullxfull"),
                "rank": img.get("rank", 0),
                "image_id": img.get("listing_image_id"),
            })

        # Sort by rank (primary image first)
        images.sort(key=lambda x: x["rank"])
        return images

    except httpx.HTTPStatusError as e:
        print(f"Error fetching images for listing {listing_id}: {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def download_image_from_url(url: str) -> Image.Image | None:
    """Download image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def fetch_and_filter_listing(listing_id: int) -> tuple[list[dict], list[Image.Image]]:
    """Fetch all images for a listing, download them, and run filter algorithm.

    Returns (filter_results, all_images) where:
    - filter_results: list of dicts with selection info
    - all_images: list of PIL Images in same order
    """
    # Fetch image URLs from Etsy
    image_infos = fetch_listing_images_from_etsy(listing_id)
    if not image_infos:
        return [], []

    # Download all images
    images = []
    valid_infos = []
    for info in image_infos:
        img = download_image_from_url(info["url"])
        if img:
            images.append(img)
            valid_infos.append(info)

    if not images:
        return [], []

    # Run filter algorithm on the images directly
    results = filter_images_pil(images, valid_infos)
    return results, images


def filter_images_pil(images: list[Image.Image], infos: list[dict], use_nobg: bool = True) -> list[dict]:
    """Run filter algorithm on PIL images instead of file paths.

    Args:
        images: List of PIL Images
        infos: List of dicts with image metadata (url, rank, image_id)
        use_nobg: If True, remove backgrounds before computing embeddings (for training)

    Returns list of result dicts with selection status.
    """
    if not images:
        return []

    n = len(images)

    # Remove backgrounds if requested (for training to match production)
    nobg_images = []
    if use_nobg:
        print(f"Removing backgrounds from {n} images (CPU mode)...")
        for i, img in enumerate(images):
            print(f"  Processing image {i+1}/{n}...")
            nobg = remove_background_cpu(img)
            nobg_images.append(nobg)
    else:
        nobg_images = images

    # Compute embeddings and edge densities on nobg images
    embeddings = []
    edge_densities = []

    for nobg_img in nobg_images:
        emb = compute_embedding(nobg_img)
        embeddings.append(emb)
        edge_densities.append(compute_edge_density(nobg_img))

    embeddings = np.array(embeddings)

    # Compute pairwise similarities
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarities[i, j] = cosine_similarity(embeddings[i], embeddings[j])
            else:
                similarities[i, j] = 1.0

    # Step 1: Identify zooms/duplicates to skip
    skip_indices = set()

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarities[i, j]

            # Near duplicate
            if sim >= DUPLICATE_SIMILARITY:
                # Keep the one with higher resolution (larger edge density as proxy)
                if edge_densities[i] < edge_densities[j]:
                    skip_indices.add(i)
                else:
                    skip_indices.add(j)

            # Potential zoom relationship
            elif ZOOM_SIMILARITY_MIN <= sim <= ZOOM_SIMILARITY_MAX:
                edge_i, edge_j = edge_densities[i], edge_densities[j]

                if edge_i < EDGE_RATIO_THRESHOLD * edge_j:
                    skip_indices.add(i)  # i is zoom of j
                elif edge_j < EDGE_RATIO_THRESHOLD * edge_i:
                    skip_indices.add(j)  # j is zoom of i

    # Step 2: Diversity selection from remaining
    primary_idx = 0  # First image is primary (rank 0 or 1)
    primary_emb = embeddings[primary_idx]

    # Start with primary always selected
    selected_indices = [primary_idx]
    candidates = [i for i in range(n) if i not in skip_indices and i != primary_idx]

    # Filter candidates by similarity bounds
    valid_candidates = []
    for idx in candidates:
        sim_to_primary = cosine_similarity(embeddings[idx], primary_emb)
        if MIN_SIMILARITY_TO_PRIMARY <= sim_to_primary <= ZOOM_SIMILARITY_MAX:
            valid_candidates.append(idx)

    # Farthest-point sampling for diversity
    while len(selected_indices) < MAX_IMAGES_TO_KEEP and valid_candidates:
        best_idx = None
        best_min_dist = -1

        for idx in valid_candidates:
            # Find minimum distance to any selected image
            min_dist = min(1 - similarities[idx, sel] for sel in selected_indices)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            valid_candidates.remove(best_idx)
        else:
            break

    # Build result list with devil's advocate reasoning
    results = []
    for i, (img, info) in enumerate(zip(images, infos)):
        sim_to_primary = similarities[i, primary_idx] if i != primary_idx else 1.0

        # Compute metrics for devil's advocate reasoning
        max_sim_to_others = 0.0
        min_sim_to_others = 1.0
        for j in range(n):
            if j != i:
                sim = similarities[i, j]
                max_sim_to_others = max(max_sim_to_others, sim)
                min_sim_to_others = min(min_sim_to_others, sim)

        avg_edge = sum(edge_densities) / n if n > 0 else 0
        edge_ratio_vs_avg = edge_densities[i] / avg_edge if avg_edge > 0 else 1.0

        if i in selected_indices:
            if i == primary_idx:
                reason = "Primary image"
                # Devil's advocate: why might primary be wrong?
                potential_rejection = []
                if edge_densities[i] < 0.5 * avg_edge:
                    potential_rejection.append(f"Low edge density ({edge_densities[i]:.4f} vs avg {avg_edge:.4f}) - could be too simple/zoomed")
                if max_sim_to_others > 0.9:
                    potential_rejection.append(f"Very similar to another image (sim={max_sim_to_others:.2f}) - redundant?")
                devils_advocate = "; ".join(potential_rejection) if potential_rejection else "Strong primary - no concerns"
            else:
                reason = f"Diverse angle (sim={sim_to_primary:.2f})"
                # Devil's advocate: why might this selection be wrong?
                potential_rejection = []
                if sim_to_primary < 0.5:
                    potential_rejection.append(f"Low similarity to primary ({sim_to_primary:.2f}) - might be different item")
                if edge_densities[i] < 0.5 * avg_edge:
                    potential_rejection.append(f"Low edge density ({edge_densities[i]:.4f}) - could be a zoom")
                for j in selected_indices:
                    if j != i and j != primary_idx:
                        sim = similarities[i, j]
                        if sim > 0.85:
                            potential_rejection.append(f"Similar to selected image {j+1} (sim={sim:.2f}) - redundant?")
                            break
                devils_advocate = "; ".join(potential_rejection) if potential_rejection else "Good diverse selection"
            selected = True
        elif i in skip_indices:
            # Determine why skipped
            skip_reason_detail = None
            for j in range(n):
                if j != i:
                    sim = similarities[i, j]
                    if sim >= DUPLICATE_SIMILARITY:
                        reason = f"Duplicate of image {j+1}"
                        skip_reason_detail = ("duplicate", j, sim)
                        break
                    elif ZOOM_SIMILARITY_MIN <= sim <= ZOOM_SIMILARITY_MAX:
                        if edge_densities[i] < EDGE_RATIO_THRESHOLD * edge_densities[j]:
                            reason = f"Zoom of image {j+1}"
                            skip_reason_detail = ("zoom", j, sim)
                            break
            else:
                reason = "Skipped (unknown)"
            selected = False

            # Devil's advocate: why might this rejection be wrong?
            potential_acceptance = []
            if skip_reason_detail:
                detail_type, other_idx, sim = skip_reason_detail
                if detail_type == "duplicate":
                    if edge_densities[i] > edge_densities[other_idx]:
                        potential_acceptance.append(f"Higher edge density than image {other_idx+1} ({edge_densities[i]:.4f} vs {edge_densities[other_idx]:.4f}) - this might be sharper")
                    if sim < 0.98:
                        potential_acceptance.append(f"Similarity {sim:.2f} - not that close, could show different detail")
                elif detail_type == "zoom":
                    if edge_densities[i] > 0.03:
                        potential_acceptance.append(f"Edge density {edge_densities[i]:.4f} not extremely low - might show useful detail close-up")
                    if abs(edge_densities[i] - edge_densities[other_idx]) < 0.01:
                        potential_acceptance.append(f"Edge densities similar ({edge_densities[i]:.4f} vs {edge_densities[other_idx]:.4f}) - zoom detection may be wrong")
            devils_advocate = "; ".join(potential_acceptance) if potential_acceptance else "Correctly rejected"
        else:
            reason = f"Not diverse enough (sim={sim_to_primary:.2f})"
            selected = False

            # Devil's advocate: why might this rejection be wrong?
            potential_acceptance = []
            if min_sim_to_others < 0.7:
                potential_acceptance.append(f"Low similarity to some images ({min_sim_to_others:.2f}) - actually shows unique angle?")
            if edge_densities[i] > avg_edge * 1.2:
                potential_acceptance.append(f"High edge density ({edge_densities[i]:.4f} vs avg {avg_edge:.4f}) - detailed shot worth keeping?")
            if len(selected_indices) < MAX_IMAGES_TO_KEEP:
                potential_acceptance.append(f"Only {len(selected_indices)} images selected - could include more diversity")
            devils_advocate = "; ".join(potential_acceptance) if potential_acceptance else "Correctly filtered for diversity"

        results.append({
            "image": img,  # Original for display
            "image_nobg": nobg_images[i] if use_nobg else img,  # Nobg for reference
            "info": info,
            "index": i,
            "rank": info.get("rank", i),
            "edge_density": edge_densities[i],  # Computed on nobg
            "sim_to_primary": sim_to_primary,  # Computed on nobg
            "selected": selected,
            "selection_reason": reason,
            "devils_advocate": devils_advocate,
            "used_nobg": use_nobg,
        })

    return results


# =============================================================================
# Streamlit UI for visual inspection
# =============================================================================

def run_streamlit_app():
    """Run Streamlit app for visual inspection of filtering.

    Fetches random listings from Etsy API and runs filter algorithm.
    """
    import streamlit as st
    global ZOOM_SIMILARITY_MIN, ZOOM_SIMILARITY_MAX, DUPLICATE_SIMILARITY
    global MIN_SIMILARITY_TO_PRIMARY, EDGE_RATIO_THRESHOLD

    st.set_page_config(page_title="Image Filter Inspector", layout="wide")
    st.title("Image Filter Inspector")
    st.caption("Fetches ALL images for a listing from Etsy, runs filter algorithm, shows results")

    # Initialize session state
    if 'initialized' not in st.session_state:
        # Load listing IDs from metadata
        st.session_state.listing_ids = load_listing_ids_from_metadata()
        if st.session_state.listing_ids:
            random.shuffle(st.session_state.listing_ids)
        st.session_state.current_listing = None
        st.session_state.results = None
        st.session_state.images = None
        st.session_state.history = []  # Track viewed listings
        st.session_state.initialized = True

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        if st.button("🎲 Random Listing", type="primary", use_container_width=True):
            if st.session_state.listing_ids:
                st.session_state.current_listing = random.choice(st.session_state.listing_ids)
                st.session_state.results = None  # Clear results to trigger re-fetch
                st.rerun()

        st.divider()

        st.header("Parameters")
        zoom_min = st.slider("Zoom similarity min", 0.5, 0.9, ZOOM_SIMILARITY_MIN, 0.05)
        zoom_max = st.slider("Zoom similarity max", 0.8, 0.99, ZOOM_SIMILARITY_MAX, 0.01)
        dup_sim = st.slider("Duplicate threshold", 0.9, 0.99, DUPLICATE_SIMILARITY, 0.01)
        min_sim = st.slider("Min similarity to primary", 0.2, 0.6, MIN_SIMILARITY_TO_PRIMARY, 0.05)
        edge_ratio = st.slider("Edge ratio threshold", 0.3, 0.7, EDGE_RATIO_THRESHOLD, 0.05)

        # Update globals
        ZOOM_SIMILARITY_MIN = zoom_min
        ZOOM_SIMILARITY_MAX = zoom_max
        DUPLICATE_SIMILARITY = dup_sim
        MIN_SIMILARITY_TO_PRIMARY = min_sim
        EDGE_RATIO_THRESHOLD = edge_ratio

        st.divider()
        st.header("Stats")
        st.write(f"Total listings: {len(st.session_state.listing_ids):,}")
        st.write(f"Listings viewed: {len(st.session_state.history)}")

        if st.session_state.history:
            st.divider()
            st.header("History")
            for lid in st.session_state.history[-10:]:
                if st.button(f"ID: {lid}", key=f"hist_{lid}"):
                    st.session_state.current_listing = lid
                    st.session_state.results = None
                    st.rerun()

    # Main content
    if not st.session_state.listing_ids:
        st.error("No listings found!")
        st.info(f"Make sure {METADATA_FILE} exists and contains listing data.")
        return

    if not st.session_state.current_listing:
        st.info("👆 Click **Random Listing** to fetch a listing from Etsy and run the filter algorithm.")
        return

    listing_id = st.session_state.current_listing
    st.subheader(f"Listing: {listing_id}")
    st.caption(f"[View on Etsy](https://www.etsy.com/listing/{listing_id})")

    # Fetch and filter if not already done
    if st.session_state.results is None:
        with st.spinner(f"Fetching images, removing backgrounds (CPU), computing embeddings..."):
            results, images = fetch_and_filter_listing(listing_id)
            st.session_state.results = results
            st.session_state.images = images

            # Add to history
            if listing_id not in st.session_state.history:
                st.session_state.history.append(listing_id)

    results = st.session_state.results

    if not results:
        st.warning(f"No images found for listing {listing_id}")
        st.info("This listing may have been removed or the API request failed.")
        return

    # Initialize user labels if not set
    if 'user_labels' not in st.session_state:
        st.session_state.user_labels = {}

    # Load training data for rules display
    training_data = load_training_data()

    # Show all images with label options
    st.markdown("### Label Each Image")
    st.caption("Top row: Original | Bottom row: Background removed (used for scoring)")

    LABEL_OPTIONS = ["keep", "zoom", "duplicate", "different", "bad_quality"]

    n_cols = min(len(results), 5)
    cols = st.columns(n_cols)

    for i, img_data in enumerate(results):
        idx = img_data['index']
        with cols[i % n_cols]:
            # Show both original and nobg images
            st.image(img_data['image'], use_container_width=True, caption="Original")
            if img_data.get('used_nobg') and img_data.get('image_nobg') is not None:
                st.image(img_data['image_nobg'], use_container_width=True, caption="No BG")
            st.caption(f"#{idx+1} | Sim: {img_data['sim_to_primary']:.2f}")

            # Radio buttons for label
            default_idx = 0 if img_data.get('selected', False) or idx == 0 else 1
            label = st.radio(
                f"Image {idx+1}",
                options=LABEL_OPTIONS,
                index=default_idx,
                key=f"label_{listing_id}_{idx}",
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state.user_labels[idx] = label

    # Save & Next button
    st.divider()
    if st.button("💾 Save & Next", type="primary", use_container_width=True):
        # Build example
        example = {
            "listing_id": listing_id,
            "used_nobg": results[0].get('used_nobg', True) if results else True,  # Scores from nobg images
            "images": []
        }

        primary_edge = results[0]['edge_density'] if results else 1

        for img_data in results:
            idx = img_data['index']
            user_label = st.session_state.user_labels.get(idx, "keep")
            example["images"].append({
                "index": idx,
                "rank": img_data['rank'],
                "label": user_label,
                "user_keep": user_label == "keep",
                "algo_keep": img_data.get('selected', False),
                "is_primary": idx == 0,
                "sim_to_primary": img_data['sim_to_primary'],  # Computed on nobg
                "edge_density": img_data['edge_density'],  # Computed on nobg
                "edge_ratio": img_data['edge_density'] / primary_edge if primary_edge > 0 else 1,
            })

        training_data["examples"].append(example)
        training_data["derived_rules"] = derive_rules_from_training(training_data)
        save_training_data(training_data)

        # Move to next
        st.session_state.current_listing = random.choice(st.session_state.listing_ids)
        st.session_state.results = None
        st.session_state.user_labels = {}
        st.rerun()

    # Show derived rules
    if training_data.get("derived_rules"):
        st.divider()
        st.markdown("### Derived Rules from Training Data")
        rules = training_data["derived_rules"]
        st.write(f"**{rules.get('n_examples', 0)} listings labeled**")

        for label in ["keep", "zoom", "duplicate", "different", "bad_quality"]:
            count = rules.get(f"{label}_count", 0)
            if count > 0:
                sim_range = f"{rules.get(f'{label}_sim_min', 0):.2f}-{rules.get(f'{label}_sim_max', 0):.2f}"
                edge_range = f"{rules.get(f'{label}_edge_min', 0):.2f}-{rules.get(f'{label}_edge_max', 0):.2f}"
                st.write(f"**{label}** ({count}): sim={sim_range}, edge_ratio={edge_range}")


def run_batch_processing():
    """Run batch processing on all listings."""
    print("Loading metadata...")

    if not METADATA_FILE.exists():
        print(f"Metadata file not found: {METADATA_FILE}")
        return

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    # Find listings that need filtering
    to_filter = []
    for lid, entry in metadata.items():
        if isinstance(entry, dict):
            if entry.get('images_obtained') and not entry.get('images_filtered'):
                to_filter.append(int(lid))

    print(f"Listings to filter: {len(to_filter)}")

    if not to_filter:
        print("Nothing to filter!")
        return

    # Process each listing
    from tqdm import tqdm

    stats = {'processed': 0, 'kept': 0, 'deleted': 0}

    for listing_id in tqdm(to_filter, desc="Filtering"):
        image_paths = get_listing_images(listing_id)
        if not image_paths:
            continue

        results = filter_listing_images(image_paths)

        # Delete discarded images
        for img_data in results:
            if img_data.get('selected'):
                stats['kept'] += 1
            else:
                # Delete the file
                img_data['path'].unlink(missing_ok=True)
                stats['deleted'] += 1

        # Update metadata
        lid_str = str(listing_id)
        if lid_str in metadata and isinstance(metadata[lid_str], dict):
            # Update images array to only include kept images
            kept_image_ids = []
            for img_data in results:
                if img_data.get('selected'):
                    # Parse image_id from filename
                    stem = img_data['path'].stem
                    if '-' in stem:
                        image_id = int(stem.split('-')[1])
                    elif '_' in stem:
                        image_id = int(stem.split('_')[1])
                    else:
                        image_id = 0
                    kept_image_ids.append(image_id)

            # Filter the images array
            if 'images' in metadata[lid_str]:
                metadata[lid_str]['images'] = [
                    img for img in metadata[lid_str]['images']
                    if img.get('image_id') in kept_image_ids
                ]

            metadata[lid_str]['images_filtered'] = True

        stats['processed'] += 1

        # Save periodically
        if stats['processed'] % 100 == 0:
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f)
            print(f"\nProgress: {stats['processed']} listings, {stats['kept']} kept, {stats['deleted']} deleted")

    # Final save
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

    print(f"\nDone!")
    print(f"  Processed: {stats['processed']} listings")
    print(f"  Kept: {stats['kept']} images")
    print(f"  Deleted: {stats['deleted']} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter images for listings")
    parser.add_argument("--batch", action="store_true", help="Run batch processing instead of UI")
    args = parser.parse_args()

    if args.batch:
        run_batch_processing()
    else:
        run_streamlit_app()
