"""Training data collector for image filter rules.

Collect labeled examples, then analyze to derive optimal thresholds.

Run: streamlit run train_filter.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import random
from pathlib import Path
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
import torch
import requests
import httpx
from PIL import Image
from dotenv import load_dotenv
import open_clip

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")
ETSY_API_KEY = os.getenv("ETSY_API_KEY")

DEV_DIR = Path(__file__).parent.parent / "dev"
METADATA_FILE = DEV_DIR.parent / "image_metadata.json"
TRAINING_DATA_FILE = Path(__file__).parent / "training_data.json"

# Global model cache
_model = None
_preprocess = None
_device = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_model():
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
    model, preprocess, device = load_clip_model()
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def compute_edge_density(image: Image.Image) -> float:
    img_gray = np.array(image.convert("L"))
    edges = cv2.Canny(img_gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)


def load_listing_ids_from_metadata() -> list[int]:
    if not METADATA_FILE.exists():
        return []
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    return [int(lid) for lid in metadata.keys()]


def fetch_listing_images_from_etsy(listing_id: int) -> list[dict]:
    if not ETSY_API_KEY:
        raise ValueError("ETSY_API_KEY not set")

    url = f"https://openapi.etsy.com/v3/application/listings/{listing_id}/images"
    headers = {"x-api-key": ETSY_API_KEY}

    try:
        response = httpx.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        images = []
        for img in data.get("results", []):
            images.append({
                "url": img.get("url_fullxfull"),
                "rank": img.get("rank", 0),
                "image_id": img.get("listing_image_id"),
            })
        images.sort(key=lambda x: x["rank"])
        return images
    except Exception as e:
        print(f"Error: {e}")
        return []


def download_image_from_url(url: str) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def load_training_data() -> dict:
    if TRAINING_DATA_FILE.exists():
        with open(TRAINING_DATA_FILE) as f:
            return json.load(f)
    return {"examples": [], "analysis": None}


def save_training_data(data: dict):
    with open(TRAINING_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def analyze_training_data(data: dict) -> dict:
    """Analyze labeled examples to derive optimal thresholds."""
    examples = data.get("examples", [])
    if not examples:
        return {"error": "No training data"}

    # Collect metrics by label type
    metrics = {
        "keep": [],      # Images that should be kept (diverse angles)
        "zoom": [],      # Zoom images (should be rejected)
        "duplicate": [], # Near duplicates (should be rejected)
        "different": [], # Different item entirely (should be rejected)
    }

    for ex in examples:
        for img in ex.get("images", []):
            label = img.get("label")
            if label and label in metrics:
                metrics[label].append({
                    "sim_to_primary": img.get("sim_to_primary", 0),
                    "edge_density": img.get("edge_density", 0),
                    "max_sim_to_others": img.get("max_sim_to_others", 0),
                    "min_sim_to_others": img.get("min_sim_to_others", 1),
                    "edge_ratio_vs_primary": img.get("edge_ratio_vs_primary", 1),
                })

    analysis = {}

    # Analyze each category
    for label, items in metrics.items():
        if items:
            sims = [i["sim_to_primary"] for i in items]
            edges = [i["edge_density"] for i in items]
            edge_ratios = [i["edge_ratio_vs_primary"] for i in items]

            analysis[label] = {
                "count": len(items),
                "sim_to_primary": {
                    "min": min(sims),
                    "max": max(sims),
                    "mean": sum(sims) / len(sims),
                },
                "edge_density": {
                    "min": min(edges),
                    "max": max(edges),
                    "mean": sum(edges) / len(edges),
                },
                "edge_ratio_vs_primary": {
                    "min": min(edge_ratios),
                    "max": max(edge_ratios),
                    "mean": sum(edge_ratios) / len(edge_ratios),
                },
            }

    # Derive recommended thresholds
    recommendations = {}

    # Different item threshold: max sim of "different" items
    if metrics["different"]:
        diff_sims = [i["sim_to_primary"] for i in metrics["different"]]
        recommendations["min_similarity_threshold"] = {
            "value": max(diff_sims) + 0.05,  # Add margin
            "reasoning": f"Different items have sim <= {max(diff_sims):.3f}"
        }

    # Duplicate threshold: min sim of duplicates
    if metrics["duplicate"]:
        dup_sims = [i["sim_to_primary"] for i in metrics["duplicate"]]
        recommendations["duplicate_threshold"] = {
            "value": min(dup_sims) - 0.02,  # Add margin
            "reasoning": f"Duplicates have sim >= {min(dup_sims):.3f}"
        }

    # Zoom detection: edge ratio of zooms vs keeps
    if metrics["zoom"] and metrics["keep"]:
        zoom_ratios = [i["edge_ratio_vs_primary"] for i in metrics["zoom"]]
        keep_ratios = [i["edge_ratio_vs_primary"] for i in metrics["keep"]]
        recommendations["zoom_edge_ratio"] = {
            "zoom_max": max(zoom_ratios),
            "keep_min": min(keep_ratios),
            "reasoning": f"Zooms have edge_ratio <= {max(zoom_ratios):.3f}, keeps >= {min(keep_ratios):.3f}"
        }

    analysis["recommendations"] = recommendations
    return analysis


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Filter Training", layout="wide")
    st.title("🏋️ Image Filter Training")
    st.caption("Label images to train the filter algorithm")

    # Initialize
    if 'training_data' not in st.session_state:
        st.session_state.training_data = load_training_data()
    if 'listing_ids' not in st.session_state:
        st.session_state.listing_ids = load_listing_ids_from_metadata()
        if st.session_state.listing_ids:
            random.shuffle(st.session_state.listing_ids)
    if 'current_listing' not in st.session_state:
        st.session_state.current_listing = None
    if 'current_images' not in st.session_state:
        st.session_state.current_images = None
    if 'labels' not in st.session_state:
        st.session_state.labels = {}

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        if st.button("🎲 Random Listing", type="primary", use_container_width=True):
            if st.session_state.listing_ids:
                st.session_state.current_listing = random.choice(st.session_state.listing_ids)
                st.session_state.current_images = None
                st.session_state.labels = {}
                st.rerun()

        st.divider()

        st.header("Training Stats")
        n_examples = len(st.session_state.training_data.get("examples", []))
        st.metric("Labeled Listings", n_examples)

        # Count by label type
        label_counts = {"keep": 0, "zoom": 0, "duplicate": 0, "different": 0}
        for ex in st.session_state.training_data.get("examples", []):
            for img in ex.get("images", []):
                lbl = img.get("label")
                if lbl in label_counts:
                    label_counts[lbl] += 1

        st.write("Label counts:")
        for lbl, cnt in label_counts.items():
            st.write(f"  {lbl}: {cnt}")

        st.divider()

        if st.button("📊 Analyze Data", use_container_width=True):
            analysis = analyze_training_data(st.session_state.training_data)
            st.session_state.training_data["analysis"] = analysis
            save_training_data(st.session_state.training_data)
            st.success("Analysis complete!")

        if st.button("💾 Export to File", use_container_width=True):
            save_training_data(st.session_state.training_data)
            st.success(f"Saved to {TRAINING_DATA_FILE}")

    # Main content
    if not st.session_state.listing_ids:
        st.error("No listings found in metadata!")
        return

    if not st.session_state.current_listing:
        st.info("👆 Click **Random Listing** to start labeling")

        # Show analysis if available
        if st.session_state.training_data.get("analysis"):
            st.subheader("📊 Current Analysis")
            analysis = st.session_state.training_data["analysis"]

            if "recommendations" in analysis:
                st.markdown("### Recommended Thresholds")
                recs = analysis["recommendations"]
                for key, val in recs.items():
                    if isinstance(val, dict):
                        st.write(f"**{key}**:")
                        for k, v in val.items():
                            st.write(f"  - {k}: {v}")

            st.markdown("### Stats by Label")
            for label in ["keep", "zoom", "duplicate", "different"]:
                if label in analysis:
                    st.write(f"**{label}** ({analysis[label]['count']} examples):")
                    st.write(f"  - sim_to_primary: {analysis[label]['sim_to_primary']}")
                    st.write(f"  - edge_ratio: {analysis[label]['edge_ratio_vs_primary']}")
        return

    listing_id = st.session_state.current_listing
    st.subheader(f"Listing: {listing_id}")
    st.caption(f"[View on Etsy](https://www.etsy.com/listing/{listing_id})")

    # Fetch images if needed
    if st.session_state.current_images is None:
        with st.spinner("Fetching images..."):
            image_infos = fetch_listing_images_from_etsy(listing_id)
            if not image_infos:
                st.error("Failed to fetch images (API rate limit?)")
                return

            images = []
            for info in image_infos:
                img = download_image_from_url(info["url"])
                if img:
                    emb = compute_embedding(img)
                    edge = compute_edge_density(img)
                    images.append({
                        "info": info,
                        "image": img,
                        "embedding": emb,
                        "edge_density": edge,
                    })

            if not images:
                st.error("No images downloaded")
                return

            # Compute pairwise similarities
            n = len(images)
            primary_emb = images[0]["embedding"]
            primary_edge = images[0]["edge_density"]

            for i, img_data in enumerate(images):
                emb = img_data["embedding"]
                sim_to_primary = float(np.dot(emb, primary_emb)) if i > 0 else 1.0

                # Similarities to all others
                sims = []
                for j in range(n):
                    if i != j:
                        sims.append(float(np.dot(emb, images[j]["embedding"])))

                img_data["sim_to_primary"] = sim_to_primary
                img_data["max_sim_to_others"] = max(sims) if sims else 0
                img_data["min_sim_to_others"] = min(sims) if sims else 1
                img_data["edge_ratio_vs_primary"] = img_data["edge_density"] / primary_edge if primary_edge > 0 else 1

            st.session_state.current_images = images
            # Initialize labels - primary is always "keep"
            st.session_state.labels = {0: "keep"}

    images = st.session_state.current_images

    # Instructions
    st.markdown("""
    **Labels:**
    - **keep**: Good image showing different angle (INCLUDE in final set)
    - **zoom**: Close-up/zoom of another image (EXCLUDE)
    - **duplicate**: Nearly identical to another image (EXCLUDE)
    - **different**: Different item entirely (EXCLUDE)
    """)

    # Display images with labeling
    st.markdown("### Images")
    cols = st.columns(min(len(images), 5))

    for i, img_data in enumerate(images):
        with cols[i % 5]:
            st.image(img_data["image"], use_container_width=True)
            st.caption(f"**#{i+1}** (Rank {img_data['info']['rank']})")
            st.caption(f"Sim: {img_data['sim_to_primary']:.3f}")
            st.caption(f"Edge: {img_data['edge_density']:.4f}")
            st.caption(f"E.ratio: {img_data['edge_ratio_vs_primary']:.2f}")

            if i == 0:
                st.success("PRIMARY (always keep)")
            else:
                label = st.selectbox(
                    f"Label #{i+1}",
                    options=["keep", "zoom", "duplicate", "different"],
                    key=f"label_{i}",
                    index=["keep", "zoom", "duplicate", "different"].index(
                        st.session_state.labels.get(i, "keep")
                    )
                )
                st.session_state.labels[i] = label

    # Save button
    st.divider()
    if st.button("✅ Save Labels & Next", type="primary", use_container_width=True):
        # Build example
        example = {
            "listing_id": listing_id,
            "timestamp": datetime.now().isoformat(),
            "images": []
        }

        for i, img_data in enumerate(images):
            example["images"].append({
                "index": i,
                "rank": img_data["info"]["rank"],
                "image_id": img_data["info"]["image_id"],
                "label": st.session_state.labels.get(i, "keep"),
                "sim_to_primary": img_data["sim_to_primary"],
                "edge_density": img_data["edge_density"],
                "edge_ratio_vs_primary": img_data["edge_ratio_vs_primary"],
                "max_sim_to_others": img_data["max_sim_to_others"],
                "min_sim_to_others": img_data["min_sim_to_others"],
            })

        # Add to training data
        st.session_state.training_data["examples"].append(example)
        save_training_data(st.session_state.training_data)

        # Move to next listing
        st.session_state.current_listing = random.choice(st.session_state.listing_ids)
        st.session_state.current_images = None
        st.session_state.labels = {}
        st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
