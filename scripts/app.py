"""Streamlit app for multi-model image similarity search."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path

from search import get_search, IMAGES_DIR, IMAGES_DIR_LEGACY
from models import MODELS

# Background-removed images directories (v2 and legacy)
IMAGES_NOBG_DIR = IMAGES_DIR.parent / "images_v2_nobg"
IMAGES_NOBG_DIR_LEGACY = IMAGES_DIR.parent / "images_nobg"


def load_image_from_url(url: str) -> Image.Image | None:
    """Load image from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
        return None


st.set_page_config(page_title="Multi-Model Image Search", layout="wide")


def load_image(listing_id: int, use_nobg: bool = False) -> tuple[Image.Image | None, bool]:
    """Load image for a listing ID.

    Checks v2 folders first, then falls back to legacy v1 folders.

    Returns (image, is_nobg) tuple.
    """
    search = get_search()

    if use_nobg:
        # Try v2 nobg folder first
        if IMAGES_NOBG_DIR.exists():
            matches = list(IMAGES_NOBG_DIR.glob(f"{listing_id}_*.jpg"))
            if matches:
                return Image.open(sorted(matches)[0]), True

        # Try legacy nobg folder
        if IMAGES_NOBG_DIR_LEGACY.exists():
            path = IMAGES_NOBG_DIR_LEGACY / f"{listing_id}.jpg"
            if path.exists():
                return Image.open(path), True
            matches = list(IMAGES_NOBG_DIR_LEGACY.glob(f"{listing_id}_*.jpg"))
            if matches:
                return Image.open(sorted(matches)[0]), True

    # Regular images (search handles v2/v1 fallback)
    path = search.get_image_path(listing_id)
    if path and path.exists():
        return Image.open(path), False
    return None, False


def display_model_results(
    model_key: str,
    results: list[tuple[int, float]],
    num_to_show: int,
    show_nobg: bool,
):
    """Display results for a single model in a grid."""
    cols_per_row = 4

    for row_start in range(0, min(len(results), num_to_show), cols_per_row):
        row_items = results[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, (listing_id, score) in zip(cols, row_items):
            with col:
                img, is_nobg = load_image(listing_id, use_nobg=show_nobg)
                if img:
                    st.image(img, use_container_width=True)
                    st.caption(f"ID: {listing_id}\nScore: {score:.3f}")


def display_consensus_results(
    consensus: list[tuple[int, float, int]],
    num_to_show: int,
    show_nobg: bool,
):
    """Display consensus results (items appearing in multiple models)."""
    cols_per_row = 5

    for row_start in range(0, min(len(consensus), num_to_show), cols_per_row):
        row_items = consensus[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, (listing_id, avg_score, num_models) in zip(cols, row_items):
            with col:
                img, is_nobg = load_image(listing_id, use_nobg=show_nobg)
                if img:
                    st.image(img, use_container_width=True)
                    st.caption(f"ID: {listing_id}\nModels: {num_models}\nAvg: {avg_score:.3f}")


def main():
    st.title("Multi-Model Image Search")

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "query_image" not in st.session_state:
        st.session_state.query_image = None
    if "num_to_show" not in st.session_state:
        st.session_state.num_to_show = 20
    if "show_nobg" not in st.session_state:
        st.session_state.show_nobg = False
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []

    # Sidebar controls
    with st.sidebar:
        st.header("Search")

        # File upload
        uploaded = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"])

        # URL input
        image_url_input = st.text_input("Or enter image URL")

        # Model selection
        search = get_search()
        available_models = search.available_models
        st.subheader("Models")
        selected_models = st.multiselect(
            "Select models to use",
            options=available_models,
            default=available_models,
            help="Results from each selected model will be shown",
        )

        num_results = st.selectbox("Results per model", [10, 20, 50, 100], index=1)

        if st.button("Search", type="primary"):
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.session_state.query_image = img
                with st.spinner("Searching across models..."):
                    st.session_state.results = search.search_by_image(
                        img, k=num_results * 2, models=selected_models
                    )
                st.session_state.num_to_show = num_results
                st.session_state.selected_models = selected_models

            elif image_url_input:
                img = load_image_from_url(image_url_input.strip())
                if img:
                    st.session_state.query_image = img
                    with st.spinner("Searching across models..."):
                        st.session_state.results = search.search_by_image(
                            img, k=num_results * 2, models=selected_models
                        )
                    st.session_state.num_to_show = num_results
                    st.session_state.selected_models = selected_models

        st.divider()

        # Display options
        st.header("Display")
        show_nobg = st.toggle(
            "Show without background",
            value=st.session_state.show_nobg,
            help="Show background-removed versions if available",
        )
        if show_nobg != st.session_state.show_nobg:
            st.session_state.show_nobg = show_nobg
            st.rerun()

        st.divider()

        # Stats
        st.header("Stats")
        st.write(f"Available models: {len(available_models)}")
        for m in available_models:
            st.write(f"  - {m} ({MODELS[m]['dim']}d)")

    # Main content
    col_query, col_results = st.columns([1, 4])

    with col_query:
        st.subheader("Query")
        if st.session_state.query_image:
            st.image(st.session_state.query_image, use_container_width=True)

    with col_results:
        if st.session_state.results:
            # View mode tabs
            tab_compare, tab_consensus = st.tabs(["Model Comparison", "Consensus"])

            with tab_compare:
                st.subheader("Results by Model")
                # Create columns for each model
                model_cols = st.columns(len(st.session_state.results))
                for col, (model_key, model_results) in zip(
                    model_cols, st.session_state.results.items()
                ):
                    with col:
                        st.markdown(f"**{model_key}**")
                        st.caption(f"{MODELS[model_key]['dim']}d, {MODELS[model_key]['library']}")
                        for listing_id, score in model_results[: st.session_state.num_to_show]:
                            img, _ = load_image(listing_id, use_nobg=st.session_state.show_nobg)
                            if img:
                                st.image(img, use_container_width=True)
                                st.caption(f"{listing_id}: {score:.3f}")

            with tab_consensus:
                st.subheader("Consensus Results")
                st.caption("Items appearing in top results across multiple models")

                min_models = st.slider(
                    "Minimum models",
                    min_value=1,
                    max_value=len(st.session_state.results),
                    value=min(2, len(st.session_state.results)),
                )

                search = get_search()
                consensus = search.get_consensus_results(
                    st.session_state.results,
                    min_models=min_models,
                    top_n=st.session_state.num_to_show,
                )

                if consensus:
                    display_consensus_results(
                        consensus,
                        st.session_state.num_to_show,
                        st.session_state.show_nobg,
                    )
                else:
                    st.info(f"No items found in top {st.session_state.num_to_show} of {min_models}+ models")
        else:
            st.info("Upload an image or enter a URL to search")


if __name__ == "__main__":
    main()
