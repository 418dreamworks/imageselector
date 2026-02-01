"""Multi-model FAISS-based similarity search for image embeddings."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from models import MODELS, get_loader

DEV_DIR = Path(__file__).parent.parent / "dev"
IMAGES_DIR = DEV_DIR / "images_v2"  # v2 uses separate image folder
IMAGES_DIR_LEGACY = DEV_DIR / "images"  # Fallback to v1 images
EMBEDDINGS_DIR = DEV_DIR / "embeddings_v2"
LISTING_IDS_FILE = DEV_DIR / "listing_ids_v2.json"  # v2 has its own listing IDs


class MultiModelSearch:
    """Search for similar images using multiple embedding models."""

    def __init__(self, models: list[str] | None = None):
        """Initialize with specified models or all available.

        Args:
            models: List of model keys to load, or None for all available.
        """
        self.models = models or list(MODELS.keys())
        self.embeddings: dict[str, np.ndarray] = {}
        self.indices: dict[str, faiss.Index] = {}
        self.listing_ids: list[int] = []
        self._load_embeddings()

    def _load_embeddings(self):
        """Load pre-computed embeddings and build FAISS indices."""
        # Load listing IDs
        if not LISTING_IDS_FILE.exists():
            raise FileNotFoundError(
                f"Listing IDs not found at {LISTING_IDS_FILE}. Run embed.py first."
            )
        with open(LISTING_IDS_FILE) as f:
            self.listing_ids = json.load(f)
        print(f"Loaded {len(self.listing_ids)} listing IDs")

        # Load embeddings for each model
        for model_key in self.models:
            emb_file = EMBEDDINGS_DIR / f"{model_key}.npy"
            if not emb_file.exists():
                print(f"Warning: Embeddings not found for {model_key}, skipping")
                continue

            print(f"Loading embeddings for {model_key}...")
            emb = np.load(emb_file).astype("float32")
            self.embeddings[model_key] = emb

            # Build FAISS index (inner product for cosine similarity on normalized vectors)
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb)
            self.indices[model_key] = index
            print(f"  {model_key}: {index.ntotal} vectors, {dim} dimensions")

    def search(
        self,
        query_embedding: dict[str, np.ndarray],
        k: int = 100,
    ) -> dict[str, list[tuple[int, float]]]:
        """Find k most similar images for each model.

        Args:
            query_embedding: Dict mapping model_key to query embedding (1, dim)
            k: Number of results per model

        Returns:
            Dict mapping model_key to list of (listing_id, score) tuples.
        """
        results = {}
        for model_key, emb in query_embedding.items():
            if model_key not in self.indices:
                continue
            distances, indices = self.indices[model_key].search(emb, k)
            model_results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:
                    model_results.append((self.listing_ids[idx], float(dist)))
            results[model_key] = model_results
        return results

    def search_by_image(
        self,
        image: Image.Image,
        k: int = 100,
        models: list[str] | None = None,
    ) -> dict[str, list[tuple[int, float]]]:
        """Search for similar images given a query image.

        Args:
            image: Query image
            k: Number of results per model
            models: Models to use, or None for all loaded models

        Returns:
            Dict mapping model_key to list of (listing_id, score) tuples.
        """
        models = models or list(self.indices.keys())
        loader = get_loader()

        # Generate embeddings for each model
        query_embeddings = {}
        for model_key in models:
            if model_key not in self.indices:
                continue
            emb = loader.embed_image(model_key, image)
            query_embeddings[model_key] = emb.cpu().numpy().astype("float32")

        return self.search(query_embeddings, k)

    def search_by_images(
        self,
        images: list[Image.Image],
        k: int = 100,
        models: list[str] | None = None,
        fusion: str = "average",
    ) -> dict[str, list[tuple[int, float]]]:
        """Search using multiple query images (multi-angle query).

        Args:
            images: List of query images
            k: Number of results per model
            models: Models to use, or None for all loaded models
            fusion: How to combine embeddings - "average" or "max"

        Returns:
            Dict mapping model_key to list of (listing_id, score) tuples.
        """
        models = models or list(self.indices.keys())
        loader = get_loader()

        # Generate and fuse embeddings for each model
        query_embeddings = {}
        for model_key in models:
            if model_key not in self.indices:
                continue

            # Get embedding for each image
            embs = []
            for img in images:
                emb = loader.embed_image(model_key, img)
                embs.append(emb.cpu().numpy())

            # Fuse embeddings
            stacked = np.vstack(embs)
            if fusion == "average":
                fused = stacked.mean(axis=0, keepdims=True)
            elif fusion == "max":
                fused = stacked.max(axis=0, keepdims=True)
            else:
                raise ValueError(f"Unknown fusion method: {fusion}")

            # Re-normalize
            fused = fused / np.linalg.norm(fused, axis=1, keepdims=True)
            query_embeddings[model_key] = fused.astype("float32")

        return self.search(query_embeddings, k)

    def search_by_listing_id(
        self,
        listing_id: int,
        k: int = 100,
        models: list[str] | None = None,
    ) -> dict[str, list[tuple[int, float]]]:
        """Search for similar images given a listing ID.

        Args:
            listing_id: Listing ID from database
            k: Number of results per model
            models: Models to use, or None for all loaded models

        Returns:
            Dict mapping model_key to list of (listing_id, score) tuples.
        """
        try:
            idx = self.listing_ids.index(listing_id)
        except ValueError:
            raise ValueError(f"Listing ID {listing_id} not in database")

        models = models or list(self.indices.keys())
        query_embeddings = {}
        for model_key in models:
            if model_key not in self.embeddings:
                continue
            query_embeddings[model_key] = self.embeddings[model_key][idx : idx + 1]

        return self.search(query_embeddings, k)

    def get_consensus_results(
        self,
        results: dict[str, list[tuple[int, float]]],
        min_models: int = 2,
        top_n: int = 50,
    ) -> list[tuple[int, float, int]]:
        """Find listings that appear in top results across multiple models.

        Args:
            results: Results from search_by_image or similar
            min_models: Minimum number of models that must include the listing
            top_n: Consider only top N results from each model

        Returns:
            List of (listing_id, avg_score, num_models) tuples, sorted by num_models desc.
        """
        # Count appearances and accumulate scores
        listing_data: dict[int, dict] = {}

        for model_key, model_results in results.items():
            for rank, (lid, score) in enumerate(model_results[:top_n]):
                if lid not in listing_data:
                    listing_data[lid] = {"scores": [], "models": 0}
                listing_data[lid]["scores"].append(score)
                listing_data[lid]["models"] += 1

        # Filter and sort
        consensus = []
        for lid, data in listing_data.items():
            if data["models"] >= min_models:
                avg_score = sum(data["scores"]) / len(data["scores"])
                consensus.append((lid, avg_score, data["models"]))

        # Sort by number of models (desc), then by avg score (desc)
        consensus.sort(key=lambda x: (-x[2], -x[1]))
        return consensus

    def get_image_path(self, listing_id: int, image_id: int | None = None) -> Path | None:
        """Get the image file path for a listing ID.

        Checks v2 folder first, then falls back to legacy v1 folder.

        Args:
            listing_id: The listing ID
            image_id: Optional specific image ID. If None, returns first available.

        Returns:
            Path to image file, or None if not found.
        """
        if image_id is not None:
            # Specific image requested: {listing_id}_{image_id}.jpg
            path = IMAGES_DIR / f"{listing_id}_{image_id}.jpg"
            if path.exists():
                return path
            return None

        # No specific image - try v2 folder first
        if IMAGES_DIR.exists():
            matches = list(IMAGES_DIR.glob(f"{listing_id}_*.jpg"))
            if matches:
                return sorted(matches)[0]

        # Fallback to legacy v1 folder: {listing_id}.jpg
        if IMAGES_DIR_LEGACY.exists():
            path_legacy = IMAGES_DIR_LEGACY / f"{listing_id}.jpg"
            if path_legacy.exists():
                return path_legacy

        return None

    def get_all_images_for_listing(self, listing_id: int) -> list[Path]:
        """Get all image paths for a listing ID.

        Returns list of paths sorted by filename.
        """
        matches = []

        # Check v2 folder: {listing_id}_{image_id}.jpg
        if IMAGES_DIR.exists():
            matches.extend(IMAGES_DIR.glob(f"{listing_id}_*.jpg"))

        # Check legacy v1 folder
        if IMAGES_DIR_LEGACY.exists():
            legacy_path = IMAGES_DIR_LEGACY / f"{listing_id}.jpg"
            if legacy_path.exists() and legacy_path not in matches:
                matches.append(legacy_path)

        return sorted(matches)

    @property
    def available_models(self) -> list[str]:
        """Return list of models with loaded embeddings."""
        return list(self.indices.keys())


# Singleton instance
_search_instance = None


def get_search(models: list[str] | None = None) -> MultiModelSearch:
    """Get or create the search instance."""
    global _search_instance
    if _search_instance is None:
        _search_instance = MultiModelSearch(models)
    return _search_instance
