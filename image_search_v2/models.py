"""Model registry for multi-model image embeddings.

Supports CLIP (via open_clip) and DINOv2 (via transformers).
"""
import torch
from PIL import Image


# Model configurations
MODELS = {
    "clip_vitb32": {
        "name": "ViT-B-32",
        "library": "open_clip",
        "pretrained": "laion2b_s34b_b79k",
        "dim": 512,
    },
    "clip_vitl14": {
        "name": "ViT-L-14",
        "library": "open_clip",
        "pretrained": "laion2b_s32b_b82k",
        "dim": 768,
    },
    "dinov2_base": {
        "name": "dinov2_vitb14",
        "library": "transformers",
        "dim": 768,
    },
    "dinov2_large": {
        "name": "dinov2_vitl14",
        "library": "transformers",
        "dim": 1024,
    },
}


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelLoader:
    """Lazy loader for embedding models."""

    def __init__(self):
        self._models = {}
        self._preprocessors = {}
        self._device = get_device()

    def get_model(self, model_key: str):
        """Load and cache a model by key."""
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        if model_key not in self._models:
            self._load_model(model_key)

        return self._models[model_key], self._preprocessors[model_key]

    def _load_model(self, model_key: str):
        """Load a model into cache."""
        config = MODELS[model_key]
        print(f"Loading {model_key} ({config['name']})...")

        if config["library"] == "open_clip":
            self._load_clip_model(model_key, config)
        elif config["library"] == "transformers":
            self._load_dinov2_model(model_key, config)
        else:
            raise ValueError(f"Unknown library: {config['library']}")

    def _load_clip_model(self, model_key: str, config: dict):
        """Load a CLIP model via open_clip."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            config["name"],
            pretrained=config["pretrained"],
            device=self._device,
        )
        model.eval()

        self._models[model_key] = model
        self._preprocessors[model_key] = preprocess

    def _load_dinov2_model(self, model_key: str, config: dict):
        """Load a DINOv2 model via transformers."""
        from transformers import AutoImageProcessor, AutoModel

        model_name = f"facebook/{config['name']}"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self._device)
        model.eval()

        self._models[model_key] = model
        self._preprocessors[model_key] = processor

    def embed_image(self, model_key: str, image: Image.Image) -> torch.Tensor:
        """Generate embedding for a single image."""
        model, preprocess = self.get_model(model_key)
        config = MODELS[model_key]

        with torch.no_grad():
            if config["library"] == "open_clip":
                img_tensor = preprocess(image).unsqueeze(0).to(self._device)
                emb = model.encode_image(img_tensor)
            else:  # transformers (DINOv2)
                inputs = preprocess(image, return_tensors="pt").to(self._device)
                outputs = model(**inputs)
                # Use CLS token embedding
                emb = outputs.last_hidden_state[:, 0, :]

            # Normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb

    def embed_batch(self, model_key: str, images: list[Image.Image]) -> torch.Tensor:
        """Generate embeddings for a batch of images."""
        model, preprocess = self.get_model(model_key)
        config = MODELS[model_key]

        with torch.no_grad():
            if config["library"] == "open_clip":
                tensors = torch.stack([preprocess(img) for img in images]).to(self._device)
                emb = model.encode_image(tensors)
            else:  # transformers (DINOv2)
                inputs = preprocess(images, return_tensors="pt").to(self._device)
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]

            # Normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb

    @property
    def device(self) -> str:
        return self._device


# Singleton instance
_loader = None


def get_loader() -> ModelLoader:
    """Get or create the model loader singleton."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
