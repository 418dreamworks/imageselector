"""Model registry for multi-model image embeddings.

Supports CLIP (via open_clip), DINOv2 and DINOv3 (via transformers).
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
        "name": "dinov2-base",
        "library": "transformers",
        "dim": 768,
    },
    "dinov2_large": {
        "name": "dinov2-large",
        "library": "transformers",
        "dim": 1024,
    },
    "dinov3_base": {
        "name": "dinov3-vitb16-pretrain-lvd1689m",
        "library": "transformers_dinov3",
        "dim": 768,
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
        self._tokenizers = {}  # For CLIP text encoding
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
        elif config["library"] in ("transformers", "transformers_dinov3"):
            self._load_dino_model(model_key, config)
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
        tokenizer = open_clip.get_tokenizer(config["name"])

        self._models[model_key] = model
        self._preprocessors[model_key] = preprocess
        self._tokenizers[model_key] = tokenizer

    def _load_dino_model(self, model_key: str, config: dict):
        """Load a DINOv2 or DINOv3 model via transformers."""
        from transformers import AutoImageProcessor, AutoModel

        model_name = f"facebook/{config['name']}"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self._device)
        model.eval()

        self._models[model_key] = model
        self._preprocessors[model_key] = processor

    def unload_all(self):
        """Unload all models from GPU to free VRAM."""
        for key in list(self._models.keys()):
            del self._models[key]
        self._models.clear()
        self._preprocessors.clear()
        self._tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def embed_image(self, model_key: str, image: Image.Image) -> torch.Tensor:
        """Generate embedding for a single image."""
        model, preprocess = self.get_model(model_key)
        config = MODELS[model_key]

        with torch.no_grad():
            if config["library"] == "open_clip":
                img_tensor = preprocess(image).unsqueeze(0).to(self._device)
                emb = model.encode_image(img_tensor)
            else:  # transformers (DINOv2/DINOv3)
                inputs = preprocess(image, return_tensors="pt").to(self._device)
                outputs = model(**inputs)
                # DINOv3 uses pooler_output, DINOv2 uses CLS token from last_hidden_state
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
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
            else:  # transformers (DINOv2/DINOv3)
                inputs = preprocess(images, return_tensors="pt").to(self._device)
                outputs = model(**inputs)
                # DINOv3 uses pooler_output, DINOv2 uses CLS token from last_hidden_state
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
                    emb = outputs.last_hidden_state[:, 0, :]

            # Normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb

    def embed_text_batch(self, model_key: str, texts: list[str]) -> torch.Tensor:
        """Generate text embeddings for CLIP models."""
        config = MODELS[model_key]
        if config["library"] != "open_clip":
            raise ValueError(f"Text embedding only supported for CLIP models, not {model_key}")

        model, _ = self.get_model(model_key)
        tokenizer = self._tokenizers[model_key]

        with torch.no_grad():
            tokens = tokenizer(texts).to(self._device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb

    def embed_batch_with_text(
        self,
        model_key: str,
        images: list[Image.Image],
        texts: list[str],
        text_weight: float = 0.3,
    ) -> torch.Tensor:
        """Generate combined image+text embeddings for CLIP models.

        Combines image and text embeddings with weighted average.
        text_weight: 0.0 = image only, 1.0 = text only, 0.3 = 70% image + 30% text
        """
        config = MODELS[model_key]
        if config["library"] != "open_clip":
            # Non-CLIP models: just return image embeddings
            return self.embed_batch(model_key, images)

        # Get image embeddings
        img_emb = self.embed_batch(model_key, images)

        # Get text embeddings
        text_emb = self.embed_text_batch(model_key, texts)

        # Weighted combination
        combined = (1 - text_weight) * img_emb + text_weight * text_emb

        # Re-normalize
        combined = combined / combined.norm(dim=-1, keepdim=True)

        return combined

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
