from __future__ import annotations

from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(component="embedding_provider")

class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        resolved_device = self._resolve_device(device)
        self.model = SentenceTransformer(model_name, device=resolved_device)
        self.batch_size = max(batch_size, 1)
        self.device = resolved_device

        # FP16 roughly halves GPU memory and speeds up inference with negligible
        # retrieval-quality impact. Only safe/beneficial on CUDA.
        self._use_fp16 = resolved_device.startswith("cuda")
        if self._use_fp16:
            self.model = self.model.half()

        logger.info(
            "Embedding model loaded",
            model_name=model_name,
            device=resolved_device,
            batch_size=self.batch_size,
            fp16=self._use_fp16,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device.lower() != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Let SentenceTransformer batch internally — a single encode() call over
        # the whole list is far more efficient than slicing in Python and calling
        # encode() repeatedly.
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
