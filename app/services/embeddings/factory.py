from __future__ import annotations

from app.core.config import Settings
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.sentence_transformer_provider import SentenceTransformerProvider


class EmbeddingFactory:
    @staticmethod
    def create(settings: Settings) -> EmbeddingProvider:
        provider = settings.embedding_provider.lower().strip()
        if provider == "sentence_transformers":
            return SentenceTransformerProvider(
                model_name=settings.embedding_model_name,
                device=settings.embedding_device,
                batch_size=settings.batch_size,
            )

        if provider in {"openai", "azure_openai", "oci_genai"}:
            raise NotImplementedError(
                f"Embedding provider '{provider}' is not implemented yet. "
                "Use sentence_transformers for now."
            )

        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
