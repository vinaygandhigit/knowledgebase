from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.services.chroma_repository import ChromaVectorRepository
from app.services.embeddings.base import EmbeddingProvider

logger = get_logger(component="retrieval_service")


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    chunk_text: str
    document_id: str
    file_name: str
    score: float
    search_type: str
    source_location: dict[str, Any]
    visual_refs: list[dict[str, str]]


class HybridRetriever:
    def __init__(
        self,
        repository: ChromaVectorRepository,
        embedding_provider: EmbeddingProvider,
        k: int = 5,
        candidate_multiplier: int = 4,
        rrf_k_constant: float = 60.0,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.k = k
        self.candidate_multiplier = max(candidate_multiplier, 1)
        self.rrf_k_constant = rrf_k_constant
        self._bm25_index: BM25Okapi | None = None
        self._chunk_texts: list[str] = []
        self._chunk_metadata: list[dict] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")

        lob_read = getattr(value, "read", None)
        if callable(lob_read):
            lob_value = lob_read()
            if isinstance(lob_value, bytes):
                return lob_value.decode("utf-8", errors="ignore")
            return str(lob_value)

        return str(value)

    def _parse_metadata(self, chunk_id: str, metadata_value: Any) -> dict:
        if isinstance(metadata_value, dict):
            return metadata_value

        metadata_text = self._to_text(metadata_value)
        if not metadata_text:
            return {"chunk_id": chunk_id}

        try:
            parsed = json.loads(metadata_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        return {"chunk_id": chunk_id}

    def _candidate_limit(self) -> int:
        return max(self.k * self.candidate_multiplier, 20)

    def build_keyword_index(self) -> None:
        rows = self.repository.fetch_all_chunks()

        # Reset state so callers never use stale index data.
        self._bm25_index = None
        self._chunk_texts = []
        self._chunk_metadata = []
        for chunk_id, chunk_text, metadata_json in rows:
            self._chunk_texts.append(self._to_text(chunk_text))
            metadata = self._parse_metadata(str(chunk_id), metadata_json)
            self._chunk_metadata.append(metadata)

        if self._chunk_texts:
            tokenized = [self._tokenize(text) for text in self._chunk_texts]
            self._bm25_index = BM25Okapi(tokenized)
            logger.info("BM25 keyword index built", total_chunks=len(self._chunk_texts))
        else:
            logger.info("BM25 keyword index empty", total_chunks=0)

    def vector_search(self, query: str) -> list[RetrievalResult]:
        query_embedding = self.embedding_provider.embed_query(query)
        candidate_limit = self._candidate_limit()

        rows = self.repository.vector_search(query_embedding, candidate_limit)

        results: list[RetrievalResult] = []
        for chunk_id, chunk_text, document_id, metadata_json, file_name, distance in rows:
            metadata = self._parse_metadata(str(chunk_id), metadata_json)
            results.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    chunk_text=self._to_text(chunk_text),
                    document_id=str(document_id),
                    file_name=self._to_text(file_name) or metadata.get("file_name", ""),
                    score=1.0 - float(distance),
                    search_type="vector",
                    source_location=metadata.get("source_location", {}),
                    visual_refs=metadata.get("visual_refs", []),
                )
            )
        return results

    def keyword_search(self, query: str) -> list[RetrievalResult]:
        if not self._bm25_index or not self._chunk_texts:
            try:
                self.build_keyword_index()
            except Exception as error:
                logger.warning("BM25 index build failed, skipping keyword search", error=str(error))
                return []

        if not self._bm25_index or not self._chunk_texts:
            logger.info("BM25 index unavailable, skipping keyword search")
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25_index.get_scores(query_tokens)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for idx, score in indexed_scores[: self._candidate_limit()]:
            if score <= 0:
                continue
            metadata = self._chunk_metadata[idx]
            results.append(
                RetrievalResult(
                    chunk_id=metadata.get("chunk_id", f"chunk_{idx}"),
                    chunk_text=self._chunk_texts[idx],
                    document_id=metadata.get("document_id", ""),
                    file_name=metadata.get("file_name", ""),
                    score=float(score),
                    search_type="keyword",
                    source_location=metadata.get("source_location", {}),
                    visual_refs=metadata.get("visual_refs", []),
                )
            )
        return results

    def hybrid_search(self, query: str, alpha: float = 0.5) -> list[RetrievalResult]:
        """Fuse vector and keyword results with weighted Reciprocal Rank Fusion.

        RRF is used because vector cosine scores and BM25 scores live on
        incomparable scales; fusing by rank avoids brittle score normalisation.
        ``alpha`` weights the vector contribution (1 - alpha weights keyword).
        Chunks surfaced by both retrievers are labelled "hybrid".
        """
        alpha = min(max(alpha, 0.0), 1.0)
        vector_results = self.vector_search(query)
        keyword_results = self.keyword_search(query)

        combined: dict[str, RetrievalResult] = {}

        def fuse(results: list[RetrievalResult], weight: float, label: str) -> None:
            for rank, result in enumerate(results, start=1):
                contribution = weight * (1.0 / (self.rrf_k_constant + rank))
                existing = combined.get(result.chunk_id)
                if existing is None:
                    combined[result.chunk_id] = replace(
                        result, score=contribution, search_type=label
                    )
                else:
                    existing.score += contribution
                    if existing.search_type != label:
                        existing.search_type = "hybrid"

        fuse(vector_results, alpha, "vector")
        fuse(keyword_results, 1.0 - alpha, "keyword")

        sorted_results = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        return sorted_results[: self.k]
