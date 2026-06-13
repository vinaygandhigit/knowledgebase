from __future__ import annotations

from app.core.config import settings
from app.services.chunking import SemanticChunkingService
from app.services.embeddings.factory import EmbeddingFactory
from app.services.ingestion_service import IngestionService, JobStore
from app.services.llm_provider import LLMFactory
from app.services.chroma_repository import ChromaVectorRepository
from app.services.parsers.factory import ParserFactory
from app.services.retrieval import HybridRetriever
from app.services.scanner import FileScanner


class AppContainer:
    def __init__(self) -> None:
        self.settings = settings

        self.embedding_provider = EmbeddingFactory.create(self.settings)
        self.chunking_service = SemanticChunkingService(self.settings, self.embedding_provider)
        self.scanner = FileScanner(self.settings)
        self.parser_factory = ParserFactory()
        self.repository = ChromaVectorRepository(self.settings)
        self.job_store = JobStore()

        self.ingestion_service = IngestionService(
            settings=self.settings,
            scanner=self.scanner,
            parser_factory=self.parser_factory,
            chunking_service=self.chunking_service,
            embedding_provider=self.embedding_provider,
            repository=self.repository,
            job_store=self.job_store,
        )

        self.retriever = HybridRetriever(
            repository=self.repository,
            embedding_provider=self.embedding_provider,
            k=self.settings.retriever_k,
            candidate_multiplier=self.settings.retrieval_candidate_multiplier,
            rrf_k_constant=self.settings.rrf_k_constant,
        )
        self.llm_provider = LLMFactory.create(self.settings)

    def initialize_retriever(self) -> None:
        self.retriever.build_keyword_index()

    def shutdown(self) -> None:
        self.repository.close()

