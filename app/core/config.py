from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor for resolving relative paths. config.py lives at
# <project_root>/app/core/config.py, so parents[2] is the project root. This
# keeps the Chroma store at a single fixed location regardless of the directory
# the program is launched from (otherwise a relative "./chroma_store" creates a
# fresh, empty store next to each working directory).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Knowledge Intelligence Platform"
    app_env: str = "dev"
    log_level: str = "INFO"

    root_path: Path = Field(default=Path("./documents"), alias="ROOT_PATH")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    supported_extensions: str | list[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".txt",
            ".md",
            ".yaml",
            ".yml",
            ".eml",
            ".sh",
            ".toml",
        ],
        alias="SUPPORTED_EXTENSIONS",
    )

    chroma_persist_dir: Path = Field(default=Path("./chroma_store"), alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="knowledge_base", alias="CHROMA_COLLECTION")

    embedding_provider: str = Field(default="sentence_transformers", alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field(
        default="BAAI/bge-large-en-v1.5", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    # "auto" picks CUDA when available (with FP16) and falls back to CPU.
    embedding_device: str = Field(default="auto", alias="EMBEDDING_DEVICE")

    chunk_breakpoint_threshold_type: str = Field(
        default="percentile", alias="CHUNK_BREAKPOINT_THRESHOLD_TYPE"
    )
    chunk_breakpoint_threshold_amount: float = Field(
        default=80.0, alias="CHUNK_BREAKPOINT_THRESHOLD_AMOUNT"
    )
    # Size guard-rails applied after semantic chunking. max_chars keeps a chunk
    # within the embedding model's window (bge-large truncates ~512 tokens);
    # min_chars merges tiny fragments so single-sentence chunks don't pollute retrieval.
    chunk_max_chars: int = Field(default=1800, alias="CHUNK_MAX_CHARS")
    chunk_min_chars: int = Field(default=250, alias="CHUNK_MIN_CHARS")
    chunk_overlap_chars: int = Field(default=150, alias="CHUNK_OVERLAP_CHARS")
    chunk_prepend_section_title: bool = Field(
        default=True, alias="CHUNK_PREPEND_SECTION_TITLE"
    )
    # "semantic" = embedding-based breakpoints (higher quality, slower — embeds
    # every sentence during chunking). "recursive" = boundary-based splitting
    # with no chunk-time embeddings (much faster ingestion).
    chunk_strategy: str = Field(default="semantic", alias="CHUNK_STRATEGY")

    batch_size: int = Field(default=32, alias="BATCH_SIZE")
    worker_count: int = Field(default=4, alias="WORKER_COUNT")

    # LLM generation. Default to Claude (Anthropic API); "ollama" remains available
    # as an offline fallback.
    llm_provider: str = Field(default="claude", alias="LLM_PROVIDER")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")
    claude_model: str = Field(default="claude-opus-4-8", alias="CLAUDE_MODEL")
    claude_max_tokens: int = Field(default=8192, alias="CLAUDE_MAX_TOKENS")
    claude_thinking: bool = Field(default=True, alias="CLAUDE_THINKING")
    claude_effort: str = Field(default="medium", alias="CLAUDE_EFFORT")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3:1b", alias="OLLAMA_MODEL")

    retriever_k: int = Field(default=5, alias="RETRIEVER_K")
    hybrid_search_alpha: float = Field(default=0.5, alias="HYBRID_SEARCH_ALPHA")
    retrieval_candidate_multiplier: int = Field(
        default=4, alias="RETRIEVAL_CANDIDATE_MULTIPLIER"
    )
    rrf_k_constant: float = Field(default=60.0, alias="RRF_K_CONSTANT")

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def parse_supported_extensions(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [v.strip().lower() for v in value]
        return [v.strip().lower() for v in value.split(",") if v.strip()]

    @field_validator("chroma_persist_dir", mode="after")
    @classmethod
    def resolve_chroma_persist_dir(cls, value: Path) -> Path:
        # Resolve relative paths against the project root so the store location
        # is stable no matter where the program is run from.
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


settings = Settings()
