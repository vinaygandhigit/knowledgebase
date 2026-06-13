from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import Settings
from app.core.logging import get_logger


logger = get_logger(component="llm_provider")


_SYSTEM_PROMPT = """You answer strictly from the provided context.
Rules:
1. Do not use outside knowledge.
2. If the answer is stated explicitly in the context, prefer that exact statement.
3. For count/list questions, return the exact count and list the exact items from the context.
4. If multiple snippets conflict, say that the context conflicts and quote the conflicting statements briefly.
5. If the context is insufficient, say you do not have enough information.
6. Cite the source file name in the answer when possible.
Keep the answer concise, factual, and grounded in the retrieved text."""


class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, context: str, query: str) -> str:
        raise NotImplementedError


class ClaudeLLMProvider(LLMProvider):
    """Anthropic Claude provider for grounded RAG answer generation.

    Uses the official ``anthropic`` SDK. Adaptive thinking + the effort
    parameter are enabled by default (they improve grounded counting and
    conflict detection); if the installed SDK or model does not support them,
    the request transparently falls back to a plain call.
    """

    def __init__(
        self,
        model_name: str = "claude-opus-4-8",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8192,
        thinking: bool = True,
        effort: str = "medium",
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.effort = effort

        client_kwargs: dict[str, object] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        # With no api_key the SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
        # from the environment, which is the recommended path for local dev.
        self.client = anthropic.Anthropic(**client_kwargs)

    def _create(self, create_kwargs: dict[str, object]):
        try:
            return self.client.messages.create(**create_kwargs)
        except TypeError:
            # Installed SDK predates the thinking / output_config kwargs.
            logger.warning("SDK too old for adaptive thinking; retrying without it")
        except self._anthropic.BadRequestError as error:
            # Model rejected adaptive thinking / effort (e.g. an older model id).
            logger.warning("Model rejected thinking/effort; retrying without it", error=str(error))

        plain = {
            key: value
            for key, value in create_kwargs.items()
            if key not in ("thinking", "output_config")
        }
        return self.client.messages.create(**plain)

    def generate_response(self, context: str, query: str) -> str:
        user_content = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        create_kwargs: dict[str, object] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
        }
        if self.thinking:
            create_kwargs["thinking"] = {"type": "adaptive"}
            create_kwargs["output_config"] = {"effort": self.effort}

        try:
            response = self._create(create_kwargs)
        except self._anthropic.APIError as error:
            logger.exception("LLM response generation failed", error=str(error))
            return f"Error generating response: {str(error)}"

        text = next(
            (block.text for block in response.content if getattr(block, "type", None) == "text"),
            "",
        ).strip()

        if not text and getattr(response, "stop_reason", None) == "max_tokens":
            logger.warning("Response truncated before producing an answer", model=self.model_name)
            return (
                "Error: the model hit its output limit before producing an answer. "
                "Increase CLAUDE_MAX_TOKENS and retry."
            )
        return text


class OllamaLLMProvider(LLMProvider):
    """Local Ollama provider, retained as an optional offline fallback."""

    def __init__(self, model_name: str = "llama3:1b", base_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.base_url = base_url
        import ollama
        self.client = ollama.Client(host=base_url)

    def generate_response(self, context: str, query: str) -> str:
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                stream=False,
            )
            return response.get("response", "").strip()
        except Exception as error:
            logger.exception("LLM response generation failed", error=str(error))
            return f"Error generating response: {str(error)}"


class LLMFactory:
    @staticmethod
    def create(settings: Settings) -> LLMProvider:
        provider = settings.llm_provider.lower().strip()

        if provider in {"claude", "anthropic"}:
            return ClaudeLLMProvider(
                model_name=settings.claude_model,
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
                max_tokens=settings.claude_max_tokens,
                thinking=settings.claude_thinking,
                effort=settings.claude_effort,
            )

        if provider == "ollama":
            return OllamaLLMProvider(
                model_name=settings.ollama_model,
                base_url=settings.ollama_base_url,
            )

        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
