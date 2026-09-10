"""Google SDK adapter. Credentials and HTTP clients are resolved only on first use."""

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass

from google import genai
from google.genai import types

from ..models import Car
from .documents import build_semantic_document
from .embeddings import Vector, normalize_vector
from .errors import EmbeddingError


@dataclass(frozen=True, slots=True)
class GoogleEmbeddingConfig:
    model: str = "gemini-embedding-001"
    output_dimensionality: int = 768
    document_task: str = "RETRIEVAL_DOCUMENT"
    query_task: str = "RETRIEVAL_QUERY"
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be nonblank")
        if type(self.output_dimensionality) is not int or self.output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be a positive integer")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 100:
            raise ValueError("batch_size must be an integer between 1 and 100")
        if self.document_task != "RETRIEVAL_DOCUMENT" or self.query_task != "RETRIEVAL_QUERY":
            raise ValueError("Google retrieval requires document and query retrieval task types")


def build_document_title(car: Car) -> str:
    """Use structured source fields only, never semantic interpretations."""
    return " ".join(
        str(v)
        for v in (car.year, car.make, car.model, car.trim)
        if v is not None and str(v).strip()
    )


class GoogleEmbeddingProvider:
    """Synchronous, ordered batches; compatible with the existing EmbeddingProvider."""

    def __init__(
        self,
        inventory: Sequence[Car] = (),
        config: GoogleEmbeddingConfig | None = None,
    ) -> None:
        self._config = config if config is not None else GoogleEmbeddingConfig()
        self._client: genai.Client | None = None
        titles = {build_document_title(car) for car in inventory}
        self._title = next(iter(titles)) if len(titles) == 1 else None
        self._documents = frozenset(build_semantic_document(car) for car in inventory)

    @property
    def provider_id(self) -> str:
        config = self._config
        title_mode = (
            "body-only" if not self._title else hashlib.sha256(self._title.encode()).hexdigest()
        )
        return (
            f"google/{config.model}/dim-{config.output_dimensionality}"
            f"/doc-{config.document_task}/query-{config.query_task}/title-v1-{title_mode}"
        )

    def _get_client(self) -> genai.Client:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not key:
                raise EmbeddingError("GEMINI_API_KEY is required for Google embedding requests")
            try:
                self._client = genai.Client(
                    api_key=key,
                    vertexai=False,
                    http_options=types.HttpOptions(timeout=60_000),
                )
            except Exception:
                raise EmbeddingError("Google embedding client initialization failed") from None
        return self._client

    def _embed(
        self, texts: Sequence[str], task: str, title: str | None = None
    ) -> tuple[Vector, ...]:
        client = self._get_client()
        try:
            response = client.models.embed_content(
                model=self._config.model,
                contents=list(texts),
                config=types.EmbedContentConfig(
                    task_type=task,
                    output_dimensionality=self._config.output_dimensionality,
                    title=title or None,
                ),
            )
        except Exception:
            # SDK exceptions can contain URLs, headers or server echoes. Never chain them.
            raise EmbeddingError(
                "Google embedding request failed; check credentials, quota and connectivity"
            ) from None
        try:
            if response.embeddings is None or len(response.embeddings) != len(texts):
                raise ValueError("wrong count")
            return tuple(
                normalize_vector(item.values, self._config.output_dimensionality)
                for item in response.embeddings
            )
        except Exception:
            raise EmbeddingError(
                "Google returned malformed embeddings (count, values or dimensions)"
            ) from None

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        if isinstance(texts, (str, bytes)) or any(
            not isinstance(t, str) or not t.strip() for t in texts
        ):
            raise EmbeddingError("Document inputs must be nonblank strings")
        if self._title and any(text not in self._documents for text in texts):
            raise EmbeddingError(
                "Document is outside the inventory used for the shared source title"
            )
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._config.batch_size):
            vectors.extend(
                self._embed(
                    texts[start : start + self._config.batch_size],
                    self._config.document_task,
                    self._title,
                )
            )
        return tuple(vectors)

    def embed_query(self, text: str) -> Vector:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Query must be a nonblank string")
        return self._embed([text], self._config.query_task)[0]

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                raise EmbeddingError("Google embedding client shutdown failed") from None
            finally:
                self._client = None

    def __enter__(self) -> "GoogleEmbeddingProvider":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
