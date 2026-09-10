"""Provider boundary and validation shared by fresh and cached embeddings."""

import math
from collections.abc import Sequence
from numbers import Real
from typing import Protocol

from .errors import EmbeddingError

Vector = tuple[float, ...]


class EmbeddingProvider(Protocol):
    """Return document vectors in input order and queries in the same vector space."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


def normalize_vector(values: Sequence[float], dimension: int | None = None) -> Vector:
    try:
        if isinstance(values, (str, bytes)):
            raise ValueError("not a numeric sequence")
        raw = tuple(values)
        if not raw or any(isinstance(v, bool) or not isinstance(v, Real) for v in raw):
            raise ValueError("expected nonempty real-valued vector")
        vector = tuple(float(v) for v in raw)
        if any(not math.isfinite(v) for v in vector):
            raise ValueError("nonfinite component")
        if dimension is not None and len(vector) != dimension:
            raise ValueError(f"dimension mismatch: expected {dimension}, got {len(vector)}")
        scale = max(abs(v) for v in vector)
        if scale == 0:
            raise ValueError("zero vector")
        scaled = tuple(v / scale for v in vector)
        norm = math.sqrt(math.fsum(v * v for v in scaled))
        return tuple(v / norm for v in scaled)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError(f"Invalid embedding: {exc}") from exc


def embed_query(provider: EmbeddingProvider, text: str) -> Vector:
    try:
        return normalize_vector(provider.embed_query(text))
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("Query embedding provider failed") from exc


def embed_documents(
    provider: EmbeddingProvider, texts: Sequence[str], dimension: int
) -> tuple[Vector, ...]:
    try:
        vectors = tuple(provider.embed_documents(texts))
        if len(vectors) != len(texts):
            raise EmbeddingError("Document embedding count does not match input count")
        return tuple(normalize_vector(vector, dimension) for vector in vectors)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("Document embedding provider failed") from exc
