"""Local cosine ranking of exactly the supplied candidate set."""

import math
from collections.abc import Iterable

from ..models import Car
from .cache import EmbeddingCache
from .documents import build_semantic_document
from .embeddings import EmbeddingProvider, embed_query
from .errors import RetrievalError
from .models import InventorySearchResult


class SemanticRanker:
    def __init__(
        self,
        provider: EmbeddingProvider,
        provider_id: str,
        cache: EmbeddingCache | None = None,
    ) -> None:
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must identify provider, model and embedding settings")
        self._provider = provider
        self._provider_id = provider_id
        self._cache = cache if cache is not None else EmbeddingCache()

    def rank(self, candidates: Iterable[Car], query: str) -> tuple[InventorySearchResult, ...]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Semantic ranking requires a nonblank query")
        cars = tuple(candidates)
        if not cars:
            return ()
        if any(not isinstance(car.listing_id, str) or not car.listing_id for car in cars):
            raise RetrievalError("Candidates require nonempty string listing IDs")
        if len({car.listing_id for car in cars}) != len(cars):
            raise RetrievalError("Duplicate candidate listing IDs")
        query_vector = embed_query(self._provider, query.strip())
        documents = {car.listing_id: build_semantic_document(car) for car in cars}
        vectors = self._cache.get_embeddings(
            documents, self._provider, self._provider_id, len(query_vector)
        )
        results = []
        for car in cars:
            score = math.fsum(
                a * b for a, b in zip(query_vector, vectors[car.listing_id], strict=True)
            )
            results.append(InventorySearchResult(car, max(-1.0, min(1.0, score))))
        return tuple(
            sorted(results, key=lambda result: (-result.semantic_score, result.car.listing_id))
        )
