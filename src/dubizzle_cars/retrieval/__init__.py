"""Application-level hybrid retrieval; importing this package performs no I/O."""

from .cache import EmbeddingCache
from .documents import build_semantic_document
from .embeddings import EmbeddingProvider
from .errors import EmbeddingCacheError, EmbeddingError, RetrievalError
from .models import InventorySearchRequest, InventorySearchResult
from .ranking import SemanticRanker
from .service import InventorySearchService

__all__ = [
    "EmbeddingCache",
    "EmbeddingCacheError",
    "EmbeddingError",
    "EmbeddingProvider",
    "InventorySearchRequest",
    "InventorySearchResult",
    "InventorySearchService",
    "RetrievalError",
    "SemanticRanker",
    "build_semantic_document",
]
