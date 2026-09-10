"""Errors callers can distinguish from inventory loading and filtering errors."""


class RetrievalError(ValueError):
    """Semantic retrieval could not produce trustworthy ranking results."""


class EmbeddingError(RetrievalError):
    """An embedding provider failed or returned invalid vectors."""


class EmbeddingCacheError(RetrievalError):
    """The generated embedding cache is unreadable or inconsistent."""
