"""Application requests and results; semantic scores are not vehicle attributes."""

from dataclasses import dataclass, field

from ..inventory import InventoryFilter
from ..models import Car


@dataclass(frozen=True, slots=True)
class InventorySearchRequest:
    filters: InventoryFilter = field(default_factory=InventoryFilter)
    semantic_query: str | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.filters, InventoryFilter):
            raise ValueError("filters must be an InventoryFilter")
        if self.semantic_query is not None:
            if not isinstance(self.semantic_query, str):
                raise ValueError("semantic_query must be a string or None")
            object.__setattr__(self, "semantic_query", self.semantic_query.strip() or None)
        if self.limit is not None and (type(self.limit) is not int or self.limit <= 0):
            raise ValueError("limit must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class InventorySearchResult:
    car: Car
    semantic_score: float | None = None
