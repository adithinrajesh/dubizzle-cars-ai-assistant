"""Hard filters first, optional soft ranking second, final limit last."""

from ..inventory import InventoryRepository
from .errors import RetrievalError
from .models import InventorySearchRequest, InventorySearchResult
from .ranking import SemanticRanker


class InventorySearchService:
    def __init__(
        self, repository: InventoryRepository, ranker: SemanticRanker | None = None
    ) -> None:
        self._repository = repository
        self._ranker = ranker

    def search(self, request: InventorySearchRequest) -> tuple[InventorySearchResult, ...]:
        candidates = self._repository.filter_inventory(request.filters)
        if not candidates:
            return ()
        if request.semantic_query is None:
            results = tuple(InventorySearchResult(car) for car in candidates)
        else:
            if self._ranker is None:
                raise RetrievalError("Semantic search requires a configured ranker")
            results = self._ranker.rank(candidates, request.semantic_query)
            # Guard the application boundary even if a future ranker implementation changes.
            expected = {car.listing_id: car for car in candidates}
            if (
                len(results) != len(candidates)
                or len({result.car.listing_id for result in results}) != len(results)
                or any(expected.get(result.car.listing_id) != result.car for result in results)
            ):
                raise RetrievalError("Ranker changed the deterministic candidate set")
        return results if request.limit is None else results[: request.limit]
