"""Thin HTTP translations; filtering and ranking stay in application services."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ..inventory import InventoryRepository
from ..retrieval import InventorySearchService, RetrievalError
from .dependencies import (
    ApiServices,
    get_inventory_repository,
    get_inventory_search_service,
    get_services,
)
from .schemas import (
    CarResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["health"])
    def health(services: Annotated[ApiServices, Depends(get_services)]) -> HealthResponse:
        """Report loaded inventory and local semantic readiness without using Google quota."""
        return HealthResponse(
            inventory_count=services.inventory_count,
            semantic_retrieval_available=services.semantic_available,
        )

    @router.get(
        "/inventory/{listing_id}",
        response_model=CarResponse,
        tags=["inventory"],
        responses={404: {"description": "Listing does not exist"}},
    )
    def car_details(
        listing_id: str,
        repository: Annotated[InventoryRepository, Depends(get_inventory_repository)],
    ) -> CarResponse:
        """Return source vehicle facts by stable ID; unknown values remain null."""
        car = repository.get_car_details(listing_id)
        if car is None:
            raise HTTPException(status_code=404, detail="Listing not found.")
        return CarResponse.model_validate(car)

    @router.post(
        "/inventory/search",
        response_model=SearchResponse,
        tags=["inventory"],
        responses={503: {"description": "Semantic retrieval unavailable"}},
    )
    def search(
        body: SearchRequest,
        service: Annotated[InventorySearchService, Depends(get_inventory_search_service)],
        services: Annotated[ApiServices, Depends(get_services)],
    ) -> SearchResponse:
        """Apply hard filters, optionally rank valid candidates, then apply the final limit."""
        query = body.to_domain()
        if query.semantic_query is not None:
            if not services.semantic_available:
                raise RetrievalError("Semantic retrieval unavailable")
            # FastAPI sync routes can run concurrently even in one worker.
            with services.semantic_lock:
                matches = service.search(query)
        else:
            matches = service.search(query)
        results = tuple(
            SearchResultResponse(
                **{name: getattr(result.car, name) for name in CarResponse.model_fields},
                semantic_score=result.semantic_score,
            )
            for result in matches
        )
        return SearchResponse(count=len(results), results=results)

    return router
