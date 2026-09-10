"""Manually invoked real-API retrieval check; never run by ordinary pytest."""

import argparse
import sys
from pathlib import Path

from . import InMemoryInventory, InventoryFilter, load_inventory
from .retrieval import (
    EmbeddingCache,
    InventorySearchRequest,
    InventorySearchService,
    RetrievalError,
    SemanticRanker,
)
from .retrieval.google_embeddings import GoogleEmbeddingProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real Google API smoke test (uses quota).")
    parser.add_argument("--workbook", type=Path, default=Path("data/sample_cars_dataset.xlsx"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/inventory_embeddings.json"))
    args = parser.parse_args(argv)
    cars = load_inventory(args.workbook)
    inventory = InMemoryInventory(cars)
    examples = (
        InventorySearchRequest(
            semantic_query="comfortable luxury car with lots of technology", limit=5
        ),
        InventorySearchRequest(semantic_query="sporty performance car", limit=5),
        InventorySearchRequest(semantic_query="practical family car for long drives", limit=5),
        InventorySearchRequest(
            InventoryFilter(make="mercedes-benz"), "luxurious and well equipped", 5
        ),
        InventorySearchRequest(InventoryFilter(year_min=2020), "sporty", 5),
    )
    print(f"Loaded {len(cars)} real listings. Scores are ranking values only, not vehicle facts.")
    try:
        with GoogleEmbeddingProvider(cars) as provider:
            service = InventorySearchService(
                inventory,
                SemanticRanker(provider, provider.provider_id, EmbeddingCache(args.cache)),
            )
            for request in examples:
                print(f"\nQuery: {request.semantic_query}\nHard filters: {request.filters}")
                results = service.search(request)
                allowed = {car.listing_id for car in inventory.filter_inventory(request.filters)}
                for result in results:
                    car = result.car
                    if (
                        car.listing_id not in allowed
                        or inventory.get_car_details(car.listing_id) != car
                    ):
                        raise RetrievalError("Smoke test detected an invalid inventory candidate")
                    print(
                        f"ID={car.listing_id} year={car.year} make={car.make} model={car.model} score={result.semantic_score:.4f}"
                    )
                print(f"{len(results)} results; hard filters and listing IDs verified.")
    except RetrievalError as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
