"""Manual real-API agent verification; not collected or invoked by pytest."""

import argparse
import json
import os
import sys
from pathlib import Path

from . import InMemoryInventory, load_inventory
from .agent import AgentService
from .agent.errors import AgentError
from .agent.openai_agent import GoogleChatProvider
from .agent.tools import InventoryTools
from .retrieval import EmbeddingCache, InventorySearchService, RetrievalError, SemanticRanker
from .retrieval.google_embeddings import GoogleEmbeddingProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manual Gemini agent check; generation/embedding calls use API quota."
    )
    parser.add_argument("--workbook", type=Path, default=Path("data/sample_cars_dataset.xlsx"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/inventory_embeddings.json"))
    parser.add_argument(
        "--model", default=os.environ.get("DUBIZZLE_CHAT_MODEL", "gemini-3.8-flash")
    )
    args = parser.parse_args(argv)
    cars = load_inventory(args.workbook)
    repository = InMemoryInventory(cars)
    unknown = next((car for car in cars if car.metadata.warranty_mention is None), None)
    examples = [
        (
            "hard filters",
            "Show me Mercedes-Benz cars from 2020 onwards that are luxurious and well equipped.",
            "search_inventory",
        ),
        ("soft preference", "I want something sporty and performance focused.", "search_inventory"),
        ("automotive knowledge", "What is the difference between an SUV and a sedan?", None),
        ("unrelated request", "Write me a Python sorting algorithm.", None),
        ("platform request", "Compare dubizzle with another used-car marketplace.", None),
    ]
    if unknown is not None:
        examples.append(
            (
                "unknown warranty",
                f"Does listing {unknown.listing_id} have warranty? Explain if the listing does not specify it.",
                "get_car_details",
            )
        )
    try:
        with (
            GoogleEmbeddingProvider(cars) as embeddings,
            GoogleChatProvider(args.model) as conversation,
        ):
            search = InventorySearchService(
                repository,
                SemanticRanker(embeddings, embeddings.provider_id, EmbeddingCache(args.cache)),
            )
            agent = AgentService(conversation, InventoryTools(repository, search))
            for label, message, expected_tool in examples:
                result = agent.chat(message)
                if expected_tool is not None and expected_tool not in result.tool_names:
                    raise AgentError("Expected inventory tool was not called")
                if expected_tool is None and result.tool_names:
                    raise AgentError("Unexpected inventory tool for a non-inventory request")
                if any(repository.get_car_details(car.listing_id) != car for car in result.cars):
                    raise AgentError("Non-inventory car returned")
                if label == "hard filters" and (
                    not result.cars
                    or any(
                        car.make != "mercedes-benz" or car.year is None or car.year < 2020
                        for car in result.cars
                    )
                ):
                    raise AgentError("Hard filters were not preserved")
                print(
                    f"\n{label}: {message}\nTools: {', '.join(result.tool_names) or 'none'}\n{result.reply}"
                )
                print(
                    json.dumps(
                        [
                            {
                                "listing_id": car.listing_id,
                                "year": car.year,
                                "make": car.make,
                                "model": car.model,
                                "warranty_mention": car.metadata.warranty_mention,
                            }
                            for car in result.cars
                        ],
                        ensure_ascii=True,
                    )
                )
                if label == "unknown warranty":
                    print("Verify the reply explains missing information, not absence of coverage.")
        print(
            "\nMechanical checks passed. Inspect relevance, scope handling and prose grounding manually."
        )
    except (AgentError, RetrievalError) as exc:
        print(
            f"Agent smoke check failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
