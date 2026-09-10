import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from uvicorn.importer import import_from_string


def test_root_entry_point_is_lazy_and_preserves_api_behavior(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("DUBIZZLE_WORKBOOK_PATH", str(root / "data/sample_cars_dataset.xlsx"))
    monkeypatch.setenv("DUBIZZLE_SEMANTIC_ENABLED", "false")
    factory_module = importlib.import_module("dubizzle_cars.api.app")
    load = Mock(wraps=factory_module.load_inventory)
    client_factory = Mock(side_effect=AssertionError("Google must not be called"))
    monkeypatch.setattr(factory_module, "load_inventory", load)
    monkeypatch.setattr("dubizzle_cars.retrieval.google_embeddings.genai.Client", client_factory)
    monkeypatch.delitem(sys.modules, "main", raising=False)
    try:
        app = import_from_string("main:app")
        assert isinstance(app, FastAPI)
        load.assert_not_called()
        client_factory.assert_not_called()
        with TestClient(app) as client:
            assert client.get("/health").json() == {
                "status": "ok",
                "inventory_count": 100,
                "semantic_retrieval_available": False,
            }
            assert client.get("/inventory/45").json()["listing_id"] == "45"
            response = client.post(
                "/inventory/search",
                json={
                    "filters": {"make": "mercedes-benz", "year_min": 2020},
                    "limit": 5,
                },
            )
            assert response.status_code == 200
            results = response.json()["results"]
            assert results
            assert all(
                car["make"] == "mercedes-benz"
                and car["year"] >= 2020
                and car["semantic_score"] is None
                for car in results
            )
            assert client.get("/docs").status_code == 200
        load.assert_called_once()
        client_factory.assert_not_called()
    finally:
        sys.modules.pop("main", None)
