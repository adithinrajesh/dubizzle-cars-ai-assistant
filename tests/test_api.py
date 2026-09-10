import importlib
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from dubizzle_cars import Car, InMemoryInventory, InventoryFilter, Metadata
from dubizzle_cars.api.app import create_app
from dubizzle_cars.api.dependencies import get_inventory_repository, get_inventory_search_service
from dubizzle_cars.api.settings import ApiSettings
from dubizzle_cars.loader import COLUMNS
from dubizzle_cars.retrieval import EmbeddingError, InventorySearchService


class FakeProvider:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(tuple(texts))
        return [(1, 0) if "Premium" in text else (0, 1) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return (1, 0)


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def cars():
    return (
        Car(
            "1",
            2020,
            "mercedes-benz",
            "c-class",
            title="Basic",
            photo_url="https://example.com/source.jpg",
            metadata=Metadata(Decimal("100000.10"), 30000, "Warranty expired", "GCC"),
        ),
        Car(
            "2",
            2021,
            "mercedes-benz",
            "e-class",
            title="Premium",
            metadata=Metadata(Decimal("110000.20"), 40000),
        ),
        Car(
            "3",
            2022,
            "bentley",
            "continental",
            title="Premium",
            metadata=Metadata(Decimal("100000.10")),
        ),
        Car(
            "4",
            2019,
            "mercedes-benz",
            "c-class",
            title="Premium",
            metadata=Metadata(Decimal("50000.00")),
        ),
        Car("5", 2023, "mercedes-benz", "e-class", title="Premium"),
    )


@pytest.fixture
def api(tmp_path, cars):
    provider = FakeProvider()
    repo = InMemoryInventory(cars)
    app = create_app(
        ApiSettings(cache_path=tmp_path / "cache.json"),
        repository=repo,
        embedding_provider=provider,
        provider_id="offline/v1",
    )
    with TestClient(app) as client:
        yield client, provider, repo


def test_health_details_and_source_semantics(api):
    client, provider, _ = api
    assert client.get("/health").json() == {
        "status": "ok",
        "inventory_count": 5,
        "semantic_retrieval_available": True,
    }
    details = client.get("/inventory/1")
    assert details.status_code == 200
    car = details.json()
    assert car["listing_id"] == "1"
    assert car["photo_url"] == "https://example.com/source.jpg"
    assert car["metadata"]["sale_price_aed"] == "100000.10"
    assert car["metadata"]["warranty_mention"] == "Warranty expired"
    assert "has_warranty" not in car["metadata"]
    unknown = client.get("/inventory/5").json()
    assert all(value is None for value in unknown["metadata"].values())
    assert unknown["photo_url"] is None
    assert client.get("/inventory/missing").status_code == 404
    assert provider.document_calls == provider.query_calls == []


@pytest.mark.parametrize("query", [None, "", " \t\n"])
def test_structured_search_preserves_domain_behavior(api, query):
    client, provider, repo = api
    filters = {
        "make": "MERCEDES-BENZ",
        "year_min": 2020,
        "year_max": 2022,
        "price_max": "110000.20",
    }
    response = client.post("/inventory/search", json={"filters": filters, "semantic_query": query})
    assert response.status_code == 200
    expected = repo.filter_inventory(
        InventoryFilter(
            make="MERCEDES-BENZ", year_min=2020, year_max=2022, price_max=Decimal("110000.20")
        )
    )
    assert (
        [r["listing_id"] for r in response.json()["results"]]
        == [c.listing_id for c in expected]
        == ["1", "2"]
    )
    assert all(r["semantic_score"] is None for r in response.json()["results"])
    assert provider.document_calls == provider.query_calls == []
    limited = client.post("/inventory/search", json={"filters": filters, "limit": 1}).json()
    assert limited["count"] == 1
    assert limited["results"][0]["listing_id"] == "1"


def test_semantic_ranking_and_limit_follow_all_hard_filters(api):
    client, provider, _ = api
    payload = {
        "filters": {
            "make": "mercedes-benz",
            "year_min": 2020,
            "year_max": 2022,
            "price_max": 120000,
        },
        "semantic_query": "luxurious",
        "limit": 1,
    }
    response = client.post("/inventory/search", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 1
    assert result["results"][0]["listing_id"] == "2"
    assert result["results"][0]["semantic_score"] == 1.0
    assert provider.query_calls == ["luxurious"]
    assert len(provider.document_calls[0]) == 2


def test_empty_search_and_exact_decimal_bound(api):
    client, provider, _ = api
    assert client.post("/inventory/search", json={"filters": {"make": "missing"}}).json() == {
        "count": 0,
        "results": [],
    }
    assert client.post(
        "/inventory/search", json={"filters": {"make": "missing"}, "semantic_query": "sporty"}
    ).json() == {"count": 0, "results": []}
    result = client.post(
        "/inventory/search",
        json={
            "filters": {"make": "mercedes-benz", "price_min": "100000.10", "price_max": "100000.10"}
        },
    ).json()
    assert [r["listing_id"] for r in result["results"]] == ["1"]
    assert provider.query_calls == []


def test_credentials_not_required_and_unavailable_semantics_returns_503(cars):
    with TestClient(create_app(repository=InMemoryInventory(cars))) as client:
        assert client.get("/health").json()["semantic_retrieval_available"] is False
        assert client.post("/inventory/search", json={}).status_code == 200
        response = client.post("/inventory/search", json={"semantic_query": "sporty"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Semantic retrieval is currently unavailable."}


@pytest.mark.parametrize(
    "payload",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": True},
        {"limit": 2.5},
        {"limit": "5"},
        {"filters": {"year_min": "2020"}},
        {"filters": {"year_min": True}},
        {"filters": {"price_min": "NaN"}},
        {"filters": {"price_min": "oops"}},
        {"filters": {"price_max": 100000.1}},
        {"filters": {"price_min": -1}},
        {"filters": {"year_min": 2022, "year_max": 2020}},
        {"filters": {"price_min": "200", "price_max": "100"}},
        {"filters": {"has_warranty_mention": "true"}},
        {"filters": {"make": " "}},
        {"semantic_query": 3},
        {"filters": {"semantic_query": "wrong layer"}},
    ],
)
def test_invalid_requests_are_422(api, payload):
    client, provider, _ = api
    response = client.post("/inventory/search", json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert provider.query_calls == []


def test_malformed_json_and_unknown_fields_are_not_echoed(api):
    client, _, _ = api
    response = client.post(
        "/inventory/search", content='{"broken":', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    response = client.post("/inventory/search", json={"unexpected": "do-not-echo-input"})
    assert response.status_code == 422
    assert "do-not-echo-input" not in response.text


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingError("private-provider-message"),
        RuntimeError("private-provider-message"),
    ],
)
def test_provider_and_internal_errors_are_safe(tmp_path, cars, caplog, error):
    provider = FakeProvider()
    provider.embed_query = Mock(side_effect=error)
    app = create_app(
        ApiSettings(cache_path=tmp_path / "cache.json"),
        repository=InMemoryInventory(cars),
        embedding_provider=provider,
        provider_id="offline/v1",
    )
    with TestClient(app) as client:
        response = client.post("/inventory/search", json={"semantic_query": "sporty"})
        # The ranker wraps unexpected embedding-provider exceptions as EmbeddingError.
        assert response.status_code == 503
        assert "private-provider-message" not in response.text
        assert "private-provider-message" not in caplog.text


def test_unexpected_service_failure_is_500_and_not_logged(cars, caplog):
    service = Mock()
    service.search.side_effect = RuntimeError("private-internal-message")
    with TestClient(
        create_app(repository=InMemoryInventory(cars), search_service=service)
    ) as client:
        response = client.post("/inventory/search", json={})
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error."}
        assert "private-internal-message" not in caplog.text


def test_lifespan_loads_temporary_workbook_once_and_fails_on_missing(tmp_path, monkeypatch):
    module = importlib.import_module("dubizzle_cars.api.app")
    path = tmp_path / "inventory.xlsx"
    book = Workbook()
    book.active.title = "cleaned dataset"
    book.active.append(COLUMNS)
    book.active.append(["01", 2020, "ford", "focus", None, None, None, None])
    book.save(path)
    book.close()
    before = path.read_bytes()
    load = Mock(wraps=module.load_inventory)
    monkeypatch.setattr(module, "load_inventory", load)
    app = create_app(ApiSettings(workbook_path=path, semantic_enabled=False))
    assert load.call_count == 0
    with TestClient(app) as client:
        assert client.get("/health").json()["inventory_count"] == 1
        assert client.get("/inventory/01").status_code == 200
        assert client.post("/inventory/search", json={}).status_code == 200
        assert load.call_count == 1
    assert not hasattr(app.state, "services")
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="Inventory initialization failed"):
        with TestClient(create_app(ApiSettings(workbook_path=tmp_path / "missing.xlsx"))):
            pass


def test_import_does_not_load_inventory(monkeypatch):
    load = Mock(side_effect=AssertionError("Import must not load workbook"))
    monkeypatch.setattr("dubizzle_cars.loader.load_inventory", load)
    module = importlib.import_module("dubizzle_cars.api.app")
    importlib.reload(module)
    assert load.call_count == 0
    # Restore the binding captured by the reloaded module for subsequent tests.
    monkeypatch.undo()
    importlib.reload(module)


def test_google_is_lazy_and_owned_provider_is_closed(cars, monkeypatch):
    provider = FakeProvider()
    provider.provider_id = "mock-google"
    provider.close = Mock()
    factory = Mock(return_value=provider)
    monkeypatch.setattr("dubizzle_cars.api.app.GoogleEmbeddingProvider", factory)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-config-placeholder")
    app = create_app(repository=InMemoryInventory(cars))
    assert factory.call_count == 0
    with TestClient(app) as client:
        assert factory.call_count == 1
        assert client.get("/health").json()["semantic_retrieval_available"] is True
        assert provider.query_calls == provider.document_calls == []
    provider.close.assert_called_once()
    with TestClient(
        create_app(ApiSettings(semantic_enabled=False), repository=InMemoryInventory(cars))
    ) as client:
        assert client.get("/health").json()["semantic_retrieval_available"] is False
    assert factory.call_count == 1


def test_prebuilt_service_and_dependency_overrides(cars):
    repo = InMemoryInventory(cars)
    service = Mock(wraps=InventorySearchService(repo))
    app = create_app(repository=repo, search_service=service)
    with TestClient(app) as client:
        assert client.post("/inventory/search", json={"limit": 1}).status_code == 200
        assert service.search.call_args.args[0].limit == 1
        app.dependency_overrides[get_inventory_repository] = lambda: InMemoryInventory([])
        assert client.get("/inventory/1").status_code == 404
        app.dependency_overrides[get_inventory_search_service] = lambda: InventorySearchService(
            InMemoryInventory([])
        )
        assert client.post("/inventory/search", json={}).json()["count"] == 0


def test_docs_and_supplied_workbook_preserved():
    path = Path(__file__).resolve().parents[1] / "data/sample_cars_dataset.xlsx"
    before = path.read_bytes()
    with TestClient(create_app(ApiSettings(workbook_path=path, semantic_enabled=False))) as client:
        assert client.get("/health").json()["inventory_count"] == 100
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {
            "/sessions",
            "/users/{user_id}/profile",
            "/health",
            "/inventory/{listing_id}",
            "/inventory/search",
            "/chat",
        }
        assert client.get("/inventory/45").status_code == 200
        assert (
            client.post(
                "/inventory/search", json={"filters": {"make": "mercedes-benz", "year_min": 2020}}
            ).status_code
            == 200
        )
    assert path.read_bytes() == before
