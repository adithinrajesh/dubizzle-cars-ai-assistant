import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path

import pytest

from dubizzle_cars import Car, InMemoryInventory, InventoryFilter, Metadata, load_inventory
from dubizzle_cars.retrieval import (
    EmbeddingCache,
    EmbeddingCacheError,
    EmbeddingError,
    InventorySearchRequest,
    InventorySearchResult,
    InventorySearchService,
    RetrievalError,
    SemanticRanker,
    build_semantic_document,
)


class FakeProvider:
    def __init__(self, vectors=None, query=(1.0, 0.0)):
        self.vectors = vectors or {}
        self.query = query
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(tuple(texts))
        return [self.vectors.get(text, (0.0, 1.0)) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self.query


@pytest.fixture
def cars():
    return (
        Car("1", 2018, "land rover", "velar", title="Basic", metadata=Metadata(Decimal(80000))),
        Car(
            "2",
            2020,
            "land rover",
            "velar",
            title="Comfortable",
            metadata=Metadata(Decimal(100000)),
        ),
        Car("3", 2024, "bentley", "continental", title="Luxury", metadata=Metadata(Decimal(90000))),
        Car("4", 2017, "land rover", "velar", title="Older", metadata=Metadata(Decimal(80000))),
        Car(
            "5", 2021, "land rover", "velar", title="Expensive", metadata=Metadata(Decimal(200000))
        ),
        Car("6", 2021, "land rover", "velar", title="Unknown price"),
    )


@pytest.fixture
def setup_search(tmp_path, cars):
    provider = FakeProvider({build_semantic_document(car): (1, 0) for car in cars[1:]})
    ranker = SemanticRanker(provider, "fake/model-v1", EmbeddingCache(tmp_path / "cache.json"))
    repository = InMemoryInventory(reversed(cars))
    return provider, ranker, repository, InventorySearchService(repository, ranker)


@pytest.mark.parametrize("query", [None, "", " \t\n"])
def test_structured_search_preserves_phase_one_and_never_embeds(setup_search, query):
    provider, _, repository, service = setup_search
    filters = InventoryFilter(make="land rover", year_min=2018, price_max=100000)
    results = service.search(InventorySearchRequest(filters, query))
    assert tuple(result.car for result in results) == repository.filter_inventory(filters)
    assert all(result.semantic_score is None for result in results)
    assert provider.document_calls == provider.query_calls == []
    assert service.search(InventorySearchRequest(filters, query, limit=1)) == results[:1]


def test_hard_constraints_precede_ranking_and_limit(setup_search, cars):
    provider, _, _, service = setup_search
    filters = InventoryFilter(make="land rover", year_min=2018, year_max=2021, price_max=100000)
    results = service.search(InventorySearchRequest(filters, " comfortable "))
    assert [result.car.listing_id for result in results] == ["2", "1"]
    assert [result.semantic_score for result in results] == [1.0, 0.0]
    assert provider.document_calls == [
        (build_semantic_document(cars[0]), build_semantic_document(cars[1]))
    ]
    assert provider.query_calls == ["comfortable"]
    assert service.search(InventorySearchRequest(filters, "comfortable", 1)) == results[:1]
    assert len(provider.document_calls) == 1  # The second search reuses document embeddings.


def test_empty_candidates_never_embed(setup_search):
    provider, _, _, service = setup_search
    assert (
        service.search(InventorySearchRequest(InventoryFilter(make="missing"), "luxurious")) == ()
    )
    assert provider.document_calls == provider.query_calls == []


def test_structured_search_needs_no_ranker_and_semantic_search_fails_clearly(cars):
    service = InventorySearchService(InMemoryInventory(cars))
    assert len(service.search(InventorySearchRequest())) == len(cars)
    with pytest.raises(RetrievalError, match="configured ranker"):
        service.search(InventorySearchRequest(semantic_query="comfortable"))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_limits(limit):
    with pytest.raises(ValueError, match="limit"):
        InventorySearchRequest(limit=limit)


def test_request_validation_and_immutability():
    with pytest.raises(ValueError, match="semantic_query"):
        InventorySearchRequest(semantic_query=42)
    with pytest.raises(ValueError, match="filters"):
        InventorySearchRequest(filters=None)
    with pytest.raises(FrozenInstanceError):
        InventorySearchRequest().limit = 2
    with pytest.raises(FrozenInstanceError):
        InventorySearchResult(Car("1")).semantic_score = 1


def test_document_is_source_only_and_handles_missing_data():
    car = Car(
        "1",
        2020,
        "Ford",
        "Focus",
        "other",
        "Actual title",
        "No warranty",
        metadata=Metadata(regional_specs="GCC"),
    )
    assert (
        build_semantic_document(car)
        == "2020 Ford Focus\n\nTrim: other\n\nActual title\n\nNo warranty"
    )
    assert build_semantic_document(Car("2")) == ""
    assert build_semantic_document(
        replace(car, listing_id="new", source_row=10)
    ) == build_semantic_document(car)


def test_cosine_ties_and_candidate_permutations(tmp_path):
    cars = (Car("2", title="A"), Car("10", title="B"), Car("1", title="C"))
    provider = FakeProvider({"A": (3, 4), "B": (6, 8), "C": (-1, 0)})
    ranker = SemanticRanker(provider, "fake", EmbeddingCache(tmp_path / "cache.json"))
    first = ranker.rank(cars, "query")
    assert [result.car.listing_id for result in first] == ["10", "2", "1"]
    assert [result.semantic_score for result in first] == pytest.approx([0.6, 0.6, -1])
    assert ranker.rank(reversed(cars), "query") == first


@pytest.mark.parametrize(
    "vector", [(), (0, 0), (float("nan"), 1), (float("inf"), 1), (True, 1), ("1", 0), None]
)
def test_invalid_vectors_fail_for_query_and_documents(tmp_path, vector):
    cache = EmbeddingCache(tmp_path / "cache.json")
    with pytest.raises(EmbeddingError, match="Invalid embedding"):
        SemanticRanker(FakeProvider(query=vector), "fake", cache).rank([Car("1", title="A")], "q")
    with pytest.raises(EmbeddingError, match="Invalid embedding"):
        SemanticRanker(FakeProvider({"A": vector}), "fake", cache).rank([Car("1", title="A")], "q")
    assert not cache.path.exists()


def test_dimension_mismatch_and_provider_count_failure(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.json")
    with pytest.raises(EmbeddingError, match="dimension"):
        SemanticRanker(FakeProvider({"A": (1, 2, 3)}), "fake", cache).rank(
            [Car("1", title="A")], "q"
        )

    class WrongCount(FakeProvider):
        def embed_documents(self, texts):
            return []

    with pytest.raises(EmbeddingError, match="count"):
        SemanticRanker(WrongCount(), "fake", cache).rank([Car("1")], "q")


def test_provider_exceptions_are_wrapped_without_fallback(tmp_path):
    class Broken(FakeProvider):
        def embed_query(self, text):
            raise RuntimeError("provider is down")

    ranker = SemanticRanker(Broken(), "fake", EmbeddingCache(tmp_path / "cache.json"))
    with pytest.raises(EmbeddingError, match="provider failed") as exc:
        ranker.rank([Car("1")], "q")
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_extreme_finite_vectors_are_normalized_safely(tmp_path):
    provider = FakeProvider({"A": (1e308, 1e308), "B": (1e-308, 1e-308)}, query=(1e308, 1e308))
    ranker = SemanticRanker(provider, "fake", EmbeddingCache(tmp_path / "cache.json"))
    assert [
        r.semantic_score for r in ranker.rank([Car("1", title="A"), Car("2", title="B")], "q")
    ] == pytest.approx([1, 1])


def test_cache_reuses_across_instances_and_maps_by_id(tmp_path):
    path = tmp_path / "cache.json"
    cars = [Car("a", title="A"), Car("b", title="B")]
    first_provider = FakeProvider({"A": (1, 0), "B": (0, 1)})
    expected = SemanticRanker(first_provider, "fake", EmbeddingCache(path)).rank(cars, "q")
    second_provider = FakeProvider()  # Would rank differently if documents were regenerated.
    ranker = SemanticRanker(second_provider, "fake", EmbeddingCache(path))
    assert ranker.rank(reversed(cars), "q") == expected
    assert ranker.rank([cars[1]], "q") == (expected[1],)
    assert second_provider.document_calls == []
    payload = json.loads(path.read_text())
    assert payload["record_count"] == 2
    assert set(payload["entries"]) == {"a", "b"}


def test_cache_invalidates_changed_text_identity_and_document_version(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    provider = FakeProvider()
    cache = EmbeddingCache(path)
    ranker = SemanticRanker(provider, "fake/v1", cache)
    cars = [Car("a", title="A"), Car("b", title="B")]
    ranker.rank(cars, "q")
    ranker.rank([replace(cars[0], title="Changed"), cars[1]], "q")
    assert provider.document_calls[-1] == ("Changed",)
    SemanticRanker(provider, "fake/v2", cache).rank(cars, "q")
    assert provider.document_calls[-1] == ("A", "B")
    monkeypatch.setattr("dubizzle_cars.retrieval.cache.DOCUMENT_VERSION", "next")
    SemanticRanker(provider, "fake/v2", cache).rank(cars, "q")
    assert len(provider.document_calls) == 4


@pytest.mark.parametrize("damage", ["json", "checksum", "count", "dimension", "mapping"])
def test_corrupt_cache_fails_explicitly_and_can_be_rebuilt(tmp_path, damage):
    cache = EmbeddingCache(tmp_path / "cache.json")
    provider = FakeProvider()
    ranker = SemanticRanker(provider, "fake", cache)
    cars = [Car("a", title="A")]
    ranker.rank(cars, "q")
    data = json.loads(cache.path.read_text())
    if damage == "json":
        cache.path.write_text("{broken")
    else:
        if damage == "checksum":
            data["entries"]["a"]["vector"] = [0, 0]
        elif damage == "count":
            data["record_count"] = 999
        elif damage == "dimension":
            data["dimension"] = 3
        else:
            data["entries"]["wrong-id"] = data["entries"].pop("a")
        cache.path.write_text(json.dumps(data))
    with pytest.raises(EmbeddingCacheError):
        ranker.rank(cars, "q")
    assert len(provider.document_calls) == 1
    cache.path.unlink()
    assert len(ranker.rank(cars, "q")) == 1
    assert len(provider.document_calls) == 2


def test_ranker_rejects_duplicate_ids_and_blank_query(tmp_path):
    provider = FakeProvider()
    ranker = SemanticRanker(provider, "fake", EmbeddingCache(tmp_path / "cache.json"))
    with pytest.raises(RetrievalError, match="Duplicate"):
        ranker.rank([Car("1"), Car("1")], "q")
    with pytest.raises(RetrievalError, match="nonblank"):
        ranker.rank([Car("1")], " ")
    assert ranker.rank([], "q") == ()
    assert provider.query_calls == []


@pytest.mark.parametrize(
    "bad_results",
    [
        (),
        (InventorySearchResult(Car("outside"), 1),),
        (InventorySearchResult(Car("1", title="altered"), 1),),
    ],
)
def test_service_rejects_ranker_that_changes_candidates(bad_results):
    class BadRanker:
        def rank(self, candidates, query):
            return bad_results

    service = InventorySearchService(InMemoryInventory([Car("1")]), BadRanker())
    with pytest.raises(RetrievalError, match="candidate set"):
        service.search(InventorySearchRequest(semantic_query="q"))


def test_workbook_unchanged_after_hybrid_search(tmp_path):
    path = Path(__file__).resolve().parents[1] / "data/sample_cars_dataset.xlsx"
    original = path.read_bytes()
    cars = load_inventory(path)
    provider = FakeProvider()
    service = InventorySearchService(
        InMemoryInventory(cars),
        SemanticRanker(provider, "fake", EmbeddingCache(tmp_path / "cache.json")),
    )
    filters = InventoryFilter(make="land rover", year_min=2018)
    results = service.search(InventorySearchRequest(filters, "comfortable", 5))
    assert results
    assert all(result.car.make == "land rover" and result.car.year >= 2018 for result in results)
    assert path.read_bytes() == original
    assert (
        hashlib.sha256(original).hexdigest()
        == "94a97f84aa742ec8ed526bb14d41ca4134ab8a2e1db1ab7f3d2f03af5c38bbb9"
    )
