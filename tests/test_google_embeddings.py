import json
import traceback
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types

from dubizzle_cars import Car, InMemoryInventory
from dubizzle_cars.retrieval import (
    EmbeddingCache,
    EmbeddingError,
    InventorySearchRequest,
    InventorySearchService,
    SemanticRanker,
    build_semantic_document,
)
from dubizzle_cars.retrieval.google_embeddings import (
    GoogleEmbeddingConfig,
    GoogleEmbeddingProvider,
    build_document_title,
)


def response(vectors):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    client = Mock()
    client.models.embed_content.side_effect = lambda **kwargs: response(
        [
            [1.0] + [0.0] * (kwargs["config"].output_dimensionality - 1)
            for text in kwargs["contents"]
        ]
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr("dubizzle_cars.retrieval.google_embeddings.genai.Client", factory)
    return client, factory


def test_correct_sdk_configuration_order_batching_and_close(sdk):
    client, factory = sdk
    vectors = [[1, 0] + [0] * 766, [0, 1] + [0] * 766, [-1, 0] + [0] * 766]
    client.models.embed_content.side_effect = [
        response(vectors[:2]),
        response(vectors[2:]),
        response(vectors[:1]),
    ]
    config = GoogleEmbeddingConfig(batch_size=2)
    with GoogleEmbeddingProvider(config=config) as provider:
        assert factory.call_count == 0
        assert provider.embed_documents(["A", "B", "C"]) == tuple(tuple(v) for v in vectors)
        assert provider.embed_query("query") == tuple(vectors[0])
        assert factory.call_count == 1
    assert client.close.call_count == 1
    calls = client.models.embed_content.call_args_list
    assert [c.kwargs["contents"] for c in calls] == [["A", "B"], ["C"], ["query"]]
    assert all(c.kwargs["model"] == "gemini-embedding-001" for c in calls)
    assert all(isinstance(c.kwargs["config"], types.EmbedContentConfig) for c in calls)
    assert [c.kwargs["config"].task_type for c in calls] == [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
    ]
    assert all(c.kwargs["config"].output_dimensionality == 768 for c in calls)
    assert factory.call_args.kwargs["vertexai"] is False


def test_factual_native_title_when_shared_and_no_false_batch_title(sdk):
    client, _ = sdk
    car = Car("1", 2020, "ford", "focus", "other", description="Source description")
    assert build_document_title(car) == "2020 ford focus other"
    assert build_document_title(Car("2", make="ford")) == "ford"
    assert build_document_title(Car("3")) == ""
    provider = GoogleEmbeddingProvider([car])
    body = build_semantic_document(car)
    provider.embed_documents([body])
    assert client.models.embed_content.call_args.kwargs["config"].title == "2020 ford focus other"
    assert client.models.embed_content.call_args.kwargs["contents"] == [body]
    with pytest.raises(EmbeddingError, match="outside"):
        provider.embed_documents(["unrelated listing"])
    provider.embed_query("comfortable")
    assert client.models.embed_content.call_args.kwargs["config"].title is None
    other = Car("2", 2021, "toyota", "camry")
    mixed = GoogleEmbeddingProvider([car, other])
    mixed.embed_documents([body, build_semantic_document(other)])
    assert client.models.embed_content.call_args.kwargs["config"].title is None
    assert provider.provider_id != mixed.provider_id


def test_missing_credentials_are_lazy_and_structured_search_never_uses_client(sdk, monkeypatch):
    _, factory = sdk
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-be-used")
    provider = GoogleEmbeddingProvider()
    service = InventorySearchService(
        InMemoryInventory([Car("1")]), SemanticRanker(provider, provider.provider_id)
    )
    assert len(service.search(InventorySearchRequest())) == 1
    assert provider.embed_documents([]) == ()
    assert factory.call_count == 0
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        provider.embed_query("query")
    assert factory.call_count == 0


@pytest.mark.parametrize(
    "bad_response",
    [
        None,
        SimpleNamespace(embeddings=None),
        response([]),
        response([[]]),
        response([[0] * 768]),
        response([[1, 2]]),
        response([[float("nan")] * 768]),
    ],
)
def test_malformed_responses_fail_safely(sdk, bad_response):
    client, _ = sdk
    client.models.embed_content.side_effect = None
    client.models.embed_content.return_value = bad_response
    with pytest.raises(EmbeddingError, match="malformed"):
        GoogleEmbeddingProvider().embed_query("query")


@pytest.mark.parametrize("stage", ["init", "request", "response", "close"])
def test_sdk_failures_never_expose_secret_in_exception_trace_or_logs(sdk, caplog, stage):
    client, factory = sdk
    secret = "offline-test-key"
    provider = GoogleEmbeddingProvider()
    if stage == "init":
        factory.side_effect = RuntimeError(f"Authentication error with key={secret}")
    elif stage == "request":
        client.models.embed_content.side_effect = RuntimeError(
            f"Quota or network error with key={secret}"
        )
    elif stage == "response":
        client.models.embed_content.side_effect = None
        client.models.embed_content.return_value = response([[secret]])
    else:
        provider.embed_query("q")
        client.close.side_effect = RuntimeError(secret)
    with pytest.raises(EmbeddingError) as exc:
        if stage == "close":
            provider.close()
        else:
            provider.embed_query("q")
    assert secret not in "".join(traceback.format_exception(exc.value))
    assert secret not in caplog.text
    assert exc.value.__cause__ is None


def test_identity_and_existing_cache_invalidation(sdk, tmp_path):
    client, _ = sdk
    cars = [Car("1", title="A")]
    cache = EmbeddingCache(tmp_path / "cache.json")
    config = GoogleEmbeddingConfig()
    provider = GoogleEmbeddingProvider(config=config)
    assert (
        "google/gemini-embedding-001/dim-768/doc-RETRIEVAL_DOCUMENT/query-RETRIEVAL_QUERY"
        in provider.provider_id
    )
    SemanticRanker(provider, provider.provider_id, cache).rank(cars, "q")
    restarted = GoogleEmbeddingProvider(config=config)
    SemanticRanker(restarted, restarted.provider_id, cache).rank(cars, "q")

    def document_calls():
        return [
            c
            for c in client.models.embed_content.call_args_list
            if c.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
        ]

    assert len(document_calls()) == 1
    changed = GoogleEmbeddingProvider(config=replace(config, output_dimensionality=1536))
    assert changed.provider_id != provider.provider_id
    SemanticRanker(changed, changed.provider_id, cache).rank(cars, "q")
    assert len(document_calls()) == 2
    assert json.loads(cache.path.read_text())["dimension"] == 1536
    assert (
        GoogleEmbeddingProvider(config=replace(config, model="future-model")).provider_id
        != provider.provider_id
    )


def test_invalid_input_does_not_call_sdk(sdk):
    _, factory = sdk
    provider = GoogleEmbeddingProvider()
    for texts in ([""], [" \n"], "not-a-sequence-of-documents"):
        with pytest.raises(EmbeddingError):
            provider.embed_documents(texts)
    with pytest.raises(EmbeddingError):
        provider.embed_query(" ")
    assert factory.call_count == 0


def test_manual_smoke_utility_with_mocked_provider(sdk, tmp_path, capsys):
    from dubizzle_cars.smoke_retrieval import main

    assert main(["--cache", str(tmp_path / "cache.json")]) == 0
    output = capsys.readouterr().out
    assert "Loaded 100 real listings" in output
    assert output.count("hard filters and listing IDs verified") == 5
    assert "ranking values only" in output
    assert "offline-test-key" not in output
