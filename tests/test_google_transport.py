"""Exercise the real SDK serializer against an in-memory HTTP transport, never Google."""

import json

import httpx
from google import genai

from dubizzle_cars.retrieval.google_embeddings import GoogleEmbeddingConfig, GoogleEmbeddingProvider


def test_sdk_batch_transport_is_offline_and_preserves_document_requests(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-transport-key")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path.endswith(":batchEmbedContents")
        return httpx.Response(
            200, json={"embeddings": [{"values": [1.0] + [0.0] * 767} for _ in payload["requests"]]}
        )

    actual_client = genai.Client

    def factory(**kwargs):
        kwargs["http_options"].client_args = {"transport": httpx.MockTransport(respond)}
        return actual_client(**kwargs)

    monkeypatch.setattr("dubizzle_cars.retrieval.google_embeddings.genai.Client", factory)
    with GoogleEmbeddingProvider(config=GoogleEmbeddingConfig(batch_size=2)) as provider:
        assert len(provider.embed_documents(["A", "B", "C"])) == 3
    assert [len(payload["requests"]) for payload in requests] == [2, 1]
    assert [
        item["content"]["parts"][0]["text"] for payload in requests for item in payload["requests"]
    ] == ["A", "B", "C"]
