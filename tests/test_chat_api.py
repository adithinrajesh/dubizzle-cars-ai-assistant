from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from dubizzle_cars import Car, InMemoryInventory
from dubizzle_cars.agent import AgentReply
from dubizzle_cars.agent.errors import AgentProviderError
from dubizzle_cars.agent.models import AgentTurn, FinalAnswer, ToolCall
from dubizzle_cars.api.app import create_app
from dubizzle_cars.api.dependencies import get_agent_service
from dubizzle_cars.api.settings import ApiSettings


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DUBIZZLE_SEMANTIC_ENABLED", "false")


class DetailsProvider:
    def __init__(self):
        self.histories = []

    def next_turn(self, message, exchanges):
        self.histories.append(exchanges)
        if not exchanges:
            return AgentTurn(calls=(ToolCall("get_car_details", {"listing_id": "1"}),))
        return AgentTurn(
            final=FinalAnswer(
                kind="inventory",
                reply="The listing does not specify price or warranty information.",
                listing_ids=["1"],
            )
        )


def test_chat_returns_source_cards_without_tool_internals_or_memory():
    car = Car("1", make="ford", photo_url="https://example.com/source.jpg")
    provider = DetailsProvider()
    app = create_app(repository=InMemoryInventory([car]), chat_provider=provider)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Does listing 1 have warranty?"})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"reply", "cars"}
        assert body["cars"][0]["listing_id"] == "1"
        assert body["cars"][0]["metadata"]["sale_price_aed"] is None
        assert body["cars"][0]["metadata"]["warranty_mention"] is None
        assert body["cars"][0]["photo_url"] == car.photo_url
        assert "semantic_score" not in body["cars"][0]
        assert (
            client.post("/chat", json={"message": "Tell me about the second one"}).json()["cars"]
            == []
        )
        assert client.post("/chat", json={"message": "Details of listing 1"}).status_code == 200
        assert provider.histories[2] == ()
        assert client.get("/inventory/1").status_code == 200
        assert client.post("/inventory/search", json={}).status_code == 200


def test_hybrid_chat_uses_existing_search_and_hard_filters(tmp_path):
    repo = InMemoryInventory(
        [
            Car("1", 2019, "mercedes-benz", title="Older car"),
            Car("2", 2021, "mercedes-benz", title="Actual match"),
            Car("3", 2024, "bmw", title="Other brand"),
        ]
    )
    provider = Mock()
    provider.next_turn.side_effect = [
        AgentTurn(
            calls=(
                ToolCall(
                    "search_inventory",
                    {
                        "make": "mercedes-benz",
                        "year_min": 2020,
                        "semantic_query": "luxurious",
                    },
                ),
            )
        ),
        AgentTurn(
            final=FinalAnswer(
                kind="inventory", reply="Listing 2 matches the search.", listing_ids=["2"]
            )
        ),
    ]
    embeddings = Mock()
    embeddings.embed_query.return_value = (1, 0)
    embeddings.embed_documents.side_effect = lambda texts: [(1, 0) for _ in texts]
    app = create_app(
        ApiSettings(semantic_enabled=True, cache_path=tmp_path / "cache.json"),
        repository=repo,
        chat_provider=provider,
        embedding_provider=embeddings,
        provider_id="fake",
    )
    with TestClient(app) as client:
        result = client.post("/chat", json={"message": "Luxurious Mercedes from 2020 onwards"})
        assert result.status_code == 200
        assert [car["listing_id"] for car in result.json()["cars"]] == ["2"]
        embeddings.embed_query.assert_called_once_with("luxurious")
        documents = embeddings.embed_documents.call_args.args[0]
        assert len(documents) == 1
        assert "Actual match" in documents[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": " \n"},
        {"message": 4},
        {"message": "a" * 8001},
        {"message": "Hi", "session_id": "not-supported"},
    ],
)
def test_chat_validation(payload):
    agent = Mock()
    with TestClient(create_app(repository=InMemoryInventory([]), agent_service=agent)) as client:
        assert client.post("/chat", json=payload).status_code == 422
        agent.chat.assert_not_called()


def test_missing_key_and_mocked_failure_have_safe_503(caplog):
    app = create_app(repository=InMemoryInventory([]))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 503
        assert response.json() == {
            "detail": "The conversational assistant is currently unavailable."
        }
        agent = Mock()
        agent.chat.side_effect = AgentProviderError("private-test-payload")
        app.dependency_overrides[get_agent_service] = lambda: agent
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 503
        assert "private-test-payload" not in response.text + caplog.text


def test_default_chat_model(monkeypatch):
    monkeypatch.delenv("DUBIZZLE_CHAT_MODEL", raising=False)
    assert ApiSettings().chat_model == "gemini-3.8-flash"
    assert ApiSettings.from_environment().chat_model == "gemini-3.8-flash"


def test_agent_injection_model_configuration_and_owned_cleanup(monkeypatch):
    monkeypatch.setenv("DUBIZZLE_CHAT_MODEL", "configured-chat-model")
    provider = Mock()
    factory = Mock(return_value=provider)
    monkeypatch.setattr("dubizzle_cars.api.app.GoogleChatProvider", factory)
    settings = ApiSettings.from_environment()
    assert settings.chat_model == "configured-chat-model"
    with TestClient(create_app(settings, repository=InMemoryInventory([]))) as client:
        factory.assert_called_once_with("configured-chat-model")
        client.get("/health")
        provider.next_turn.assert_not_called()
    provider.close.assert_called_once()
    agent = Mock()
    agent.chat.return_value = AgentReply("Hello!")
    with TestClient(create_app(repository=InMemoryInventory([]), agent_service=agent)) as client:
        assert client.post("/chat", json={"message": "Hello"}).json() == {
            "reply": "Hello!",
            "cars": [],
        }
    assert factory.call_count == 1


def test_chat_keeps_supplied_workbook_unchanged():
    path = Path(__file__).resolve().parents[1] / "data/sample_cars_dataset.xlsx"
    before = path.read_bytes()
    with TestClient(
        create_app(
            ApiSettings(workbook_path=path, semantic_enabled=False), chat_provider=DetailsProvider()
        )
    ) as client:
        assert client.get("/health").json()["inventory_count"] == 100
        assert client.post("/chat", json={"message": "Details of listing 1"}).status_code == 200
        assert "/chat" in client.get("/openapi.json").json()["paths"]
    assert path.read_bytes() == before
