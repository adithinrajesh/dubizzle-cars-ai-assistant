"""Offline diagnostics checks with all external providers replaced."""

from unittest.mock import MagicMock, Mock

import pytest

from dubizzle_cars import smoke_agent
from dubizzle_cars.agent.errors import AgentProviderError
from dubizzle_cars.retrieval import RetrievalError


@pytest.mark.parametrize(
    "error",
    [
        AgentProviderError("Malformed or incomplete conversational response"),
        RetrievalError("Semantic retrieval failed"),
    ],
)
@pytest.mark.parametrize("model", [None, "configured-chat-model"])
def test_controlled_failure_prints_only_safe_type_and_message(monkeypatch, capsys, error, model):
    if model is None:
        monkeypatch.delenv("DUBIZZLE_CHAT_MODEL", raising=False)
    else:
        monkeypatch.setenv("DUBIZZLE_CHAT_MODEL", model)
    error.__cause__ = RuntimeError("private-provider-payload")
    monkeypatch.setattr(smoke_agent, "load_inventory", Mock(return_value=[]))
    for name in ("GoogleEmbeddingProvider", "GoogleChatProvider", "SemanticRanker"):
        monkeypatch.setattr(smoke_agent, name, MagicMock())
    agent = Mock()
    agent.chat.side_effect = error
    monkeypatch.setattr(smoke_agent, "AgentService", Mock(return_value=agent))

    assert smoke_agent.main([]) == 1
    smoke_agent.GoogleChatProvider.assert_called_once_with(model or "gemini-3.8-flash")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"Agent smoke check failed: {type(error).__name__}: {error}\n"
    assert "private-provider-payload" not in captured.err
