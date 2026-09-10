"""Provider-neutral per-request messages; no conversation state lives on a provider."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..models import Car


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal[
        "inventory",
        "automotive",
        "chitchat",
        "clarification",
        "out_of_scope",
        "competitor",
        "booking_unavailable",
        "currency_unsupported",
    ]
    reply: str = Field(min_length=1, max_length=12000)
    listing_ids: list[str] = Field(
        default_factory=list,
        max_length=80,
        description="Only IDs returned by successful inventory tools in this request and discussed in the reply.",
    )


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, object]
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentTurn:
    calls: tuple[ToolCall, ...] = ()
    final: FinalAnswer | None = None
    # Opaque provider content preserves signed parts; never exposed over HTTP or logged.
    provider_content: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ToolResult:
    data: Mapping[str, object]
    cars: tuple[Car, ...] = ()


@dataclass(frozen=True, slots=True)
class Exchange:
    turn: AgentTurn
    results: tuple[ToolResult, ...]


class ChatProvider(Protocol):
    def next_turn(self, message: str, exchanges: tuple[Exchange, ...]) -> AgentTurn: ...


@dataclass(frozen=True, slots=True)
class AgentReply:
    reply: str
    cars: tuple[Car, ...] = ()
    # Internal audit for offline/manual verification; not part of POST /chat.
    tool_names: tuple[str, ...] = ()
