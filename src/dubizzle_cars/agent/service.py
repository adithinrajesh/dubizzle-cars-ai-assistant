"""Bounded manual orchestration. All exchanges and evidence are local to one request."""

from .errors import (
    AgentError,
    AgentProviderError,
    GroundingError,
    ToolRoundLimitError,
    ToolValidationError,
)
from .models import AgentReply, AgentTurn, ChatProvider, Exchange, FinalAnswer
from .policy import controlled_reply, preflight_reply
from .tools import InventoryTools


class AgentService:
    def __init__(
        self,
        provider: ChatProvider,
        tools: InventoryTools,
        *,
        max_tool_rounds: int = 4,
        stateful: bool = False,
    ) -> None:
        if type(max_tool_rounds) is not int or not 1 <= max_tool_rounds <= 4:
            raise ValueError("max_tool_rounds must be between 1 and 4")
        self._provider = provider
        self._tools = tools
        self._max_tool_rounds = max_tool_rounds
        self._stateful = stateful

    def chat(self, message: str) -> AgentReply:
        if not isinstance(message, str) or not message.strip() or len(message) > 8000:
            raise AgentError("Message must contain 1 to 8000 characters")
        message = message.strip()
        blocked = preflight_reply(message, stateful=self._stateful)
        if blocked is not None:
            return AgentReply(blocked)
        exchanges: tuple[Exchange, ...] = ()
        evidence = {}
        tool_names: list[str] = []
        for round_number in range(self._max_tool_rounds + 1):
            try:
                turn = self._provider.next_turn(message, exchanges)
            except AgentError:
                raise
            except Exception:
                raise AgentProviderError("Conversational provider failed") from None
            if not isinstance(turn, AgentTurn) or bool(turn.calls) == (turn.final is not None):
                raise AgentProviderError("Invalid conversational response")
            if turn.calls:
                if round_number == self._max_tool_rounds or len(turn.calls) > 4:
                    raise ToolRoundLimitError("Conversation tool budget exceeded")
                if (
                    self._stateful
                    and len(turn.calls) > 1
                    and any(call.name in {"book_test_drive", "save_lead"} for call in turn.calls)
                ):
                    raise ToolValidationError("Submit one persistence action at a time")
                for call in turn.calls:
                    self._tools.validate(call)  # Validate the entire batch before executing any.
                results = tuple(self._tools.execute(call) for call in turn.calls)
                for call, result in zip(turn.calls, results, strict=True):
                    tool_names.append(call.name)
                    for car in result.cars:
                        evidence[car.listing_id] = car
                if self._stateful:
                    confirmations = [
                        r.data["application_reply"]
                        for r in results
                        if "application_reply" in r.data
                    ]
                    if confirmations:
                        return AgentReply("\n".join(confirmations), (), tuple(tool_names))
                exchanges += (Exchange(turn, results),)
                continue
            final = turn.final
            if not isinstance(final, FinalAnswer):
                raise AgentProviderError("Invalid final response")
            if not final.reply.strip():
                raise AgentProviderError("Empty final response")
            if final.kind == "inventory" and not tool_names:
                raise GroundingError("Inventory answers require tool evidence")
            if final.listing_ids and final.kind != "inventory":
                raise GroundingError("Car references require an inventory answer")
            chosen_ids = tuple(dict.fromkeys(final.listing_ids))
            if any(key not in evidence for key in chosen_ids):
                raise GroundingError("Final car references lack tool evidence")
            cars = tuple(evidence[key] for key in chosen_ids)
            try:
                if any(
                    self._tools.repository.get_car_details(car.listing_id) != car for car in cars
                ):
                    raise ValueError("Changed inventory")
            except Exception:
                raise GroundingError("Inventory evidence is no longer valid") from None
            reply = controlled_reply(final.kind, final.reply.strip())
            # A policy redirect cannot carry inventory cards from a disallowed answer.
            if reply != final.reply.strip():
                cars = ()
            return AgentReply(reply, cars, tuple(tool_names))
        raise ToolRoundLimitError("Conversation tool budget exceeded")
