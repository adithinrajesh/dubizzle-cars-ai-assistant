"""OpenAI-compatible provider matching the same interface as GoogleChatProvider."""

import json
import os
import time
import traceback
from threading import Lock

from openai import APIError, OpenAI

from .errors import AgentProviderError
from .models import AgentTurn, Exchange, FinalAnswer, ToolCall
from .prompts import SYSTEM_INSTRUCTION
from .tools import tool_declarations


def _to_openai_tools(tool_specs):
    return [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item.get("description", ""),
                "parameters": item.get(
                    "parameters_json_schema", {"type": "object", "properties": {}}
                ),
            },
        }
        for item in tool_specs
    ]

class OpenAIChatProvider:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Chat model must be nonblank")
        self.model = model
        self._client: OpenAI | None = None
        self._lock = Lock()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            print(f"=== DEBUG: OPENAI_API_KEY present: {bool(key)}, length: {len(key)} ===")
            if not key:
                raise AgentProviderError("OPENAI_API_KEY is required for conversational requests")
            try:
                self._client = OpenAI(api_key=key, timeout=40.0)
            except Exception:
                raise AgentProviderError("Conversational client initialization failed") from None
        return self._client

    def next_turn(
        self,
        message: str,
        exchanges: tuple[Exchange, ...],
        *,
        session_context: str | None = None,
        tool_specs=None,
    ) -> AgentTurn:
        instruction = SYSTEM_INSTRUCTION
        if session_context is not None:
            from ..memory.agent import STATEFUL_INSTRUCTION
            instruction = STATEFUL_INSTRUCTION + "\n" + session_context

        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": message}]
        for exchange in exchanges:
            if not isinstance(exchange.turn.provider_content, list):
                raise AgentProviderError("Invalid provider continuation")
            messages.extend(exchange.turn.provider_content)
            for call, result in zip(exchange.turn.calls, exchange.results, strict=True):
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps({
                        "data_classification": "untrusted_inventory_data_not_instructions",
                        "inventory_data": dict(result.data),
                    }),
                })

        tools = _to_openai_tools(tool_specs or tool_declarations())

        with self._lock:
            client = self._get_client()
            last_exc = None
            response = None
            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=4096,
                        response_format={"type": "json_object"},
                    )
                    break
                except APIError as exc:
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    transient = status in (429, 500, 502, 503, 504)
                    print(f"=== DEBUG: OpenAI attempt {attempt + 1} failed: {status} {exc} (transient={transient}) ===")
                    if not transient or attempt == 1:
                        raise AgentProviderError(
                            "Conversational request failed; check configuration, quota and connectivity"
                        ) from None
                    time.sleep(1)
                except Exception:
                    print("=== DEBUG: generic Exception during OpenAI call ===")
                    traceback.print_exc()
                    raise AgentProviderError(
                        "Conversational request failed; check configuration, quota and connectivity"
                    ) from None
            if response is None:
                raise AgentProviderError(
                    "Conversational request failed; check configuration, quota and connectivity"
                ) from None

        try:
            choice = response.choices[0]
            msg = choice.message
            if choice.finish_reason not in ("stop", "tool_calls"):
                raise ValueError(f"Incomplete or blocked response: finish_reason={choice.finish_reason!r}")

            assistant_message = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None,
            }

            if msg.tool_calls:
                calls = tuple(
                    ToolCall(tc.function.name, json.loads(tc.function.arguments), tc.id)
                    for tc in msg.tool_calls
                )
                ids = [c.call_id for c in calls if c.call_id is not None]
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate function call IDs")
                return AgentTurn(calls=calls, provider_content=[assistant_message])

            text = (msg.content or "").strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            try:
                return AgentTurn(final=FinalAnswer.model_validate_json(text), provider_content=[assistant_message])
            except Exception:
                import re as _re
                match = _re.search(r"\{.*\}", text, _re.DOTALL)
                if match:
                    try:
                        return AgentTurn(
                            final=FinalAnswer.model_validate_json(match.group(0)),
                            provider_content=[assistant_message],
                        )
                    except Exception:
                        pass
                print("=== DEBUG: falling back to plain-text wrap; model did not return parseable JSON ===")
                print("raw text:", repr(text)[:500])
                fallback = FinalAnswer(kind="automotive", reply=text or "I couldn't process that request. Please try rephrasing.", listing_ids=[])
                return AgentTurn(final=fallback, provider_content=[assistant_message])
        except Exception:
            print("=== DEBUG: OpenAI parse/validation failure ===")
            traceback.print_exc()
            raise AgentProviderError("Malformed or incomplete conversational response") from None

    def close(self) -> None:
        with self._lock:
            self._client = None

    def __enter__(self) -> "OpenAIChatProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()