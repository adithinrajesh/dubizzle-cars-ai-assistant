"""Optional stateful chat transport; services own policy, persistence and tools."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import Field, field_validator, model_validator

from ..agent import AgentService
from .dependencies import get_agent_service
from .schemas import ApiModel, CarResponse
from .state import get_stateful


class ChatRequest(ApiModel):
    user_id: str | None = Field(
        default=None, min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$"
    )
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    message: str = Field(
        min_length=1,
        max_length=8000,
        description="User message; supply user_id for persistent sessions.",
    )

    @field_validator("message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def identity_required(self):
        if self.session_id is not None and self.user_id is None:
            raise ValueError("A session requires user_id")
        return self


class ChatResponse(ApiModel):
    session_id: str | None = None
    reply: str
    cars: tuple[CarResponse, ...] = Field(
        description="Only real tool-returned cars discussed in the reply. Facts retain the inventory API schema; no model-generated car objects."
    )


def create_chat_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "/chat",
        response_model=ChatResponse,
        response_model_exclude_unset=True,
        tags=["chat"],
        responses={503: {"description": "Conversational assistant unavailable"}},
    )
    def chat(
        body: ChatRequest,
        request: Request,
        agent: Annotated[AgentService, Depends(get_agent_service)],
    ) -> ChatResponse:
        """Handle a stateless message or continue an explicitly identified conversation."""
        if body.user_id is not None:
            result, session_id = get_stateful(request).chat(
                body.message, body.user_id, body.session_id
            )
            return ChatResponse(
                reply=result.reply,
                cars=tuple(CarResponse.model_validate(car) for car in result.cars),
                session_id=session_id,
            )
        result = agent.chat(body.message)
        return ChatResponse(
            reply=result.reply, cars=tuple(CarResponse.model_validate(car) for car in result.cars)
        )

    return router
