"""State endpoints delegate all SQL and business rules to the memory service."""

from fastapi import APIRouter, Request
from pydantic import Field

from .schemas import ApiModel


class SessionRequest(ApiModel):
    user_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    display_name: str | None = Field(default=None, min_length=1, max_length=100)


def get_stateful(request: Request):
    return request.app.state.get_stateful()


def create_state_router():
    router = APIRouter(tags=["state"])

    @router.post("/sessions")
    def create_session(body: SessionRequest, request: Request):
        return get_stateful(request).state.create_session(body.user_id, body.display_name)

    @router.get("/users/{user_id}/profile")
    def profile(user_id: str, request: Request):
        return get_stateful(request).state.profile(user_id)

    return router
