"""Centralized safe HTTP errors; never return or log provider exception text."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from ..agent.errors import AgentError
from ..retrieval.errors import RetrievalError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AgentError)
    async def agent_error(request: Request, exc: AgentError) -> JSONResponse:
        import traceback
        print(f"=== DEBUG: AgentError handler caught {type(exc).__name__}: {exc} ===")
        traceback.print_exc()
        print("=== END DEBUG ===")
        return JSONResponse(
            status_code=503,
            content={"detail": "The conversational assistant is currently unavailable."},
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error(request: Request, exc: RetrievalError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": "Semantic retrieval is currently unavailable."}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo rejected input, exception context, or user-controlled field names.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "loc": ["body"],
                        "msg": "Invalid request value or filter bounds.",
                        "type": "request_validation",
                    }
                ]
            },
        )

    @app.middleware("http")
    async def safe_internal_error(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception:
            import traceback
            print("=== DEBUG: unhandled exception in middleware ===")
            traceback.print_exc()
            print("=== END DEBUG ===")
            return JSONResponse(status_code=500, content={"detail": "Internal server error."})