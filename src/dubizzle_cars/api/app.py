"""Application factory with one inventory load per lifespan and no import-time I/O."""

import os
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..actions import LeadService
from ..agent import AgentService, ChatProvider
from ..agent.openai_agent import OpenAIChatProvider
from ..agent.tools import InventoryTools
from ..inventory import InMemoryInventory, InventoryRepository
from ..loader import load_inventory
from ..memory.repository import SQLiteState, StateError
from ..memory.service import StatefulChat
from ..retrieval import EmbeddingCache, EmbeddingProvider, InventorySearchService, SemanticRanker
from ..retrieval.google_embeddings import GoogleEmbeddingProvider
from .chat import create_chat_router
from .dependencies import ApiServices
from .errors import register_error_handlers
from .routes import create_router
from .settings import ApiSettings
from .state import create_state_router


def create_app(
    settings: ApiSettings | None = None,
    *,
    repository: InventoryRepository | None = None,
    search_service: InventorySearchService | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    provider_id: str | None = None,
    semantic_available: bool = False,
    agent_service: AgentService | None = None,
    chat_provider: ChatProvider | None = None,
) -> FastAPI:
    """Injected services/providers are caller-owned; factory-created Google clients are closed."""
    config = settings if settings is not None else ApiSettings.from_environment()
    if agent_service is not None and chat_provider is not None:
        raise ValueError("Inject an agent service or chat provider, not both")
    if embedding_provider is not None and (
        not isinstance(provider_id, str) or not provider_id.strip()
    ):
        raise ValueError("An injected embedding provider requires its provider_id")
    if search_service is not None and embedding_provider is not None:
        raise ValueError("Inject a search service or embedding provider, not both")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned_provider: GoogleEmbeddingProvider | None = None
        owned_chat: OpenAIChatProvider | None = None
        try:
            try:
                repo = (
                    repository
                    if repository is not None
                    else InMemoryInventory(load_inventory(config.workbook_path))
                )
                cars = repo.filter_inventory()
            except Exception:
                raise RuntimeError(
                    "Inventory initialization failed; check the configured workbook or repository."
                ) from None
            available = False
            if search_service is not None:
                service = search_service
                available = config.semantic_enabled and semantic_available
            else:
                ranker = None
                if config.semantic_enabled:
                    provider = embedding_provider
                    identity = provider_id
                    # Read only readiness here. The adapter owns credential use; never print it.
                    if provider is None and bool(os.environ.get("GEMINI_API_KEY", "").strip()):
                        owned_provider = GoogleEmbeddingProvider(cars)
                        provider = owned_provider
                        identity = owned_provider.provider_id
                    if provider is not None:
                        assert identity is not None
                        ranker = SemanticRanker(
                            provider, identity, EmbeddingCache(config.cache_path)
                        )
                        available = True
                service = InventorySearchService(repo, ranker)
            semantic_lock = Lock()
            conversation_provider = chat_provider
            agent = agent_service
            if agent is None:
                conversation_provider = chat_provider
                if conversation_provider is None:
                    owned_chat = OpenAIChatProvider(config.chat_model)
                    conversation_provider = owned_chat
                agent = AgentService(
                    conversation_provider,
                    InventoryTools(
                        repo,
                        service,
                        semantic_available=available,
                        semantic_lock=semantic_lock,
                    ),
                )
            app.state.services = ApiServices(
                repo, service, len(cars), available, semantic_lock, agent
            )
            stateful_lock = Lock()
            stateful = None

            def get_stateful():
                nonlocal stateful
                with stateful_lock:
                    if stateful is None:
                        if conversation_provider is None:
                            raise StateError("Stateful chat requires a conversational provider")
                        stateful = StatefulChat(
                            SQLiteState(config.state_db_path),
                            LeadService(config.leads_csv_path),
                            repo,
                            service,
                            conversation_provider,
                            semantic_available=available,
                            semantic_lock=semantic_lock,
                        )
                    return stateful

            app.state.get_stateful = get_stateful
            yield
        finally:
            if hasattr(app.state, "services"):
                del app.state.services
            if hasattr(app.state, "get_stateful"):
                del app.state.get_stateful
            try:
                if owned_chat is not None:
                    owned_chat.close()
            finally:
                if owned_provider is not None:
                    owned_provider.close()

    app = FastAPI(
        title="dubizzle Cars Inventory API",
        version="0.6.0",
        lifespan=lifespan,
        description="Inventory, optional persistent chat, and simulated local bookings and enquiries.",
    )
    register_error_handlers(app)
    app.include_router(create_router())
    app.include_router(create_chat_router())
    app.include_router(create_state_router())

    @app.exception_handler(StateError)
    async def state_error(request: Request, exc: StateError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
