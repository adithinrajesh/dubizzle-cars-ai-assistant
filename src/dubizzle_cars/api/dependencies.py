"""Per-application services and reusable FastAPI dependencies."""

from _thread import LockType
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from ..agent import AgentService
from ..agent.errors import AgentProviderError
from ..inventory import InventoryRepository
from ..retrieval import InventorySearchService


@dataclass(frozen=True, slots=True)
class ApiServices:
    repository: InventoryRepository
    search_service: InventorySearchService
    inventory_count: int
    semantic_available: bool
    semantic_lock: LockType = field(default_factory=Lock)
    agent_service: AgentService | None = None


def get_services(request: Request) -> ApiServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Inventory service is not initialized.")
    return services


def get_inventory_repository(
    services: Annotated[ApiServices, Depends(get_services)],
) -> InventoryRepository:
    return services.repository


def get_inventory_search_service(
    services: Annotated[ApiServices, Depends(get_services)],
) -> InventorySearchService:
    return services.search_service


def get_agent_service(services: Annotated[ApiServices, Depends(get_services)]) -> AgentService:
    if services.agent_service is None:
        raise AgentProviderError("Conversational assistant is not configured")
    return services.agent_service
