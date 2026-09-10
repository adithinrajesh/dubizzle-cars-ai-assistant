"""Two validated local tools; no dynamic imports, eval, or model-selected Python access."""

import re
from _thread import LockType
from collections.abc import Callable, Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..inventory import InventoryFilter, InventoryRepository
from ..models import Car
from ..retrieval import InventorySearchRequest, InventorySearchService
from .errors import ToolExecutionError, ToolValidationError
from .models import ToolCall, ToolResult
from .prompts import DETAILS_DESCRIPTION, SEARCH_DESCRIPTION


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    make: str | None = Field(
        default=None,
        description="Explicit make only, e.g. bmw or mercedes-benz. Never infer from soft preferences.",
    )
    model: str | None = Field(
        default=None, description="Explicit source model, exact case-insensitive match."
    )
    trim: str | None = Field(default=None, description="Explicit trim, not inferred equipment.")
    regional_specs: str | None = Field(
        default=None,
        description="Explicit regional specification, e.g. GCC. Unknown does not match.",
    )
    year_min: int | None = Field(default=None, description="Inclusive explicit minimum model year.")
    year_max: int | None = Field(default=None, description="Inclusive explicit maximum model year.")
    price_min: StrictInt | StrictStr | None = Field(
        default=None,
        description="Inclusive AED minimum, integer or plain decimal string; no currency conversion.",
    )
    price_max: StrictInt | StrictStr | None = Field(
        default=None,
        description="Inclusive AED budget, integer or plain decimal string. 100k = 100000. Unknown price excluded.",
    )
    mileage_min: int | None = Field(
        default=None, description="Inclusive explicit kilometre minimum."
    )
    mileage_max: int | None = Field(
        default=None, description="Inclusive explicit kilometre maximum. Unknown mileage excluded."
    )
    has_warranty_mention: bool | None = Field(
        default=None,
        description="Presence of a literal warranty mention, including negatives. NEVER active-warranty coverage.",
    )
    semantic_query: str | None = Field(
        default=None,
        description="Only soft preferences: sporty, luxury, comfort, equipment, family road trips. Do not invent hard constraints.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Final result limit after ranking, default 5 and maximum 20 per agent tool call.",
    )

    @field_validator("price_min", "price_max")
    @classmethod
    def price_text(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, str) and not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            raise ValueError("Price must be an integer or plain decimal string")
        return value

    def to_request(self) -> InventorySearchRequest:
        fields = self.model_dump(exclude={"semantic_query", "limit"})
        for name in ("price_min", "price_max"):
            if isinstance(fields[name], str):
                fields[name] = Decimal(fields[name])
        return InventorySearchRequest(InventoryFilter(**fields), self.semantic_query, self.limit)

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        self.to_request()
        return self


class DetailsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    listing_id: str = Field(
        min_length=1,
        description="Exact stable listing ID from the user or a tool result in this request.",
    )

    @field_validator("listing_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Listing ID cannot be blank")
        return value


def serialize_car(car: Car, semantic_score: float | None = None) -> dict[str, object]:
    return {
        "listing_id": car.listing_id,
        "year": car.year,
        "make": car.make,
        "model": car.model,
        "trim": car.trim,
        "title": car.title,
        "description": car.description,
        "photo_url": car.photo_url,
        "sale_price_aed": None
        if car.metadata.sale_price_aed is None
        else format(car.metadata.sale_price_aed, "f"),
        "mileage_km": car.metadata.mileage_km,
        "warranty_mention": car.metadata.warranty_mention,
        "regional_specs": car.metadata.regional_specs,
        "semantic_score": semantic_score,
    }


def tool_declarations() -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "search_inventory",
            "description": SEARCH_DESCRIPTION,
            "parameters_json_schema": SearchArguments.model_json_schema(),
        },
        {
            "name": "get_car_details",
            "description": DETAILS_DESCRIPTION,
            "parameters_json_schema": DetailsArguments.model_json_schema(),
        },
    )


class InventoryTools:
    def __init__(
        self,
        repository: InventoryRepository,
        search: InventorySearchService,
        *,
        semantic_available: bool = True,
        semantic_lock: LockType | None = None,
    ) -> None:
        self.repository = repository
        self._search = search
        self._semantic_available = semantic_available
        self._semantic_lock = semantic_lock
        self._registry: Mapping[str, tuple[type[BaseModel], Callable]] = MappingProxyType(
            {
                "search_inventory": (SearchArguments, self._search_inventory),
                "get_car_details": (DetailsArguments, self._get_car_details),
            }
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._registry)

    def validate(self, call: ToolCall) -> BaseModel:
        try:
            if call.name not in self._registry or not isinstance(call.arguments, Mapping):
                raise ValueError("Unknown tool or invalid argument object")
            return self._registry[call.name][0].model_validate(dict(call.arguments))
        except Exception:
            raise ToolValidationError("Unregistered tool or invalid tool arguments") from None

    def execute(self, call: ToolCall) -> ToolResult:
        arguments = self.validate(call)
        try:
            result = self._registry[call.name][1](arguments)
            if any(self.repository.get_car_details(car.listing_id) != car for car in result.cars):
                raise ValueError("Tool returned a car outside the repository")
            return result
        except Exception:
            import traceback
            print("=== DEBUG: ToolExecutionError root cause ===")
            traceback.print_exc()
            print("=== END DEBUG ===")
            raise ToolExecutionError("Inventory tool execution failed") from None

    def _search_inventory(self, arguments: SearchArguments) -> ToolResult:
        request = arguments.to_request()
        if request.semantic_query is not None and not self._semantic_available:
            raise ToolExecutionError("Semantic search is not configured")
        if request.semantic_query is not None and self._semantic_lock is not None:
            with self._semantic_lock:
                results = self._search.search(request)
        else:
            results = self._search.search(request)
        return ToolResult(
            {
                "status": "ok",
                "count": len(results),
                "cars": [serialize_car(r.car, r.semantic_score) for r in results],
            },
            tuple(r.car for r in results),
        )

    def _get_car_details(self, arguments: DetailsArguments) -> ToolResult:
        car = self.repository.get_car_details(arguments.listing_id)
        if car is None:
            return ToolResult(
                {"status": "not_found", "listing_id": arguments.listing_id, "car": None}
            )
        return ToolResult({"status": "ok", "car": serialize_car(car)}, (car,))
