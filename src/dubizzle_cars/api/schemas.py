"""Explicit HTTP contracts; domain models remain the authority for search validation."""

import re
from decimal import Decimal
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from ..inventory import InventoryFilter
from ..retrieval.models import InventorySearchRequest


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, from_attributes=True)


class FilterRequest(ApiModel):
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    regional_specs: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    price_min: StrictInt | StrictStr | None = Field(
        default=None,
        description="Inclusive AED bound: integer or exact decimal string; floats are rejected.",
    )
    price_max: StrictInt | StrictStr | None = Field(
        default=None,
        description="Inclusive AED bound: integer or exact decimal string; unknown prices never match.",
    )
    mileage_min: int | None = None
    mileage_max: int | None = None
    has_warranty_mention: bool | None = Field(
        default=None,
        description="Literal mention presence, including expired/negative statements. Not warranty coverage.",
    )

    @field_validator("price_min", "price_max")
    @classmethod
    def validate_price_text(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, str) and not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            raise ValueError("Price must be an integer or plain decimal string")
        return value

    def to_domain(self) -> InventoryFilter:
        fields = self.model_dump()
        for name in ("price_min", "price_max"):
            if isinstance(fields[name], str):
                fields[name] = Decimal(fields[name])
        return InventoryFilter(**fields)

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        self.to_domain()
        return self


class SearchRequest(ApiModel):
    filters: FilterRequest = Field(default_factory=FilterRequest)
    semantic_query: str | None = Field(
        default=None,
        description="Optional soft preference. Blank means structured-only; never parses hard filters.",
    )
    limit: int | None = Field(
        default=None,
        description="Positive integer; applied after ranking. Null returns all matches.",
    )

    def to_domain(self) -> InventorySearchRequest:
        return InventorySearchRequest(self.filters.to_domain(), self.semantic_query, self.limit)

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        self.to_domain()
        return self


class MetadataResponse(ApiModel):
    sale_price_aed: Decimal | None = Field(
        description="Exact AED decimal string in JSON, or null when unknown."
    )
    mileage_km: int | None
    warranty_mention: str | None = Field(
        description="Source mention only; does not confirm active warranty."
    )
    regional_specs: str | None

    @field_serializer("sale_price_aed")
    def serialize_price(self, value: Decimal | None) -> str | None:
        return None if value is None else format(value, "f")


class CarResponse(ApiModel):
    listing_id: str
    year: int | None
    make: str | None
    model: str | None
    trim: str | None
    title: str | None
    description: str | None
    photo_url: str | None
    metadata: MetadataResponse


class SearchResultResponse(CarResponse):
    semantic_score: float | None = Field(
        description="Ranking metadata only, not confidence, probability, or vehicle quality."
    )


class SearchResponse(ApiModel):
    count: int = Field(description="Number of returned results, after the optional limit.")
    results: tuple[SearchResultResponse, ...]


class HealthResponse(ApiModel):
    status: str = "ok"
    inventory_count: int
    semantic_retrieval_available: bool = Field(
        description="Local configuration readiness only; no Google API request is made."
    )
