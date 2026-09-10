"""Immutable domain records, independent of workbook and future application adapters."""

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Metadata:
    sale_price_aed: Decimal | None = None
    mileage_km: int | None = None
    # Literal mention only; never a claim of current warranty coverage.
    warranty_mention: str | None = None
    regional_specs: str | None = None


@dataclass(frozen=True, slots=True)
class Car:
    listing_id: str
    year: int | None = None
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    title: str | None = None
    description: str | None = None
    photo_url: str | None = None
    metadata: Metadata = field(default_factory=Metadata)
    source_row: int | None = None
