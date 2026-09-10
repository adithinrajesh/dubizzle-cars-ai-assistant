"""Structured AND filtering with stable ID ordering and explicit unknown handling."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Protocol

from .models import Car


@dataclass(frozen=True, slots=True)
class InventoryFilter:
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    regional_specs: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    price_min: Decimal | None = None
    price_max: Decimal | None = None
    mileage_min: int | None = None
    mileage_max: int | None = None
    has_warranty_mention: bool | None = None

    def __post_init__(self):
        for name in ("make", "model", "trim", "regional_specs"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("year", "price", "mileage"):
            low, high = getattr(self, name + "_min"), getattr(self, name + "_max")
            for value in (low, high):
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
                    raise ValueError(f"{name} bounds must be integers or Decimal amounts")
                if not Decimal(value).is_finite() or value < 0:
                    raise ValueError(f"{name} bounds must be finite and nonnegative")
                if name != "price" and not isinstance(value, int):
                    raise ValueError(f"{name} bounds must be integers")
            if low is not None and high is not None and low > high:
                raise ValueError(f"{name}_min cannot exceed {name}_max")
        if self.has_warranty_mention is not None and type(self.has_warranty_mention) is not bool:
            raise ValueError("has_warranty_mention must be bool or None")


class InventoryRepository(Protocol):
    def filter_inventory(self, filters: InventoryFilter | None = None) -> tuple[Car, ...]: ...
    def get_car_details(self, listing_id: str) -> Car | None: ...


def _matches(car: Car, filters: InventoryFilter) -> bool:
    for name in ("make", "model", "trim", "regional_specs"):
        wanted = getattr(filters, name)
        actual = getattr(car.metadata if name == "regional_specs" else car, name)
        if wanted is not None and (
            actual is None or actual.strip().casefold() != wanted.strip().casefold()
        ):
            return False
    for name, value in (
        ("year", car.year),
        ("price", car.metadata.sale_price_aed),
        ("mileage", car.metadata.mileage_km),
    ):
        low, high = getattr(filters, name + "_min"), getattr(filters, name + "_max")
        if (low is not None or high is not None) and value is None:
            return False
        if low is not None and value < low:
            return False
        if high is not None and value > high:
            return False
    mention = filters.has_warranty_mention
    return mention is None or (car.metadata.warranty_mention is not None) == mention


class InMemoryInventory:
    def __init__(self, cars: Iterable[Car]):
        self._cars = tuple(sorted(cars, key=lambda car: car.listing_id))
        self._by_id = {car.listing_id: car for car in self._cars}
        if len(self._by_id) != len(self._cars):
            raise ValueError("Duplicate listing IDs")

    def filter_inventory(self, filters: InventoryFilter | None = None) -> tuple[Car, ...]:
        query = filters if filters is not None else InventoryFilter()
        return tuple(car for car in self._cars if _matches(car, query))

    def get_car_details(self, listing_id: str) -> Car | None:
        return self._by_id.get(listing_id)
