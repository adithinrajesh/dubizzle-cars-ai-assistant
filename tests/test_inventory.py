from decimal import Decimal

import pytest

from dubizzle_cars import Car, InMemoryInventory, InventoryFilter, Metadata


@pytest.fixture
def inventory():
    return InMemoryInventory(
        [
            Car(
                "2",
                2020,
                "Ford",
                "Explorer",
                metadata=Metadata(Decimal("40000"), 60000, "No warranty", "GCC"),
            ),
            Car("10", 2018, "Ford", "Focus", metadata=Metadata(Decimal("20000"), 0)),
            Car("1", make="Toyota"),
        ]
    )


def test_deterministic_order_and_details(inventory):
    assert [c.listing_id for c in inventory.filter_inventory()] == ["1", "10", "2"]
    assert inventory.get_car_details("2").description is None
    assert inventory.get_car_details("missing") is None
    assert inventory.get_car_details("1").year is None


def test_combined_inclusive_exact_filters(inventory):
    query = InventoryFilter(
        make=" ford ",
        model="EXPLORER",
        regional_specs="gcc",
        year_min=2020,
        year_max=2020,
        price_min=40000,
        price_max=40000,
        mileage_max=60000,
    )
    assert [c.listing_id for c in inventory.filter_inventory(query)] == ["2"]
    assert inventory.filter_inventory(InventoryFilter(model="explor")) == ()


def test_unknowns_excluded_only_when_filtered(inventory):
    assert len(inventory.filter_inventory(InventoryFilter(price_min=0))) == 2
    assert len(inventory.filter_inventory(InventoryFilter(mileage_max=0))) == 1
    assert len(inventory.filter_inventory(InventoryFilter(has_warranty_mention=True))) == 1
    assert len(inventory.filter_inventory(InventoryFilter(has_warranty_mention=False))) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"year_min": 2020, "year_max": 2010},
        {"price_min": -1},
        {"price_min": Decimal("NaN")},
        {"price_max": Decimal("Infinity")},
        {"mileage_min": True},
        {"year_min": 2019.5},
        {"make": " "},
        {"has_warranty_mention": 1},
        {"mileage_max": Decimal("1.5")},
    ],
)
def test_invalid_queries(kwargs):
    with pytest.raises(ValueError):
        InventoryFilter(**kwargs)


def test_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        InMemoryInventory([Car("1"), Car("1")])
