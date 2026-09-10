from decimal import Decimal

import pytest

from dubizzle_cars import extract_metadata


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Cash price: AED 119,750.00", Decimal("119750.00")),
        ("AED 119,750.00 in cash", Decimal("119750.00")),
        ("Price: DHS 30000", Decimal("30000")),
        ("AED 2,111 monthly; AED 119,750 in cash", Decimal("119750")),
        ("Price: AED 2000 monthly", None),
        ("Price: AED 2000 down payment", None),
        ("Price: AED 30,000; Price: AED 35,000", None),
        ("Price: AED 30,000 or AED 35,000", None),
        ("Price: AED 30,000?", None),
        ("Price: AED 30,000-35,000", None),
        ("Price: AED 30,000 approximately", None),
        ("30500, call +971552011671", None),
        ("Price: 30,000", None),
        ("AED 30,000", None),
        ("Price: USD 30,000", None),
        ("Price: AED 0", None),
        ("Price: AED 50k", None),
        ("Price: AED 50,00", None),
        ("Price: AED 50 000", None),
        ("Price: AED 50 K", None),
        ("Price: AED 30,000 to 35,000", None),
    ],
)
def test_price(text, expected):
    assert extract_metadata(None, text).sale_price_aed == expected


@pytest.mark.parametrize(
    "title,description,expected",
    [
        ("Mini GCC 120,000 km 30500", None, 120000),
        ("-50,000 km", None, None),
        (None, "Mileage: Just 68,000 km", 68000),
        (None, "Mileage - 56000 KM", 56000),
        (None, "Mileage: 0 km", 0),
        (None, "Warranty for 100,000 km", None),
        (None, "Mileage: 50,000 miles", None),
        (None, "Mileage: 50,000", None),
        (None, "Mileage: about 50,000 km", None),
        (None, "Mileage: 50,000 km?", None),
        ("50,000 km", "Mileage: 60,000 km", None),
        ("50,000 km", "Mileage: 50,000 km", 50000),
        ("40,000-50,000 km", None, None),
    ],
)
def test_mileage(title, description, expected):
    assert extract_metadata(title, description).mileage_km == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2018 GCC", "GCC"),
        ("Japanese specs", "Japanese"),
        ("American specifications", "American"),
        ("USA", None),
        ("Dubai showroom", None),
        ("Not GCC", None),
        ("Non-GCC", None),
        ("GCC or Japanese specs", None),
        ("GCC?", None),
        ("GCC; Japanese specs", None),
    ],
)
def test_regions(text, expected):
    assert extract_metadata(text, None).regional_specs == expected


def test_warranty_is_source_mention_not_coverage():
    result = extract_metadata(None, "No warranty; warranty expired in 2020")
    assert result.warranty_mention == "No warranty\nwarranty expired in 2020"
    assert extract_metadata(None, None).warranty_mention is None


def test_warranty_geography_does_not_establish_specs():
    assert extract_metadata(None, "Comes with GCC warranty").regional_specs is None


def test_uncertain_description_cancels_title_mileage():
    assert extract_metadata("50,000 km", "Mileage: about 60,000 km").mileage_km is None
