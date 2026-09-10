import hashlib
from pathlib import Path

import pytest
from openpyxl import Workbook

from dubizzle_cars import InventoryDataError, load_inventory
from dubizzle_cars.loader import COLUMNS


def workbook(tmp_path, rows, headers=COLUMNS, sheet="cleaned dataset"):
    path = tmp_path / "fixture.xlsx"
    book = Workbook()
    book.active.title = sheet
    book.active.append(list(headers))
    for row in rows:
        book.active.append(row)
    book.save(path)
    book.close()
    return path


def test_load_preserves_source_and_unknowns(tmp_path):
    path = workbook(
        tmp_path,
        [
            [1, 2019, " Ford ", "Focus", "other", "2020 GCC", None, None],
            [None] * 8,
            ["002", "unknown", None, None, None, None, None, None],
        ],
    )
    before = path.read_bytes()
    cars = load_inventory(path)
    assert path.read_bytes() == before
    assert len(cars) == 2
    assert cars[0].listing_id == "1"
    assert cars[0].year == 2019  # Structured field, not the title's year.
    assert cars[0].make == "Ford"
    assert cars[0].trim == "other"  # Preserve source category.
    assert cars[1].listing_id == "002"
    assert cars[1].year is None
    assert cars[1].source_row == 4


@pytest.mark.parametrize(
    "rows",
    [
        [[1], [1]],
        [[None, 2020]],
        [["=1+1"]],
        [[1, "=2000+20"]],
    ],
)
def test_bad_rows_rejected(tmp_path, rows):
    with pytest.raises(InventoryDataError):
        load_inventory(workbook(tmp_path, rows))


def test_missing_sheet_and_columns(tmp_path):
    with pytest.raises(InventoryDataError, match="sheet"):
        load_inventory(workbook(tmp_path, [], sheet="raw dataset"))
    with pytest.raises(InventoryDataError, match="column"):
        load_inventory(workbook(tmp_path, [], headers=["Listing_ID"]))
    with pytest.raises(InventoryDataError, match="column"):
        load_inventory(workbook(tmp_path, [], headers=[*COLUMNS, "year"]))


def test_supplied_workbook():
    path = Path(__file__).resolve().parents[1] / "data/sample_cars_dataset.xlsx"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cars = load_inventory(path)
    assert len(cars) == 100
    assert len({car.listing_id for car in cars}) == 100
    third = next(car for car in cars if car.listing_id == "3")
    assert third.metadata.sale_price_aed == 119750
    assert third.metadata.mileage_km == 68000
    assert third.metadata.regional_specs == "GCC"
    assert third.metadata.warranty_mention is not None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
