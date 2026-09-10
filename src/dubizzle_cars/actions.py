"""Deterministic local booking and qualified lead persistence."""

import csv
import os
import re
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

from .memory.repository import StateError, identity, now

CSV_LOCK = Lock()


class BookingService:
    def __init__(self, state, inventory, clock=None):
        self.state, self.inventory = state, inventory
        self.zone = ZoneInfo("Asia/Dubai")
        self.clock = clock or (lambda: datetime.now(self.zone))

    def book(self, user_id, session_id, listing_id, requested_datetime):
        if self.inventory.get_car_details(listing_id) is None:
            raise StateError("Listing does not exist")
        if not isinstance(requested_datetime, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::00)?", requested_datetime
        ):
            raise StateError(
                "Provide a Dubai-local date and time as YYYY-MM-DDTHH:MM without an offset"
            )
        try:
            local = datetime.fromisoformat(requested_datetime).replace(tzinfo=self.zone)
        except ValueError:
            raise StateError("Invalid booking date or time") from None
        if local.weekday() == 6 or not time(8) <= local.time() <= time(20):
            raise StateError(
                "Bookings are available Monday through Saturday, 08:00–20:00 Dubai time"
            )
        if local <= self.clock():
            raise StateError("Booking time must be in the future")
        return self.state.book(user_id, session_id, listing_id, local.isoformat())


class LeadService:
    fields = (
        "user_id",
        "budget_min_aed",
        "budget_max_aed",
        "needs",
        "preferred_make",
        "preferred_model",
        "created_at",
    )

    def __init__(self, path: Path):
        self.path = Path(path)

    def save(
        self,
        user_id,
        *,
        budget_min_aed=None,
        budget_max_aed=None,
        needs=None,
        preferred_make=None,
        preferred_model=None,
    ):
        identity(user_id)
        budgets = []
        for value in (budget_min_aed, budget_max_aed):
            try:
                if value is None:
                    budgets.append(None)
                    continue
                if isinstance(value, bool):
                    raise ValueError
                number = Decimal(str(value))
                if not number.is_finite() or number < 0 or number > Decimal("1000000000"):
                    raise ValueError
                budgets.append(number)
            except (ValueError, InvalidOperation):
                raise StateError("Budget must be a finite nonnegative AED amount") from None
        low, high = budgets
        if (
            low is None
            and high is None
            or high == 0
            or (high is None and low == 0)
            or (low is not None and high is not None and low > high)
        ):
            raise StateError("Provide a usable AED budget range")
        if (
            not isinstance(needs, str)
            or not 5 <= len(needs.strip()) <= 1000
            or len(re.findall(r"[A-Za-z]+", needs)) < 2
        ):
            raise StateError("Provide meaningful automotive needs")
        row = dict(
            zip(
                self.fields,
                (
                    user_id,
                    "" if low is None else format(low, "f"),
                    "" if high is None else format(high, "f"),
                    needs.strip(),
                    preferred_make or "",
                    preferred_model or "",
                    now(),
                ),
                strict=True,
            )
        )
        for key, value in row.items():
            if not isinstance(value, str) or len(value) > 1000:
                raise StateError("Invalid lead field")
            # Quoting alone does not prevent spreadsheet formula execution.
            if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(
                ("\t", "\r", "\n")
            ):
                row[key] = "'" + value
        try:
            with CSV_LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a+", newline="", encoding="utf-8") as handle:
                    handle.seek(0)
                    header = next(csv.reader(handle), None)
                    if header is not None and header != list(self.fields):
                        raise StateError("Lead CSV schema does not match")
                    handle.seek(0, 2)
                    writer = csv.DictWriter(handle, fieldnames=self.fields)
                    if header is None:
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError:
            raise StateError("Lead persistence failed") from None
        return {"status": "saved"}
