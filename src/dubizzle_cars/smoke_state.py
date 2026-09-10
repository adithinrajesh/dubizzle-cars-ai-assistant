"""Offline manual state smoke; all artifacts live in a disposable temporary directory."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from . import Car, InMemoryInventory
from .actions import BookingService, LeadService
from .memory.repository import SQLiteState, StateError
from .memory.service import StatefulChat
from .retrieval import InventorySearchService


def main():
    with TemporaryDirectory(prefix="dubizzle-state-") as directory:
        root = Path(directory)
        state = SQLiteState(root / "state.db")
        first = state.create_session("smoke-user")
        state.remember("smoke-user", {"make": "honda"}, "Show Hondas")
        second = state.create_session("smoke-user")
        assert first["session_id"] != second["session_id"] and second["returning_user"]
        repo = InMemoryInventory([Car("1", make="honda")])
        leads = LeadService(root / "leads.csv")
        chat = StatefulChat(state, leads, repo, InventorySearchService(repo), None)
        reply, _ = chat.chat(
            "What was I looking for last time?", "smoke-user", second["session_id"]
        )
        assert "honda" in reply.reply
        booking = BookingService(
            state, repo, lambda: datetime(2029, 1, 1, tzinfo=ZoneInfo("Asia/Dubai"))
        )
        booking.book("smoke-user", second["session_id"], "1", "2030-01-07T12:00")
        try:
            booking.book("smoke-user", second["session_id"], "1", "2030-01-13T12:00")
        except StateError:
            pass
        else:
            raise AssertionError("Sunday booking was accepted")
        leads.save("smoke-user", budget_max_aed=90000, needs="comfortable long drives")
    print("Offline state smoke passed; temporary database and CSV removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
