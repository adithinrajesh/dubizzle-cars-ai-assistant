"""Versioned SQLite state. Each operation owns a short transaction and connection."""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4


class StateError(ValueError):
    """Safe application validation error."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value):
        raise StateError(
            "Identity must contain 1–100 letters, digits, dots, underscores or hyphens"
        )
    return value


class SQLiteState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = RLock()  # Serializes local stateful turns, including CSV actions.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise StateError("Unsupported state database version")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY, display_name TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
                    returning_user INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
                    last_search_result_ids TEXT NOT NULL DEFAULT '[]', active_listing_id TEXT,
                    updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT NOT NULL REFERENCES users(user_id), preference_type TEXT NOT NULL,
                    preference_value TEXT NOT NULL, source_message TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, preference_type));
                CREATE TABLE IF NOT EXISTS bookings (
                    booking_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
                    session_id TEXT NOT NULL REFERENCES sessions(session_id), listing_id TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL,
                    UNIQUE(listing_id, scheduled_at));
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        except sqlite3.Error:
            raise StateError("State persistence failed") from None
        finally:
            db.close()

    def create_session(self, user_id: str, display_name: str | None = None) -> dict:
        identity(user_id)
        if display_name is not None and (not display_name.strip() or len(display_name) > 100):
            raise StateError("Display name must contain 1–100 characters")
        with self.lock, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            returning = (
                db.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone() is not None
            )
            timestamp, session_id = now(), str(uuid4())
            db.execute(
                """INSERT INTO users VALUES (?,?,?,?) ON CONFLICT(user_id) DO UPDATE
                SET display_name=COALESCE(excluded.display_name, users.display_name), updated_at=excluded.updated_at""",
                (user_id, display_name, timestamp, timestamp),
            )
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?)",
                (session_id, user_id, returning, timestamp, timestamp),
            )
            db.execute(
                "INSERT INTO session_state(session_id,updated_at) VALUES (?,?)",
                (session_id, timestamp),
            )
        return {"user_id": user_id, "session_id": session_id, "returning_user": returning}

    def session(self, user_id: str, session_id: str) -> dict:
        identity(user_id)
        with self.connection() as db:
            row = db.execute(
                """SELECT s.*, t.last_search_result_ids, t.active_listing_id
                FROM sessions s JOIN session_state t USING(session_id)
                WHERE s.session_id=? AND s.user_id=?""",
                (session_id, user_id),
            ).fetchone()
        if row is None:
            raise StateError("Session is unavailable for this user")
        result = dict(row)
        result["last_search_result_ids"] = json.loads(result["last_search_result_ids"])
        return result

    def profile(self, user_id: str) -> dict:
        identity(user_id)
        with self.connection() as db:
            user = db.execute(
                "SELECT user_id,display_name FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
            if user is None:
                raise StateError("User profile is unavailable")
            preferences = db.execute(
                "SELECT preference_type,preference_value FROM user_preferences WHERE user_id=? ORDER BY preference_type",
                (user_id,),
            ).fetchall()
        return {**dict(user), "preferences": {r[0]: json.loads(r[1]) for r in preferences}}

    def update_reference(self, user_id: str, session_id: str, *, results=None, active=None):
        self.session(user_id, session_id)
        with self.connection() as db:
            if results is not None:
                db.execute(
                    "UPDATE session_state SET last_search_result_ids=?, active_listing_id=NULL, updated_at=? WHERE session_id=?",
                    (json.dumps(results), now(), session_id),
                )
            if active is not None:
                db.execute(
                    "UPDATE session_state SET active_listing_id=?, updated_at=? WHERE session_id=?",
                    (active, now(), session_id),
                )

    def remember(self, user_id: str, preferences: dict, source: str):
        with self.connection() as db:
            for kind, value in preferences.items():
                db.execute(
                    """INSERT INTO user_preferences VALUES (?,?,?,?,?,?)
                    ON CONFLICT(user_id,preference_type) DO UPDATE SET
                    preference_value=excluded.preference_value, source_message=excluded.source_message,
                    updated_at=excluded.updated_at""",
                    (user_id, kind, json.dumps(value), source, now(), now()),
                )

    def message(self, user_id: str, session_id: str, role: str, content: str):
        self.session(user_id, session_id)
        with self.connection() as db:
            db.execute(
                "INSERT INTO messages(session_id,role,content,created_at) VALUES (?,?,?,?)",
                (session_id, role, content, now()),
            )
            db.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now(), session_id))

    def book(self, user_id: str, session_id: str, listing_id: str, scheduled_at: str) -> dict:
        self.session(user_id, session_id)
        with self.lock, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM bookings WHERE listing_id=? AND scheduled_at=?",
                (listing_id, scheduled_at),
            ).fetchone():
                raise StateError("That listing already has a local booking at this time")
            booking = {
                "booking_id": str(uuid4()),
                "user_id": user_id,
                "session_id": session_id,
                "listing_id": listing_id,
                "scheduled_at": scheduled_at,
                "created_at": now(),
                "status": "confirmed",
            }
            db.execute("INSERT INTO bookings VALUES (?,?,?,?,?,?,?)", tuple(booking.values()))
        return booking
