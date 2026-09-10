"""Automated tests must never connect to the network or use runtime state paths."""

import socket
import sys

import pytest


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("DUBIZZLE_STATE_DB_PATH", str(tmp_path / "isolated.db"))
    monkeypatch.setenv("DUBIZZLE_LEADS_CSV_PATH", str(tmp_path / "isolated-leads.csv"))
    original = socket.socket.connect

    def connect(sock, address):
        # Windows implements asyncio's local wakeup socketpair using loopback TCP.
        caller = sys._getframe(1)
        if (
            caller.f_globals.get("__name__") == "socket"
            and caller.f_code.co_name == "_fallback_socketpair"
        ):
            return original(sock, address)
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Network access is forbidden during automated tests")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
