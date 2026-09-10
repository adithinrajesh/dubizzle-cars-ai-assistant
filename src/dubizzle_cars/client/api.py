"""Centralized synchronous HTTP client with controlled, non-leaking errors."""

import os
from urllib.parse import quote, urlsplit

import httpx
from pydantic import ValidationError

from .models import Chat, Health, Profile, Session


class ClientError(Exception):
    """Safe user-facing failure; never contains raw HTTP or provider data."""


class ApiClient:
    def __init__(self, base_url=None, *, transport=None):
        self.base_url = (base_url or os.environ.get("DUBIZZLE_API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        try:
            url = urlsplit(self.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError
        except ValueError:
            raise ClientError("Invalid backend URL. Use an HTTP(S) address without credentials or query parameters.") from None
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(90, connect=10), transport=transport, follow_redirects=False)

    def _request(self, method, path, model, **kwargs):
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise ClientError("The request timed out. The backend may still have completed it; check before resubmitting a booking or enquiry.") from None
        except httpx.HTTPError:
            raise ClientError("Backend unavailable. Start FastAPI before using the assistant.") from None
        if response.status_code != 200:
            message = {
                400: "The request or session was not accepted. Check your user ID and start a new conversation if needed.",
                422: "Please check the entered values and try again.",
                503: "The AI assistant is temporarily unavailable. Your saved session remains intact; you can try again shortly.",
            }.get(response.status_code, "The backend could not complete this request. Please try again later.")
            raise ClientError(message)
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError):
            raise ClientError("The backend returned an unexpected response. Please check that the frontend and backend versions match.") from None

    def health(self):
        return self._request("GET", "/health", Health)

    def create_session(self, user_id, display_name=None):
        return self._request("POST", "/sessions", Session, json={"user_id": user_id, "display_name": display_name or None})

    def profile(self, user_id):
        return self._request("GET", "/users/" + quote(user_id, safe="") + "/profile", Profile)

    def chat(self, message, user_id, session_id):
        return self._request("POST", "/chat", Chat, json={"message": message, "user_id": user_id, "session_id": session_id})

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
