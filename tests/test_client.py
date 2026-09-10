"""Offline HTTP contract and Streamlit display-state tests."""

import ast
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from dubizzle_cars.client.api import ApiClient, ClientError
from dubizzle_cars.client.models import Chat, Health, Profile, Session

ROOT = Path(__file__).resolve().parents[1]
CAR = {"listing_id": "1", "year": 2020, "make": "honda", "model": "civic", "trim": None,
       "title": "source", "photo_url": None, "metadata": {"sale_price_aed": None,
       "mileage_km": None, "warranty_mention": None, "regional_specs": None}}


def test_base_url(monkeypatch):
    monkeypatch.delenv("DUBIZZLE_API_BASE_URL", raising=False)
    with ApiClient() as client:
        assert client.base_url == "http://127.0.0.1:8000"
        assert client._http.timeout.read == 90
    monkeypatch.setenv("DUBIZZLE_API_BASE_URL", "http://localhost:8888/")
    with ApiClient() as client:
        assert client.base_url == "http://localhost:8888"


@pytest.mark.parametrize("url", ["file:///secret", "https://key@example.com", "https://example.com?key=private", "not a URL"])
def test_invalid_url_is_safe(url):
    with pytest.raises(ClientError) as caught:
        ApiClient(url)
    assert url not in str(caught.value)


def test_contracts_and_session_payload():
    requests = []

    def respond(request):
        requests.append(request)
        data = {"/health": {"inventory_count": 100, "semantic_retrieval_available": False},
                "/sessions": {"user_id": "demo", "session_id": "s1", "returning_user": True},
                "/users/demo/profile": {"user_id": "demo", "display_name": "Demo", "preferences": {"make": "honda"}},
                "/chat": {"reply": "Details", "cars": [CAR], "session_id": "s1"}}
        return httpx.Response(200, json=data[request.url.path])

    with ApiClient(transport=httpx.MockTransport(respond)) as client:
        assert client.health().inventory_count == 100
        assert client.create_session("demo", "Demo").returning_user
        assert client.profile("demo").preferences == {"make": "honda"}
        for message in ["Show Hondas", "The first one", "Does it have warranty?"]:
            result = client.chat(message, "demo", "s1")
            assert result.cars[0].metadata.sale_price_aed is None
            assert result.cars[0].metadata.warranty_mention is None
            assert result.cars[0].listing_id == "1"
            assert json.loads(requests[-1].content) == {"message": message, "user_id": "demo", "session_id": "s1"}


@pytest.mark.parametrize("status", [400, 422, 503, 500, 404, 302])
def test_http_errors_never_echo_bodies(status, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-marker")
    with ApiClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, text="private-test-marker"))) as client:
        with pytest.raises(ClientError) as caught:
            client.chat("Hi", "demo", "s1")
    assert "private-test-marker" not in str(caught.value)
    if status == 503:
        assert "saved session remains intact" in str(caught.value)


@pytest.mark.parametrize("error", [httpx.ConnectError("private"), httpx.ReadTimeout("private")])
def test_transport_errors(error):
    def fail(request):
        raise error

    with ApiClient(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(ClientError) as caught:
            client.health()
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("body", ["not JSON", "{}", '{"reply":"Hi","cars":"wrong"}'])
def test_malformed_response(body):
    with ApiClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body))) as client:
        with pytest.raises(ClientError, match="unexpected response"):
            client.chat("Hi", "demo", "s1")


def test_frontend_has_no_backend_imports():
    paths = list((ROOT / "src/dubizzle_cars/client").glob("*.py")) + [ROOT / "streamlit_app.py"]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dubizzle_cars"):
                assert node.module.startswith("dubizzle_cars.client")
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(("sqlite3", "google", "dubizzle_cars.agent", "dubizzle_cars.memory")) for n in node.names)


def test_import_and_photo_validation():
    spec = importlib.util.spec_from_file_location("frontend_entry", ROOT / "streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for value in [None, "", "file:///local", "javascript:alert(1)", "http://user:secret@example.com"]:
        assert module.image_url(value) is None
    assert module.image_url("https://example.com/photo.jpg") == "https://example.com/photo.jpg"


def test_streamlit_session_reruns_and_new_conversation(monkeypatch):
    from streamlit.testing.v1 import AppTest

    class FakeClient:
        sessions = 0
        payloads = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def health(self):
            return Health(inventory_count=100, semantic_retrieval_available=False)

        def create_session(self, user_id, display_name=None):
            FakeClient.sessions += 1
            return Session(user_id=user_id, session_id=f"s{self.sessions}", returning_user=self.sessions > 1)

        def profile(self, user_id):
            return Profile(user_id=user_id, display_name=None, preferences={"make": "honda"})

        def chat(self, message, user_id, session_id):
            self.payloads.append((message, user_id, session_id))
            return Chat(reply="The listing does not specify warranty information.", cars=[CAR], session_id=session_id)

    monkeypatch.setattr("dubizzle_cars.client.api.ApiClient", FakeClient)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    app.text_input[0].set_value("demo")
    next(b for b in app.button if b.label == "Start conversation").click().run()
    app.chat_input[0].set_value("Show Hondas").run()
    assert len(app.session_state.messages) == 2
    assert FakeClient.payloads[-1] == ("Show Hondas", "demo", "s1")
    app.run()
    assert len(app.session_state.messages) == 2
    assert any("Price: Not specified" in element.value for element in app.text)
    next(b for b in app.button if b.label == "New conversation").click().run()
    assert app.session_state.session_id == "s2"
    assert app.session_state.user_id == "demo"
    assert app.session_state.messages == []
    assert app.session_state.returning_user
    assert not app.exception
