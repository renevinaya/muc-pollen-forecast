"""Retries on the data release: GitHub's 5xx took two runs down in November 2026."""

import httpx
import pytest

from src import store


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(store.time, "sleep", waited.append)
    return waited


class FakeStream:
    """What httpx.stream returns: a context manager over one response."""

    def __init__(self, status: int, body: bytes = b""):
        self.response = httpx.Response(
            status, content=body, request=httpx.Request("GET", "https://example.test/a")
        )

    def __enter__(self):
        return self.response

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, outcomes):
    calls = []

    def fake_stream(method, url, **kwargs):
        calls.append(url)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(store.httpx, "stream", fake_stream)
    monkeypatch.setattr(store, "_get_release", lambda: {"assets": [
        {"name": "models.tar.gz", "browser_download_url": "https://example.test/a"}]})
    return calls


def test_a_download_is_retried_after_a_gateway_timeout(monkeypatch, tmp_path, sleeps):
    """2026-11-09 05:20: one 504 on models.tar.gz ended the run."""
    calls = _serve(monkeypatch, [FakeStream(504), FakeStream(200, b"payload")])

    assert store._download_asset("models.tar.gz", tmp_path / "m.tar.gz")

    assert len(calls) == 2 and sleeps == [store._RETRY_DELAYS[0]]
    assert (tmp_path / "m.tar.gz").read_bytes() == b"payload"


def test_a_download_is_retried_after_a_dropped_connection(monkeypatch, tmp_path, sleeps):
    calls = _serve(monkeypatch, [httpx.ReadError("reset"), FakeStream(200, b"payload")])

    assert store._download_asset("models.tar.gz", tmp_path / "m.tar.gz")
    assert len(calls) == 2


def test_a_client_error_on_a_download_is_not_retried(monkeypatch, tmp_path, sleeps):
    calls = _serve(monkeypatch, [FakeStream(404)])

    with pytest.raises(httpx.HTTPStatusError):
        store._download_asset("models.tar.gz", tmp_path / "m.tar.gz")
    assert len(calls) == 1 and sleeps == []


def test_a_download_gives_up_after_the_last_wait(monkeypatch, tmp_path, sleeps):
    attempts = len(store._RETRY_DELAYS) + 1
    _serve(monkeypatch, [FakeStream(503)] * attempts)

    with pytest.raises(httpx.HTTPStatusError):
        store._download_asset("models.tar.gz", tmp_path / "m.tar.gz")
    assert sleeps == list(store._RETRY_DELAYS)


def test_api_requests_wait_longer_between_server_errors(monkeypatch, sleeps):
    """2026-11-05 23:22: an asset DELETE answered 500 four times within 7 s."""
    statuses = iter([500, 500, 204])

    def fake_request(method, url, **kwargs):
        return httpx.Response(next(statuses), request=httpx.Request(method, url))

    monkeypatch.setattr(store.httpx, "request", fake_request)

    assert store._request("DELETE", "https://example.test/x").status_code == 204
    assert sleeps == list(store._RETRY_DELAYS[:2])
    assert sum(store._RETRY_DELAYS) >= 30
