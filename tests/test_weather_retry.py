"""Tests for the Open-Meteo retry.

The case that motivated this is the one in ``test_retries_a_2xx_with_a_non_json_body``:
on 2026-08-29 the archive endpoint answered with a successful status and an
empty body, which passed raise_for_status() and then failed on the decode,
taking the whole scheduled run down.
"""

import httpx
import pytest

import src.weather as weather


class FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("GET", "https://example.test/")

    def json(self):
        import json

        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Keep the backoff out of the test runtime."""
    monkeypatch.setattr(weather.time, "sleep", lambda _seconds: None)


def queue_responses(monkeypatch, outcomes):
    """Serve *outcomes* one per call; an Exception instance is raised instead."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(weather.httpx, "get", fake_get)
    return calls


GOOD = '{"hourly": {"time": []}}'


def test_retries_a_2xx_with_a_non_json_body(monkeypatch):
    calls = queue_responses(monkeypatch, [FakeResponse(200, ""), FakeResponse(200, GOOD)])

    payload = weather._fetch_json("https://example.test/", {}, timeout=1)

    assert payload == {"hourly": {"time": []}}
    assert len(calls) == 2


def test_retries_a_server_error(monkeypatch):
    calls = queue_responses(
        monkeypatch, [FakeResponse(503, "unavailable"), FakeResponse(200, GOOD)]
    )

    assert weather._fetch_json("https://example.test/", {}, timeout=1) is not None
    assert len(calls) == 2


def test_retries_a_transport_error(monkeypatch):
    calls = queue_responses(
        monkeypatch, [httpx.ReadTimeout("timed out"), FakeResponse(200, GOOD)]
    )

    assert weather._fetch_json("https://example.test/", {}, timeout=1) is not None
    assert len(calls) == 2


def test_gives_up_after_the_attempt_limit(monkeypatch):
    calls = queue_responses(monkeypatch, [FakeResponse(200, "<html>nope</html>")])

    with pytest.raises(RuntimeError) as excinfo:
        weather._fetch_json("https://example.test/", {}, timeout=1)

    assert len(calls) == weather._MAX_ATTEMPTS
    # The message has to name what came back, or the next incident is opaque.
    assert "non-JSON body" in str(excinfo.value)
    assert "nope" in str(excinfo.value)


def test_does_not_retry_a_client_error(monkeypatch):
    """4xx is deterministic — retrying only delays the soil-param fallback."""
    calls = queue_responses(monkeypatch, [FakeResponse(400, "bad param")])

    with pytest.raises(httpx.HTTPStatusError):
        weather._fetch_json("https://example.test/", {}, timeout=1)

    assert len(calls) == 1


def test_soil_params_are_dropped_after_a_client_error(monkeypatch):
    """The pre-existing fallback must still work through the retry helper."""
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append(params["hourly"])
        if "soil_temperature_6cm" in params["hourly"]:
            return FakeResponse(400, "unknown variable")
        return FakeResponse(200, GOOD)

    monkeypatch.setattr(weather.httpx, "get", fake_get)
    monkeypatch.setattr(weather, "_parse_hourly_response", lambda payload: payload)

    weather._get_weather(
        "https://example.test/", {}, weather.FORECAST_SOIL_PARAMS, timeout=1
    )

    assert len(seen) == 2
    assert "soil_temperature_6cm" in seen[0]
    assert "soil_temperature_6cm" not in seen[1]


def test_a_transient_failure_does_not_trigger_the_soil_fallback(monkeypatch):
    """An empty body says nothing about the parameters — keep asking for soil."""
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append(params["hourly"])
        return FakeResponse(200, "" if len(seen) == 1 else GOOD)

    monkeypatch.setattr(weather.httpx, "get", fake_get)
    monkeypatch.setattr(weather, "_parse_hourly_response", lambda payload: payload)

    weather._get_weather(
        "https://example.test/", {}, weather.FORECAST_SOIL_PARAMS, timeout=1
    )

    assert len(seen) == 2
    assert all("soil_temperature_6cm" in hourly for hourly in seen)



def test_retries_a_rate_limit(monkeypatch):
    """429 is the service's limit, not a wrong request: wait and ask again."""
    calls = queue_responses(
        monkeypatch,
        [FakeResponse(429, '{"error":true,"reason":"Minutely API request limit exceeded."}'),
         FakeResponse(200, GOOD)],
    )

    assert weather._fetch_json("https://example.test/", {}, timeout=1) is not None
    assert len(calls) == 2


def test_a_rate_limit_does_not_trigger_the_soil_fallback(monkeypatch):
    """A persistent 429 fails as a service failure, with the soil parameters still asked for.

    Read as a client error it used to drop the soil parameters and retry —
    into the same limit, or into a forecast without soil features.
    """
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append(params["hourly"])
        return FakeResponse(429, '{"error":true,"reason":"Daily API request limit exceeded."}')

    monkeypatch.setattr(weather.httpx, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        weather._get_weather("https://example.test/", {}, weather.FORECAST_SOIL_PARAMS, timeout=1)

    assert len(seen) == weather._MAX_ATTEMPTS
    assert all("soil_temperature_6cm" in hourly for hourly in seen)
    assert "429" in str(excinfo.value) and "Daily API request limit" in str(excinfo.value)
