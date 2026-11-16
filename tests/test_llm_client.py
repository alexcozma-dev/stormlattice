"""Tests for the LLM client. A fake transport stands in for both providers, so nothing touches the network."""

import json
import logging
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel

from stormlattice.llm import client
from stormlattice.llm.client import LLMConfig, LLMError, cache_key, complete_json, write_briefing
from stormlattice.llm.guardrails import ungrounded_numbers

KEY = "FAKE-KEY-0123456789"


class Answer(BaseModel):
    level: Literal["green", "amber", "red"]
    note: str


def gemini(tmp_path, api_key=KEY):
    return LLMConfig(provider="gemini", model="gemini-test", api_key=api_key, cache_dir=tmp_path / "llm")


def openrouter(tmp_path, api_key=KEY):
    return LLMConfig(provider="openrouter", model="google/gemini-test", api_key=api_key,
                     cache_dir=tmp_path / "llm")


def gemini_reply(text, status="completed"):
    """A Gemini Interaction body whose model output is `text`."""
    return {"id": "v1_abc", "status": status, "model": "gemini-test",
            "steps": [{"type": "thought", "signature": "x"},
                      {"type": "model_output", "content": [{"type": "text", "text": text}]}]}


def openrouter_reply(text):
    """An OpenRouter chat completion body whose message content is `text`."""
    return {"id": "gen-abc", "model": "google/gemini-test",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


class FakeAPI:
    """Stands in for the provider: answers with the queued (status, body) pairs and keeps the requests."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def transport(self):
        def handler(request):
            self.requests.append(request)
            status, body = self.replies.pop(0)
            return httpx.Response(status, json=body)
        return httpx.MockTransport(handler)

    def body(self, i=0):
        return json.loads(self.requests[i].content)


def no_network():
    """A transport that fails the test if anything is sent."""
    def handler(request):
        raise AssertionError(f"unexpected request to {request.url}")
    return httpx.MockTransport(handler)


GOOD = json.dumps({"level": "amber", "note": "storms near the route"})


def test_gemini_request_and_answer(tmp_path):
    api = FakeAPI((200, gemini_reply(GOOD)))
    out = complete_json("How is the route?", Answer, gemini(tmp_path), system="Be brief.",
                        transport=api.transport())
    assert out == {"level": "amber", "note": "storms near the route"}

    request = api.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert request.headers["x-goog-api-key"] == KEY
    assert request.headers["api-revision"] == "2026-05-20"
    assert api.body() == {
        "model": "gemini-test",
        "input": "How is the route?",
        "system_instruction": "Be brief.",
        "response_format": {"type": "text", "mime_type": "application/json",
                            "schema": Answer.model_json_schema()},
        "store": False,
    }


def test_openrouter_request_and_answer(tmp_path):
    api = FakeAPI((200, openrouter_reply(GOOD)))
    out = complete_json("How is the route?", Answer, openrouter(tmp_path), system="Be brief.",
                        transport=api.transport())
    assert out == {"level": "amber", "note": "storms near the route"}

    request = api.requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert api.body() == {
        "model": "google/gemini-test",
        "messages": [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": "How is the route?"}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "Answer", "strict": True,
                                            "schema": Answer.model_json_schema()}},
        "provider": {"require_parameters": True},
    }


def test_no_system_text_is_left_out_of_the_request(tmp_path):
    api = FakeAPI((200, gemini_reply(GOOD)), (200, openrouter_reply(GOOD)))
    complete_json("a", Answer, gemini(tmp_path), transport=api.transport())
    complete_json("b", Answer, openrouter(tmp_path), transport=api.transport())
    assert "system_instruction" not in api.body(0)
    assert api.body(1)["messages"] == [{"role": "user", "content": "b"}]


def test_second_call_comes_from_the_cache(tmp_path):
    api = FakeAPI((200, gemini_reply(GOOD)))
    first = complete_json("q", Answer, gemini(tmp_path), transport=api.transport())
    second = complete_json("q", Answer, gemini(tmp_path), transport=no_network())
    assert first == second
    assert len(api.requests) == 1


def test_cache_hit_needs_no_key_and_no_provider_match(tmp_path):
    # replay: the cache was filled with Gemini; the replay machine has no key and another provider set
    complete_json("q", Answer, gemini(tmp_path), transport=FakeAPI((200, gemini_reply(GOOD))).transport())
    out = complete_json("q", Answer, openrouter(tmp_path, api_key=""), transport=no_network())
    assert out["level"] == "amber"


def test_cache_file_is_named_by_the_input_hash_and_holds_no_key(tmp_path):
    complete_json("q", Answer, gemini(tmp_path), system="s",
                  transport=FakeAPI((200, gemini_reply(GOOD))).transport())
    files = list((tmp_path / "llm").iterdir())
    assert [f.name for f in files] == [cache_key("q", "s", Answer.model_json_schema()) + ".json"]
    assert KEY not in files[0].read_text()
    assert json.loads(files[0].read_text())["provider"] == "gemini"


def test_cache_key_depends_on_every_part_of_the_input():
    schema = Answer.model_json_schema()
    keys = {cache_key("q", "s", schema), cache_key("q2", "s", schema), cache_key("q", "s2", schema),
            cache_key("q", "s", {"type": "object"})}
    assert len(keys) == 4
    assert cache_key("q", "s", schema) == cache_key("q", "s", dict(reversed(list(schema.items()))))


def test_missing_key_raises_before_any_request(tmp_path):
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        complete_json("q", Answer, gemini(tmp_path, api_key=""), transport=no_network())


def test_unknown_provider_raises(tmp_path):
    llm = LLMConfig(provider="anthropic", model="m", api_key=KEY, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="gemini or openrouter"):
        complete_json("q", Answer, llm, transport=no_network())


@pytest.mark.parametrize("text", ["not json at all",
                                  json.dumps({"level": "purple", "note": "x"}),     # not in the enum
                                  json.dumps({"level": "green"})])                  # missing field
def test_answer_that_does_not_fit_the_schema_raises_and_is_not_cached(tmp_path, text):
    with pytest.raises(LLMError, match="does not fit Answer"):
        complete_json("q", Answer, gemini(tmp_path), transport=FakeAPI((200, gemini_reply(text))).transport())
    assert not (tmp_path / "llm").exists()


def test_gemini_status_other_than_completed_raises(tmp_path):
    api = FakeAPI((200, gemini_reply(GOOD[:10], status="incomplete")))
    with pytest.raises(LLMError, match="incomplete"):
        complete_json("q", Answer, gemini(tmp_path), transport=api.transport())


def test_openrouter_error_inside_a_200_raises(tmp_path):
    api = FakeAPI((200, {"error": {"code": 502, "message": "model is down"}}))
    with pytest.raises(LLMError, match="502: model is down"):
        complete_json("q", Answer, openrouter(tmp_path), transport=api.transport())


def test_openrouter_empty_content_raises(tmp_path):
    api = FakeAPI((200, {"choices": [{"finish_reason": "length", "message": {"content": None}}]}))
    with pytest.raises(LLMError, match="length"):
        complete_json("q", Answer, openrouter(tmp_path), transport=api.transport())


@pytest.mark.parametrize("make", [gemini, openrouter])
def test_key_never_reaches_logs_errors_or_repr(tmp_path, caplog, make):
    caplog.set_level(logging.DEBUG)
    llm = make(tmp_path)
    # a provider that echoes the key back in its error message
    api = FakeAPI((429, {"error": {"code": 429, "message": f"quota exceeded for key {KEY}"}}))
    with pytest.raises(LLMError, match="HTTP 429") as raised:
        complete_json("q", Answer, llm, transport=api.transport())
    assert "quota exceeded" in str(raised.value)
    assert KEY not in str(raised.value)
    assert "HTTP Request" in caplog.text        # httpx did log the request, so the next line means something
    assert KEY not in caplog.text
    assert KEY not in repr(llm)


def test_network_failure_raises_llm_error_without_the_key(tmp_path):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)
    with pytest.raises(LLMError, match="request failed") as raised:
        complete_json("q", Answer, gemini(tmp_path), transport=httpx.MockTransport(handler))
    assert KEY not in str(raised.value)


FACTS = {"ROUTE-1": "route risk 0.41 for a 15:10 UTC departure",
         "MRMS-1808": "radar frame 18:08 UTC, 3 storms within 20 km of the route"}


def briefing_reply(text):
    return (200, gemini_reply(json.dumps({"briefing": text})))


def test_write_briefing_returns_a_grounded_draft(tmp_path):
    draft = "Route risk is 0.41 for the 15:10 UTC departure [ROUTE-1]. 3 storms lie within 20 km [MRMS-1808]."
    api = FakeAPI(briefing_reply(draft))
    assert write_briefing(FACTS, gemini(tmp_path), transport=api.transport()) == draft
    assert len(api.requests) == 1
    assert api.body()["input"].startswith("FACTS:\n")
    assert "0.41" in api.body()["input"]
    assert api.body()["system_instruction"] == client.BRIEFING_SYSTEM


def test_write_briefing_regenerates_once_after_an_invented_number(tmp_path):
    good = "Route risk is 0.41 [ROUTE-1]."
    api = FakeAPI(briefing_reply("Route risk is about 0.4 [ROUTE-1]."), briefing_reply(good))
    assert write_briefing(FACTS, gemini(tmp_path), transport=api.transport()) == good
    assert len(api.requests) == 2
    assert client.BRIEFING_RETRY_NOTE not in api.body(0)["input"]
    assert client.BRIEFING_RETRY_NOTE in api.body(1)["input"]

    # replay: both drafts are cached, so the same facts give the same briefing with no request
    assert write_briefing(FACTS, gemini(tmp_path, api_key=""), transport=no_network()) == good


def test_write_briefing_falls_back_to_the_template(tmp_path):
    api = FakeAPI(briefing_reply("Risk is 0.4 [ROUTE-1]."), briefing_reply("Risk is 41% [ROUTE-1]."))
    text = write_briefing(FACTS, gemini(tmp_path), transport=api.transport())
    assert len(api.requests) == 2
    assert text == client.template_briefing(FACTS)
    assert "route risk 0.41 for a 15:10 UTC departure [ROUTE-1]" in text
    assert ungrounded_numbers(text, json.dumps(FACTS)) == set()


def test_write_briefing_lets_a_provider_failure_through(tmp_path):
    with pytest.raises(LLMError, match="HTTP 503"):
        write_briefing(FACTS, gemini(tmp_path), transport=FakeAPI((503, {"error": {}})).transport())


def test_load_config_reads_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("STORM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    llm = client.load_config()      # Gemini is the default
    assert (llm.provider, llm.model, llm.api_key) == ("gemini", client.DEFAULT_MODEL["gemini"], "g-key")
    assert llm.cache_dir == tmp_path / "llm"

    monkeypatch.setenv("LLM_PROVIDER", "OpenRouter")
    monkeypatch.setenv("LLM_MODEL", "some/model")
    llm = client.load_config()
    assert (llm.provider, llm.model, llm.api_key) == ("openrouter", "some/model", "o-key")

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="gemini or openrouter"):
        client.load_config()


def test_load_config_without_a_key_still_loads(tmp_path, monkeypatch):
    # replay runs from the cache alone, so a missing key is only an error once a request is needed
    monkeypatch.setenv("STORM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.load_config().api_key == ""
