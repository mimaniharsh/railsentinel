"""The Gemini wrapper driven through the real google-genai client, with the network replaced by a mock
transport: checks the requests the SDK sends and that responses are parsed the way the agents expect."""
import json
from types import SimpleNamespace

import httpx
from google import genai
from google.genai import types

from railsentinel.agents.llm import LLM, run_tool_loop

SETTINGS = SimpleNamespace(gemini_api_key=None, llm_model="gemini-test-flash", llm_judge_model="gemini-test-lite")


def _client(replies: list[dict], sent: list[dict]) -> genai.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), "headers": {k.lower(): v for k, v in request.headers.items()},
                     "body": json.loads(request.content)})
        return httpx.Response(200, json=replies[len(sent) - 1])

    return genai.Client(api_key="test-key", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))


def _reply(*parts: dict) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": list(parts)}, "finishReason": "STOP"}]}


def test_tool_loop_round_trip_through_the_sdk():
    sent: list[dict] = []
    replies = [
        _reply({"text": "working it out", "thought": True, "thoughtSignature": "c2lnLTE="},
               {"functionCall": {"id": "call-1", "name": "add", "args": {"a": 2, "b": 3}},
                "thoughtSignature": "c2lnLTI="}),
        _reply({"text": "The answer is 5."}),
    ]
    llm = LLM(SETTINGS, client=_client(replies, sent))
    tools = [{"name": "add", "description": "Add two numbers.",
              "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                             "required": ["a", "b"]}}]
    steps = []
    out = run_tool_loop(llm, "You are a calculator.", "What is 2 + 3?", tools,
                        {"add": lambda args: {"sum": args["a"] + args["b"]}}, on_step=steps.append)

    assert out == {"text": "The answer is 5.", "turns": 2, "stop_reason": "STOP"}
    assert steps[0]["tool"] == "add" and steps[0]["ok"]
    first, second = sent
    assert first["url"].endswith("/models/gemini-test-flash:generateContent")
    assert first["headers"]["x-goog-api-key"] == "test-key"
    assert first["body"]["systemInstruction"]["parts"][0]["text"] == "You are a calculator."
    assert first["body"]["tools"][0]["functionDeclarations"][0]["name"] == "add"
    model_turn, result_turn = second["body"]["contents"][1:]
    # The model's turn goes back unchanged, thought signatures included.
    assert model_turn["role"] == "model"
    assert [p.get("thoughtSignature") for p in model_turn["parts"]] == ["c2lnLTE=", "c2lnLTI="]
    response = result_turn["parts"][0]["functionResponse"]
    assert response == {"id": "call-1", "name": "add", "response": {"output": {"sum": 5}}}


def test_tool_errors_are_reported_back_to_the_model():
    sent: list[dict] = []
    replies = [_reply({"functionCall": {"id": "c1", "name": "lookup", "args": {}}}), _reply({"text": "Sorry."})]
    llm = LLM(SETTINGS, client=_client(replies, sent))

    def broken(args):
        raise ValueError("no such record")

    out = run_tool_loop(llm, "system", "question", [{"name": "lookup", "description": "Look up.",
                                                     "parameters": {"type": "object", "properties": {}}}],
                        {"lookup": broken})
    assert out["text"] == "Sorry."
    payload = sent[1]["body"]["contents"][-1]["parts"][0]["functionResponse"]["response"]
    assert payload == {"error": "ValueError: no such record"}


def test_text_and_json_helpers_skip_thoughts_and_use_the_judge_model():
    sent: list[dict] = []
    replies = [_reply({"text": "private reasoning", "thought": True},
                      {"text": 'Verdict: {"faithful": true, "score": 5}'})]
    llm = LLM(SETTINGS, client=_client(replies, sent))
    result = llm.json("Audit the answer.", "QUESTION ...", model=SETTINGS.llm_judge_model)
    assert result == {"faithful": True, "score": 5}
    assert sent[0]["url"].endswith("/models/gemini-test-lite:generateContent")
    assert "tools" not in sent[0]["body"]


def test_api_errors_fall_back_quietly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": 429, "message": "Resource exhausted",
                                                   "status": "RESOURCE_EXHAUSTED"}})

    client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    llm = LLM(SETTINGS, client=client)
    assert llm.text("system", "prompt") is None
    assert "429" in llm.last_error


def test_wrapper_is_disabled_without_a_key_and_enabled_with_one():
    assert not LLM(SETTINGS).enabled
    keyed = LLM(SimpleNamespace(gemini_api_key="test-key", llm_model="m", llm_judge_model="j"))
    assert keyed.enabled and keyed.model == "m"
