"""Thin wrapper around the Google Gemini API (google-genai SDK).

With no GEMINI_API_KEY the wrapper reports `enabled == False` and every
caller uses its deterministic offline path, so the platform always runs.

No `max_output_tokens` is set on requests: Gemini's thinking models spend part
of the output budget on reasoning, so a small cap can leave the visible answer
empty. Replies are kept short by the prompts instead.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

log = logging.getLogger("railsentinel.llm")

REQUEST_TIMEOUT_MS = 60_000
RETRY_ATTEMPTS = 3        # the first call plus two retries on 408, 429 and 5xx responses
RETRY_MAX_DELAY_S = 8.0


class LLM:
    def __init__(self, settings, client: Any | None = None):
        self.settings = settings
        self.last_error: str | None = None
        self._client = client
        if self._client is None and settings.gemini_api_key:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS, max_delay=RETRY_MAX_DELAY_S)))

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self.settings.llm_model

    def generate(self, system: str, contents: Any, tools: list | None = None, model: str | None = None) -> Any:
        """One `generate_content` call. Tool calls are returned to the caller, never executed by the SDK."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if tools else None)
        return self._client.models.generate_content(model=model or self.model, contents=contents, config=config)

    def text(self, system: str, prompt: str, model: str | None = None) -> str | None:
        if not self.enabled:
            return None
        try:
            return response_text(self.generate(system, prompt, model=model)) or None
        except Exception as exc:  # network, auth, quota and API errors fall back to offline behaviour
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("LLM call failed: %s", self.last_error)
            return None

    def json(self, system: str, prompt: str, model: str | None = None) -> dict | None:
        raw = self.text(system, prompt, model=model)
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _parts(response: Any) -> list:
    candidates = getattr(response, "candidates", None) or []
    content = candidates[0].content if candidates else None
    return list(content.parts or []) if content is not None else []


def response_text(response: Any) -> str:
    """Visible text of the first candidate (thought summaries are excluded)."""
    return "".join(p.text for p in _parts(response) if isinstance(p.text, str) and not p.thought).strip()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def run_tool_loop(llm: LLM, system: str, user_message: str, tools: list[dict],
                  handlers: dict[str, Callable[[dict], Any]], max_steps: int = 8,
                  on_step: Callable[[dict], None] | None = None) -> dict:
    """Manual function-calling loop: call the model, run the functions it asks for, send the
    results back, and stop when it answers without a function call or after `max_steps` turns.

    `tools` use a provider-neutral shape: {"name", "description", "parameters": <JSON Schema>}."""
    from google.genai import types

    declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                              parameters_json_schema=t["parameters"]) for t in tools]
    gemini_tools = [types.Tool(function_declarations=declarations)]
    contents: list = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]
    final_text = ""
    for turn in range(max_steps):
        resp = llm.generate(system, contents, tools=gemini_tools)
        candidate = resp.candidates[0] if resp.candidates else None
        final_text = response_text(resp)
        calls = [p.function_call for p in _parts(resp) if p.function_call is not None]
        if not calls:
            reason = getattr(candidate, "finish_reason", None)
            return {"text": final_text, "turns": turn + 1,
                    "stop_reason": getattr(reason, "value", reason) or "stop"}
        # The model's turn is sent back unchanged: it carries the thought signatures Gemini
        # needs to continue its reasoning after the function results.
        contents.append(candidate.content)
        results = []
        for call in calls:
            args = dict(call.args or {})
            handler = handlers.get(call.name)
            try:
                if handler is None:
                    raise KeyError(f"unknown tool {call.name}")
                output = handler(args)
                is_error = False
            except Exception as exc:
                output = {"error": f"{type(exc).__name__}: {exc}"}
                is_error = True
            if on_step:
                on_step({"tool": call.name, "args": args, "ok": not is_error,
                         "result_preview": json.dumps(output, default=str)[:300]})
            payload = {"error": output["error"]} if is_error else {"output": _json_safe(output)}
            results.append(types.Part(function_response=types.FunctionResponse(
                id=call.id, name=call.name, response=payload)))
        contents.append(types.Content(role="user", parts=results))
    return {"text": final_text, "turns": max_steps, "stop_reason": "max_steps"}
