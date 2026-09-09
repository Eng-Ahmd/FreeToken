"""Client-side reasoning budget (llama.cpp-style ``--reasoning-budget``).

llama.cpp enforces thinking budgets server-side by appending a nudge message once
the think trace overflows. Any OpenAI-compatible server gets the same observable
behavior in two non-streaming calls:

1. Ask with ``max_tokens = think_budget``. If the turn ends with the think block
   closed (``</think>`` seen) or never opened, return it untouched.
2. Otherwise the model was cut mid-think: re-post the history with its partial
   think closed off plus a user nudge ("Time to stop thinking..."), budgeted at
   ``answer_budget`` tokens, and concatenate.

Costs one extra cached-prefix prefill on the nudge (cheap under prefix caching);
token accounting sums both calls. Grade on the combined text.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from compare_backends import _post, _split_thinking  # noqa: E402

DEFAULT_NUDGE = "Time to stop thinking. Give the final answer now."


def _single(base_url: str, model: str, messages: list, max_tokens: int,
            api_key: str | None) -> dict:
    data = _post(
        f"{base_url}/chat/completions",
        {"model": model, "messages": messages, "temperature": 0.0,
         "max_tokens": max_tokens},
        api_key=api_key,
    )
    choice = data["choices"][0]
    reasoning, content = _split_thinking(choice["message"])
    usage = data.get("usage", {})
    return {
        "reasoning": reasoning,
        "content": content,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": choice.get("finish_reason"),
        "raw": choice["message"],
    }


def _think_open(raw_message: dict, reasoning: str, content: str) -> bool:
    """True when the turn was cut inside an unclosed think block."""
    text = (raw_message.get("content") or "") + (raw_message.get("reasoning_content") or "")
    if "</think>" in text:
        return False
    if "<think>" in text or reasoning.strip():
        return True
    return False


def complete_with_budget(
    base_url: str,
    model: str,
    messages: list,
    think_budget: int = 2048,
    answer_budget: int = 1024,
    api_key: str | None = None,
    nudge: str = DEFAULT_NUDGE,
) -> dict:
    """Chat completion with a capped think trace (see module docstring)."""
    first = _single(base_url, model, messages, think_budget, api_key)
    if first["finish_reason"] != "length" or not _think_open(
        first["raw"], first["reasoning"], first["content"]
    ):
        return {**first, "think_tokens": first["completion_tokens"],
                "answer_tokens": 0, "nudged": False, "calls": 1}
    # Close the partial think ourselves so history re-renders cleanly on both
    # template conventions (FT parses </think>; plain templates keep it verbatim).
    partial = (first["reasoning"] + "\n" + first["content"]).strip()
    continued = messages + [
        {"role": "assistant", "content": f"<think>\n{partial}\n</think>\n"},
        {"role": "user", "content": nudge},
    ]
    second = _single(base_url, model, continued, answer_budget, api_key)
    return {
        "reasoning": (first["reasoning"] + "\n" + second["reasoning"]).strip(),
        "content": (first["content"] + "\n" + second["content"]).strip(),
        "think_tokens": first["completion_tokens"],
        "answer_tokens": second["completion_tokens"],
        "completion_tokens": first["completion_tokens"] + second["completion_tokens"],
        "finish_reason": second["finish_reason"],
        "nudged": True,
        "calls": 2,
    }


__all__ = ["DEFAULT_NUDGE", "complete_with_budget"]
