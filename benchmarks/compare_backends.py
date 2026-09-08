"""A/B a dense checkpoint across two OpenAI-compatible servers (e.g. FreeToken FTW
vs llama.cpp GGUF) at deterministic settings.

One GPU workflow: point at server A, run, unload A, load server B, run again with
--out a different file, then diff the two result files.

Usage:
    python benchmarks/compare_backends.py --base-url http://127.0.0.1:1919/v1 \\
        --model Qwen3.8-27B-NVFP4-RTX50-S2 --out results_ft.jsonl
    python benchmarks/compare_backends.py --base-url http://127.0.0.1:8080/v1 \\
        --model Qwen3.8-27B-UD-IQ3_S --out results_llama.jsonl

Determinism: temperature=0 everywhere, identical prompts and max_tokens. Thought
traces are normalized (reasoning_content when the server splits it, <think> parse
otherwise) so both engines are judged on the same fields.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

PROMPTS: list[dict] = [
    {
        "id": "factual-1",
        "category": "factual",
        "messages": [{"role": "user", "content": "What is the capital of Japan? Reply with only the name."}],
        "max_tokens": 256,
        "must_contain": ["Tokyo"],
    },
    {
        "id": "math-1",
        "category": "math",
        "messages": [{"role": "user", "content": "If a train travels 120 km in 1.5 hours, what is its speed in m/s? Reply with only the number."}],
        "max_tokens": 512,
        "must_contain": ["22.22"],
    },
    {
        "id": "math-2",
        "category": "math",
        "messages": [{"role": "user", "content": "Janet has 3 bags with 12 apples each. She gives away 14 apples. How many apples are left? Reply with only the number."}],
        "max_tokens": 512,
        "must_contain": ["22"],
    },
    {
        "id": "math-3",
        "category": "math",
        "messages": [{"role": "user", "content": "Notebooks cost $4 each and pens cost $2 each. Maya buys 3 notebooks and 5 pens and pays with a $50 bill. How much change does she get? Reply with only the number."}],
        "max_tokens": 512,
        "must_contain": ["28"],
    },
    {
        "id": "code-1",
        "category": "code",
        "messages": [{"role": "user", "content": "Write a one-line Python lambda that squares a number. Reply with only the code."}],
        "max_tokens": 256,
        "must_contain_any": [["x**2", "x*x", "x ** 2"]],
    },
    {
        "id": "code-2",
        "category": "code",
        "messages": [{"role": "user", "content": "Write a Python function fib(n) returning the n-th Fibonacci number (fib(0)=0, fib(1)=1). Reply with only the code, no explanations."}],
        "max_tokens": 512,
        "exec_check": {"call": "fib(10)", "expect": 55},
    },
    {
        "id": "code-3",
        "category": "code-hard",
        "messages": [{"role": "user", "content": "Write a Python function longest_palindrome(s) returning the longest palindromic substring of s. Reply with only the code, no explanations."}],
        "max_tokens": 1024,
        "exec_check": {"call": "[longest_palindrome('babad') in ('bab', 'aba'), longest_palindrome('cbbd') == 'bb']", "expect": [True, True]},
    },
    {
        "id": "json-1",
        "category": "structured",
        "messages": [{"role": "user", "content": 'List the first 5 prime numbers as JSON like {"primes": [...]}. Reply with only the JSON.'}],
        "max_tokens": 512,
        "json_check": {"key": "primes", "expect": [2, 3, 5, 7, 11]},
    },
    {
        "id": "logic-1",
        "category": "reasoning",
        "messages": [{"role": "user", "content": "Three boxes labeled Apples, Oranges, Mixed are ALL mislabeled. You may take one fruit from one box. Which box do you pick from to correctly relabel all three? Answer in one sentence."}],
        "max_tokens": 512,
        "must_contain": ["Mixed"],
    },
    {
        "id": "summarize-1",
        "category": "open",
        "messages": [{"role": "user", "content": "Summarize in two sentences: The Qwen3.8-27B model uses a hybrid architecture with 48 Gated-DeltaNet linear-attention layers and 16 full-attention layers. Its 27B parameters include 17408-wide MLPs quantized to NVFP4, while attention runs in FP8. On a 32GB card it decodes at about 90 tokens per second without speculation."}],
        "max_tokens": 512,
    },
]


def _post(url: str, payload: dict, timeout: int = 600, api_key: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _split_thinking(message: dict) -> tuple[str, str]:
    """(reasoning, content) across both server conventions."""
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    if not reasoning and "</think>" in content:
        parts = content.split("</think>", 1)
        reasoning = parts[0].split("<think>")[-1].strip()
        content = parts[1].strip()
    return reasoning, content


def _strip_fences(text: str) -> str:
    """Remove ```lang fences some servers leave around code/JSON answers."""
    text = text.strip()
    m = re.fullmatch(r"```(?:\w+)?\s*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _exec_check(code: str, call: str, expect) -> tuple[bool, str]:
    """Run model code with a neutered builtins set and compare one expression."""
    code = _strip_fences(code)
    safe_builtins = {"range": range, "len": len, "min": min, "max": max}
    env: dict = {"__builtins__": safe_builtins}
    try:
        exec(compile(code, "<model>", "exec"), env)
        got = eval(compile(call, "<check>", "eval"), dict(env))
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        return False, f"{type(exc).__name__}: {exc}"
    return (got == expect, f"got {got!r}, want {expect!r}")


def run_one(base_url: str, model: str, prompt: dict, api_key: str | None = None) -> dict:
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "temperature": 0.0,
        "max_tokens": prompt["max_tokens"],
    }
    t0 = time.perf_counter()
    try:
        data = _post(f"{base_url}/chat/completions", payload, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return {"id": prompt["id"], "error": f"{type(exc).__name__}: {exc}"}
    wall = time.perf_counter() - t0
    choice = data["choices"][0]
    reasoning, content = _split_thinking(choice["message"])
    usage = data.get("usage", {})
    completion = int(usage.get("completion_tokens") or 0)
    checks: dict[str, object] = {}
    # Search content + reasoning: a length-truncated turn can hold the right answer
    # in the think trace on either backend; this judges correctness, not budgets.
    hay = content + "\n" + reasoning
    for needle in prompt.get("must_contain", []):
        checks[f"contains:{needle}"] = needle in hay
    for group in prompt.get("must_contain_any", []):
        checks["contains-any:" + "|".join(group)] = any(n in hay for n in group)
    if "json_check" in prompt:
        spec = prompt["json_check"]
        try:
            parsed = json.loads(_strip_fences(content))
            checks["json"] = parsed.get(spec["key"]) == spec["expect"]
        except Exception as exc:  # noqa: BLE001
            checks["json"] = f"parse failed: {exc}"
    if "exec_check" in prompt:
        spec = prompt["exec_check"]
        ok, detail = _exec_check(content, spec["call"], spec["expect"])
        checks["exec"] = ok
        checks["exec_detail"] = detail
    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "wall_s": round(wall, 3),
        "tok_per_s": round(completion / wall, 2) if wall > 0 else 0.0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "finish_reason": choice.get("finish_reason"),
        "reasoning_chars": len(reasoning),
        "checks": checks,
        "reasoning": reasoning,
        "content": content,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:1919/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="results JSONL path")
    ap.add_argument("--api-key", default=None, help="Bearer token if the server needs one")
    ap.add_argument("--only", default=None, help="comma-separated prompt ids to run")
    ns = ap.parse_args()

    only = set(ns.only.split(",")) if ns.only else None
    results = []
    for prompt in PROMPTS:
        if only and prompt["id"] not in only:
            continue
        print(f"[{prompt['id']}] ...", flush=True)
        results.append(run_one(ns.base_url.rstrip("/"), ns.model, prompt, api_key=ns.api_key))

    ok = sum(
        1
        for r in results
        if "error" not in r and all(v is True for k, v in r["checks"].items() if not k.endswith("_detail"))
    )
    total = sum(1 for r in results if "error" not in r)
    with open(ns.out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'id':12} {'tok/s':>7} {'checks':>10}  note")
    for r in results:
        if "error" in r:
            print(f"{r['id']:12} {'ERR':>7} {'-':>10}  {r['error'][:80]}")
            continue
        flags = [k for k, v in r["checks"].items() if v is True]
        fails = [k for k, v in r["checks"].items() if v is not True and not k.endswith("_detail")]
        print(f"{r['id']:12} {r['tok_per_s']:>7} {len(flags):>3}/{len(flags)+len(fails):<6}  {'; '.join(fails) or r['finish_reason']}")
    wall = sum(r.get("wall_s", 0) for r in results)
    toks = sum(r.get("completion_tokens", 0) for r in results)
    print(f"\nchecks passed: {ok}/{total}   overall: {toks/wall:.1f} tok/s over {wall:.1f}s -> {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
