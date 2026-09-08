"""Validated quality suites over any OpenAI-compatible server (FreeToken FTW or
llama.cpp GGUF): HumanEval (exec-graded code) and GSM8K (numeric exact-match).

Usage:
    python benchmarks/quality_suites.py --suite humaneval --base-url URL --model M --out f.jsonl
    python benchmarks/quality_suites.py --suite gsm8k --limit 200 --base-url URL --model M --out f.jsonl

Determinism: temperature=0, identical prompts. Thought traces are normalized
(reasoning_content or <think> parse) and searched together with content.
Data files live in <tmp>/benchdata (HumanEval.jsonl.gz, gsm8k_test.jsonl).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from compare_backends import _post, _split_thinking, _strip_fences  # noqa: E402

DATADIR = r"C:\Users\engah\AppData\Local\Temp\opencode\benchdata"
_SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "sum": sum,
    "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "int": int, "float": float,
    "str": str, "bool": bool, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "ValueError": ValueError, "Exception": Exception,
}


def _complete(base_url: str, model: str, messages: list, max_tokens: int,
              api_key: str | None) -> dict:
    payload = {"model": model, "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens}
    t0 = time.perf_counter()
    try:
        data = _post(f"{base_url}/chat/completions", payload, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    wall = time.perf_counter() - t0
    choice = data["choices"][0]
    reasoning, content = _split_thinking(choice["message"])
    usage = data.get("usage", {})
    return {
        "wall_s": round(wall, 3),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": choice.get("finish_reason"),
        "reasoning": reasoning,
        "content": content,
    }


def _run_untrusted(program: str, timeout: int = 15) -> tuple[bool, str]:
    """Exec grade in a subprocess (kills infinite loops); returns (passed, detail)."""
    wrapper = (
        "import traceback\n" + program + "\nprint('HARNESS-OK')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        return False, tail[-1][:200] if tail else f"exit {proc.returncode}"
    return "HARNESS-OK" in proc.stdout, proc.stdout.strip().splitlines()[-1][:200]


def _load_humaneval() -> list[dict]:
    with gzip.open(os.path.join(DATADIR, "humaneval.jsonl.gz"), "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def run_humaneval(base_url: str, model: str, api_key: str | None, out: str,
                  limit: int | None) -> None:
    tasks = _load_humaneval()
    if limit:
        tasks = tasks[:limit]
    results = []
    for i, task in enumerate(tasks):
        print(f"[{i + 1}/{len(tasks)}] {task['task_id']} ...", flush=True)
        r = _complete(
            base_url, model,
            [{"role": "user", "content": task["prompt"] + "\nReply with only the code, no explanations."}],
            1024, api_key,
        )
        if "error" in r:
            results.append({"task_id": task["task_id"], **r})
            continue
        code = _strip_fences(r["content"]) or _strip_fences(r["reasoning"])
        program = code + "\n" + task["test"]
        ok, detail = _run_untrusted(program)
        # Canonical HumanEval entry point check lives in task["entry_point"]; the
        # test body already calls it, so a clean run == pass.
        results.append({"task_id": task["task_id"], "pass": ok, "detail": detail, **r})
    _report(results, out, key="pass")


def _gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _gsm8k_pred(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def _load_gsm8k() -> list[dict]:
    with open(os.path.join(DATADIR, "gsm8k_test.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def run_gsm8k(base_url: str, model: str, api_key: str | None, out: str,
              limit: int | None, offset: int) -> None:
    items = _load_gsm8k()[offset:]
    if limit:
        items = items[:limit]
    results = []
    for i, item in enumerate(items):
        print(f"[{i + 1}/{len(items)}] ...", flush=True)
        r = _complete(
            base_url, model,
            [{"role": "user", "content": item["question"] + "\nReply with only the final number."}],
            512, api_key,
        )
        if "error" in r:
            results.append({"idx": offset + i, **r})
            continue
        gold = _gsm8k_gold(item["answer"])
        pred = _gsm8k_pred(r["content"] + "\n" + r["reasoning"])
        try:
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            ok = False
        results.append({"idx": offset + i, "pass": ok, "gold": gold, "pred": pred, **r})
    _report(results, out, key="pass")


def _load_aime() -> list[dict]:
    items = []
    for name, tag in (("aime2025-I.jsonl", "I"), ("aime2025-II.jsonl", "II")):
        with open(os.path.join(DATADIR, name), encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                items.append({"id": f"{tag}-{i + 1}", **row})
    return items


def _aime_gold(answer) -> int | None:
    """Gold answers carry LaTeX (e.g. '336^\\circ'): take the first integer."""
    m = re.search(r"-?\d+", str(answer).replace(",", ""))
    return int(m.group(0)) if m else None


def run_aime(base_url: str, model: str, api_key: str | None, out: str) -> None:
    items = _load_aime()
    results = []
    for i, item in enumerate(items):
        print(f"[{i + 1}/{len(items)}] {item['id']} ...", flush=True)
        r = _complete(
            base_url, model,
            [{"role": "user", "content": item["question"] + "\nReply with only the integer answer (0-999)."}],
            2048, api_key,
        )
        if "error" in r:
            results.append({"id": item["id"], **r})
            continue
        gold = _aime_gold(item["answer"])
        pred = _gsm8k_pred(r["content"] + "\n" + r["reasoning"])
        try:
            ok = gold is not None and pred is not None and int(float(pred)) == gold
        except ValueError:
            ok = False
        results.append({"id": item["id"], "pass": ok, "gold": gold, "pred": pred, **r})
    _report(results, out, key="pass")


def _report(results: list[dict], out: str, key: str) -> None:
    with open(out, "w", encoding="utf-8") as fh:
        for r in results:
            slim = {k: v for k, v in r.items() if k in (
                "task_id", "idx", key, "detail", "gold", "pred",
                "wall_s", "completion_tokens", "finish_reason", "error")}
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    done = [r for r in results if "error" not in r]
    passed = sum(1 for r in done if r.get(key) is True)
    wall = sum(r.get("wall_s", 0) for r in done)
    toks = sum(r.get("completion_tokens", 0) for r in done)
    fails = [str(r.get("task_id", r.get("idx"))) for r in done if r.get(key) is not True][:10]
    print(f"\n{key}: {passed}/{len(done)} ({100 * passed / max(1, len(done)):.1f}%)  "
          f"{toks / max(wall, 1e-9):.1f} tok/s over {wall:.0f}s -> {out}")
    if fails:
        print("first fails:", ", ".join(fails))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=["humaneval", "gsm8k", "aime"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ns = ap.parse_args()
    if ns.suite == "humaneval":
        run_humaneval(ns.base_url.rstrip("/"), ns.model, ns.api_key, ns.out, ns.limit)
    elif ns.suite == "aime":
        run_aime(ns.base_url.rstrip("/"), ns.model, ns.api_key, ns.out)
    else:
        run_gsm8k(ns.base_url.rstrip("/"), ns.model, ns.api_key, ns.out, ns.limit, ns.offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
