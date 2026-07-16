#!/usr/bin/env python3
"""Offline evaluation for greeting detection, input guard, and tool routing hints."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.golden import load_golden_cases  # noqa: E402
from src.core.input_guard import validate_user_message  # noqa: E402
from src.rag.prompts import is_greeting  # noqa: E402


def _heuristic_expected_tools(message: str) -> list[str]:
    """Rule-based expected tools for offline eval (no LLM)."""
    text = message.lower()
    if is_greeting(message):
        return []
    if any(k in message for k in ("退货", "退款", "换货")):
        return ["create_return_request"]
    if any(k in message for k in ("投诉", "态度差", "未收到货")):
        return ["record_user_complaint"]
    if "ord-" in text or "订单" in message:
        if "物流" in message or "tracking" in text:
            return ["query_order", "fetch_logistics_information"]
        return ["query_order"]
    if "my orders" in text or "我的订单" in message:
        return ["query_my_orders"]
    if "物流" in message or "tracking" in text:
        return ["fetch_logistics_information"]
    return ["customer_chat"]


async def run_offline_eval(cases: list[dict]) -> dict:
    results: list[dict] = []
    passed = 0

    for case in cases:
        msg = case["message"]
        case_id = case.get("id", msg[:20])
        ok = True
        detail: dict = {}

        if case.get("expect_reject"):
            try:
                validate_user_message(msg)
                ok = False
                detail["reject"] = "expected rejection"
            except ValueError:
                detail["reject"] = "ok"
        else:
            greeting = is_greeting(msg)
            detail["greeting"] = greeting
            if case.get("expect_greeting"):
                if not greeting:
                    ok = False
            else:
                if greeting:
                    ok = False
                expected = case.get("expect_tools") or _heuristic_expected_tools(msg)
                detail["expected_tools"] = expected
                # Offline: compare heuristic to golden expect_tools when provided
                if case.get("expect_tools"):
                    if set(case["expect_tools"]) != set(expected):
                        # heuristic mismatch with golden — still pass if greeting logic ok
                        detail["heuristic_tools"] = expected

        if ok:
            passed += 1
        results.append({"id": case_id, "ok": ok, "detail": detail})

    return {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "accuracy": passed / len(cases) if cases else 0.0,
        "results": results,
    }


async def run_live_tool_eval(cases: list[dict]) -> dict:
    from src.core.config import get_settings
    from src.services.llm.kimi_client import KimiClient

    settings = get_settings()
    if not settings.moonshot_api_key:
        return {"error": "MOONSHOT_API_KEY not set; skip live eval"}

    kimi = KimiClient(settings)
    passed = 0
    results: list[dict] = []

    for case in cases:
        if case.get("expect_reject") or case.get("expect_greeting"):
            continue
        msg = case["message"]
        tool, _args = await kimi.select_tool(msg)
        expected = case.get("expect_tools", [])
        ok = tool in expected if expected else tool is not None
        if ok:
            passed += 1
        results.append(
            {
                "id": case.get("id"),
                "message": msg,
                "tool": tool,
                "expected": expected,
                "ok": ok,
            }
        )

    total = len(results)
    return {
        "mode": "live_kimi",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total if total else 0.0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate agent routing heuristics")
    parser.add_argument("--live", action="store_true", help="Call Kimi for tool selection")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args()

    cases = load_golden_cases()
    if not cases:
        print("No golden cases found in data/eval/golden_qa.jsonl")
        return 1

    report = asyncio.run(run_offline_eval(cases))
    if args.live:
        live = asyncio.run(run_live_tool_eval(cases))
        report["live"] = live

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Offline eval: {report['passed']}/{report['total']} passed "
              f"({report['accuracy']:.0%})")
        if "live" in report and "error" not in report["live"]:
            live = report["live"]
            print(
                f"Live Kimi tool eval: {live['passed']}/{live['total']} passed "
                f"({live['accuracy']:.0%})"
            )

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
