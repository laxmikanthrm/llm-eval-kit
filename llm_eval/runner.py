"""Run metrics over a JSONL dataset and compare runs (e.g. two prompt versions)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .metrics import answer_relevance, check_format, detect_refusal, groundedness


@dataclass
class EvalConfig:
    min_groundedness: float = 0.8
    min_relevance: float = 0.1
    allow_refusal: bool = False


@dataclass
class ItemResult:
    id: str
    groundedness: float
    relevance: float
    refusal: bool
    format_ok: bool
    passed: bool
    errors: list[str]


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e.msg})") from e
    return rows


def evaluate_item(row: dict, cfg: EvalConfig) -> ItemResult:
    answer = row.get("answer", "")
    errors: list[str] = []

    g = groundedness(answer, row.get("context", "")) if row.get("context") else None
    g_score = g.score if g else 1.0
    if g and g_score < cfg.min_groundedness:
        errors.append(f"low groundedness ({g_score}); unsupported: {g.unsupported[:2]}")

    structured = (row.get("format") or {}).get("type") == "json"
    rel = 1.0 if structured else answer_relevance(row.get("question", ""), answer)
    if rel < cfg.min_relevance:
        errors.append(f"low relevance ({rel})")

    refusal = detect_refusal(answer)
    if refusal and not cfg.allow_refusal and not row.get("expect_refusal", False):
        errors.append("unexpected refusal")

    fmt = check_format(answer, row.get("format"))
    if not fmt.ok:
        errors.append(f"format: {fmt.reason}")

    return ItemResult(
        id=str(row.get("id", "")),
        groundedness=g_score,
        relevance=rel,
        refusal=refusal,
        format_ok=fmt.ok,
        passed=not errors,
        errors=errors,
    )


def evaluate_dataset(rows: list[dict], cfg: EvalConfig | None = None) -> dict:
    cfg = cfg or EvalConfig()
    results = [evaluate_item(r, cfg) for r in rows]
    n = len(results) or 1
    summary = {
        "n": len(results),
        "pass_rate": round(sum(r.passed for r in results) / n, 3),
        "error_rate": round(sum(not r.passed for r in results) / n, 3),
        "mean_groundedness": round(sum(r.groundedness for r in results) / n, 3),
        "mean_relevance": round(sum(r.relevance for r in results) / n, 3),
        "format_failures": sum(not r.format_ok for r in results),
        "refusals": sum(r.refusal for r in results),
    }
    return {"summary": summary, "items": [asdict(r) for r in results]}


def compare_runs(baseline: dict, candidate: dict) -> dict:
    """Compare two evaluate_dataset() outputs; relative change in error rate."""
    b, c = baseline["summary"], candidate["summary"]
    rel = None
    if b["error_rate"] > 0:
        rel = round((b["error_rate"] - c["error_rate"]) / b["error_rate"], 3)
    return {
        "baseline_error_rate": b["error_rate"],
        "candidate_error_rate": c["error_rate"],
        "relative_error_reduction": rel,
        "groundedness_delta": round(c["mean_groundedness"] - b["mean_groundedness"], 3),
        "relevance_delta": round(c["mean_relevance"] - b["mean_relevance"], 3),
    }


def to_markdown(report: dict, title: str = "LLM Evaluation Report") -> str:
    s = report["summary"]
    lines = [
        f"# {title}",
        "",
        f"- Items: **{s['n']}**",
        f"- Pass rate: **{s['pass_rate']:.1%}** (error rate {s['error_rate']:.1%})",
        f"- Mean groundedness: {s['mean_groundedness']}",
        f"- Mean relevance: {s['mean_relevance']}",
        f"- Format failures: {s['format_failures']} | Refusals: {s['refusals']}",
        "",
        "| id | pass | grounded | relevance | errors |",
        "|---|---|---|---|---|",
    ]
    for it in report["items"]:
        err = "; ".join(it["errors"]).replace("|", "\\|") or "-"
        mark = "PASS" if it["passed"] else "FAIL"
        lines.append(
            f"| {it['id']} | {mark} | {it['groundedness']} | {it['relevance']} | {err} |"
        )
    return "\n".join(lines) + "\n"
