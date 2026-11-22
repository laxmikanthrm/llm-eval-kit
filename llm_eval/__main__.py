"""Command line interface.

    python -m llm_eval run examples/baseline.jsonl --out report.md
    python -m llm_eval compare examples/baseline.jsonl examples/improved.jsonl
    python -m llm_eval run data.jsonl --fail-under 0.9   # exit 1 in CI if pass rate drops
"""

from __future__ import annotations

import argparse
import json
import sys

from .runner import EvalConfig, compare_runs, evaluate_dataset, load_jsonl, to_markdown


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llm_eval", description="Evaluate LLM outputs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate one JSONL file")
    r.add_argument("path")
    r.add_argument("--out", help="write Markdown report here")
    r.add_argument("--json", dest="json_out", help="write JSON report here")
    r.add_argument("--min-groundedness", type=float, default=0.8)
    r.add_argument("--fail-under", type=float, help="exit 1 if pass rate is below this")

    c = sub.add_parser("compare", help="compare a baseline and a candidate run")
    c.add_argument("baseline")
    c.add_argument("candidate")

    a = p.parse_args(argv)

    if a.cmd == "run":
        cfg = EvalConfig(min_groundedness=a.min_groundedness)
        report = evaluate_dataset(load_jsonl(a.path), cfg)
        md = to_markdown(report)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(md)
        if a.json_out:
            with open(a.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        print(md)
        if a.fail_under is not None and report["summary"]["pass_rate"] < a.fail_under:
            print(f"pass rate below {a.fail_under}", file=sys.stderr)
            return 1
        return 0

    base = evaluate_dataset(load_jsonl(a.baseline))
    cand = evaluate_dataset(load_jsonl(a.candidate))
    print(json.dumps(compare_runs(base, cand), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
