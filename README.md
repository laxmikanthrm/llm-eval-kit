# llm-eval-kit

![CI](https://github.com/laxmikanthrm/llm-eval-kit/actions/workflows/ci.yml/badge.svg)

Lightweight, **offline** evaluation for LLM outputs. Score every response for
hallucinations, relevance, refusals, and output format, then compare two prompt
or model versions to see whether a change actually reduced errors.

It runs in milliseconds with no API key, so you can put it in CI and fail a build
when a prompt change makes answers worse.

## Why

When I integrated LLMs into product features, the hardest part was not calling
the model. It was knowing whether a prompt change made answers better or worse.
This kit turns that into a number you can track.

## Metrics

| Metric | What it catches | How |
|---|---|---|
| **Groundedness** | Hallucinated claims not in the source context | Per-sentence content-word coverage against the context; returns the unsupported sentences |
| **Relevance** | Answers that drift off the question | Cosine similarity of stemmed bag-of-words vectors |
| **Refusal** | "I'm sorry, I can't help" when an answer was expected | Regex patterns |
| **Format** | Broken JSON, missing keys, too long, forbidden content (e.g. phone numbers) | JSON parsing + rules |

## Quick start

```bash
pip install -e ".[dev]"
python -m llm_eval run examples/baseline.jsonl --out report.md
python -m llm_eval compare examples/baseline.jsonl examples/improved.jsonl
```

Example comparison of a baseline prompt and a prompt that tells the model to
answer only from context and return strict JSON:

```json
{
  "baseline_error_rate": 0.6,
  "candidate_error_rate": 0.0,
  "relative_error_reduction": 1.0,
  "groundedness_delta": 0.35
}
```

The report shows exactly which sentence failed, e.g.
`low groundedness (0.0); unsupported: ['Yes, the premium warranty covers accidental drops...']`.

## Dataset format (JSONL)

```json
{"id": "q5", "question": "Return the Pro plan price as JSON.",
 "context": "The Pro plan costs 25 dollars per month...",
 "answer": "{\"price_per_month\": 25, \"users\": 20}",
 "format": {"type": "json", "required_keys": ["price_per_month", "users"]}}
```

Optional fields: `context` (enables groundedness), `format`
(`type`, `required_keys`, `max_words`, `must_include`, `must_not_include`),
`expect_refusal`.

## Use in CI

```bash
python -m llm_eval run eval_set.jsonl --fail-under 0.9
```

Exits with code 1 if the pass rate falls below 90%. See `.github/workflows/ci.yml`.

## Use as a library

```python
from llm_eval import groundedness, evaluate_dataset

r = groundedness(answer, context)
print(r.score, r.unsupported)
```

## Project layout

```
llm_eval/metrics.py   # individual metrics
llm_eval/runner.py    # dataset evaluation, run comparison, Markdown report
llm_eval/__main__.py  # CLI
tests/                # pytest suite
examples/             # baseline vs improved runs
```

## Limitations and next steps

- Groundedness and relevance are lexical, so paraphrases can be under-scored.
  Next step: optional sentence-embedding backend and an LLM-as-judge mode.
- Add per-metric thresholds per dataset row.

## License

MIT
