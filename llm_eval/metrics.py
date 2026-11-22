"""Individual metrics for scoring a single LLM response.

Every metric is deterministic and runs offline, so evaluations are cheap,
reproducible, and can run in CI on every prompt change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "and",
    "or", "in", "on", "at", "for", "with", "by", "it", "this", "that", "as", "from",
    "its", "their", "has", "have", "had", "can", "will", "do", "does", "not", "but",
    "if", "so", "than", "then", "there", "which", "who", "what", "when", "how",
}

_REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) (?:sorry|unable|not able)\b",
    r"\bI can(?:not|'t) (?:help|assist|provide|answer)\b",
    r"\bas an AI(?: language model)?\b",
    r"\bI don't have (?:access|enough information)\b",
]


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(_tokens(p)) >= 3]


@dataclass
class GroundednessResult:
    score: float
    unsupported: list[str] = field(default_factory=list)


def groundedness(answer: str, context: str, threshold: float = 0.6) -> GroundednessResult:
    """Share of answer sentences whose content words mostly appear in the context.

    A sentence is "supported" when at least ``threshold`` of its content tokens are
    found in the context. Unsupported sentences are returned so you can see exactly
    which claims are likely hallucinated.
    """
    ctx = set(_tokens(context))
    sents = _sentences(answer)
    if not sents:
        return GroundednessResult(score=1.0)
    unsupported = []
    for s in sents:
        toks = _tokens(s)
        coverage = sum(t in ctx for t in toks) / len(toks)
        if coverage < threshold:
            unsupported.append(s)
    score = 1 - len(unsupported) / len(sents)
    return GroundednessResult(score=round(score, 3), unsupported=unsupported)


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def answer_relevance(question: str, answer: str) -> float:
    """Cosine similarity of stemmed bag-of-words vectors (0..1).

    Cheap proxy for "does the answer talk about what was asked". Swap in an
    embedding model for semantic matching (see README).
    """
    q = [_stem(t) for t in _tokens(question)]
    a = [_stem(t) for t in _tokens(answer)]
    if not q or not a:
        return 0.0
    vec = CountVectorizer(analyzer=lambda x: x).fit([q, a])
    m = vec.transform([q, a])
    return round(float(cosine_similarity(m[0], m[1])[0, 0]), 3)


def detect_refusal(answer: str) -> bool:
    return any(re.search(p, answer, re.IGNORECASE) for p in _REFUSAL_PATTERNS)


@dataclass
class FormatResult:
    ok: bool
    reason: str = ""


def check_format(answer: str, spec: dict | None) -> FormatResult:
    """Validate output format.

    ``spec`` keys (all optional):
      - ``type``: "json" -> answer must parse as a JSON object
      - ``required_keys``: keys that must exist in the JSON object
      - ``max_words``: maximum number of words
      - ``must_include`` / ``must_not_include``: lists of regex patterns
    """
    if not spec:
        return FormatResult(True)
    if spec.get("type") == "json":
        try:
            obj = json.loads(answer)
        except json.JSONDecodeError as e:
            return FormatResult(False, f"invalid JSON: {e.msg}")
        if not isinstance(obj, dict):
            return FormatResult(False, "JSON is not an object")
        missing = [k for k in spec.get("required_keys", []) if k not in obj]
        if missing:
            return FormatResult(False, f"missing keys: {', '.join(missing)}")
    max_words = spec.get("max_words")
    if max_words is not None and len(answer.split()) > max_words:
        return FormatResult(False, f"{len(answer.split())} words > max {max_words}")
    for pat in spec.get("must_include", []):
        if not re.search(pat, answer, re.IGNORECASE):
            return FormatResult(False, f"missing required pattern /{pat}/")
    for pat in spec.get("must_not_include", []):
        if re.search(pat, answer, re.IGNORECASE):
            return FormatResult(False, f"contains forbidden pattern /{pat}/")
    return FormatResult(True)
