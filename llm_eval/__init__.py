"""llm-eval-kit: lightweight, dependency-light evaluation for LLM outputs."""

from .metrics import (
    answer_relevance,
    check_format,
    detect_refusal,
    groundedness,
)
from .runner import EvalConfig, evaluate_dataset, compare_runs

__all__ = [
    "answer_relevance",
    "check_format",
    "detect_refusal",
    "groundedness",
    "EvalConfig",
    "evaluate_dataset",
    "compare_runs",
]
__version__ = "0.1.0"
