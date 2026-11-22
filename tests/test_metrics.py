from llm_eval import answer_relevance, check_format, detect_refusal, groundedness
from llm_eval.runner import EvalConfig, compare_runs, evaluate_dataset

CTX = "Standard shipping takes 3 to 7 business days. Express shipping costs 15 dollars."


def test_groundedness_supported():
    r = groundedness("Standard shipping takes 3 to 7 business days.", CTX)
    assert r.score == 1.0 and r.unsupported == []


def test_groundedness_flags_hallucination():
    r = groundedness(
        "Standard shipping takes 3 to 7 business days. Drones deliver packages within minutes.",
        CTX,
    )
    assert r.score == 0.5
    assert "Drones" in r.unsupported[0]


def test_relevance_handles_plurals():
    assert answer_relevance("Where does my refund go?", "Refunds go to your card.") > 0.3


def test_relevance_unrelated_is_low():
    assert answer_relevance("How do I reset my password?", "Bananas are yellow fruit.") == 0.0


def test_refusal_detection():
    assert detect_refusal("I'm sorry, I cannot help with that.")
    assert not detect_refusal("Express shipping costs 15 dollars.")


def test_json_format():
    spec = {"type": "json", "required_keys": ["price"]}
    assert check_format('{"price": 10}', spec).ok
    assert not check_format("price is 10", spec).ok
    assert "missing keys" in check_format('{"cost": 10}', spec).reason


def test_word_limit_and_patterns():
    assert not check_format("one two three", {"max_words": 2}).ok
    assert not check_format("call 555-1234", {"must_not_include": [r"\d{3}-\d{4}"]}).ok
    assert check_format("see docs", {"must_include": ["docs"]}).ok


def test_dataset_and_compare():
    bad = [{"id": "1", "question": "How fast is shipping?", "context": CTX,
            "answer": "Shipping is instant via teleportation portals."}]
    good = [{"id": "1", "question": "How fast is shipping?", "context": CTX,
             "answer": "Standard shipping takes 3 to 7 business days."}]
    b = evaluate_dataset(bad, EvalConfig())
    g = evaluate_dataset(good, EvalConfig())
    assert b["summary"]["error_rate"] == 1.0
    assert g["summary"]["error_rate"] == 0.0
    assert compare_runs(b, g)["relative_error_reduction"] == 1.0
