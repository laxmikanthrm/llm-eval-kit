"""Generates the two example runs: a baseline prompt and an improved prompt.

The improved prompt told the model to answer only from the provided context and
to return strict JSON where requested, which removes most hallucinations and
format errors.
"""

import json
from pathlib import Path

CTX = {
    "refund": "Customers can request a refund within 30 days of purchase. Refunds are issued to the original payment method within 5 business days.",
    "shipping": "Standard shipping takes 3 to 7 business days. Express shipping takes 1 to 2 business days and costs 15 dollars.",
    "warranty": "All laptops include a one year limited warranty covering hardware defects. Accidental damage is not covered.",
    "password": "To reset a password, open Settings, choose Security, and click Reset Password. A reset link is emailed to the account address.",
    "plans": "The Basic plan costs 10 dollars per month and includes 5 users. The Pro plan costs 25 dollars per month and includes 20 users and priority support.",
}

ITEMS = [
    ("q1", "refund", "How long do I have to request a refund?", None,
     "You can request a refund within 30 days of purchase. Refunds usually take two months and require a store visit.",
     "You can request a refund within 30 days of purchase. Refunds go back to the original payment method within 5 business days."),
    ("q2", "shipping", "How fast is express shipping and what does it cost?", None,
     "Express shipping takes 1 to 2 business days and costs 15 dollars.",
     "Express shipping takes 1 to 2 business days and costs 15 dollars."),
    ("q3", "warranty", "Does the laptop warranty cover accidental damage?", None,
     "Yes, the premium warranty covers accidental drops and liquid spills for three years.",
     "No. Laptops include a one year limited warranty for hardware defects, and accidental damage is not covered."),
    ("q4", "password", "How do I reset my password?", {"max_words": 40},
     "To reset a password, open Settings, choose Security, and click Reset Password. A reset link is emailed to the account address.",
     "Open Settings, choose Security, and click Reset Password. A reset link is emailed to the account address."),
    ("q5", "plans", "Return the Pro plan price and user limit as JSON.", {"type": "json", "required_keys": ["price_per_month", "users"]},
     "The Pro plan costs 25 dollars per month and includes 20 users.",
     "{\"price_per_month\": 25, \"users\": 20}"),
    ("q6", "plans", "Return the Basic plan price and user limit as JSON.", {"type": "json", "required_keys": ["price_per_month", "users"]},
     "{\"price\": 10}",
     "{\"price_per_month\": 10, \"users\": 5}"),
    ("q7", "shipping", "How long does standard shipping take?", None,
     "I'm sorry, I cannot help with shipping questions.",
     "Standard shipping takes 3 to 7 business days."),
    ("q8", "refund", "Where does my refund go?", None,
     "Refunds are issued to the original payment method within 5 business days.",
     "Refunds are issued to the original payment method within 5 business days."),
    ("q9", "warranty", "How long is the laptop warranty?", None,
     "All laptops include a one year limited warranty covering hardware defects.",
     "All laptops include a one year limited warranty covering hardware defects."),
    ("q10", "plans", "Which plan includes priority support?", None,
     "The Enterprise plan includes priority support, dedicated managers, and unlimited storage.",
     "The Pro plan includes priority support, along with 20 users for 25 dollars per month."),
]


def write(path: Path, which: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for id_, ctx, q, fmt, base, imp in ITEMS:
            row = {"id": id_, "question": q, "context": CTX[ctx], "answer": (base, imp)[which]}
            if fmt:
                row["format"] = fmt
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    here = Path(__file__).parent
    write(here / "baseline.jsonl", 0)
    write(here / "improved.jsonl", 1)
    print("wrote baseline.jsonl and improved.jsonl")
