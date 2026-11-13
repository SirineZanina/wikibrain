"""Retrieval quality measurement.

Answers one question: when I change chunk size, overlap, or k, does retrieval get better
or worse? Without a number here, tuning is guesswork.

`hit@k` = the fraction of questions where the correct policy file appears anywhere in the
top k results. `top-1` = the fraction where it is the single best result, which is the
stricter and more useful signal.

Run:  python -m server.eval_retrieval [--k 4]
"""

import argparse
import logging

from server.config import settings
from server.retriever import retrieve

# (question, filename that should answer it)
EVAL_SET: list[tuple[str, str]] = [
    ("How many days of paid annual leave do I get per year?", "pto-and-leave-policy.md"),
    ("How much annual leave can I carry over to next year?", "pto-and-leave-policy.md"),
    ("How long is paid leave for a primary caregiver?", "pto-and-leave-policy.md"),
    ("How many paid sick days are there?", "pto-and-leave-policy.md"),
    ("When am I eligible for a sabbatical?", "pto-and-leave-policy.md"),
    ("What is the home office setup stipend?", "remote-work-policy.md"),
    ("How many days a month must hybrid staff be in the office?", "remote-work-policy.md"),
    ("What are core working hours?", "remote-work-policy.md"),
    ("Can I work from another country temporarily?", "remote-work-policy.md"),
    ("What is the monthly internet allowance?", "remote-work-policy.md"),
    ("What is the meal per diem when travelling?", "expense-and-travel-policy.md"),
    ("What is the nightly hotel cap in London?", "expense-and-travel-policy.md"),
    ("When am I allowed to fly business class?", "expense-and-travel-policy.md"),
    ("What is the deadline for submitting an expense claim?", "expense-and-travel-policy.md"),
    ("How much can I spend on client entertainment per person?", "expense-and-travel-policy.md"),
    ("Can I buy extra annual leave days?", "pto-and-leave-policy.md"),
    ("Can I swap a public holiday for a religious observance?", "pto-and-leave-policy.md"),
    ("How many paid volunteering days do I get?", "pto-and-leave-policy.md"),
    ("How much is the coworking space allowance?", "remote-work-policy.md"),
    ("Is there a day with no recurring meetings?", "remote-work-policy.md"),
    ("When does new equipment arrive for a remote joiner?", "remote-work-policy.md"),
    ("What do I do if I lost a receipt?", "expense-and-travel-policy.md"),
    ("Do I get a rest day after a long-haul flight?", "expense-and-travel-policy.md"),
]


def run(k: int) -> tuple[float, float]:
    top1 = 0
    hitk = 0

    for question, expected in EVAL_SET:
        hits = retrieve(question, top_k=k)
        sources = [h.source for h in hits]

        is_top1 = bool(sources) and sources[0] == expected
        in_topk = expected in sources
        top1 += is_top1
        hitk += in_topk

        flag = "ok  " if is_top1 else ("~   " if in_topk else "MISS")
        print(f"{flag} {question[:52]:<52} -> {sources[0] if sources else '(none)'}")

    n = len(EVAL_SET)
    print(f"\nk={k}  chunk_size={settings.chunk_size}  overlap={settings.chunk_overlap}")
    print(f"top-1 : {top1}/{n} ({top1 / n:.0%})")
    print(f"hit@{k} : {hitk}/{n} ({hitk / n:.0%})")
    return top1 / n, hitk / n


def main() -> None:
    logging.disable(logging.INFO)
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()
    run(args.k)


if __name__ == "__main__":
    main()
