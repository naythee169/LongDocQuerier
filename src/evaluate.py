"""
Evaluation script.

Runs the full RAG pipeline against a set of ground-truth question-answer pairs
and reports structured metrics.

Metrics:
  - Answer accuracy : does the generated answer contain the ground-truth answer?
  - Abstention rate : how often did the system abstain?
  - Hallucination   : answered confidently but incorrectly
  - Page recall@k   : was the correct page in the top-k retrieved chunks?

Input format (.jsonl — one JSON object per line):
    {"question": "...", "answer": "...", "evidence_page": 42}

The evidence_page field is optional.

Usage:
    python evaluate.py path/to/qa_pairs.jsonl

Required environment variables:
    JINA_API_KEY
    GROQ_API_KEY
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import time
from generation import generate_answer
from retrieval import Retriever

# Data models

@dataclass
class EvalSample:
    question: str
    expected_answer: str
    evidence_page: int | None = None


@dataclass
class EvalResult:
    sample: EvalSample
    generated_answer: str
    abstained: bool
    page_in_top_k: bool | None
    answer_correct: bool


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def abstention_rate(self) -> float:
        return sum(r.abstained for r in self.results) / self.n if self.n else 0

    @property
    def answer_accuracy(self) -> float:
        answered = [r for r in self.results if not r.abstained]
        if not answered:
            return 0.0
        return sum(r.answer_correct for r in answered) / len(answered)

    @property
    def hallucination_rate(self) -> float:
        return sum(
            not r.abstained and not r.answer_correct for r in self.results
        ) / self.n if self.n else 0

    @property
    def page_recall(self) -> float | None:
        with_page = [r for r in self.results if r.page_in_top_k is not None]
        if not with_page:
            return None
        return sum(r.page_in_top_k for r in with_page) / len(with_page)

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print(f"EVALUATION REPORT  ({self.n} questions)")
        print("=" * 60)
        print(f"  Answer accuracy (when not abstained): {self.answer_accuracy:.1%}")
        print(f"  Abstention rate:                      {self.abstention_rate:.1%}")
        print(f"  Hallucination rate:                   {self.hallucination_rate:.1%}")
        if self.page_recall is not None:
            print(f"  Page recall@k:                        {self.page_recall:.1%}")
        print("=" * 60)

        failures = [r for r in self.results if not r.answer_correct and not r.abstained]
        if failures:
            print(f"\nFailed answers ({len(failures)}):")
            for r in failures[:10]:
                print(f"\n  Q: {r.sample.question}")
                print(f"  Expected: {r.sample.expected_answer}")
                print(f"  Got:      {r.generated_answer[:200]}")


# Answer matching

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\.\-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,\.]*%?", text))


def _answer_correct(generated: str, expected: str) -> bool:
    gen_norm = _normalise(generated)
    exp_norm = _normalise(expected)

    expected_numbers = _extract_numbers(exp_norm)
    if expected_numbers:
        return expected_numbers.issubset(_extract_numbers(gen_norm))

    return exp_norm in gen_norm


# Evaluation loop

def evaluate(qa_path: str | Path, retriever: Retriever) -> EvalReport:
    qa_path = Path(qa_path)
    samples: list[EvalSample] = []

    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(EvalSample(
                question=obj["question"],
                expected_answer=str(obj["answer"]),
                evidence_page=obj.get("evidence_page"),
            ))

    print(f"Evaluating {len(samples)} questions...")
    report = EvalReport()

    for i, sample in enumerate(samples, 1):
        time.sleep(2) # prevent overloading of groqg
        print(f"  [{i}/{len(samples)}] {sample.question[:60]}...")

        retrieved = retriever.retrieve(sample.question)
        answer = generate_answer(sample.question, retrieved)

        page_in_top_k: bool | None = None
        if sample.evidence_page is not None:
            retrieved_pages = {
                p
                for rc in retrieved
                for p in range(rc.chunk.page_start, rc.chunk.page_end + 1)
            }
            page_in_top_k = sample.evidence_page in retrieved_pages

        correct = _answer_correct(answer.answer_text, sample.expected_answer)

        report.results.append(EvalResult(
            sample=sample,
            generated_answer=answer.answer_text,
            abstained=answer.abstained,
            page_in_top_k=page_in_top_k,
            answer_correct=correct,
        ))

    return report


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("qa_file", help="Path to .jsonl file with Q&A pairs")
    args = parser.parse_args()

    retriever = Retriever()
    report = evaluate(args.qa_file, retriever)
    report.print_report()

    output_path = Path("eval_results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in report.results:
            f.write(json.dumps({
                "question": r.sample.question,
                "expected": r.sample.expected_answer,
                "generated": r.generated_answer,
                "abstained": r.abstained,
                "correct": r.answer_correct,
                "page_in_top_k": r.page_in_top_k,
            }) + "\n")
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
