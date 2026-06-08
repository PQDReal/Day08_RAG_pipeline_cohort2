"""
RAG Evaluation Pipeline.

This script uses a lightweight offline evaluator so the group can run scoring
without paid judge-model calls. The four reported metrics mirror common RAGAS /
DeepEval concepts:
    - Faithfulness: answer terms are supported by retrieved context
    - Answer Relevance: answer overlaps with the user question
    - Context Recall: expected evidence appears in retrieved contexts
    - Context Precision: retrieved chunks contain expected evidence/answer terms

It also runs an A/B comparison:
    Config A: hybrid retrieval + reranking
    Config B: hybrid retrieval without reranking
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import reorder_for_llm

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RESULTS_JSON_PATH = Path(__file__).parent / "results.json"


@dataclass
class EvalConfig:
    name: str
    use_reranking: bool
    top_k: int = 5


CONFIGS = [
    EvalConfig(name="Config A - hybrid + rerank", use_reranking=True),
    EvalConfig(name="Config B - hybrid no rerank", use_reranking=False),
]


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
    stopwords = {
        "là", "và", "của", "có", "cho", "theo", "về", "trong", "nào",
        "những", "các", "một", "đến", "từ", "bị", "được", "này", "gì",
    }
    return {token for token in tokens if token not in stopwords and len(token) > 1}


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 25]


def citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    return metadata.get("source") or metadata.get("title") or f"Source {index}"


def extractive_answer(question: str, contexts: list[dict]) -> str:
    """Generate a deterministic answer from retrieved contexts for evaluation."""
    query_terms = tokenize(question)
    if not query_terms or not contexts:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    scored = []
    for index, chunk in enumerate(reorder_for_llm(contexts), 1):
        label = citation_label(chunk, index)
        for sentence in split_sentences(chunk.get("content", "")):
            overlap = len(query_terms & tokenize(sentence))
            if overlap:
                scored.append((overlap, sentence, label))

    if not scored:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence, label in scored:
        key = sentence.lower()[:100]
        if key in seen:
            continue
        seen.add(key)
        selected.append(f"{sentence} [{label}]")
        if len(selected) == 3:
            break
    return "\n".join(selected)


def overlap_ratio(left: str, right: str) -> float:
    left_terms = tokenize(left)
    right_terms = tokenize(right)
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def metric_faithfulness(answer: str, contexts: list[dict]) -> float:
    context_text = " ".join(chunk.get("content", "") for chunk in contexts)
    return round(overlap_ratio(answer, context_text), 3)


def metric_answer_relevance(question: str, answer: str) -> float:
    return round(overlap_ratio(question, answer), 3)


def metric_context_recall(expected_context: str, contexts: list[dict]) -> float:
    context_text = " ".join(
        f"{chunk.get('content', '')} {json.dumps(chunk.get('metadata', {}), ensure_ascii=False)}"
        for chunk in contexts
    )
    return round(overlap_ratio(expected_context, context_text), 3)


def metric_context_precision(expected_answer: str, expected_context: str, contexts: list[dict]) -> float:
    if not contexts:
        return 0.0
    useful = 0
    expected_terms = tokenize(expected_answer) | tokenize(expected_context)
    for chunk in contexts:
        chunk_terms = tokenize(chunk.get("content", "")) | tokenize(json.dumps(chunk.get("metadata", {}), ensure_ascii=False))
        if expected_terms & chunk_terms:
            useful += 1
    return round(useful / len(contexts), 3)


def run_config(config: EvalConfig, dataset: list[dict]) -> dict:
    cases = []
    for item in dataset:
        contexts = retrieve(
            item["question"],
            top_k=config.top_k,
            score_threshold=-1.0,
            use_reranking=config.use_reranking,
        )
        answer = extractive_answer(item["question"], contexts)
        scores = {
            "faithfulness": metric_faithfulness(answer, contexts),
            "answer_relevance": metric_answer_relevance(item["question"], answer),
            "context_recall": metric_context_recall(item["expected_context"], contexts),
            "context_precision": metric_context_precision(item["expected_answer"], item["expected_context"], contexts),
        }
        scores["average"] = round(statistics.mean(scores.values()), 3)
        cases.append({
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "expected_context": item["expected_context"],
            "answer": answer,
            "sources": [citation_label(chunk, i + 1) for i, chunk in enumerate(contexts)],
            "scores": scores,
        })

    summary = {}
    for metric in ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average"]:
        summary[metric] = round(statistics.mean(case["scores"][metric] for case in cases), 3)

    return {"config": config.name, "summary": summary, "cases": cases}


def compare_configs(dataset: list[dict]) -> dict:
    return {config.name: run_config(config, dataset) for config in CONFIGS}


def worst_performers(result: dict, limit: int = 3) -> list[dict]:
    cases = result["cases"]
    return sorted(cases, key=lambda case: case["scores"]["average"])[:limit]


def export_results(results: dict) -> None:
    config_names = list(results.keys())
    config_a = results[config_names[0]]
    config_b = results[config_names[1]]

    def delta(metric: str) -> float:
        return round(config_a["summary"][metric] - config_b["summary"][metric], 3)

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "Custom offline evaluator inspired by RAGAS/DeepEval metrics. This avoids judge-model API requirements while still reporting Faithfulness, Answer Relevance, Context Recall, and Context Precision.",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Δ |",
        "|--------|-----------------------------|------------------------------|---|",
    ]

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "**Average**",
    }
    for metric, label in metric_labels.items():
        lines.append(
            f"| {label} | {config_a['summary'][metric]:.3f} | "
            f"{config_b['summary'][metric]:.3f} | {delta(metric):+.3f} |"
        )

    lines.extend([
        "",
        "## A/B Comparison Analysis",
        "",
        "**Config A:** Hybrid retrieval combines semantic and BM25 results with RRF, then reranks candidates.",
        "",
        "**Config B:** Hybrid retrieval combines semantic and BM25 results with RRF but skips reranking.",
        "",
    ])

    winner = "Config A" if config_a["summary"]["average"] >= config_b["summary"]["average"] else "Config B"
    lines.extend([
        f"**Kết luận:** {winner} có điểm trung bình cao hơn trong bộ đánh giá hiện tại. Điểm này nên được xem là tín hiệu tương đối vì evaluator dùng token overlap, không phải LLM judge.",
        "",
        "## Worst Performers (Bottom 3 - Config A)",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |",
        "|---|----------|--------------|-----------|--------|---------------|------------|",
    ])

    for index, case in enumerate(worst_performers(config_a), 1):
        scores = case["scores"]
        question = case["question"].replace("|", "\\|")
        failure_stage = "retrieval" if scores["context_recall"] < 0.5 else "generation"
        root_cause = "Expected evidence not ranked in top contexts" if failure_stage == "retrieval" else "Extractive answer lacks enough overlap"
        lines.append(
            f"| {index} | {question} | {scores['faithfulness']:.3f} | "
            f"{scores['answer_relevance']:.3f} | {scores['context_recall']:.3f} | "
            f"{failure_stage} | {root_cause} |"
        )

    lines.extend([
        "",
        "## Recommendations",
        "",
        "### Cải tiến 1",
        "**Action:** Bổ sung văn bản pháp luật đầy đủ hơn, đặc biệt Bộ luật Hình sự Điều 249 và bản PDF/DOCX gốc có nội dung đầy đủ.",
        "**Expected impact:** Tăng context recall cho câu hỏi về hình phạt và điều luật cụ thể.",
        "",
        "### Cải tiến 2",
        "**Action:** Làm sạch news crawl để loại bỏ menu, footer, quảng cáo và text điều hướng.",
        "**Expected impact:** Tăng context precision và giảm nhiễu trong câu trả lời.",
        "",
        "### Cải tiến 3",
        "**Action:** Dùng LLM judge hoặc RAGAS/DeepEval thật khi có API key để chấm ngữ nghĩa thay vì token overlap.",
        "**Expected impact:** Điểm faithfulness/relevance phản ánh chất lượng câu trả lời tự nhiên tốt hơn.",
        "",
    ])

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    RESULTS_JSON_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    dataset = load_golden_dataset()
    if len(dataset) < 15:
        raise ValueError(f"Golden dataset must have at least 15 items; got {len(dataset)}")
    results = compare_configs(dataset)
    export_results(results)
    print(f"Loaded {len(dataset)} test cases")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
