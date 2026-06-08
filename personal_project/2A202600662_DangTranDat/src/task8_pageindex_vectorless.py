"""
Task 8 — PageIndex Vectorless RAG

Requirements:
1. Upload documents to PageIndex.
2. Implement:
    def pageindex_search(query: str, top_k: int = 5) -> list[dict]
3. Return:
    [
        {
            "content": str,
            "score": float,
            "metadata": dict
        }
    ]

Design:
- PageIndex is used as a vectorless / agentic retrieval fallback.
- Official PageIndex SDK docs show uploading PDF via submit_document().
- This script uploads PDF files from data/landing/.
- Search uses PageIndex Chat API and normalizes the answer into list[dict].
"""

import json
import os
import re
import sys
import time
import textwrap
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pageindex import PageIndexClient


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

LANDING_DIR = Path("data/landing")
PAGEINDEX_DIR = Path("data/pageindex")
REGISTRY_PATH = PAGEINDEX_DIR / "pageindex_documents.json"
RESULTS_DIR = PAGEINDEX_DIR / "results"

DEFAULT_TOP_K = 5
DEFAULT_WAIT_SECONDS = 5
DEFAULT_MAX_ATTEMPTS = 60


# ============================================================
# CLIENT & FILE HELPERS
# ============================================================

def ensure_dirs() -> None:
    """Create PageIndex output folders if they do not exist."""
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_client() -> PageIndexClient:
    """
    Create PageIndex client from PAGEINDEX_API_KEY in .env.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError(
            "Missing PAGEINDEX_API_KEY. "
            "Hãy thêm PAGEINDEX_API_KEY vào file .env"
        )

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def collect_pdf_documents() -> list[Path]:
    """
    Collect PDF documents for PageIndex upload.

    Official PageIndex SDK docs demonstrate PDF upload, so this task
    implementation uses PDF files under data/landing/.
    """
    if not LANDING_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {LANDING_DIR}")

    pdf_files = sorted(LANDING_DIR.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "Không tìm thấy file PDF trong data/landing/. "
            "PageIndex SDK docs ưu tiên upload PDF, nên hãy đảm bảo có ít nhất 1 file PDF."
        )

    return pdf_files


def save_registry(records: list[dict]) -> None:
    """
    Save uploaded PageIndex document records to JSON.
    """
    ensure_dirs()

    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_registry() -> list[dict]:
    """
    Load PageIndex document registry.

    Returns:
        List of records, or [] if registry does not exist.
    """
    if not REGISTRY_PATH.exists():
        return []

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("records", [])


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

def wait_until_completed(
    client: PageIndexClient,
    doc_id: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleep_seconds: int = DEFAULT_WAIT_SECONDS,
) -> str:
    """
    Wait until PageIndex finishes processing a document.

    Returns:
        completed / failed / error / timeout / unknown
    """
    for attempt in range(1, max_attempts + 1):
        try:
            info = client.get_document(doc_id)
            status = info.get("status", "unknown")
        except Exception as error:
            print(f"  Could not get status for doc_id={doc_id}: {error}")
            status = "unknown"

        print(f"  Attempt {attempt}/{max_attempts}: doc_id={doc_id}, status={status}")

        if status in {"completed", "failed", "error"}:
            return status

        time.sleep(sleep_seconds)

    return "timeout"


def upload_documents(force: bool = False) -> list[dict]:
    """
    Upload PDF documents to PageIndex.

    Args:
        force:
            If False and registry exists, reuse previous upload records.
            If True, upload again.

    Returns:
        [
            {
                "source_path": str,
                "source_file": str,
                "doc_id": str,
                "status": str,
                "error": str | None
            }
        ]
    """
    ensure_dirs()

    existing_records = load_registry()

    if existing_records and not force:
        print(f"Found existing registry: {REGISTRY_PATH}")
        print(f"Reusing {len(existing_records)} uploaded document record(s).")
        return existing_records

    client = get_client()
    pdf_files = collect_pdf_documents()

    print(f"Uploading documents: {len(pdf_files)} PDF file(s) found.")

    records: list[dict] = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{index}/{len(pdf_files)}] Uploading: {pdf_path}")

        record = {
            "source_path": str(pdf_path),
            "source_file": pdf_path.name,
            "doc_id": None,
            "status": "unknown",
            "error": None,
        }

        try:
            result = client.submit_document(str(pdf_path))
            doc_id = result["doc_id"]

            print(f"  Uploaded. doc_id={doc_id}")

            status = wait_until_completed(client, doc_id)

            record["doc_id"] = doc_id
            record["status"] = status
            record["raw_submit_result"] = result

            print(f"  Final status: {status}")

        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            print(f"  Failed: {error}")

        records.append(record)
        save_registry(records)

    print(f"\nSaved registry to: {REGISTRY_PATH}")
    return records


def get_available_doc_ids() -> list[str]:
    """
    Get PageIndex doc_ids from registry.
    Prefer completed documents. If none are completed, use uploaded non-failed docs.
    """
    records = load_registry()

    completed_doc_ids = [
        record["doc_id"]
        for record in records
        if record.get("doc_id") and record.get("status") == "completed"
    ]

    if completed_doc_ids:
        return completed_doc_ids

    available_doc_ids = [
        record["doc_id"]
        for record in records
        if record.get("doc_id") and record.get("status") not in {"failed", "error"}
    ]

    return available_doc_ids


# ============================================================
# RESPONSE PARSING
# ============================================================

def remove_markdown_fences(text: str) -> str:
    """
    Remove common markdown code fences if PageIndex returns fenced JSON.
    """
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def parse_json_from_answer(raw_answer: str) -> list[dict]:
    """
    Parse PageIndex answer into a JSON list.

    The prompt asks PageIndex to return only JSON, but LLM responses can still
    contain code fences or extra text, so this function tries to recover JSON.
    """
    text = remove_markdown_fences(raw_answer)

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            for key in ["results", "items", "evidence", "passages"]:
                value = parsed.get(key)
                if isinstance(value, list):
                    return value

    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]

        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return []


def split_text_fallback(raw_answer: str, top_k: int) -> list[str]:
    """
    Fallback parser when PageIndex does not return valid JSON.

    It removes low-value blocks such as intro lines and separators.
    """
    blocks: list[str] = []

    for block in raw_answer.split("\n\n"):
        block = block.strip()

        if not block:
            continue

        lowered = block.lower()

        if block in {"---", "-", "--"}:
            continue

        if "dưới đây là" in lowered:
            continue

        if "json" == lowered:
            continue

        blocks.append(block)

    return blocks[:top_k]


def normalize_pageindex_answer(
    raw_answer: str,
    query: str,
    top_k: int,
    doc_ids: list[str],
) -> list[dict]:
    """
    Normalize PageIndex raw answer into Task 8 required format:
        [
            {
                "content": str,
                "score": float,
                "metadata": dict
            }
        ]
    """
    parsed_items = parse_json_from_answer(raw_answer)
    results: list[dict] = []

    if parsed_items:
        for rank, item in enumerate(parsed_items[:top_k], start=1):
            if not isinstance(item, dict):
                content = str(item).strip()
                source = "PageIndex Chat API"
                section = None
                raw_score = 1.0 / rank
            else:
                content = str(item.get("content", "")).strip()
                source = item.get("source", "PageIndex Chat API")
                section = item.get("section")
                raw_score = item.get("score", 1.0 / rank)

            if not content:
                continue

            try:
                score = float(raw_score)
            except Exception:
                score = float(1.0 / rank)

            results.append(
                {
                    "content": content,
                    "score": score,
                    "source": "pageindex",
                    "metadata": {
                        "retrieval_type": "pageindex_vectorless",
                        "rank": rank,
                        "query": query,
                        "doc_ids": doc_ids,
                        "source": source,
                        "section": section,
                        "parse_mode": "json",
                        "raw_answer": raw_answer if rank == 1 else None,
                    },
                }
            )

    if not results:
        text_blocks = split_text_fallback(raw_answer, top_k=top_k)

        for rank, block in enumerate(text_blocks, start=1):
            results.append(
                {
                    "content": block,
                    "score": float(1.0 / rank),
                    "source": "pageindex",
                    "metadata": {
                        "retrieval_type": "pageindex_vectorless",
                        "rank": rank,
                        "query": query,
                        "doc_ids": doc_ids,
                        "source": "PageIndex Chat API",
                        "section": None,
                        "parse_mode": "fallback_text_blocks",
                        "raw_answer": raw_answer if rank == 1 else None,
                    },
                }
            )

    results = sorted(results, key=lambda item: item["score"], reverse=True)

    return results[:top_k]


def save_last_search(
    query: str,
    top_k: int,
    doc_ids: list[str],
    raw_answer: str,
    results: list[dict],
) -> None:
    """
    Save the latest PageIndex search result for debugging/demo evidence.
    """
    ensure_dirs()

    result_path = RESULTS_DIR / "last_pageindex_search.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "query": query,
                "top_k": top_k,
                "doc_ids": doc_ids,
                "raw_answer": raw_answer,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# TASK 8 REQUIRED FUNCTION
# ============================================================

def pageindex_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Search uploaded PageIndex documents.

    Args:
        query:
            Query string.
        top_k:
            Number of evidence snippets to return.

    Returns:
        [
            {
                "content": str,
                "score": float,
                "metadata": dict
            }
        ]
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên lớn hơn 0.")

    client = get_client()
    doc_ids = get_available_doc_ids()

    if not doc_ids:
        raise RuntimeError(
            "Chưa có doc_id PageIndex. Hãy chạy upload trước:\n"
            "python src\\task8_pageindex_vectorless.py upload"
        )

    prompt = textwrap.dedent(
        f"""
        Bạn là retrieval engine.

        Nhiệm vụ:
        Tìm tối đa {top_k} đoạn bằng chứng liên quan nhất đến câu hỏi từ các tài liệu đã upload.

        Câu hỏi:
        {query}

        Yêu cầu bắt buộc:
        - Chỉ trả về JSON hợp lệ.
        - Không viết lời mở đầu.
        - Không dùng markdown.
        - Không dùng dấu ```json.
        - Không dùng dấu ---.
        - Không giải thích ngoài JSON.
        - Nội dung phải là đoạn bằng chứng trích từ tài liệu, không bịa ngoài tài liệu.

        Định dạng trả về chính xác:
        [
          {{
            "content": "đoạn bằng chứng liên quan, tự đủ ngữ cảnh",
            "score": 0.95,
            "source": "tên tài liệu hoặc nguồn nếu có",
            "section": "mục/điều nếu có"
          }}
        ]
        """
    ).strip()

    response = client.chat_completions(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        doc_id=doc_ids,
    )

    raw_answer = response["choices"][0]["message"]["content"]

    results = normalize_pageindex_answer(
        raw_answer=raw_answer,
        query=query,
        top_k=top_k,
        doc_ids=doc_ids,
    )

    save_last_search(
        query=query,
        top_k=top_k,
        doc_ids=doc_ids,
        raw_answer=raw_answer,
        results=results,
    )

    return results


# ============================================================
# CLI
# ============================================================

def print_results(results: list[dict]) -> None:
    for index, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        preview = item.get("content", "").replace("\n", " ")[:500]

        print(f"\n--- PageIndex Result {index} ---")
        print(f"Score: {item.get('score')}")
        print(f"Retrieval type: {metadata.get('retrieval_type')}")
        print(f"Rank: {metadata.get('rank')}")
        print(f"Source: {metadata.get('source')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Parse mode: {metadata.get('parse_mode')}")
        print(f"Preview: {preview}...")


def interactive_search() -> None:
    print("\n=== Task 8: PageIndex Vectorless Search ===")
    print("Gõ query để tìm kiếm bằng PageIndex.")
    print("Gõ 'q', 'quit' hoặc 'exit' để thoát.\n")

    while True:
        query = input("Nhập query: ").strip()

        if query.lower() in {"q", "quit", "exit"}:
            print("Thoát PageIndex search.")
            break

        if not query:
            print("Query không được rỗng.")
            continue

        top_k_input = input("Nhập top_k, mặc định 5: ").strip()

        try:
            top_k = int(top_k_input) if top_k_input else DEFAULT_TOP_K
        except ValueError:
            print("top_k không hợp lệ. Dùng mặc định top_k=5.")
            top_k = DEFAULT_TOP_K

        try:
            results = pageindex_search(query=query, top_k=top_k)
            print_results(results)
        except Exception as error:
            print(f"Lỗi PageIndex search: {error}")

        print("\n" + "-" * 80 + "\n")


def main() -> None:
    ensure_dirs()

    if len(sys.argv) >= 2:
        command = sys.argv[1].lower()

        if command == "upload":
            force = "--force" in sys.argv
            upload_documents(force=force)
            return

        if command == "search":
            if len(sys.argv) < 3:
                raise ValueError(
                    "Thiếu query. Ví dụ:\n"
                    "python src\\task8_pageindex_vectorless.py search \"tội phạm ma túy\" 5"
                )

            query = sys.argv[2]
            top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_TOP_K

            results = pageindex_search(query=query, top_k=top_k)
            print_results(results)
            return

    interactive_search()


if __name__ == "__main__":
    main()
