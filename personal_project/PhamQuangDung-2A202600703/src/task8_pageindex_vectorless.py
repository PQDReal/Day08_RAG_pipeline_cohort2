"""
Task 8 — PageIndex Vectorless RAG.

When PAGEINDEX_API_KEY is available, this module uses the real PageIndex API:
    1. Build one PDF corpus from standardized markdown files.
    2. Upload the PDF with PageIndexClient.submit_document().
    3. Submit retrieval queries and poll for results.

If the cloud API is not configured or a document is still processing, it falls
back to a local vectorless BM25 section index so the pipeline remains runnable.
"""

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from textwrap import wrap

from dotenv import load_dotenv

try:
    from pageindex import PageIndexAPIError, PageIndexClient
except ImportError:
    PageIndexAPIError = Exception
    PageIndexClient = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from src.task6_lexical_search import tokenize
except ModuleNotFoundError:
    from task6_lexical_search import tokenize

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
PAGEINDEX_LOCAL_FILE = INDEX_DIR / "pageindex_local_sections.jsonl"
PAGEINDEX_DOCS_FILE = INDEX_DIR / "pageindex_docs.json"
PAGEINDEX_UPLOAD_PDF = INDEX_DIR / "pageindex_corpus.pdf"


def _client():
    if not PAGEINDEX_API_KEY or PageIndexClient is None:
        return None
    return PageIndexClient(PAGEINDEX_API_KEY)


def _markdown_corpus() -> list[tuple[Path, str]]:
    return [
        (md_file, md_file.read_text(encoding="utf-8"))
        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md"))
    ]


def _write_pdf_from_markdown(pdf_path: Path) -> Path:
    """Create a simple PDF corpus because PageIndex SDK uploads PDF files."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin = 48
    y = height - margin
    font_name = "Helvetica"
    font_size = 10
    line_height = 14
    max_width = width - 2 * margin

    def draw_line(line: str) -> None:
        nonlocal y
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        if y < margin:
            c.showPage()
            c.setFont(font_name, font_size)
            y = height - margin
        c.drawString(margin, y, safe_line)
        y -= line_height

    c.setFont(font_name, font_size)
    for md_file, content in _markdown_corpus():
        draw_line(f"===== SOURCE: {md_file.relative_to(STANDARDIZED_DIR)} =====")
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                y -= line_height
                continue

            current = ""
            for word in line.split():
                candidate = word if not current else f"{current} {word}"
                if stringWidth(candidate, font_name, font_size) <= max_width:
                    current = candidate
                else:
                    draw_line(current)
                    current = word
            if current:
                for part in wrap(current, width=110) or [""]:
                    draw_line(part)
        c.showPage()
        c.setFont(font_name, font_size)
        y = height - margin

    c.save()
    return pdf_path


def _save_doc_id(doc_id: str) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_DOCS_FILE.write_text(
        json.dumps({"doc_id": doc_id, "pdf": str(PAGEINDEX_UPLOAD_PDF)}, indent=2),
        encoding="utf-8",
    )


def _load_doc_id() -> str | None:
    if not PAGEINDEX_DOCS_FILE.exists():
        return None
    data = json.loads(PAGEINDEX_DOCS_FILE.read_text(encoding="utf-8"))
    return data.get("doc_id")


def _split_sections(content: str) -> list[str]:
    sections = []
    buffer = []
    for line in content.splitlines():
        is_heading = line.lstrip().startswith("#")
        if is_heading and buffer:
            section = "\n".join(buffer).strip()
            if section:
                sections.append(section)
            buffer = [line]
        else:
            buffer.append(line)

    final_section = "\n".join(buffer).strip()
    if final_section:
        sections.append(final_section)

    paragraphs = []
    for section in sections:
        if len(section) <= 900:
            paragraphs.append(section)
        else:
            paragraphs.extend(part.strip() for part in section.split("\n\n") if part.strip())
    return [section for section in paragraphs if len(section) > 40]


def _load_local_sections() -> list[dict]:
    sections = []
    for md_file, content in _markdown_corpus():
        doc_type = "legal" if "legal" in md_file.parts else "news"
        for section_index, section in enumerate(_split_sections(content)):
            sections.append({
                "content": section,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                    "type": doc_type,
                    "section_index": section_index,
                },
            })
    return sections


def _prepare_local_pageindex() -> Path:
    sections = _load_local_sections()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with PAGEINDEX_LOCAL_FILE.open("w", encoding="utf-8") as f:
        for section in sections:
            f.write(json.dumps(section, ensure_ascii=False) + "\n")
    return PAGEINDEX_LOCAL_FILE


def upload_documents() -> dict:
    """
    Upload corpus to PageIndex when an API key is configured.

    Returns {'mode': 'pageindex', 'doc_id': ...} on cloud upload, otherwise
    {'mode': 'local', 'path': ...}.
    """
    client = _client()
    if client is None:
        path = _prepare_local_pageindex()
        print(f"PAGEINDEX_API_KEY not set; prepared local fallback: {path}")
        return {"mode": "local", "path": str(path)}

    existing_doc_id = _load_doc_id()
    if existing_doc_id:
        try:
            metadata = client.get_document(existing_doc_id)
            print(f"✓ Reusing PageIndex document: {existing_doc_id} ({metadata.get('status', 'unknown')})")
            return {"mode": "pageindex", "doc_id": existing_doc_id, "metadata": metadata}
        except PageIndexAPIError:
            pass

    pdf_path = _write_pdf_from_markdown(PAGEINDEX_UPLOAD_PDF)
    response = client.submit_document(str(pdf_path))
    doc_id = response.get("doc_id") or response.get("id")
    if not doc_id:
        raise RuntimeError(f"PageIndex upload did not return a document id: {response}")

    _save_doc_id(doc_id)
    _prepare_local_pageindex()
    print(f"✓ Uploaded corpus to PageIndex: {doc_id}")
    return {"mode": "pageindex", "doc_id": doc_id, "response": response}


@lru_cache(maxsize=1)
def _get_local_pageindex_state():
    if not PAGEINDEX_LOCAL_FILE.exists():
        _prepare_local_pageindex()

    sections = []
    with PAGEINDEX_LOCAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sections.append(json.loads(line))

    if BM25Okapi is None:
        raise ImportError("rank-bm25 is required. Install with: pip install rank-bm25")
    return sections, BM25Okapi([tokenize(section["content"]) for section in sections])


def _local_pageindex_search(query: str, top_k: int) -> list[dict]:
    sections, bm25 = _get_local_pageindex_state()
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results = []
    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append({
            "content": sections[idx]["content"],
            "score": round(score, 6),
            "metadata": sections[idx]["metadata"],
            "source": "pageindex",
        })
    return results


def _extract_pageindex_results(payload: dict, top_k: int) -> list[dict]:
    if payload.get("retrieved_nodes"):
        results = []
        for node_rank, node in enumerate(payload["retrieved_nodes"], 1):
            relevant_groups = node.get("relevant_contents", [])
            for group in relevant_groups:
                for item in group:
                    content = item.get("relevant_content", "")
                    if not content:
                        continue
                    results.append({
                        "content": content,
                        "score": round(1.0 / node_rank, 6),
                        "metadata": {
                            "node_id": node.get("id"),
                            "title": node.get("title"),
                            "section_title": item.get("section_title"),
                            "physical_index": item.get("physical_index"),
                        },
                        "source": "pageindex",
                    })
                    if len(results) >= top_k:
                        return results
        return results

    raw_results = (
        payload.get("results")
        or payload.get("retrieval_result")
        or payload.get("contexts")
        or payload.get("chunks")
        or []
    )
    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", [])

    results = []
    for item in raw_results[:top_k]:
        if isinstance(item, str):
            content = item
            score = 1.0
            metadata = {}
        else:
            content = (
                item.get("text")
                or item.get("content")
                or item.get("markdown")
                or item.get("answer")
                or json.dumps(item, ensure_ascii=False)
            )
            score = float(item.get("score", item.get("relevance_score", 1.0)))
            metadata = item.get("metadata", {})
        results.append({
            "content": content,
            "score": round(score, 6),
            "metadata": metadata,
            "source": "pageindex",
        })
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex API, with local vectorless fallback.
    """
    if top_k <= 0:
        return []

    client = _client()
    doc_id = _load_doc_id()
    if client is None:
        return _local_pageindex_search(query, top_k)

    if not doc_id:
        upload_info = upload_documents()
        doc_id = upload_info.get("doc_id")

    if doc_id:
        try:
            retrieval = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = retrieval.get("retrieval_id") or retrieval.get("id")
            if retrieval_id:
                for _ in range(12):
                    payload = client.get_retrieval(retrieval_id)
                    status = str(payload.get("status", "")).lower()
                    results = _extract_pageindex_results(payload, top_k)
                    if results:
                        return results
                    if status in {"failed", "error"}:
                        break
                    time.sleep(5)
        except Exception as exc:
            print(f"⚠ PageIndex API query failed; using local fallback ({exc})")

    return _local_pageindex_search(query, top_k)


if __name__ == "__main__":
    info = upload_documents()
    print(f"Upload/index mode: {info.get('mode')}")

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        source = r.get("metadata", {}).get("source", "pageindex")
        print(f"[{r['score']:.3f}] {source} | {r['content'][:100]}...")
