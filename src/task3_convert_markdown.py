"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_DIR = Path(__file__).parent.parent


def _markitdown():
    """Import MarkItDown lazily so JSON conversion still works without it."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None
    return MarkItDown()


def _write_markdown_strict(output_path: Path, content: str, source_path: Path) -> None:
    """Write markdown only when conversion produced real, non-empty content."""
    text = content.strip()
    if len(text) < 200:
        raise RuntimeError(
            f"Convert failed for {source_path.name}: output is too short "
            f"({len(text)} chars). This usually means the PDF is scanned/image-only "
            "and needs OCR before Markdown conversion."
        )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(output_path)


def _ocr_pdf_with_tesseract(pdf_path: Path) -> str:
    """
    OCR scanned/image PDFs with local Tesseract.

    This is not fallback content: it extracts actual text from rendered PDF pages.
    Requirements:
        pip install pytesseract Pillow pypdfium2
        Install Tesseract binary and ensure `tesseract` is on PATH.
    """
    tesseract_cmd = shutil.which("tesseract")
    default_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if tesseract_cmd is None and default_tesseract.exists():
        tesseract_cmd = str(default_tesseract)
    if tesseract_cmd is None:
        raise RuntimeError(
            "PDF appears scanned/image-only and Tesseract OCR is not installed/on PATH. "
            "Install Tesseract OCR, then rerun Task 3."
        )

    try:
        import pytesseract
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "Missing OCR Python packages. Run: "
            "python -m pip install pytesseract pypdfium2 Pillow"
        ) from exc

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    local_tessdata = PROJECT_DIR / "data"
    local_vie = local_tessdata / "vie.traineddata"
    if local_vie.exists() and not os.getenv("TESSDATA_PREFIX"):
        os.environ["TESSDATA_PREFIX"] = str(local_tessdata)
    ocr_lang = os.getenv("TESSERACT_LANG", "vie" if local_vie.exists() else "eng")

    pdf = pdfium.PdfDocument(str(pdf_path))
    pages_text = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = page.render(scale=2.0)
            image = bitmap.to_pil()
            image_path = tmp_path / f"page_{page_index + 1:04d}.png"
            image.save(image_path)
            page_text = pytesseract.image_to_string(
                image,
                lang=ocr_lang,
            ).strip()
            if page_text:
                pages_text.append(f"## Page {page_index + 1}\n\n{page_text}")

    text = f"# {pdf_path.stem}\n\n" + "\n\n".join(pages_text)
    return text.strip()


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = _markitdown()
    if md is None:
        raise RuntimeError(
            "MarkItDown chưa được cài. Chạy: "
            "python -m pip install 'markitdown[pdf]'"
        )

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            result = md.convert(str(filepath))
            text = result.text_content
            if len(text.strip()) < 200 and filepath.suffix.lower() == ".pdf":
                print("  MarkItDown output too short; trying OCR...")
                text = _ocr_pdf_with_tesseract(filepath)
            _write_markdown_strict(output_path, text, filepath)
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            content = header + (
                data.get("content_markdown")
                or data.get("markdown")
                or data.get("content")
                or data.get("text")
                or ""
            )
            _write_markdown_strict(output_path, content, filepath)
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
