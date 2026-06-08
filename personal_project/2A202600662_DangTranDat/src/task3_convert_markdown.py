import json
from pathlib import Path
from datetime import datetime

from markitdown import MarkItDown


LANDING_DIR = Path("data/landing")
STANDARDIZED_DIR = Path("data/standardized")

SUPPORTED_MARKITDOWN_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".html",
    ".htm",
    ".txt",
}


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_markdown_metadata(metadata: dict) -> str:
    """
    Tạo metadata block ở đầu file .md.
    Đây là phần rất quan trọng để Task 4/10 còn biết source/citation.
    """
    lines = ["---"]

    for key, value in metadata.items():
        if value is None:
            value = ""
        value = str(value).replace("\n", " ").strip()
        lines.append(f'{key}: "{value}"')

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def convert_json_news_to_markdown(input_path: Path, output_path: Path) -> None:
    """
    Convert file JSON bài báo đã crawl ở Task 2 sang Markdown.
    JSON có sẵn content_markdown nên không cần MarkItDown.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        article = json.load(f)

    title = article.get("title", input_path.stem)
    url = article.get("url", "")
    crawl_date = article.get("crawl_date", "")
    crawl_datetime = article.get("crawl_datetime", "")
    source_type = article.get("source_type", "news")
    content = article.get("content_markdown", "")

    metadata = {
        "title": title,
        "source_type": source_type,
        "url": url,
        "crawl_date": crawl_date,
        "crawl_datetime": crawl_datetime,
        "original_file": str(input_path),
        "converted_at": datetime.now().isoformat(timespec="seconds"),
    }

    markdown_text = make_markdown_metadata(metadata)
    markdown_text += f"# {title}\n\n"
    markdown_text += f"**Nguồn:** {url}\n\n"
    markdown_text += f"**Ngày crawl:** {crawl_date}\n\n"
    markdown_text += "---\n\n"
    markdown_text += content.strip()
    markdown_text += "\n"

    ensure_output_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)


def convert_with_markitdown(input_path: Path, output_path: Path) -> None:
    """
    Convert PDF/DOCX/HTML/TXT sang Markdown bằng MarkItDown.
    """
    md = MarkItDown()
    result = md.convert(str(input_path))
    text_content = result.text_content or ""

    relative_parts = input_path.relative_to(LANDING_DIR).parts
    source_type = relative_parts[0] if len(relative_parts) > 1 else "unknown"

    metadata = {
        "title": input_path.stem,
        "source_type": source_type,
        "original_file": str(input_path),
        "converted_at": datetime.now().isoformat(timespec="seconds"),
    }

    markdown_text = make_markdown_metadata(metadata)
    markdown_text += f"# {input_path.stem}\n\n"
    markdown_text += text_content.strip()
    markdown_text += "\n"

    ensure_output_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)


def get_output_path(input_path: Path) -> Path:
    """
    Giữ nguyên cấu trúc thư mục con:
    data/landing/news/a.json -> data/standardized/news/a.md
    data/landing/legal/a.pdf -> data/standardized/legal/a.md
    """
    relative_path = input_path.relative_to(LANDING_DIR)
    output_relative_path = relative_path.with_suffix(".md")
    return STANDARDIZED_DIR / output_relative_path


def convert_all() -> None:
    if not LANDING_DIR.exists():
        print(f"❌ Không tìm thấy thư mục {LANDING_DIR}")
        return

    converted_count = 0
    skipped_count = 0

    files = [p for p in LANDING_DIR.rglob("*") if p.is_file()]

    if not files:
        print(f"⚠ Không có file nào trong {LANDING_DIR}")
        return

    for input_path in files:
        output_path = get_output_path(input_path)
        suffix = input_path.suffix.lower()

        try:
            if suffix == ".json":
                convert_json_news_to_markdown(input_path, output_path)
                print(f"✅ JSON -> MD: {input_path} -> {output_path}")
                converted_count += 1

            elif suffix in SUPPORTED_MARKITDOWN_EXTENSIONS:
                convert_with_markitdown(input_path, output_path)
                print(f"✅ MarkItDown -> MD: {input_path} -> {output_path}")
                converted_count += 1

            else:
                print(f"⚠ Skip unsupported file: {input_path}")
                skipped_count += 1

        except Exception as e:
            print(f"❌ Failed converting: {input_path}")
            print(f"   Error: {e}")
            skipped_count += 1

    print("\nDone.")
    print(f"Converted: {converted_count}")
    print(f"Skipped/Failed: {skipped_count}")
    print(f"Output folder: {STANDARDIZED_DIR}")


if __name__ == "__main__":
    convert_all()