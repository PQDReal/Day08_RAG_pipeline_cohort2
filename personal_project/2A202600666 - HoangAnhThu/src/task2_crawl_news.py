"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://tienphong.vn/truy-to-ca-si-chi-dan-nguoi-mau-an-tay-va-225-bi-can-trong-duong-day-ma-tuy-post1832551.tpo",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
    "https://vietnamnet.vn/su-nghiep-on-ao-cua-ca-si-long-nhat-truoc-khi-sup-do-vi-vuong-vao-ma-tuy-2517570.html",
    "https://eva.vn/lang-sao/lan-song-sao-viet-cong-khai-test-ma-tuy-minh-bach-hay-ap-luc-phai-tu-chung-minh-vo-toi-c20a672064.html",
]


def list_existing_news_files() -> list[Path]:
    """Liệt kê article files đang có trong data/landing/news."""
    setup_directory()
    valid_extensions = {".json", ".html", ".md", ".txt"}
    return sorted(
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions and not f.name.startswith(".")
    )


def validate_collection(min_files: int = 5) -> bool:
    """Kiểm tra data/landing/news đã đủ tối thiểu 5 bài có nội dung."""
    files = list_existing_news_files()
    valid_files = [f for f in files if f.stat().st_size > 500]
    print(f"✓ Có {len(valid_files)} file news hợp lệ: {[f.name for f in valid_files]}")
    return len(valid_files) >= min_files


def _fallback_title_from_url(url: str) -> str:
    slug = Path(urlparse(url).path).stem
    return slug.replace("-", " ").strip().title() or "Unknown"


def _extract_title_from_html(html: str, url: str) -> str:
    import re

    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if not match:
        return _fallback_title_from_url(url)
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title or _fallback_title_from_url(url)


def _html_to_markdownish(html: str) -> str:
    """Fallback extractor khi chưa cài Crawl4AI."""
    import re

    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    return re.sub(r"\s+", " ", text).strip()


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return {
                "url": url,
                "title": result.metadata.get("title", _fallback_title_from_url(url)),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": result.markdown or "",
            }
    except Exception:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 Day08-RAG-Crawler/1.0"},
        )
        response.raise_for_status()
        html = response.text
        return {
            "url": url,
            "title": _extract_title_from_html(html, url),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": _html_to_markdownish(html),
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: crawl or validate news articles.")
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Crawl lại ARTICLE_URLS. Mặc định chỉ dùng file đã có trong data/landing/news.",
    )
    args = parser.parse_args()

    if args.crawl:
        asyncio.run(crawl_all())
    elif not validate_collection():
        raise SystemExit("Chưa đủ tối thiểu 5 bài báo hợp lệ trong data/landing/news.")
