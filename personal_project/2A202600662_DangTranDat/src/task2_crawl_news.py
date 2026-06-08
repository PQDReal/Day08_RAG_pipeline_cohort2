import asyncio
import json
import re
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler


ARTICLE_URLS = [
    "https://vnexpress.net/nha-thiet-ke-nguyen-cong-tri-bi-bat-vi-lien-quan-ma-tuy-4917929.html",
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://baochinhphu.vn/khoi-to-le-anh-nhat-ca-si-miu-le-ve-hanh-vi-to-chuc-su-dung-trai-phep-chat-ma-tuy-102260516224626903.htm",
    "https://laodong.vn/phap-luat/toan-canh-vu-ca-si-chau-viet-cuong-nhet-toi-vao-mieng-ban-tinh-661444.ldo",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
]

OUTPUT_DIR = Path("data/landing/news")


def remove_vietnamese_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text


def slugify(text: str, max_length: int = 80) -> str:
    text = remove_vietnamese_accents(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text[:max_length].strip("-")
    return text or "article"


def get_markdown_from_result(result) -> str:
    """
    Crawl4AI mỗi version có thể trả markdown hơi khác nhau.
    Hàm này giúp lấy nội dung markdown an toàn hơn.
    """
    markdown = getattr(result, "markdown", "")

    if markdown is None:
        return ""

    if isinstance(markdown, str):
        return markdown

    for attr in ["raw_markdown", "fit_markdown", "markdown"]:
        value = getattr(markdown, attr, None)
        if isinstance(value, str):
            return value

    return str(markdown)


def extract_title(result, markdown: str, url: str) -> str:
    """
    Lấy tiêu đề bài báo từ metadata.
    Nếu metadata không có thì lấy heading đầu tiên trong markdown.
    Nếu vẫn không có thì dùng domain.
    """
    metadata = getattr(result, "metadata", {}) or {}

    title = metadata.get("title") or metadata.get("og:title")
    if title:
        return title.strip()

    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.replace("#", "").strip()

    domain = urlparse(url).netloc
    return f"Bài báo từ {domain}"


async def crawl_article(url: str, index: int) -> dict:
    """
    Crawl 1 bài báo và trả về dữ liệu dạng dict để lưu JSON.
    """
    print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    markdown = get_markdown_from_result(result)
    title = extract_title(result, markdown, url)

    now = datetime.now()
    domain = urlparse(url).netloc

    article = {
        "title": title,
        "url": url,
        "crawl_date": now.strftime("%Y-%m-%d"),
        "crawl_datetime": now.isoformat(timespec="seconds"),
        "source_type": "news",
        "content_format": "markdown",
        "content_markdown": markdown,
        "metadata": {
            "original_url": url,
            "title": title,
            "domain": domain,
            "crawler": "crawl4ai"
        }
    }

    return article


def save_article_json(article: dict, index: int) -> Path:
    """
    Lưu 1 bài báo thành 1 file JSON trong data/landing/news/
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    title = article.get("title", "article")
    url = article.get("url", "")

    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    file_name = f"news_{index:02d}_{slugify(title)}_{url_hash}.json"

    output_path = OUTPUT_DIR / file_name

    article["metadata"]["file_name"] = file_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)

    return output_path


async def crawl_all():
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for index, url in enumerate(ARTICLE_URLS, start=1):
        try:
            article = await crawl_article(url, index)
            output_path = save_article_json(article, index)

            content_length = len(article.get("content_markdown", ""))

            print(f"✅ Saved: {output_path}")
            print(f"   Title: {article['title']}")
            print(f"   Content length: {content_length} characters")

            success_count += 1

        except Exception as e:
            print(f"❌ Failed: {url}")
            print(f"   Error: {e}")

    print(f"\nDone. Saved {success_count}/{len(ARTICLE_URLS)} articles to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())