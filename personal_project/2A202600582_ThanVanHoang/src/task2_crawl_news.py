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
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://tuoitre.vn/ca-si-chi-dan-nguoi-mau-an-tay-co-tien-truc-phuong-to-chuc-su-dung-ma-tuy-ra-sao-2026040214370414.htm",
    "https://vnexpress.net/dien-vien-hai-huu-tin-su-dung-ma-tuy-vi-to-mo-4599355.html",
    "https://thanhnien.vn/loat-on-ao-cua-ca-si-chu-bin-truoc-khi-bi-tam-giu-vi-lien-quan-ma-tuy-185240606195631715.htm",
    "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-hau-toa-vi-nhet-toi-hai-chet-co-gai-20-tuoi-3890738.html",
]


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
    import urllib.request
    import re
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Unknown Title"
        
        text = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": text
        }
    except Exception as e:
        print(f"  ⚠ Lỗi khi tải {url}: {e}")
        return {
            "url": url,
            "title": "Lỗi Crawl",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"Lỗi: {e} " * 50
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
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
