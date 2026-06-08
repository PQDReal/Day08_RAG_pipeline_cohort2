"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


LEGAL_DOCUMENTS = [
    {
        "title": "Luật Phòng, chống ma túy 2021",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "source_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/06/73.signed.pdf",
    },
    {
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng, chống ma túy",
        "filename": "nghi-dinh-105.pdf",
        "source_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed.pdf",
    },
    {
        "title": "Nghị định 57/2022/NĐ-CP về danh mục chất ma túy và tiền chất",
        "filename": "nghi-dinh-57.pdf",
        "source_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/08/57-cp.signed.pdf",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str, overwrite: bool = False) -> Path:
    """Download một văn bản pháp luật về DATA_DIR."""
    setup_directory()
    filepath = DATA_DIR / filename
    if filepath.exists() and not overwrite:
        print(f"✓ Đã có sẵn: {filepath.name}")
        return filepath

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    if len(response.content) < 1024:
        raise ValueError(f"File tải về quá nhỏ, có thể URL lỗi: {url}")

    filepath.write_bytes(response.content)
    print(f"✓ Đã tải: {filepath}")
    return filepath


def collect_legal_docs(overwrite: bool = False) -> list[Path]:
    """Tải/kiểm tra tối thiểu 3 văn bản pháp luật theo manifest."""
    downloaded = []
    for doc in LEGAL_DOCUMENTS:
        downloaded.append(
            download_file(
                doc["source_url"],
                doc["filename"],
                overwrite=overwrite,
            )
        )
    return downloaded


def list_existing_legal_docs() -> list[Path]:
    """Liệt kê PDF/DOC/DOCX đang có trong data/landing/legal."""
    setup_directory()
    valid_extensions = {".pdf", ".doc", ".docx"}
    return sorted(
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    )


def validate_collection(min_files: int = 3) -> bool:
    """Kiểm tra collection đạt yêu cầu bài: >=3 file và mỗi file >1KB."""
    files = list_existing_legal_docs()
    valid_files = [f for f in files if f.stat().st_size > 1024]
    print(f"✓ Có {len(valid_files)} file legal hợp lệ: {[f.name for f in valid_files]}")
    return len(valid_files) >= min_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1: collect legal documents.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Tải lại legal docs từ source_url. Mặc định chỉ dùng file đã có trong data/landing/legal.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè file đã có khi dùng --download.",
    )
    args = parser.parse_args()

    if args.download:
        collect_legal_docs(overwrite=args.overwrite)

    if not validate_collection():
        raise SystemExit("Chưa đủ tối thiểu 3 văn bản pháp luật hợp lệ.")
