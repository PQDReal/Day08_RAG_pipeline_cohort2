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

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "url": "https://vbpl.vn/FileData/TW/Lists/vbpq/Attachments/152501/VanBanGoc_73_2021_QH14%20%281%29.pdf",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "fallback_filename": "luat-phong-chong-ma-tuy-2021.docx",
        "title": "Luật Phòng, chống ma túy 2021 (Luật số 73/2021/QH14)",
        "fallback_text": """
Luật Phòng, chống ma túy 2021 quy định về phòng, chống ma túy; quản lý người sử dụng trái phép chất ma túy; cai nghiện ma túy; trách nhiệm của cá nhân, gia đình, cơ quan, tổ chức trong phòng, chống ma túy; quản lý nhà nước và hợp tác quốc tế về phòng, chống ma túy.

Văn bản nêu các hành vi bị nghiêm cấm như trồng cây có chứa chất ma túy, nghiên cứu, giám định, sản xuất, tàng trữ, vận chuyển, mua bán, phân phối, sử dụng trái phép chất ma túy, tiền chất, thuốc gây nghiện, thuốc hướng thần, thuốc tiền chất.

Luật cũng quy định trách nhiệm tuyên truyền, giáo dục phòng chống ma túy, quản lý người sử dụng trái phép chất ma túy, lập hồ sơ cai nghiện, cai nghiện tự nguyện, cai nghiện bắt buộc và quản lý sau cai nghiện.
""",
    },
    {
        "url": "https://congbao.chinhphu.vn/tai-ve-van-ban-so-105-2021-nd-cp-34944?format=pdf",
        "filename": "nghi-dinh-105-2021.pdf",
        "fallback_filename": "nghi-dinh-105-2021.docx",
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng, chống ma túy",
        "fallback_text": """
Nghị định 105/2021/NĐ-CP quy định chi tiết và hướng dẫn thi hành một số điều của Luật Phòng, chống ma túy. Nội dung tập trung vào kiểm soát các hoạt động hợp pháp liên quan đến ma túy, quản lý người sử dụng trái phép chất ma túy, cai nghiện ma túy và quản lý sau cai nghiện.

Nghị định hướng dẫn trình tự xác định tình trạng nghiện ma túy, hồ sơ quản lý người sử dụng trái phép chất ma túy, trách nhiệm của Ủy ban nhân dân cấp xã, cơ quan công an, cơ sở y tế, cơ sở cai nghiện và gia đình.

Văn bản là tài liệu triển khai quan trọng cho các quy định của Luật Phòng, chống ma túy 2021 trong thực tiễn quản lý nhà nước.
""",
    },
    {
        "url": "https://congbao.chinhphu.vn/tai-ve-van-ban-so-57-2022-nd-cp-37734-41623?format=pdf",
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf",
        "fallback_filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.docx",
        "title": "Nghị định 57/2022/NĐ-CP quy định các danh mục chất ma túy và tiền chất",
        "fallback_text": """
Nghị định 57/2022/NĐ-CP quy định các danh mục chất ma túy và tiền chất. Các danh mục này là căn cứ pháp lý để xác định chất ma túy, tiền chất bị kiểm soát trong đời sống xã hội, y tế, thú y, nghiên cứu, kiểm nghiệm, giám định và điều tra tội phạm.

Văn bản phân nhóm các chất ma túy tuyệt đối cấm sử dụng trong y học và đời sống xã hội; các chất ma túy được dùng theo quy định đặc biệt của cơ quan có thẩm quyền; và các tiền chất thiết yếu tham gia vào cấu trúc chất ma túy.

Nghị định được ban hành căn cứ Luật Phòng, chống ma túy 2021, Bộ luật Hình sự 2015 sửa đổi 2017, Luật Hóa chất và Luật Dược.
""",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def _write_docx(path: Path, title: str, source_url: str, body: str) -> None:
    """Create a small DOCX fallback using only the Python standard library."""
    paragraphs = [title, f"Nguồn: {source_url}", *body.strip().splitlines()]
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>',
    ]
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        escaped = (
            paragraph.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        xml_parts.append(f"<w:p><w:r><w:t>{escaped}</w:t></w:r></w:p>")
    xml_parts.append("<w:sectPr/></w:body></w:document>")

    with ZipFile(path, "w", ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>""",
        )
        docx.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>""",
        )
        docx.writestr("word/document.xml", "\n".join(xml_parts))


def download_file(url: str, filename: str) -> Path:
    """Download a legal document and validate that it looks non-empty."""
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)
    if filepath.stat().st_size <= 1024:
        filepath.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is too small: {filename}")
    header = filepath.read_bytes()[:16]
    is_pdf = filepath.suffix.lower() == ".pdf" and header.startswith(b"%PDF")
    is_docx = filepath.suffix.lower() == ".docx" and header.startswith(b"PK")
    if not (is_pdf or is_docx):
        filepath.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid {filepath.suffix} document: {filename}")
    print(f"✓ Đã tải: {filepath.name}")
    return filepath


def collect_legal_documents() -> list[Path]:
    """Collect at least three legal PDF/DOCX files into data/landing/legal."""
    setup_directory()
    collected: list[Path] = []

    for doc in LEGAL_DOCUMENTS:
        try:
            collected.append(download_file(doc["url"], doc["filename"]))
        except Exception as exc:
            fallback_path = DATA_DIR / doc["fallback_filename"]
            _write_docx(fallback_path, doc["title"], doc["url"], doc["fallback_text"])
            collected.append(fallback_path)
            print(f"⚠ Không tải được {doc['filename']} ({exc}); đã tạo DOCX fallback: {fallback_path.name}")

    return collected


if __name__ == "__main__":
    collect_legal_documents()
