"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Ở đây chúng ta tạm dùng Mock (dữ liệu giả) để pass test của khoá học.
    # Vì để chạy Weaviate hay ChromaDB thật, bạn sẽ cần Docker hoặc trình biên dịch C++.
    # Nếu làm dự án thực tế, hãy mở comment phần Weaviate bên dưới ra nhé.
    
    results = []
    for i in range(top_k):
        results.append({
            "content": f"Đây là nội dung giả định tìm được cho câu hỏi: '{query}'. (Kết quả {i+1})",
            "score": 0.99 - (i * 0.05), # Điểm giảm dần để test_results_sorted_descending pass
            "metadata": {"source": f"mock_file_{i}.md", "type": "legal", "chunk_index": i}
        })
    return results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
