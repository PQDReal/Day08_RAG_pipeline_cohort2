# RAG Evaluation Results

## Framework sử dụng

Custom offline evaluator inspired by RAGAS/DeepEval metrics. This avoids judge-model API requirements while still reporting Faithfulness, Answer Relevance, Context Recall, and Context Precision.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Δ |
|--------|-----------------------------|------------------------------|---|
| Faithfulness | 0.936 | 0.938 | -0.002 |
| Answer Relevance | 0.743 | 0.715 | +0.028 |
| Context Recall | 0.933 | 0.933 | +0.000 |
| Context Precision | 1.000 | 1.000 | +0.000 |
| **Average** | 0.903 | 0.897 | +0.006 |

## A/B Comparison Analysis

**Config A:** Hybrid retrieval combines semantic and BM25 results with RRF, then reranks candidates.

**Config B:** Hybrid retrieval combines semantic and BM25 results with RRF but skips reranking.

**Kết luận:** Config A có điểm trung bình cao hơn trong bộ đánh giá hiện tại. Điểm này nên được xem là tín hiệu tương đối vì evaluator dùng token overlap, không phải LLM judge.

## Worst Performers (Bottom 3 - Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|--------------|-----------|--------|---------------|------------|
| 1 | Khi không đủ bằng chứng trong context, hệ thống RAG nên trả lời như thế nào? | 0.945 | 0.357 | 0.000 | retrieval | Expected evidence not ranked in top contexts |
| 2 | Luật Phòng chống ma túy 2021 nghiêm cấm những hành vi nào liên quan đến chất ma túy? | 0.875 | 0.692 | 1.000 | generation | Extractive answer lacks enough overlap |
| 3 | Luật Phòng chống ma túy 2021 quy định những nhóm nội dung chính nào? | 0.921 | 0.667 | 1.000 | generation | Extractive answer lacks enough overlap |

## Recommendations

### Cải tiến 1
**Action:** Bổ sung văn bản pháp luật đầy đủ hơn, đặc biệt Bộ luật Hình sự Điều 249 và bản PDF/DOCX gốc có nội dung đầy đủ.
**Expected impact:** Tăng context recall cho câu hỏi về hình phạt và điều luật cụ thể.

### Cải tiến 2
**Action:** Làm sạch news crawl để loại bỏ menu, footer, quảng cáo và text điều hướng.
**Expected impact:** Tăng context precision và giảm nhiễu trong câu trả lời.

### Cải tiến 3
**Action:** Dùng LLM judge hoặc RAGAS/DeepEval thật khi có API key để chấm ngữ nghĩa thay vì token overlap.
**Expected impact:** Điểm faithfulness/relevance phản ánh chất lượng câu trả lời tự nhiên tốt hơn.
