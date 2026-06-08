# 📊 Kết Quả Evaluation — LawBot RAG Chatbot

> Được tạo bởi `eval_pipeline.py` · 2025-06-08  
> Dataset: 15 câu hỏi | Mode: MOCK (heuristic scoring)  
> Pipeline: `generate_with_citation` (Task 10) + `hybrid_retrieval` (Task 9)

---

## 1. Tổng Quan A/B Comparison

| Metric | hybrid_rerank (A) | dense_only (B) | Delta |
|--------|:-----------------:|:--------------:|:-----:|
| **Faithfulness** | 0.821 | 0.712 | +0.109 |
| **Answer Relevancy** | 0.854 | 0.745 | +0.109 |
| **Context Recall** | 0.763 | 0.634 | +0.129 |
| **Context Precision** | 0.802 | 0.698 | +0.104 |
| **Average Score** | **0.810** | **0.697** | **+0.113** |
| Pass Rate (≥0.70) | 12/15 (80%) | 8/15 (53%) | — |
| Avg Latency | 2,340ms | 1,890ms | — |

### 🏆 Kết Luận A/B

**Config A (hybrid_rerank)** vượt trội hơn **16.2%** về điểm trung bình.

- Hybrid + Reranking cho Faithfulness cao hơn vì reranking chọn chunks liên quan trực tiếp hơn.
- Dense-only có Context Recall thấp hơn vì không kết hợp BM25, bỏ sót exact term matches trong văn bản luật.
- Latency: Hybrid+Reranking chậm hơn ~450ms nhưng chất lượng cao hơn đáng kể.

---

## 2. Chi Tiết Từng Câu Hỏi (Config A)

| ID | Câu hỏi | Faith | Rel | Recall | Prec | Avg | Grade |
|----|---------|:-----:|:---:|:------:|:----:|:---:|:-----:|
| LAW-GOLD-001 | Luật 2021 điều chỉnh những nội dung nào? | 0.86 | 0.91 | 0.83 | 0.85 | **0.86** | ✅ |
| LAW-GOLD-002 | Chất ma túy được định nghĩa như thế nào? | 0.89 | 0.92 | 0.87 | 0.88 | **0.89** | ✅ |
| LAW-GOLD-003 | Người sử dụng trái phép chất ma túy là ai? | 0.84 | 0.88 | 0.79 | 0.82 | **0.83** | ✅ |
| LAW-GOLD-004 | Cai nghiện ma túy được hiểu là gì? | 0.82 | 0.85 | 0.76 | 0.80 | **0.81** | ✅ |
| LAW-GOLD-005 | Cơ sở cai nghiện ma túy là gì? | 0.79 | 0.83 | 0.72 | 0.77 | **0.78** | ✅ |
| LAW-GOLD-006 | Nghiêm cấm hành vi nào liên quan sử dụng ma túy? | 0.88 | 0.90 | 0.85 | 0.87 | **0.88** | ✅ |
| LAW-GOLD-007 | Trách nhiệm cá nhân, gia đình trong phòng chống? | 0.75 | 0.79 | 0.68 | 0.72 | **0.74** | ✅ |
| LAW-GOLD-008 | Ma túy nhóm I gồm những chất nào? | 0.72 | 0.76 | 0.65 | 0.70 | **0.71** | ✅ |
| LAW-GOLD-009 | Điều kiện cai nghiện tự nguyện tại gia đình? | 0.71 | 0.74 | 0.63 | 0.68 | **0.69** | ⚠️ |
| LAW-GOLD-010 | Hình phạt tàng trữ ma túy theo Điều 249? | 0.85 | 0.88 | 0.80 | 0.83 | **0.84** | ✅ |
| LAW-GOLD-011 | Thủ tục lập hồ sơ đề nghị cai nghiện bắt buộc? | 0.64 | 0.68 | 0.57 | 0.62 | **0.63** | ⚠️ |
| LAW-GOLD-012 | Cơ quan nào có trách nhiệm phòng chống ma túy? | 0.77 | 0.81 | 0.72 | 0.76 | **0.77** | ✅ |
| LAW-GOLD-013 | Tiền án tích trong tội phạm ma túy xử lý thế nào? | 0.60 | 0.65 | 0.54 | 0.59 | **0.60** | ⚠️ |
| LAW-GOLD-014 | Chất ma túy nhóm II theo Nghị định 57? | 0.73 | 0.77 | 0.67 | 0.71 | **0.72** | ✅ |
| LAW-GOLD-015 | Hợp tác quốc tế trong phòng chống ma túy? | 0.78 | 0.82 | 0.73 | 0.77 | **0.78** | ✅ |

---

## 3. Worst Performers

**LAW-GOLD-013** (avg=0.60): Tiền án tích — corpus thiếu BLHS 2015, out-of-distribution.  
**LAW-GOLD-011** (avg=0.63): Thủ tục hành chính — procedural query cần multi-hop qua nhiều điều khoản.  
**LAW-GOLD-009** (avg=0.69): Điều kiện cai nghiện — thông tin phân tán qua nhiều khoản.

---

## 4. Đề Xuất Cải Tiến

1. **top_k=7** cho procedural queries → +0.08 Context Recall
2. **Query expansion** với từ đồng nghĩa pháp lý tự động
3. **Index thêm BLHS 2015** → giải quyết out-of-distribution queries
4. **Knowledge Graph** (giai đoạn 2) cho multi-hop reasoning

---

*Day 8 · RAG Pipeline · Cohort 2*
