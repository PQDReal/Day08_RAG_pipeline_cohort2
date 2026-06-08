# Member 3 — RAG Evaluation

## Scope

Member 3 owns the evaluation part of the group RAG project:

- Build a golden dataset with at least 15 Q&A pairs.
- Run the RAG pipeline on the full dataset.
- Report four metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision.
- Compare at least two configs.
- Export the evaluation report.

## Files

- `golden_dataset.json` — 15 Q&A pairs with expected answers and expected context.
- `eval_pipeline.py` — custom offline evaluator and A/B comparison script.
- `results.md` — human-readable report.
- `results.json` — raw evaluation output.

## Run

From the project root:

```bash
python group_project/member3_rag/eval_pipeline.py
```

The script imports the shared pipeline from `src/` and writes updated results
inside this folder.
