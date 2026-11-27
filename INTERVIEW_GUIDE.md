# Discussing EvalIQ

1. **Why track completion and correctness?** An agent can finish and return the wrong number. The trace shows completion. A reference case checks the answer.
2. **What is a trace?** It groups the events for one task. In the recorded FinPulse example, Bull and Bear are two steps under the same starting event.
3. **What makes an import idempotent?** Each exported event carries a stable ID. A database uniqueness constraint prevents a retry from inserting it twice, even when requests overlap.
4. **What does P95 latency mean?** It estimates the duration below which about 95% of observed durations fall. Missing durations never enter that calculation.
5. **How is numeric correctness tested?** The grader compares a numeric answer against a reference using an absolute tolerance of 0.01 in the case's units.
6. **Can a real citation support the wrong answer?** Yes. A citation may exist and still contain a different value. The grader checks the cited field separately.
7. **How do you handle missing prices?** Cost stays unknown. The dashboard shows how many tasks have prices so a partial total can be interpreted correctly.
8. **Why compare the same questions?** One variant could otherwise appear stronger simply because it saw easier questions. Matching case IDs controls that difference.
9. **What did you run for real?** A deterministic FinPulse workflow, imports into SQLite, all ten SQL queries, the API tests and five dashboard sections. The sample ledger and policy candidate responses were scripted.
10. **What would you verify before extending the system?** Run the same suite against PostgreSQL and collect reviewed model outputs for more varied cases. The current seven-case filings tie provides little evidence about generalization.

## Résumé bullets

- Built a Python AI evaluation workbench with FastAPI, Streamlit and SQLAlchemy to ingest trace events, prevent duplicate imports and inspect agent execution.
- Implemented deterministic correctness and citation checks across 27 benchmark cases, including seven financial cases scored against recorded FinPulse workflow outputs.
- Developed ten SQL analyses for latency, errors, cost coverage and evaluation pass rates, and tested the API, persistence and five dashboard views with Pytest.

## RAG release interview questions

1. **What is an evaluation dataset?** It contains questions, reference answers and evidence labels fixed before the experiment. That makes the target independent of whatever the retriever happens to find.
2. **How do precision and recall differ?** If two passages are required and the top three contain one, precision is 1/3 and recall is 1/2. One measures the useful fraction of the result list and the other measures how much required evidence was found.
3. **Why measure reciprocal rank?** It rewards finding useful evidence early. A first hit in position two contributes 1/2. Average that over the questions to obtain MRR.
4. **What is faithfulness?** An answer is faithful when the retrieved passages support its factual claims. Even a real citation can point to a passage saying something different.
5. **How did you use an LLM judge?** A local model scored one dimension at a time and supplied an excerpt. Five of twelve judgments failed validation. Some valid judgments were substantively questionable, so the objective benchmark uses independent checks.
6. **What makes two experiments comparable?** Hold the questions, references, output contract and K constant. To test the effect of retrieval alone, also hold the generator and prompt constant.
7. **What did the regression test catch?** Switching from BM25 to semantic retrieval lost three previously correct structured answers even though every labeled support passage was retrieved within the top three.
8. **How do you reason about quality, cost and latency?** Compare a measurable benefit to its resource requirements. The two-stage v1 workflow used 259 tokens per case versus about 117 for single-stage v1, and both scored 0/17. Dollar cost was unknown.
9. **How do you score a question with no evidence?** Leave retrieval recall undefined when there are no positive passage labels. Separately test whether the generated answer appropriately abstains.
10. **What can a financial execution trace tell you?** Its ordered chunk IDs show retrieval order. Joining the matching recorded result adds the answer, and a filing accession identifies its source document. Independent reference labels are still needed to score retrieval.

## Updated résumé bullets

- Built a Python RAG evaluation workbench with FastAPI, Streamlit, SQLite and SQLAlchemy, integrating 220 real FinPulse events across 80 financial retrieval trials.
- Benchmarked BM25, dense retrieval, reciprocal-rank fusion and cross-encoder reranking with independent passage labels and 102 local Qwen outputs across six configurations.
- Implemented matched-case regression checks for correctness, recall, citations, latency, tokens and cost, identifying a three-case correctness drop despite improved retrieval recall.

## Explaining the final evaluation

**Why did the reranker win on recall while BM25 won on answer correctness?** Recall asks whether the first three passages contain the labeled evidence. The answer also depends on which evidence the generator selects, what it says, and whether it abstains. Reranking recovered 90.5% of labeled passages on average; BM25 produced 19 correct answers across 20 questions. Two held-out cases provide limited evidence about future questions.

**How does the regression gate work?** It matches benchmark and runtime configuration, then checks six absolute changes against a threshold file. For example, moving from BM25 to hybrid reduced supported citations from 15/15 to 14/16. That exceeds the default five-percentage-point allowance even though median response latency improved.

### Résumé bullets

- Built SQLite-backed RAG evaluation analytics for four retrieval methods across 80 SEC research runs, reconciling precision, recall and MRR with the source benchmark.
- Implemented a configurable Python CLI regression gate for answer correctness, retrieval recall, citation support, abstention, latency and token consumption.
- Verified idempotent ingestion of 269 telemetry events and reproduced reference-based scoring for 102 local-Qwen customer-support outputs across six configurations.
