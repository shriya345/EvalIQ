# Release verification

## Frozen RAG release

SQLite, 15 tests, all ten original SQL queries and four experiment queries verified. Six live configurations produced 102 outputs on 17 support cases. The captured FinPulse export contains 220 RAG events in addition to the older 49-event agent export. See RAG_EXPERIMENTS.md for denominators, failed cases, judge reliability, source provenance and reproducible startup. PostgreSQL remained unavailable. No paid API or publication was used.

## Original MVP verification

**Status.** Local MVP verified with SQLite. PostgreSQL execution remains open. No accessible server or running Docker daemon was available and a local package-install attempt found no compatible pgserver distribution. The included Compose and test paths have not been exercised against a live PostgreSQL instance here.

| Evaluation corpus | Unique cases | Baseline passes | Revised passes | Output origin |
| --- | ---: | ---: | ---: | --- |
| Fictional financial ledger | 5 | 3 | 5 | Scripted candidate answers |
| Fictional support policies | 15 | 11 | 14 | Scripted candidate answers |
| FY2024 financial filings | 7 | 7 | 7 | Recorded FinPulse deterministic extraction, with and without numeric verification |

The filings workflows tied on six numerical facts and one unavailable forecast. The synthetic score differences come from deliberately constructed errors. They measure grader behavior. No live LLM comparison or semantic judge was used.

The synthetic telemetry import inserted 118 events across 40 started tasks. Of these, 38 completed and two were missing a response. Median observed latency was 165 ms and P95 was 254.65 ms. Only 26 tasks had a price, summing to 0.0039 fixture USD. These values describe the demonstration inputs. The independent FinPulse recording contributed four additional real execution events from one deterministic research task.

**Checks executed.** The commands below ran with the existing Python 3.13.5 environment. The release verification used a fresh SQLite database. The recorded integration test reimports the same four events and checks that all four are duplicates.

```sh
python cli.py demo
python cli.py import fixtures/finpulse-recorded.jsonl
python cli.py run filings baseline
python cli.py run filings revised
python cli.py sql
python -m pytest -q
ruff format .
ruff check .
```

All ten SQL queries executed. Ten tests passed, including a Streamlit AppTest through all five sections and FastAPI TestClient requests. Ruff reported “All checks passed!”. Browser rendering and full network-server operation were not independently tested. Raw prompts are excluded from ordinary telemetry. There is no access control for a shared server. The optional prompt callback and custom candidate imports require the caller to select suitable data.

**Suggested commit:** feat: add local AI evaluation workbench with FinPulse trace ingestion

## Frozen targeted RAG release

EvalIQ joined 80 recorded SEC question/method runs to retrieval and response events by trace ID. It recomputed ranking metrics from the original chunk labels and checked cited excerpts, company, filing cutoff and source metadata against 643 extracted passages. Answer correctness and responsive citation counts reuse FinPulse's frozen answer-specific reviews. No new human review was performed. All original summary fields reconcile across all, development, evaluation and held-out slices.

| Method | P@3 | R@3 | MRR@3 | Hit@3 | Correct /20 | Valid citations | Reviewed support | Median retrieval / end-to-end ms | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | .381 | .845 | .750 | 13/14 | 19 | 15/15 | 15/15 | 0.37 / 1135 | 5552 |
| Semantic | .238 | .536 | .536 | 9/14 | 14 | 13/13 | 10/13 | 29.28 / 737 | 4753 |
| Hybrid | .381 | .833 | .881 | 13/14 | 18 | 16/16 | 14/16 | 0.51 / 805 | 6059 |
| Hybrid + reranker | .405 | .905 | .845 | 14/14 | 18 | 16/16 | 14/16 | 102.29 / 915 | 6257 |

Ranking averages cover 14 questions with positive labels at K=3. All four methods abstained on 6/6 unanswerable cases. Only two questions were strictly held out. Precision uses the available positive judgments and may undercount unlabeled relevant passages. Recorded timings include cache and execution-order effects. Token totals sum reported agent calls and count zero for traces without a model call. Citation validity measures provenance and excerpt integrity. Support measures the original reviewer's assessment of responsiveness.

The unchanged fictional support dataset supplied a second domain. Re-scoring 102 recorded local-Qwen outputs against the preset reference checks reproduced 5/17 (BM25), 2/17 (semantic), 4/17 (hybrid), 2/17 (reranked), 0/17 (earlier prompt) and 0/17 (review workflow). These are six configurations on the same questions. Strict structured-answer checks can reject paraphrases. No new inference or semantic judge was required.

Verification on SQLite: 21 tests passed, including Streamlit rendering, metric reconciliation on four splits, regression thresholds, missing usage, compatibility checks and duplicate ingestion. A fresh DB accepted 220 RAG events and 49 earlier financial events, then accepted zero additional rows on replay. PostgreSQL was outside this verification.
