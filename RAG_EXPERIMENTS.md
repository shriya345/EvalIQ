# RAG experiments and results

## Reproduce the recorded demo

From the project directory with the base requirements installed:

```sh
python cli.py demo
python experiment.py load experiments/*.json
python cli.py import fixtures/finpulse-live-49.jsonl
python cli.py import fixtures/finpulse-rag-enriched.jsonl
python cli.py rag-sql
python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Retrieval Intelligence shows rankings and imported financial traces. Toggle **Inspect RAG experiment** in Evaluation Studio for generated answers, references, citations and case checks. Experiment Comparison uses matching cases. Regression Monitoring compares an explicit baseline and candidate. The raw generations under experiments/ are genuine local outputs and can be inspected without any model installed.

## Support policy benchmark

The original fictional corpus has nine labeled passages in the RAG benchmark. Seventeen independently specified questions cover one or more relevant passages, numbers, archive conflicts, insufficient information and an injected instruction. Fifteen questions have positive passage labels. The two questions without known supporting passages still count toward generation correctness and abstention. A shipping question requires two passages to establish that the price is ambiguous.

All four main configurations used the same Qwen2.5-1.5B-Instruct Q4_K_M model, greedy decoding, single generation stage, v2 prompt, top three passages and the same dataset digest. The reference answer and gold passage IDs were withheld from the generator. Runs include measured subprocess startup and CPU inference time. Concurrent local activity can affect latency. The small, authored benchmark is a diagnostic and has no population-level confidence claim.

| Retrieval | Precision@3 | Recall@3 | MRR@3 | nDCG@3 | Strict correct / 17 | P95 end-to-end | Mean tokens / task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | .333 | .900 | .722 | .761 | 5 | 1.20 s | 149.7 |
| MPNet semantic | .378 | 1.000 | .933 | .951 | 2 | 1.45 s | 149.2 |
| Hybrid reciprocal-rank fusion | .378 | 1.000 | .856 | .888 | 4 | 1.84 s | 148.6 |
| Hybrid + MiniLM reranking | .378 | 1.000 | .967 | .975 | 2 | 2.41 s | 151.4 |

With BM25 and prompt v1, the single-stage workflow passed 0/17 and averaged 117.2 tokens. Adding a second model call to review the draft also passed 0/17 and averaged 259.0 tokens. Each workflow used the same local model. These two runs and the four v2 runs comprise 102 case outputs. The model often copied a source sentence or used the singular citation key where the contract required a numeric answer and a list of citation IDs. Exact output requirements penalize useful paraphrases. Strict correctness requires valid structure and answer correctness. Citation support is scored separately.

## Financial recordings

FinPulse's recorded filing evaluation covers 20 questions across four methods. Its 220-event snapshot has 80 retrievals paired to 80 response traces. The adapter joined its result export by trace ID, checked the retrieved order, and resolved filing IDs using its chunk catalog. Fourteen questions had positive chunk labels. Mean Recall@3 across those questions was .845 for BM25, .536 for semantic search, .833 for fusion and .905 for reranking. See reports/financial-retrieval.json for individual trace IDs, split, labels and ranks. The financial results come from the recorded producer run. EvalIQ ran the support experiments directly. The datasets share measurement definitions but are evaluated separately.

## Regression example

```sh
python experiment.py compare experiments/bm25.json experiments/semantic.json fixtures/thresholds.json
```

The command returns 1 for a measured regression and 0 when no configured threshold fails. Incompatible comparisons raise an error. Thresholds use absolute changes in metric units. Missing or partially paired measurements report insufficient data. Inspect that status when CI requires a particular metric. In the saved report, strict correctness drops by 3/17, while recall rises .10. Cases support-12, support-16 and support-17 lose correctness. Cost is unknown across all 17 pairs. See reports/regression.json.

## Run new inference

For BM25 retrieval alone use `python experiment.py run bm25 --output /tmp/bm25.json`. For live generation, point to an existing CPU build of llama.cpp b5430 and local Qwen weights. The included script compiles its upstream llama-simple example into a temporary executable without changing the producer project.

```sh
export LLAMA_SOURCE=/absolute/path/to/llama.cpp-b5430
export LLAMA_LIB=/absolute/path/to/build/bin
sh scripts/build_local_runner.sh
export EVALIQ_LLAMA=/tmp/evaliq-llama-simple
export EVALIQ_GGUF=/absolute/path/to/qwen.gguf
export EVALIQ_MODEL_ID=Qwen2.5-1.5B-Instruct-Q4_K_M
python experiment.py run bm25 --live --prompt v2 --output /tmp/support-live.json
```

For semantic, hybrid or rerank, install requirements-rag.txt and set EVALIQ_EMBEDDING and EVALIQ_RERANKER to local snapshots for sentence-transformers/all-mpnet-base-v2 and cross-encoder/ms-marco-MiniLM-L-6-v2. The runner uses CPU and local files only. Experiment records include the snapshot revision. Set --workflow review for two calls per case. The Qwen chat template is specific to this model family. Other compatible local weights need their own identity and a suitable prompt template. Optional cost estimates use the configured provider/model rates from the base README.

## Semantic judge

```sh
python judge.py experiments/bm25.json /tmp/judge.json support-01 support-13 support-16
```

The local model scored relevance, faithfulness, completeness and absence of unsupported claims on three recorded answers. Each call retains its raw output, dimension, model, prompt version, explanation and uncertainty. A confident rating needs a valid schema and a verbatim evidence excerpt. Seven of twelve judgments passed these checks. Five were unusable. The judge gave the correct phone-number abstention zero on faithfulness, and gave a directly responsive return-window answer zero on relevance. A quote alone cannot establish sound judgment. The model also judged its own outputs. Treat the judge as exploratory. No independent human labels were available for this release. The optional --human-review JSON maps case IDs to dimension scores from 0 to 2. Agreement counts only paired, valid, confident judgments.

Ragas was considered for its context-based faithfulness metric. Neither Ragas nor DeepEval was installed. The documented claim-support approach informed the rubric review. The shipped benchmark uses transparent Python metrics and the existing local model. See the [Ragas faithfulness documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/).
