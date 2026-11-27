# EvalIQ

EvalIQ is a local workbench for answering two different questions about an AI application. Did a request finish? Did its answer satisfy the task and match the evidence? A successful HTTP response alone answers neither.

## RAG evaluation release

[Recorded experiments, startup instructions and results](RAG_EXPERIMENTS.md) cover 102 genuine local Qwen outputs, four retrieval methods, matched-case regression checks and FinPulse filing traces. The archive includes the raw outputs. The recorded demo needs only the base dependencies.

## Run the demo

Use Python 3.11–3.13 from the project directory. The verified environment used Python 3.13.5 and the packages listed in requirements.txt.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python cli.py demo
python cli.py import fixtures/finpulse-recorded.jsonl
python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Evaluation Studio opens each saved case and its evidence checks. In Model Comparison, select finance to compare the same five ledger questions across variants. In Agent Trace Explorer, select FinPulse AI to inspect the recorded Bull/Bear run. Demo imports are idempotent by event ID. Each benchmark execution intentionally creates a new run.

From a second terminal in the same directory and virtual environment:

```sh
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive endpoint contract. The demo creates evaliq.db in the current directory. Override EVALIQ_DATABASE_URL to select another database.

## Architecture

```mermaid
flowchart TD
  A["Python SDK or FinPulse JSONL"] --> B["Event validation and adapter"]
  B --> C["SQLAlchemy event store"]
  R["Ranked passages and independent labels"] --> X["Matched experiments"]
  X --> Y["Regression comparison"]
  X --> H
  D["Reference cases and candidate answers"] --> E["Deterministic evaluation"]
  E --> F["Saved runs and case results"]
  C --> G["Task and SQL analytics"]
  F --> G
  G --> H["Streamlit dashboard"]
  G --> I["FastAPI"]
```

contract.py and sdk.py can be copied together into a Python application with Pydantic 2 installed. core.py handles transactions, pricing and task metrics. evaluation.py runs cases and stores a snapshot of the expected answer, submitted output and failed checks. api.py, cli.py and dashboard.py are the three entry points. SQL is in sql/analytics.sql. PRD.md describes the product scope.

## Instrument a Python application

```python
from sdk import Client

client = Client("events.jsonl", "Helpdesk")
client.start_trace(model="local-rules", prompt_version="v1", workflow="single")
client.log_tool_call(tool_name="policy_lookup")
client.finish_trace(status="ok")
```

Use one client per in-flight trace. Parent IDs default to the request event. Pass parent_event_id explicitly for nested operations. finish_trace measures elapsed wall time with a monotonic clock. On failure, log_error(error_type="TimeoutError") and finish_trace(status="error"). Prompts are discarded by default. A caller can supply an explicit capture_prompt(trace_id, redacted_prompt) callback to start_trace for separately approved local capture. Basic redaction covers common secret patterns and email addresses. Keep private data out of identifiers and evaluation outputs. The evaluation store retains submitted case outputs for inspection.

## Import, evaluate and analyze

```sh
python cli.py init
python cli.py import fixtures/finpulse-recorded.jsonl
python cli.py run support revised
python cli.py run finance my-workflow --outputs fixtures/finance-baseline.json
python cli.py sql
python -m pytest -q
ruff check .
```

Submit JSONL to POST /events/import as text/plain. Each valid line commits independently. The result identifies invalid line numbers and counts duplicates. For transport retries use a stable event_id. POST /events returns 409 for an existing ID. Use GET /applications, /traces, /traces/{trace_id}, /metrics and /health to inspect telemetry. GET /evaluations and /evaluations/{run_id} retrieve evaluations. POST /evaluations/run accepts dataset, variant and optional outputs keyed by case ID. An omitted candidate fails its case. Reserve baseline and revised for the bundled runs.

Pricing is optional. A producer estimate takes precedence. Alternatively, set EVALIQ_PRICES_JSON to a JSON object keyed by provider/model with numeric input_per_million and output_per_million in USD. Estimates require both usage counts and a matching rate. Pricing changes apply to future ingests. Use the rates agreed with your provider.

## PostgreSQL path

SQLite was verified end to end. PostgreSQL runtime verification is pending. The repository includes a PostgreSQL dialect path, native percentile SQL, a loopback-only Compose service and an integration-test switch. For a local Docker environment, use the following from the project directory. Installing this optional driver is unnecessary for the SQLite demo.

```sh
python -m pip install 'psycopg[binary]>=3.2,<4'
export EVALIQ_PG_PASSWORD="$(python -c 'import secrets; print(secrets.token_hex(24))')"
docker compose up -d --wait db
export EVALIQ_DATABASE_URL="postgresql+psycopg://evaliq:${EVALIQ_PG_PASSWORD}@127.0.0.1:55432/evaliq"
EVALIQ_TEST_POSTGRES_URL="$EVALIQ_DATABASE_URL" python -m pytest -q
python cli.py demo
python cli.py sql
```

Retain the database password locally when reusing a Compose volume. The PostgreSQL test switch creates isolated schemas and requires permission to create and remove those schemas.

## Results and limitations

See PROJECT_STATUS.md for measured results, FINPULSE.md for the verified adapter, METHODOLOGY.md for denominators and evidence rules, and INTERVIEW_GUIDE.md for interview preparation. This release runs locally and covers small datasets that fit in memory.

## FinPulse experiment comparison

Launch the verified working copy on Shriya's Mac from Terminal:

```sh
cd "/Users/shriyahr/Documents/Codex/2026-11-25/new-chat-2/evaliq"
/opt/anaconda3/bin/python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Open **Experiment Comparison** for the SEC benchmark and **Regression Monitoring** for paired methods. The packaged SQLite snapshot preserves the existing dashboard data. From the project directory, reproduce the measurements offline:

```sh
/opt/anaconda3/bin/python financial_eval.py
/opt/anaconda3/bin/python financial_eval.py --split held-out
/opt/anaconda3/bin/python financial_eval.py --compare BM25 Hybrid --limits fixtures/financial-regression.json
/opt/anaconda3/bin/python -m pytest -q
```

For CI, save a candidate's JSON from the first command and pass its path with `--candidate-report` alongside `--compare`. Use the same scoring protocol and machine for timing comparisons. The CLI checks the baseline against FinPulse, matches the corpus and case fingerprints, K, split, model and prompt, then compares the two selected methods. Exit 0 passes, 1 flags deterioration, and 2 means incompatible inputs or incomplete comparisons. The JSON allowances use absolute rate differences, median end-to-end milliseconds and mean model tokens per question. They are example tolerances, not statistical significance thresholds. See PROJECT_STATUS.md for results and limitations.
