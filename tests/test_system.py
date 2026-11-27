import json
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, event

import core
from api import app
from cli import queries
from evaluation import details, evaluate, fixture, numeric, run
from sdk import Client


@pytest.fixture(autouse=True)
def database(monkeypatch, tmp_path):
    import api
    import cli
    import evaluation

    url = os.getenv("EVALIQ_TEST_POSTGRES_URL")
    admin = None
    schema = "evaliq_test_" + uuid4().hex
    if url:
        admin = create_engine(url)
        assert admin.dialect.name == "postgresql"
        with admin.begin() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        engine = create_engine(
            url, connect_args={"options": f"-c search_path={schema}"}
        )
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def register(conn, _):
            conn.create_aggregate("p95", 1, core.Percentile)

    for module in (core, cli, api, evaluation):
        monkeypatch.setattr(module, "db", engine)
    core.init()
    yield
    engine.dispose()
    if admin is not None:
        with admin.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        admin.dispose()


def test_validation_and_utc():
    base = {"application": "test", "trace_id": "trace", "event_type": "request"}
    item = core.Event(**base, timestamp="2026-11-25T01:00:00+05:30")
    assert item.timestamp.isoformat() == "2026-11-24T19:30:00+00:00"
    for extra in [
        {"timestamp": "2026-11-24"},
        {"input_tokens": -1},
        {"prompt": "private"},
        {"latency_ms": float("nan")},
    ]:
        with pytest.raises(ValidationError):
            core.Event(**base, **extra)


def test_recorded_import_persistence():
    data = (core.ROOT / "fixtures/finpulse-recorded.jsonl").read_text()
    assert core.import_lines(data) == {"inserted": 4, "duplicates": 0, "invalid": []}
    report = core.import_lines(data + "{bad\n{}")
    assert report["duplicates"] == 4 and [r["line"] for r in report["invalid"]] == [
        5,
        6,
    ]
    rows = core.records(core.events).sort_values("timestamp")
    assert rows.event_type.tolist() == ["request", "agent_run", "agent_run", "response"]
    assert rows.iloc[1].parent_event_id == rows.iloc[0].event_id
    assert rows.iloc[2].agent_name == "Bear"


def test_sdk_privacy_and_parent(tmp_path):
    path = tmp_path / "sdk.jsonl"
    client = Client(path, "test")
    client.start_trace(model="sk-abcdefgh")
    client.log_tool_call(tool_name="search")
    client.finish_trace()
    assert "sk-" not in path.read_text()
    core.import_lines(path.read_text())
    m = core.metrics(core.records(core.events))
    assert m["requests"] == 1 and m["median_ms"] >= 0
    assert m["known_cost_usd"] is None and m["input_tokens"] is None


def test_cost_latency_coverage():
    for i, cost in enumerate([0, 0.04]):
        for kind in ["request", "response"]:
            core.ingest(
                core.Event(
                    application="a",
                    trace_id=str(i),
                    event_type=kind,
                    latency_ms=(10 + i * 20) if kind == "response" else None,
                    estimated_cost_usd=cost if kind == "response" else None,
                )
            )
    m = core.metrics(core.records(core.events))
    assert m["cost_per_success"] == 0.02 and m["p95_ms"] == 29 and m["median_ms"] == 20
    core.ingest(
        core.Event(application="a", trace_id="incomplete", event_type="request")
    )
    m = core.metrics(core.records(core.events))
    assert m["cost_per_success"] is None and m[
        "missing_response_rate"
    ] == pytest.approx(1 / 3)


def test_correctness_and_citations():
    docs = fixture("documents")
    case = fixture("finance")[0]
    assert numeric(120.009, 120) and not numeric(True, 1)
    good = {"answer": 120, "citations": ["ledger"]}
    assert evaluate(case, good, docs)["passed"]
    assert (
        "numeric_evidence"
        in evaluate(case, {**good, "citations": ["fake"]}, docs)["failures"]
    )
    assert not evaluate(case, {"answer": 120}, docs)["checks"]["structured_valid"]
    assert not evaluate(case, {"answer": "120", "citations": ["ledger"]}, docs)[
        "passed"
    ]
    case = fixture("support")[1]
    assert evaluate(case, {"answer": " NO ", "citations": ["returns"]}, docs)["passed"]
    case = fixture("support")[12]
    assert not evaluate(case, {"answer": "UNKNOWN", "citations": []}, docs)["passed"]


def test_benchmarks_and_sql():
    core.import_lines((core.ROOT / "fixtures/demo.jsonl").read_text())
    for ds, variant, passed in [
        ("finance", "baseline", 3),
        ("finance", "revised", 5),
        ("support", "baseline", 11),
        ("support", "revised", 14),
    ]:
        assert sum(x["passed"] for x in details(run(ds, variant))) == passed
    output = list(queries())
    assert len(output) == 10 and output[0][1].requests.sum() == 40
    assert output[3][1].p95_ms.iloc[0] == pytest.approx(254.65)


def test_http_endpoints():
    with TestClient(app) as c:
        data = json.loads(
            (core.ROOT / "fixtures/finpulse-recorded.jsonl").read_text().splitlines()[0]
        )
        assert c.get("/health").status_code == 200
        assert c.post("/events", json=data).status_code == 201
        assert c.post("/events", json=data).status_code == 409
        assert c.post("/events", json={"prompt": "personal"}).status_code == 422
        assert c.get("/applications").json() == ["FinPulse AI"]
        assert len(c.get("/traces").json()) == 1
        assert c.get(f"/traces/{data['trace_id']}").json()[0]["event_type"] == "request"
        assert c.get("/metrics").json()["missing_response_rate"] == 1
        assert c.get("/traces/absent").status_code == 404
        assert c.post(
            "/events/import", content="{}", headers={"Content-Type": "text/plain"}
        ).json()["invalid"]
        rid = c.post(
            "/evaluations/run", json={"dataset": "finance", "variant": "baseline"}
        ).json()["run_id"]
        assert len(c.get(f"/evaluations/{rid}").json()) == 5
        assert c.get("/evaluations").json()[0]["id"] == rid


def test_all_dashboard_views():
    from streamlit.testing.v1 import AppTest

    core.import_lines((core.ROOT / "fixtures/demo.jsonl").read_text())
    run("finance", "baseline")
    view = AppTest.from_file(str(core.ROOT / "dashboard.py"), default_timeout=15).run()
    for section in view.radio[0].options:
        view.radio[0].set_value(section).run()
        assert not view.exception, section


def test_custom_outputs_and_recorded_filings():
    for variant in ["baseline", "revised"]:
        trial = details(run("filings", variant))
        assert len(trial) == 7 and all(r["passed"] for r in trial)
    trial = details(
        run(
            "finance",
            "candidate",
            {"finance-01": {"answer": 120, "citations": ["ledger"]}},
        )
    )
    assert sum(r["passed"] for r in trial) == 1
    with pytest.raises(ValueError):
        run("finance", "candidate", {"unrelated": {}})


def test_pricing_requires_both_counts(monkeypatch):
    monkeypatch.setenv(
        "EVALIQ_PRICES_JSON",
        '{"local/m":{"input_per_million":1,"output_per_million":3}}',
    )
    event = core.Event(
        application="bench",
        trace_id="one",
        event_type="response",
        model="m",
        provider="local",
        input_tokens=500,
        output_tokens=100,
    )
    assert core.estimated_cost(event) == 0.0008
    assert core.estimated_cost(event.model_copy(update={"input_tokens": None})) is None
    assert core.estimated_cost(event.model_copy(update={"model": "other"})) is None
    assert core.estimated_cost(event.model_copy(update={"estimated_cost_usd": 0})) == 0
