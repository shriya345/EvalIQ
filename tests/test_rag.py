import json
import math
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from streamlit.testing.v1 import AppTest

import core
from contract import Event
from experiment import compare, save
from retrieval import scores

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def test_hand_calculated_metrics():
    a = scores(["noise", "a", "b"], {"a": 3, "b": 1, "c": 1}, 3)
    assert a["precision"] == pytest.approx(2 / 3) and a["recall"] == pytest.approx(
        2 / 3
    )
    assert a["hit"] == 1 and a["rr"] == 0.5
    assert a["ndcg"] == pytest.approx(
        (7 / math.log2(3) + 0.5) / (7 + 1 / math.log2(3) + 0.5)
    )
    assert scores(["a", "a"], {"a": 1, "b": 1}, 3)["precision"] == pytest.approx(1 / 3)
    assert set(scores([], {"x": 1}, 2).values()) == {0}
    assert set(scores(["a"], {}, 2).values()) == {None}
    with pytest.raises(ValueError):
        scores([], {}, 0)


def test_regression_coverage_and_pairing():
    a = read("experiments/bm25.json")
    b = read("experiments/semantic.json")
    r = compare(a, b, {"correct": 0.1})
    assert (
        r["correct"]["state"] == "regression" and len(r["correct"]["worse_cases"]) == 3
    )
    assert r["recall"]["n"] == 15 and r["cost"]["state"] == "insufficient"
    assert compare(a, b, {"correct": 0.2})["correct"]["state"] == "pass"
    assert (
        compare(a, {**b, "rows": b["rows"][::-1]}, {"tokens": 1})["tokens"]["state"]
        == "pass"
    )
    b["rows"][0]["tokens"] = None
    assert compare(a, b, {"tokens": 1})["tokens"]["state"] == "insufficient"
    with pytest.raises(ValueError):
        compare(a, {**b, "digest": "other"}, {})
    with pytest.raises(ValueError):
        compare(a, {**b, "rows": []}, {})


def test_real_retrieval_import_and_old_schema(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    monkeypatch.setattr(core, "db", engine)
    core.events.create(engine)
    core.init()
    raw = (ROOT / "fixtures/finpulse-live-49.jsonl").read_text()
    assert core.import_lines(raw)["inserted"] == 49
    data = (ROOT / "fixtures/finpulse-rag-enriched.jsonl").read_text()
    n = len(data.splitlines())
    assert core.import_lines(data) == {"inserted": n, "duplicates": 0, "invalid": []}
    assert core.import_lines(data)["duplicates"] == n
    frame = core.records(core.retrievals)
    assert len(frame) == 80 and frame.answer.notna().all()
    ranked = next(json.loads(v) for v in frame.passages if json.loads(v))
    assert (
        ranked[0]["rank"] == 1
        and ranked[0]["document_id"]
        and ranked[0]["score"] is None
    )
    sample = json.loads(raw.splitlines()[0])
    sample["schema_version"] = "3.0"
    with pytest.raises(ValueError):
        Event.model_validate(sample)


def test_experiment_sql_and_render(tmp_path, monkeypatch):
    import cli

    engine = create_engine(f"sqlite:///{tmp_path / 'exp.db'}")
    monkeypatch.setattr(core, "db", engine)
    monkeypatch.setattr(cli, "db", engine)
    core.init()
    for name in ["bm25", "semantic"]:
        exp = read("experiments/" + name + ".json")
        save(exp)
        save(exp)
    assert len(core.records(core.experiment_cases)) == 34
    assert len(list(cli.queries("experiments.sql"))) == 4
    view = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=15).run()
    for section in [
        "Retrieval Intelligence",
        "Experiment Comparison",
        "Regression Monitoring",
        "Evaluation Studio",
    ]:
        view.radio[0].set_value(section).run()
        if section == "Evaluation Studio":
            view.toggle[0].set_value(True).run()
        assert not view.exception, section


def test_financial_raw_then_enriched(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'finance.db'}")
    monkeypatch.setattr(core, "db", engine)
    core.init()
    raw = next(
        json.loads(x)
        for x in (ROOT / "fixtures/finpulse-rag.jsonl").read_text().splitlines()
        if json.loads(x).get("tool_name")
    )
    enriched = next(
        json.loads(x)
        for x in (ROOT / "fixtures/finpulse-rag-enriched.jsonl")
        .read_text()
        .splitlines()
        if json.loads(x)["event_id"] == raw["event_id"]
    )
    assert core.ingest(Event.model_validate(raw))
    assert core.records(core.retrievals).answer.iloc[0] is None
    assert not core.ingest(Event.model_validate(enriched))
    assert (
        core.records(core.retrievals).answer.iloc[0] == enriched["retrieval"]["answer"]
    )
