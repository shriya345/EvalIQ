import json
import math
import pytest
from financial_eval import LIMITS, metric_rows, read, regress
from contract import Event
import core
from sqlalchemy import create_engine


@pytest.mark.parametrize("split", ["all", "dev", "evaluation", "held-out"])
def test_producer_reconciliation(split):
    rows = {r["method"]: r for r in metric_rows(split)}
    for expected in read("summary")[split]:
        actual = rows[expected["method"]]
        for key, value in expected.items():
            if key != "method":
                assert math.isclose(actual[key], value)
    if split == "all":
        assert rows["BM25"]["hit_rate"] == 13 / 14
        assert rows["BM25"]["tokens"] == 5552


def test_regression():
    base = metric_rows()[0]
    limits = json.loads(LIMITS.read_text())
    candidate = dict(base, correct=10, tokens=base["tokens"] + 2000)
    result = regress(base, candidate, limits)
    assert {k for k, v in result.items() if v["status"] == "regression"} == {
        "correctness",
        "tokens",
    }
    with pytest.raises(ValueError):
        regress(base, dict(candidate, configuration=[]), limits)
    assert (
        regress(base, dict(candidate, token_cases=0), limits)["tokens"]["status"]
        == "unavailable"
    )


def test_all_original_events(tmp_path, monkeypatch):
    monkeypatch.setattr(
        core, "db", create_engine("sqlite:///" + str(tmp_path / "import.db"))
    )
    core.init()
    records = [
        Event.model_validate_json(line)
        for name in ("finpulse-rag", "finpulse-live-49")
        for line in (core.ROOT / "fixtures" / (name + ".jsonl"))
        .read_text()
        .splitlines()
    ]
    assert sum(core.ingest(record) for record in records) == 269
    assert sum(core.ingest(record) for record in records) == 0
