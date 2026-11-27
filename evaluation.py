import json
import math
from datetime import UTC, datetime

from contract import uid
from core import ROOT, db, records, results, runs

DATASETS = ("finance", "support", "filings")
VARIANTS = ("baseline", "revised")


def fixture(name):
    return json.loads((ROOT / "fixtures" / (name + ".json")).read_text())


def numeric(value, expected):
    return type(value) in (int, float) and math.isclose(
        value, expected, rel_tol=0, abs_tol=0.01
    )


def evaluate(case, output, documents):
    obj = isinstance(output, dict)
    fields = obj and {"answer", "citations"} <= output.keys()
    citations = output.get("citations") if obj else None
    shape = (
        fields
        and isinstance(citations, list)
        and all(isinstance(s, str) for s in citations)
    )
    refs = citations if shape else []
    answer = output.get("answer") if obj else None
    expected = case["expected"]
    if case["mode"] == "numeric":
        correct = numeric(answer, expected)
    elif case["mode"] == "normalized":
        correct = isinstance(answer, str) and " ".join(
            answer.casefold().split()
        ) == " ".join(expected.casefold().split())
    else:
        correct = type(answer) is type(expected) and answer == expected
    source = case["source"]
    checks = {
        "required_fields": bool(fields),
        "structured_valid": bool(shape),
        "correct": correct,
        "valid_references": all(ref in documents for ref in refs) if shape else False,
        "source_coverage": source in refs if source else None,
    }
    if type(answer) in (float, int) or case["fact"]:
        checks["numeric_evidence"] = any(
            case["fact"] in documents.get(ref, {}).get("facts", {})
            and numeric(answer, documents[ref]["facts"][case["fact"]])
            for ref in refs
        )
    failed = [k for k, v in checks.items() if v is False]
    return {
        "passed": not failed,
        "checks": checks,
        "failures": failed,
        "output": output,
        **case,
    }


def run(dataset, variant, outputs=None):
    if dataset not in DATASETS:
        raise ValueError("Unknown fixture selection")
    rid = uid()
    if outputs is None:
        if variant not in VARIANTS:
            raise ValueError("Supply a candidate file for custom variants")
        outputs = fixture(f"{dataset}-{variant}")
    elif variant in VARIANTS or not variant.strip():
        raise ValueError("Use a distinct label for your candidate outputs")
    if not isinstance(outputs, dict) or set(outputs) - {
        c["id"] for c in fixture(dataset)
    }:
        raise ValueError("Expected a mapping keyed by known case IDs")
    docs = fixture("documents")
    with db.begin() as conn:
        conn.execute(
            runs.insert().values(
                id=rid,
                dataset=dataset,
                variant=variant,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )
        for case in fixture(dataset):
            detail = evaluate(case, outputs.get(case["id"]), docs)
            conn.execute(
                results.insert().values(
                    run_id=rid,
                    case_id=case["id"],
                    passed=int(detail["passed"]),
                    detail=json.dumps(detail),
                )
            )
    return rid


def details(rid):
    return [json.loads(t) for t in records(results, run_id=rid).detail]
