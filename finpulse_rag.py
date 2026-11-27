"""Join a FinPulse recorded evaluation export to its original JSONL and filing catalog."""

import argparse
import gzip
import json
from pathlib import Path

from retrieval import scores


def enrich(events, runs, cases, catalog):
    from collections import Counter

    by_trace = {r["trace_id"]: r for r in runs}
    if len(by_trace) != len(runs):
        raise ValueError("Duplicate result trace")
    questions = {c["id"]: c for c in cases}
    rows = []
    matched = Counter()
    for event in events:
        if event.get("event_type") != "tool_call" or not (
            event.get("tool_name") or ""
        ).startswith("rag."):
            continue
        result = by_trace.get(event["trace_id"])
        if result is None:
            continue
        ids = result["retrieved"]
        if ids != event["evidence_ids"]:
            raise ValueError("Passage order differs between export and trace")
        matched[event["trace_id"]] += 1
        case = questions[result["case"]]
        labels = {key: 1 for group in case["gold"] for key in group["chunk_ids"]}
        method = event["tool_name"][4:]
        passages = [
            {
                "rank": i,
                "chunk_id": key,
                "document_id": catalog[key]["accession"],
                "score": None,
            }
            for i, key in enumerate(ids, 1)
        ]
        event["retrieval"] = {
            "query_id": event["evaluation_case_id"],
            "method": method,
            "passages": passages,
            "answer": result["answer"],
        }
        rows.append(
            dict(
                case=case["id"],
                trace=event["trace_id"],
                method=method,
                split=case["split"],
                labels=labels,
                ranked=ids,
                **scores(ids, labels, result["k"]),
            )
        )
    if set(by_trace) != set(matched) or any(n != 1 for n in matched.values()):
        raise ValueError("Results need paired retrieval events")
    return events, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["events", "runs", "cases", "catalog", "enriched", "report"]:
        p.add_argument(key, type=Path)
    a = p.parse_args()
    events = [json.loads(line) for line in a.events.read_text().splitlines()]
    out, report = enrich(
        events,
        json.loads(a.runs.read_text()),
        json.loads(a.cases.read_text()),
        json.loads(gzip.decompress(a.catalog.read_bytes())),
    )
    a.enriched.write_text("".join(json.dumps(x) + "\n" for x in out))
    a.report.write_text(json.dumps(report, indent=2) + "\n")
