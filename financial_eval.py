"""Re-score recorded FinPulse traces against frozen labels and citation reviews."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from retrieval import scores

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "fixtures/finpulse"
LIMITS = ROOT / "fixtures/financial-regression.json"


def read(name):
    return json.loads((DATA / f"document-{name}.json").read_text())


def metric_rows(split="all"):
    cases = {c["id"]: c for c in read("cases")}
    reviews = read("review")
    chunks = {c["id"]: c for c in json.load(gzip.open(DATA / "chunks.json.gz"))}
    events = defaultdict(list)
    for line in (DATA.parent / "finpulse-rag.jsonl").read_text().splitlines():
        event = json.loads(line)
        events[event["trace_id"]].append(event)
    groups = defaultdict(list)
    for run in read("runs"):
        case = cases[run["case"]]
        if split != "all" and case["split"] != split:
            continue
        trace = events[run["trace_id"]]
        (tool,) = [e for e in trace if e["event_type"] == "tool_call"]
        (response,) = [e for e in trace if e["event_type"] == "response"]
        ranked = tool["evidence_ids"]
        if ranked != run["retrieved"] or run["k"] != 3:
            raise ValueError("Incompatible ranking or cutoff")
        gold = {i: 1 for g in case["gold"] for i in g["chunk_ids"]}
        metrics = scores(ranked, gold, 3)
        citations = run["citations"]
        key = [case["id"], run["answer"], [c["excerpt"] for c in citations]]
        review = reviews[
            hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()
        ]
        if len(review["responsive_citations"]) != len(citations):
            raise ValueError("Citation review coverage mismatch")
        valid = 0
        for citation in citations:
            source = chunks[citation["chunk_id"]]
            valid += bool(
                citation["excerpt"]
                and citation["excerpt"] in source["text"]
                and source["id"] in ranked
                and source["ticker"] == case["ticker"]
                and source["filed"] <= case["cutoff"]
                and all(
                    source[k] == citation[k]
                    for k in (
                        "ticker",
                        "section",
                        "url",
                        "filed",
                        "period",
                        "accession",
                    )
                )
            )
        agents = [e for e in trace if e["event_type"] == "agent_run"]
        counts = [e[k] for e in agents for k in ("input_tokens", "output_tokens")]
        groups[run["method"]].append(
            dict(
                metrics,
                gold=len(gold),
                hits=len(set(ranked[:3]) & gold.keys()),
                correct=review["fully_correct"],
                citations=len(citations),
                valid=valid,
                support=sum(review["responsive_citations"]),
                abstained=run["status"] == "abstained",
                retrieval=tool["latency_ms"],
                latency=response["latency_ms"],
                tokens=sum(counts) if None not in counts else None,
                case=case["id"],
                configs={(e["model"], e["prompt_version"]) for e in agents},
            )
        )
    results = []
    for method, rows in groups.items():
        pos = [r for r in rows if r["gold"]]
        neg = [r for r in rows if not r["gold"]]
        token_counts = [r["tokens"] for r in rows if r["tokens"] is not None]
        ids = sorted(r["case"] for r in rows)
        expected = sorted(
            c["id"] for c in cases.values() if split in ("all", c["split"])
        )
        if ids != expected:
            raise ValueError("Incomplete or duplicate case set")
        results.append(
            dict(
                method=method,
                queries=len(rows),
                answerable=len(pos),
                unanswerable=len(neg),
                precision_at_3=mean(r["precision"] for r in pos),
                recall_at_3=mean(r["recall"] for r in pos),
                mrr_at_3=mean(r["rr"] for r in pos),
                hit_rate=mean(r["hit"] for r in pos),
                gold_chunks=sum(r["gold"] for r in pos),
                relevant_retrieved=sum(r["hits"] for r in pos),
                correct=sum(r["correct"] for r in rows),
                citations=sum(r["citations"] for r in rows),
                valid=sum(r["valid"] for r in rows),
                reviewed_support=sum(r["support"] for r in rows),
                appropriate_abstentions=sum(r["abstained"] for r in neg),
                median_retrieval_ms=median(r["retrieval"] for r in rows),
                median_end_to_end_ms=median(r["latency"] for r in rows),
                tokens=sum(token_counts),
                token_cases=len(token_counts),
                configuration=[
                    3,
                    split,
                    sorted(set.union(*(r["configs"] for r in rows))),
                    hashlib.sha256(
                        json.dumps([cases[i] for i in ids], sort_keys=True).encode()
                    ).hexdigest(),
                    hashlib.sha256((DATA / "chunks.json.gz").read_bytes()).hexdigest(),
                ],
            )
        )
    return results


def regress(base, candidate, limits):
    if base["configuration"] != candidate["configuration"]:
        raise ValueError("Incompatible cases, corpus, split, K, model or prompt")

    def rates(r):
        def ratio(n, d):
            return r[n] / r[d] if r[d] else None

        return dict(
            correctness=ratio("correct", "queries"),
            recall=r["recall_at_3"],
            citation_support=ratio("reviewed_support", "citations"),
            abstention=ratio("appropriate_abstentions", "unanswerable"),
            latency_ms=r["median_end_to_end_ms"],
            tokens=(
                ratio("tokens", "queries") if r["token_cases"] == r["queries"] else None
            ),
        )

    a, b = rates(base), rates(candidate)
    if set(limits) != set(a) or any(
        not math.isfinite(v) or v < 0 for v in limits.values()
    ):
        raise ValueError("Supply a finite nonnegative allowance for each metric")
    result = {}
    for key, limit in limits.items():
        delta = None if None in (a[key], b[key]) else b[key] - a[key]
        loss = delta if key in ("latency_ms", "tokens") or delta is None else -delta
        result[key] = dict(
            baseline=a[key],
            candidate=b[key],
            delta=delta,
            status=(
                "unavailable"
                if loss is None
                else "regression" if loss > limit else "pass"
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="all", choices=list(read("summary")))
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"))
    parser.add_argument("--limits", type=Path, default=LIMITS)
    parser.add_argument(
        "--candidate-report",
        type=Path,
        help="Use a saved candidate report for the same split and scoring protocol",
    )
    args = parser.parse_args()
    if args.candidate_report and not args.compare:
        parser.error("--candidate-report requires --compare")
    rows = {r["method"]: r for r in metric_rows(args.split)}
    for ref in read("summary")[args.split]:
        for k, v in ref.items():
            if k != "method" and not math.isclose(rows[ref["method"]][k], v):
                raise ValueError(f"Producer mismatch: {ref['method']} {k}")
    if args.compare:
        candidates = (
            rows
            if not args.candidate_report
            else {r["method"]: r for r in json.loads(args.candidate_report.read_text())}
        )
        left, right = args.compare
        if left not in rows or right not in candidates:
            parser.error("Unknown method")
        try:
            checks = regress(
                rows[left], candidates[right], json.loads(args.limits.read_text())
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(checks, indent=2))
        states = {c["status"] for c in checks.values()}
        return 2 if "unavailable" in states else int("regression" in states)
    print(json.dumps(list(rows.values()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
