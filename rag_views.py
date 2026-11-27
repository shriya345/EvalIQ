"""Views over measured experiments saved in the shared database."""

import json

import pandas as pd
import streamlit as st
from sqlalchemy import select

import core
from experiment import compare, dataset, summary


def stored():
    with core.db.connect() as conn:
        return [
            json.loads(x)
            for x in conn.execute(select(core.experiments.c.payload)).scalars()
        ]


def label(e):
    return f"{e['method']} · K={e['k']} · {e['model'] or 'retrieval only'} · {e['prompt'] or '—'} · {e['id'][:8]}"


def financial_view(view):
    from financial_eval import LIMITS, metric_rows, regress

    st.subheader("FinPulse SEC document benchmark")
    split = st.selectbox(
        "Financial benchmark split", ["all", "dev", "evaluation", "held-out"]
    )
    rows = metric_rows(split)
    st.caption(
        "Only two questions were strictly held out. Retrieval averages use positive-label cases. Answer correctness includes abstentions."
    )
    if view == "Experiment Comparison":
        frame = pd.DataFrame(rows).set_index("method")
        st.write("Retrieval quality at K=3")
        st.dataframe(
            frame[
                [
                    "answerable",
                    "precision_at_3",
                    "recall_at_3",
                    "mrr_at_3",
                    "hit_rate",
                    "median_retrieval_ms",
                ]
            ]
        )
        st.write("Final answer quality and execution")
        st.dataframe(
            frame[
                [
                    "correct",
                    "queries",
                    "valid",
                    "reviewed_support",
                    "citations",
                    "appropriate_abstentions",
                    "unanswerable",
                    "median_end_to_end_ms",
                    "tokens",
                    "token_cases",
                ]
            ]
        )
    else:
        methods = {r["method"]: r for r in rows}
        left = st.selectbox("Financial baseline", methods)
        right = st.selectbox("Financial candidate", methods, index=2)
        st.dataframe(
            pd.DataFrame(
                regress(methods[left], methods[right], json.loads(LIMITS.read_text()))
            ).T
        )
    st.divider()


def render(view):
    if view in ("Experiment Comparison", "Regression Monitoring"):
        financial_view(view)
    if view == "Retrieval Intelligence":
        with st.expander("Imported application retrievals"):
            events = core.records(core.events)
            imported = core.records(core.retrievals).merge(
                events[["event_id", "application", "trace_id", "latency_ms", "status"]],
                on="event_id",
            )
            st.dataframe(imported, hide_index=True)
            finance = pd.DataFrame(
                json.loads((core.ROOT / "reports/financial-retrieval.json").read_text())
            )
            finance = finance[finance.trace.isin(imported.trace_id)]
            if len(finance):
                st.caption("FinPulse filing questions · mean over labeled cases at K=3")
                st.dataframe(
                    finance.groupby("method").agg(
                        labeled=("recall", "count"),
                        recall=("recall", "mean"),
                        precision=("precision", "mean"),
                        mrr=("rr", "mean"),
                    )
                )
    exps = stored()
    if not exps:
        st.info(
            "Load the recorded runs with python experiment.py load experiments/*.json"
        )
        return
    if view == "Regression Monitoring":
        a = st.selectbox("Baseline", exps, format_func=label)
        b = st.selectbox(
            "Candidate", exps, index=min(1, len(exps) - 1), format_func=label
        )
        limits = json.loads((core.ROOT / "fixtures/thresholds.json").read_text())
        try:
            report = compare(a, b, limits)
        except ValueError as exc:
            st.info(str(exc))
            return
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
        return
    if view == "Experiment Comparison":
        options = st.multiselect("Experiments", exps, default=exps, format_func=label)
        if options:
            selected = options[0]
            for e in options:
                try:
                    compare(selected, e, {})
                except ValueError as exc:
                    st.warning(str(exc))
                    return
        rows = []
        for e in options:
            s = summary(e)
            rows.append(
                {
                    "configuration": label(e),
                    "cases": len(e["rows"]),
                    "correct": s["correct"]["value"],
                    "mean_tokens": s["tokens"]["value"],
                    "p95_ms": s["p95_ms"]["value"],
                    "recall": s["recall"]["value"],
                    "scored": s["correct"]["n"],
                    "workflow": e["workflow"],
                }
            )
        st.dataframe(rows, hide_index=True)
        if rows:
            for axis in ["p95_ms", "mean_tokens"]:
                st.scatter_chart(
                    pd.DataFrame(rows), x=axis, y="correct", color="configuration"
                )
        return
    exp = st.selectbox("Recorded experiment", exps, format_func=label)
    if view == "Retrieval Intelligence":
        st.dataframe(
            [
                dict(
                    method=label(e),
                    **{
                        k: summary(e)[k]["value"]
                        for k in [
                            "precision",
                            "recall",
                            "hit",
                            "rr",
                            "ndcg",
                            "retrieval_ms",
                        ]
                    },
                )
                for e in exps
            ],
            hide_index=True,
        )
        failed = st.checkbox("Show only cases with incomplete retrieval")
        rows = [
            r
            for r in exp["rows"]
            if not failed or r["recall"] is not None and r["recall"] < 1
        ]
    else:
        rows = exp["rows"]
    if not rows:
        return
    row = st.selectbox("Question", rows, format_func=lambda r: r["question"])
    st.json(
        {
            "expected": row["expected"],
            "generated": row.get("output"),
            "checks": row.get("checks"),
            "required_passages": row["labels"],
            "retrieved": [
                {**r, "text": dataset()["documents"][r["chunk_id"]]["text"]}
                for r in row["ranked"]
            ],
        }
    )
    rating_file = core.ROOT / "reports/judge.json"
    if rating_file.exists():
        record = json.loads(rating_file.read_text())
        if record["experiment_id"] == exp["id"]:
            ratings = [r for r in record["ratings"] if r["case_id"] == row["case_id"]]
            if ratings:
                st.dataframe(ratings, hide_index=True)
