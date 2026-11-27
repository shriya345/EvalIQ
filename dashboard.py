import pandas as pd
import plotly.express as px
import streamlit as st

from core import events, init, metrics, records, runs
from evaluation import DATASETS, VARIANTS, details, fixture, run

st.set_page_config(page_title="EvalIQ", layout="wide")
init()
st.title("EvalIQ")
section = st.sidebar.radio(
    "Workspace",
    [
        "AI Performance Overview",
        "Evaluation Studio",
        "Model Comparison",
        "Agent Trace Explorer",
        "Product Intelligence",
        "Retrieval Intelligence",
        "Experiment Comparison",
        "Regression Monitoring",
    ],
)
from rag_views import render as render_experiments

if section in {
    "Retrieval Intelligence",
    "Experiment Comparison",
    "Regression Monitoring",
}:
    st.header(section)
    render_experiments(section)
    st.stop()
if section == "Evaluation Studio" and st.toggle("Inspect RAG experiment"):
    render_experiments(section)
    st.stop()
frame = records(events)
registry = records(runs).sort_values("timestamp", ascending=False)
if section == "Evaluation Studio":
    dataset = st.selectbox("Dataset", DATASETS)
    variant = st.selectbox(
        "Response variant",
        list(
            dict.fromkeys(
                [
                    *VARIANTS,
                    *registry.loc[registry.dataset == dataset, "variant"].tolist(),
                ]
            )
        ),
    )
    if dataset == "filings":
        st.caption(
            "Recorded FinPulse workflows • baseline uses single extraction, revised adds numeric verification."
        )
    if variant in VARIANTS and st.button("Evaluate fixture"):
        run(dataset, variant)
        st.rerun()
    options = registry[(registry.dataset == dataset) & (registry.variant == variant)]
    if len(options):
        rid = st.selectbox("Saved run", options.id)
        trial = pd.DataFrame(details(rid))
        st.dataframe(
            trial[["question", "expected", "passed", "failures"]], hide_index=True
        )
        st.json(
            trial.set_index("id").loc[st.selectbox("Inspect case", trial.id)].to_dict(),
            expanded=True,
        )
        st.download_button(
            "Export run JSON", trial.to_json(orient="records"), "evaluation.json"
        )
    st.stop()
if frame.empty:
    st.info(
        "Import telemetry or run python cli.py demo to explore the fictional benchmarks."
    )
    st.stop()
for col in ["application", "model", "prompt_version", "workflow", "app_version"]:
    choice = st.sidebar.multiselect(
        col.replace("_", " ").title(), sorted(frame[col].dropna().unique())
    )
    if choice:
        frame = frame[frame[col].isin(choice)]
dates = pd.to_datetime(frame.timestamp, utc=True)
if len(dates):
    window = st.sidebar.date_input(
        "UTC date range", (dates.min().date(), dates.max().date())
    )
    if len(window) == 2:
        frame = frame[dates.dt.date.between(*window)]
st.header(section)
st.caption(
    "Demo applications and both response variants use scripted fixtures. Imported applications retain their own labels."
)
m = metrics(frame)


def display(value):
    return "Unknown" if value is None else f"{value:,.4g}"


def chart(data, x, y):
    if len(data):
        st.plotly_chart(px.bar(data, x=x, y=y), use_container_width=True)


def ratings(group):
    rows = []
    for app, ds in [("FinPulse-fixture", "finance"), ("PolicyQA-fixture", "support")]:
        subset = group[group.application == app]
        for variant in subset.prompt_version.dropna().unique():
            saved = registry[(registry.dataset == ds) & (registry.variant == variant)]
            if saved.empty:
                continue
            ids = set(subset[subset.prompt_version == variant].evaluation_case_id)
            rows.extend(x for x in details(saved.iloc[0].id) if x["id"] in ids)
    return rows


if section == "AI Performance Overview":
    items = [
        ("Requests", m["requests"]),
        ("Evaluated cases", len(ratings(frame))),
        ("Task success", m["completion_rate"]),
        ("Error rate", m["error_rate"]),
        ("Median ms", m["median_ms"]),
        ("P95 ms", m["p95_ms"]),
        ("Known input tokens", m["input_tokens"]),
        ("Known output tokens", m["output_tokens"]),
        ("Known spend · USD", m["known_cost_usd"]),
    ]
    for i, (label, value) in enumerate(items):
        if i % 3 == 0:
            cols = st.columns(3)
        cols[i % 3].metric(label, display(value))
    st.caption(
        f"Priced outcomes {m['cost_known']} / {m['requests']} · Timed outcomes {m['latency_known']} / {m['requests']}"
    )
    chart(
        frame[frame.event_type == "request"]
        .assign(day=lambda d: d.timestamp.str[:10])
        .groupby("day")
        .size()
        .reset_index(name="requests"),
        "day",
        "requests",
    )
elif section == "Agent Trace Explorer":
    ids = frame[["application", "trace_id"]].drop_duplicates()
    if len(ids):
        app, trace = st.selectbox(
            "Application and trace", list(ids.itertuples(index=False, name=None))
        )
        st.dataframe(
            frame[(frame.application == app) & (frame.trace_id == trace)]
            .sort_values("timestamp")
            .dropna(axis=1, how="all"),
            hide_index=True,
        )
elif section == "Model Comparison":
    ds = st.selectbox("Benchmark dataset", DATASETS)
    with st.expander("Saved benchmark runs · all telemetry dates"):
        summaries = []
        for entry in (
            registry[registry.dataset == ds].drop_duplicates("variant").itertuples()
        ):
            trial = details(entry.id)
            summaries.append(
                {
                    "variant": entry.variant,
                    "cases": len(trial),
                    "passed": sum(r["passed"] for r in trial),
                    "correct": sum(r["checks"]["correct"] for r in trial),
                }
            )
        st.dataframe(summaries, hide_index=True)
    frame = frame[frame.evaluation_case_id.isin([r["id"] for r in fixture(ds)])]
    groups = list(
        frame.groupby(
            ["application", "model", "prompt_version", "workflow", "app_version"],
            dropna=False,
        )
    )
    shared = (
        set.intersection(*(set(g.evaluation_case_id) for _, g in groups))
        if groups
        else set()
    )
    st.caption(
        f"{len(shared)} shared case IDs across the visible configurations. Scores refer to stored fixture runs."
    )
    rows = []
    for key, g in groups:
        matched = g[g.evaluation_case_id.isin(shared)]
        assessed = ratings(matched)
        info = metrics(matched)
        rows.append(
            {
                "configuration": " / ".join(map(str, key)),
                "evaluated": len(assessed),
                "correctness": pd.Series(
                    [r["checks"]["correct"] for r in assessed], dtype=float
                ).mean(),
                "source_coverage": pd.Series(
                    [r["checks"]["source_coverage"] for r in assessed], dtype=float
                ).mean(),
                **info,
            }
        )
    st.dataframe(rows, hide_index=True)
    if rows and any(r["evaluated"] for r in rows):
        for x in ["average_ms", "cost_per_request"]:
            st.plotly_chart(
                px.scatter(
                    pd.DataFrame(rows), x=x, y="correctness", color="configuration"
                ),
                use_container_width=True,
            )
else:
    rows = [{"application": a, **metrics(g)} for a, g in frame.groupby("application")]
    st.dataframe(rows, hide_index=True)
    errors = frame[
        (frame.status.isin(["invalid", "error"])) | (frame.event_type == "error")
    ]
    chart(
        errors.assign(category=errors.error_type.fillna(errors.event_type))
        .groupby("category")
        .size()
        .reset_index(name="failures"),
        "category",
        "failures",
    )
    if len(errors):
        st.write(
            "Inspect failed traces for",
            ", ".join(errors.tool_name.dropna().unique()) or "error details",
        )
    trend = [
        {"day": day, "success_rate": metrics(g)["completion_rate"]}
        for day, g in frame.groupby(frame.timestamp.str[:10])
    ]
    chart(pd.DataFrame(trend), "day", "success_rate")
