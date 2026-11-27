"""Validated telemetry, SQL persistence and shared analytics."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from contract import Event, redact

ROOT = Path(__file__).resolve().parent
db = create_engine(os.getenv("EVALIQ_DATABASE_URL", "sqlite:///evaliq.db"))
if db.dialect.name == "sqlite":

    @event.listens_for(db, "connect")
    def configure(conn, record):
        conn.execute("PRAGMA foreign_keys=ON")
        conn.create_aggregate("p95", 1, Percentile)


class Percentile:
    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        return float(np.percentile(self.values, 95)) if self.values else None


meta = MetaData()
apps = Table("applications", meta, Column("application", String, primary_key=True))
integer = {"input_tokens", "output_tokens"}
real = {"latency_ms", "estimated_cost_usd"}
events = Table(
    "telemetry_events",
    meta,
    Column("event_id", String, primary_key=True),
    Column("application", ForeignKey("applications.application"), nullable=False),
    *(
        Column(k, Integer if k in integer else Float if k in real else Text)
        for k in Event.model_fields
        if k not in {"event_id", "application", "retrieval"}
    ),
)
Index("ix_trace", events.c.application, events.c.trace_id, events.c.timestamp)
runs = Table(
    "evaluation_runs",
    meta,
    Column("id", String, primary_key=True),
    Column("dataset", String),
    Column("variant", String),
    Column("timestamp", String),
)
results = Table(
    "evaluation_results",
    meta,
    Column("run_id", ForeignKey("evaluation_runs.id"), primary_key=True),
    Column("case_id", String, primary_key=True),
    Column("passed", Integer),
    Column("detail", Text),
)


retrievals = Table(
    "rag_retrievals",
    meta,
    Column("event_id", ForeignKey("telemetry_events.event_id"), primary_key=True),
    Column("query_id", String),
    Column("method", String),
    Column("passages", Text),
    Column("answer", Text),
)
experiments = Table(
    "rag_experiments",
    meta,
    Column("id", String, primary_key=True),
    Column("payload", Text, nullable=False),
)
experiment_cases = Table(
    "rag_cases",
    meta,
    Column("experiment_id", ForeignKey("rag_experiments.id"), primary_key=True),
    Column("case_id", String, primary_key=True),
    Column("category", String),
    Column("method", String),
    Column("recall", Float),
    Column("correct", Integer),
    Column("retrieval_ms", Float),
    Column("tokens", Integer),
    Column("created", String),
    Column("dataset_digest", String),
)


def init():
    meta.create_all(db)


def insert_once(conn, table, values, key):
    insert = pg_insert if db.dialect.name == "postgresql" else sqlite_insert
    return conn.execute(
        insert(table).values(**values).on_conflict_do_nothing(index_elements=[key])
    ).rowcount


class Price(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)


def estimated_cost(item):
    if item.estimated_cost_usd is not None:
        return item.estimated_cost_usd
    if item.input_tokens is None or item.output_tokens is None or not item.model:
        return None
    rates = json.loads(os.getenv("EVALIQ_PRICES_JSON", "{}"))
    entry = rates.get(f"{item.provider}/{item.model}")
    if entry is None:
        return None
    rate = Price.model_validate(entry)
    return (
        item.input_tokens * rate.input_per_million
        + item.output_tokens * rate.output_per_million
    ) / 1_000_000


def ingest(item):
    data = item.model_dump(mode="json")
    retrieval = data.pop("retrieval", None)
    data["estimated_cost_usd"] = estimated_cost(item)
    data["timestamp"] = item.timestamp.isoformat(timespec="microseconds")
    data = {k: redact(v) if isinstance(v, str) else v for k, v in data.items()}
    data["evidence_ids"] = json.dumps([redact(v) for v in item.evidence_ids])
    with db.begin() as conn:
        insert_once(conn, apps, {"application": data["application"]}, "application")
        inserted = bool(insert_once(conn, events, data, "event_id"))
        if inserted and retrieval:
            insert_once(
                conn,
                retrievals,
                {
                    "event_id": data["event_id"],
                    "query_id": retrieval["query_id"],
                    "method": retrieval["method"],
                    "passages": json.dumps(retrieval["passages"]),
                    "answer": redact(retrieval["answer"])
                    if retrieval["answer"]
                    else None,
                },
                "event_id",
            )
        elif (
            retrieval
            and not inserted
            and (
                retrieval["answer"]
                or any(
                    p["document_id"] or p["score"] is not None
                    for p in retrieval["passages"]
                )
            )
        ):
            # Only fill missing details when the existing immutable event and ranking agree.
            saved = (
                conn.execute(
                    select(events).where(events.c.event_id == data["event_id"])
                )
                .mappings()
                .one()
            )
            if all(
                saved[key] == data[key]
                for key in (
                    "application",
                    "trace_id",
                    "timestamp",
                    "tool_name",
                    "evidence_ids",
                    "evaluation_case_id",
                )
            ):
                row = (
                    conn.execute(
                        select(retrievals).where(
                            retrievals.c.event_id == data["event_id"]
                        )
                    )
                    .mappings()
                    .first()
                )
                if (
                    row
                    and row["method"] == retrieval["method"]
                    and row["query_id"] == retrieval["query_id"]
                ):
                    old = json.loads(row["passages"])
                    new = retrieval["passages"]
                    if [(p["rank"], p["chunk_id"]) for p in old] == [
                        (p["rank"], p["chunk_id"]) for p in new
                    ]:
                        for previous, extra in zip(old, new):
                            for key in ("document_id", "score"):
                                if previous.get(key) is None:
                                    previous[key] = extra.get(key)
                        conn.execute(
                            retrievals.update()
                            .where(
                                retrievals.c.event_id == data["event_id"],
                                retrievals.c.passages == row["passages"],
                            )
                            .values(
                                passages=json.dumps(old),
                                answer=row["answer"]
                                or (
                                    redact(retrieval["answer"])
                                    if retrieval["answer"]
                                    else None
                                ),
                            )
                        )
        return inserted


def import_lines(content):
    report = {"inserted": 0, "duplicates": 0, "invalid": []}
    for line_no, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if (
                not isinstance(raw, dict)
                or not raw.get("event_id")
                or not raw.get("timestamp")
            ):
                report["invalid"].append(
                    {
                        "line": line_no,
                        "reason": "missing persisted event ID or timestamp",
                    }
                )
                continue
            item = Event.model_validate(raw)
        except ValueError as exc:
            report["invalid"].append(
                {
                    "line": line_no,
                    "reason": exc.errors(include_input=False, include_context=False)[0][
                        "type"
                    ]
                    if hasattr(exc, "errors")
                    else "invalid_json",
                }
            )
            continue
        report["inserted" if ingest(item) else "duplicates"] += 1
    return report


def records(table, **filters):
    stmt = select(table)
    for k, v in filters.items():
        if v is not None:
            stmt = stmt.where(table.c[k] == v)
    with db.connect() as conn:
        return pd.read_sql(stmt, conn)


def metrics(frame):
    req = frame[frame.event_type == "request"]
    keys = ["application", "trace_id"]
    responses = (
        frame[frame.event_type == "response"]
        .sort_values(["timestamp", "event_id"])
        .drop_duplicates(keys, keep="last")
    )
    tasks = req.drop_duplicates(keys).merge(
        responses, on=keys, how="left", suffixes=("_start", "")
    )
    tools = frame[frame.event_type == "tool_call"]
    costs = tasks.estimated_cost_usd.dropna()
    latency = tasks.latency_ms.dropna()
    success = int((tasks.status == "ok").sum())
    errors = frame[(frame.event_type == "error") | (frame.status == "error")]
    with_errors = tasks[keys].merge(errors[keys].drop_duplicates(), on=keys)

    def ratio(n, d):
        return float(n / d) if d else None

    def stat(v):
        return None if pd.isna(v) else float(v)

    return {
        "requests": len(tasks),
        "successful_tasks": success,
        "completion_rate": ratio(success, len(tasks)),
        "error_rate": ratio(len(with_errors), len(tasks)),
        "tool_failure_rate": ratio((tools.status == "error").sum(), len(tools)),
        "invalid_output_rate": ratio((tasks.status == "invalid").sum(), len(tasks)),
        "missing_response_rate": ratio(
            (tasks.status.isna() | tasks.status.eq("missing")).sum(), len(tasks)
        ),
        "average_ms": stat(latency.mean()),
        "median_ms": stat(latency.median()),
        "p95_ms": stat(latency.quantile(0.95)),
        "latency_known": len(latency),
        "cost_known": len(costs),
        "known_cost_usd": stat(costs.sum(min_count=1)),
        "cost_per_request": stat(costs.mean()),
        "cost_per_success": ratio(costs.sum(), success)
        if len(costs) == len(tasks)
        else None,
        **{k: stat(tasks[k].sum(min_count=1)) for k in integer},
        **{k + "_known": int(tasks[k].count()) for k in integer},
    }
