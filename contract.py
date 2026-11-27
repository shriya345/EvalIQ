"""EvalIQ wire format and FinPulse v1 adapter."""

import re
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def uid():
    return str(uuid4())


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    chunk_id: str = Field(min_length=1, max_length=200)
    document_id: str | None = None
    rank: int = Field(ge=1)
    score: float | None = None


class Retrieval(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    query_id: str | None = None
    method: str = Field(min_length=1, max_length=120)
    passages: list[Passage] = Field(default_factory=list, max_length=100)
    answer: str | None = None

    @model_validator(mode="after")
    def unique_ranks(self):
        if len({p.rank for p in self.passages}) != len(self.passages):
            raise ValueError("Duplicate rank")
        return self


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_max_length=160)
    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(default_factory=uid, min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    application: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    session_id: str | None = None
    parent_event_id: str | None = None
    event_type: Literal[
        "request", "response", "tool_call", "agent_run", "error", "evaluation"
    ]
    status: Literal["ok", "error", "invalid", "missing"] = "ok"
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    app_version: str | None = None
    workflow: str | None = None
    token_usage_source: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    error_type: str | None = None
    input_tokens: int | None = Field(None, ge=0)
    output_tokens: int | None = Field(None, ge=0)
    estimated_cost_usd: float | None = Field(None, ge=0)
    latency_ms: float | None = Field(None, ge=0)
    evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    retrieval: Retrieval | None = None
    evaluation_case_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def finpulse_contract(cls, data):
        if isinstance(data, dict) and data.get("application") == "FinPulse AI":
            aliases = {"research_started": "request", "research_completed": "response"}
            kind = data.get("event_type")
            name = data.get("tool_name")
            if (
                kind == "tool_call"
                and isinstance(name, str)
                and name.startswith("rag.")
                and data.get("retrieval") is None
                and isinstance(data.get("evidence_ids", []), list)
            ):
                return {
                    **data,
                    "retrieval": {
                        "query_id": data.get("evaluation_case_id"),
                        "method": name[4:],
                        "passages": [
                            {"chunk_id": key, "rank": i}
                            for i, key in enumerate(data.get("evidence_ids", []), 1)
                        ],
                    },
                }
            if isinstance(kind, str) and kind in aliases:
                return {**data, "event_type": aliases[kind]}
        return data

    @field_validator("timestamp")
    @classmethod
    def utc(cls, value):
        if value.tzinfo is None:
            raise ValueError("Timezone required")
        return value.astimezone(UTC)


def redact(value):
    return re.sub(
        r"(?i)(?:bearer\s+\S+|(?:sk-|ghp_|AKIA)\S+|\S+@\S+|(?:password|token|api_key)=\S+)",
        "[redacted]",
        value,
    )
