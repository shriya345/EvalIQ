from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Body, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from core import (
    Event,
    apps,
    db,
    events,
    import_lines,
    ingest,
    init,
    metrics,
    records,
    runs,
)
from evaluation import details, run


@asynccontextmanager
async def lifespan(app):
    init()
    yield


app = FastAPI(title="EvalIQ", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": [{"field": e["loc"], "reason": e["type"]} for e in exc.errors()]
        },
    )


@app.get("/health")
def health():
    with db.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/events", status_code=201)
def submit(item: Event):
    if not ingest(item):
        raise HTTPException(409, "Duplicate event ID")
    return {"accepted": True}


@app.post("/events/import")
def bulk(content: str = Body(media_type="text/plain")):
    return import_lines(content)


@app.get("/applications")
def applications():
    return records(apps).application.tolist()


@app.get("/traces")
def traces():
    return (
        records(events)[["application", "trace_id"]]
        .drop_duplicates()
        .to_dict("records")
    )


@app.get("/traces/{trace_id}")
def trace(trace_id: str, application: str | None = None):
    frame = records(events, trace_id=trace_id, application=application).sort_values(
        "timestamp"
    )
    if frame.empty:
        raise HTTPException(404, "Unknown trace")
    from core import retrievals

    side = records(retrievals)
    if not side.empty:
        side = side[side.event_id.isin(frame.event_id)].copy()
        if not side.empty:
            import json

            side["retrieval"] = [
                {
                    "query_id": r.query_id,
                    "method": r.method,
                    "passages": json.loads(r.passages),
                    "answer": r.answer,
                }
                for r in side.itertuples()
            ]
            frame = frame.merge(
                side[["event_id", "retrieval"]], on="event_id", how="left"
            )
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


@app.get("/metrics")
def overview(application: str | None = None):
    return metrics(records(events, application=application))


@app.get("/evaluations")
def evaluations():
    return records(runs).to_dict("records")


class Selection(BaseModel):
    dataset: Literal["support", "finance", "filings"]
    variant: str
    outputs: dict | None = None


@app.post("/evaluations/run")
def execute(selection: Selection):
    try:
        return {"run_id": run(selection.dataset, selection.variant, selection.outputs)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/evaluations/{run_id}")
def show(run_id: str):
    output = details(run_id)
    if not output:
        raise HTTPException(404, "Unknown run")
    return output
