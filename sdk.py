"""Metadata-only JSONL SDK. One client instance per trace."""

from pathlib import Path
from time import perf_counter

from contract import Event, redact, uid


class Client:
    def __init__(self, path, application, session_id=None):
        self.path = Path(path)
        self.application = application
        self.session_id = session_id or uid()

    def start_trace(self, *, prompt=None, capture_prompt=None, **metadata):
        self.trace_id = uid()
        self.started = perf_counter()
        self.root = None
        self.defaults = {
            k: v
            for k, v in metadata.items()
            if k
            in {
                "model",
                "provider",
                "prompt_version",
                "workflow",
                "app_version",
                "evaluation_case_id",
            }
        }
        self.root = self.log_request(**metadata)
        if prompt is not None and capture_prompt is not None:
            capture_prompt(self.trace_id, redact(prompt))
        return self.trace_id

    def _log(self, kind, **metadata):
        metadata = {**self.defaults, **metadata}
        item = Event(
            application=self.application,
            session_id=self.session_id,
            trace_id=self.trace_id,
            event_type=kind,
            parent_event_id=metadata.pop("parent_event_id", self.root),
            **metadata,
        )
        data = item.model_dump(mode="json")
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = redact(value)
        data["evidence_ids"] = [redact(v) for v in data["evidence_ids"]]
        with self.path.open("a", encoding="utf8") as out:
            out.write(Event.model_validate(data).model_dump_json() + "\n")
        return item.event_id

    def log_request(self, **kw):
        return self._log("request", **kw)

    def log_response(self, **kw):
        return self._log("response", **kw)

    def log_tool_call(self, **kw):
        return self._log("tool_call", **kw)

    def log_error(self, **kw):
        return self._log("error", status="error", **kw)

    def log_evaluation(self, **kw):
        return self._log("evaluation", **kw)

    def finish_trace(self, **kw):
        kw.setdefault("latency_ms", (perf_counter() - self.started) * 1000)
        return self.log_response(**kw)
