import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from core import ROOT, db, import_lines, init
from evaluation import DATASETS, VARIANTS, details, run


def queries(workbook="analytics.sql"):
    parts = (ROOT / "sql" / workbook).read_text().split(";")
    with db.connect() as conn:
        for part in filter(str.strip, parts):
            if db.dialect.name == "sqlite":
                part = part.replace(
                    "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)",
                    "p95(latency_ms)",
                )
            yield part.strip().splitlines()[0][3:], pd.read_sql(text(part), conn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for cmd in ["init", "demo", "sql", "rag-sql"]:
        commands.add_parser(cmd)
    commands.add_parser("import").add_argument("file", type=Path)
    trial = commands.add_parser("run")
    trial.add_argument("dataset", choices=DATASETS)
    trial.add_argument("variant")
    trial.add_argument("--outputs", type=Path)
    args = parser.parse_args()
    init()
    if args.command == "import":
        print(import_lines(args.file.read_text()))
    if args.command == "demo":
        print(import_lines((ROOT / "fixtures/demo.jsonl").read_text()))
        for dataset in DATASETS:
            for variant in VARIANTS:
                out = details(run(dataset, variant))
                print(dataset, variant, sum(r["passed"] for r in out), "/", len(out))
    if args.command == "run":
        print(
            json.dumps(
                details(
                    run(
                        args.dataset,
                        args.variant,
                        json.loads(args.outputs.read_text()) if args.outputs else None,
                    )
                ),
                indent=2,
            )
        )
    if args.command in {"sql", "rag-sql"}:
        for question, frame in queries(
            "experiments.sql" if args.command == "rag-sql" else "analytics.sql"
        ):
            print(question, frame.to_string(index=False), sep="\n")
