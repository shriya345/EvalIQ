"""Optional local semantic review. Human ratings must be supplied separately."""

import argparse
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiment import dataset
from local_model import complete


class Rating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: Literal[0, 1, 2] | None
    evidence: str
    explanation: str = Field(min_length=1)
    uncertain: bool

    @model_validator(mode="after")
    def check(self):
        if self.score is None and not self.uncertain:
            raise ValueError("Missing uncertainty")
        return self


DIMENSIONS = {
    "faithfulness": "Are all answer claims supported by the supplied passages?",
    "relevance": "Does the answer address the question?",
    "completeness": "Does the answer cover every part of the question?",
    "unsupported_claims": "Is the answer free of unsupported factual claims?",
}


def judge(exp, selected, review=None):
    docs = dataset()["documents"]
    ratings = []
    for row in exp["rows"]:
        if row["case_id"] not in selected:
            continue
        evidence = [docs[x["chunk_id"]]["text"] for x in row["ranked"]]
        for dim, criterion in DIMENSIONS.items():
            prompt = (
                "Assess only this dimension: "
                + criterion
                + " Score 0 for fail, 1 for partial, 2 for pass. Return exactly JSON with score, evidence (verbatim excerpt), explanation, uncertain (boolean). Use null and uncertain=true to abstain."
            )
            response = complete(
                prompt,
                json.dumps(
                    {
                        "question": row["question"],
                        "answer": row.get("output"),
                        "passages": evidence,
                    }
                ),
                120,
            )
            item = {
                "case_id": row["case_id"],
                "dimension": dim,
                "judge_model": os.environ["EVALIQ_MODEL_ID"],
                "judge_prompt_version": "rubric-v1",
                "raw": response["raw"],
            }
            try:
                rating = Rating.model_validate(response["output"])
                if not rating.uncertain and (
                    not rating.evidence.strip()
                    or not any(rating.evidence in e for e in evidence)
                ):
                    raise ValueError("Unverified quote")
                item.update(rating.model_dump(), valid=True)
            except ValueError:
                item.update(valid=False, score=None, uncertain=True)
            ratings.append(item)
    agreements = [
        int(r["score"] == review[r["case_id"]][r["dimension"]])
        for r in ratings
        if review
        and r["valid"]
        and not r["uncertain"]
        and review.get(r["case_id"], {}).get(r["dimension"]) in (0, 1, 2)
    ]
    return {
        "experiment_id": exp["id"],
        "ratings": ratings,
        "human_comparisons": len(agreements),
        "exact_agreement": sum(agreements) / len(agreements) if agreements else None,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("cases", nargs="+")
    p.add_argument("--human-review", type=Path)
    a = p.parse_args()
    result = judge(
        json.loads(a.experiment.read_text()),
        a.cases,
        json.loads(a.human_review.read_text()) if a.human_review else None,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
