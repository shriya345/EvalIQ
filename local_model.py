"""Greedy offline Qwen chat using the upstream llama-simple example."""

import json
import os
import re
import subprocess
from time import perf_counter


def complete(instruction, content, limit=80):
    prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    start = perf_counter()
    p = subprocess.run(
        [
            os.environ["EVALIQ_LLAMA"],
            "-m",
            os.environ["EVALIQ_GGUF"],
            "-ngl",
            "0",
            "-n",
            str(limit),
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=150,
        check=True,
    )
    _prefix, found, raw = p.stdout.partition("<|im_start|>assistant\n")
    if not found:
        raise ValueError("Runner omitted chat boundary")
    raw = raw.strip()

    def count(pattern):
        match = re.search(pattern, p.stderr)
        return int(match.group(1)) if match else None

    try:
        payload = json.loads(
            re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", raw)
        )
    except ValueError:
        payload = None
    return {
        "raw": raw,
        "output": payload,
        "input_tokens": count(r"prompt eval time.*?/\s*(\d+) tokens"),
        "output_tokens": count(r"decoded (\d+) tokens"),
        "llm_ms": (perf_counter() - start) * 1000,
    }
