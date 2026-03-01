import argparse
import json
import os
import time

from groq import Groq
import yaml

from run_one import (
    JSON_INSTRUCTIONS,
    SYSTEM_BASELINE,
    SYSTEM_CONSTITUTIONAL,
    call_groq,
    load_dotenv_file,
    try_parse_json,
    write_jsonl,
)


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset evaluation across conditions.")
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Dataset config YAML path (default: configs/dataset.yaml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv_file(".env")
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Add it to .env or export it in your shell before running."
        )
    client = Groq(api_key=api_key)

    prompts = list(load_jsonl(cfg["input_path"]))
    limit = cfg.get("limit")
    if limit is not None:
        prompts = prompts[: int(limit)]

    completed = set()
    if os.path.exists(cfg["output_path"]):
        for row in load_jsonl(cfg["output_path"]):
            existing_prompt_id = row.get("prompt_id")
            existing_condition = row.get("condition")
            if existing_prompt_id is not None and existing_condition is not None:
                completed.add((str(existing_prompt_id), str(existing_condition)))

    conditions = [
        ("baseline", SYSTEM_BASELINE),
        ("constitutional", SYSTEM_CONSTITUTIONAL),
    ]
    total = len(prompts) * len(conditions)
    idx = 0

    for prompt in prompts:
        prompt_id = prompt.get("variant_id") or prompt.get("prompt_id") or prompt["base_id"]
        for condition, system_prompt in conditions:
            idx += 1
            key = (str(prompt_id), str(condition))
            if key in completed:
                print(f"[{idx}/{total}] skip {condition} {prompt_id} (already exists)")
                continue

            print(f"[{idx}/{total}] {condition} {prompt_id}")

            messages = [
                {"role": "system", "content": system_prompt + "\n\n" + JSON_INSTRUCTIONS},
                {"role": "user", "content": prompt["text"]},
            ]

            t0 = time.time()
            out = call_groq(
                client=client,
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=messages,
            )
            dt_ms = int((time.time() - t0) * 1000)
            parsed, parse_error = try_parse_json(out)
            parse_ok = parse_error is None

            record = {
                "run_id": cfg["run_id"],
                "prompt_id": prompt_id,
                "base_id": prompt.get("base_id"),
                "shift_type": prompt.get("shift_type"),
                "category": prompt.get("category"),
                "condition": condition,
                "raw_output": out,
                "parsed": parsed,
                "parse_error": parse_error,
                "parse_ok": parse_ok,
                "behavior_label": None,
                "actionability": None,
                "latency_ms": dt_ms,
            }
            write_jsonl(cfg["output_path"], record)
            completed.add(key)


if __name__ == "__main__":
    main()
