import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI

from run_one import (
    JSON_INSTRUCTIONS,
    SYSTEM_BASELINE,
    SYSTEM_CONSTITUTIONAL,
    load_dotenv_file,
    try_parse_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT responder over adversarial prompts.")
    parser.add_argument(
        "--input",
        default="data/adversarial_prompts_v2.jsonl",
        help="Input JSONL with adversarial prompts.",
    )
    parser.add_argument(
        "--output",
        default="results/current/gpt_baseline_responses.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "constitutional", "both"],
        default="baseline",
        help="Which condition(s) to run.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max completion tokens.",
    )
    return parser.parse_args()


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_id(row: dict[str, Any]) -> str | None:
    row_id = row.get("id") or row.get("variant_id") or row.get("prompt_id") or row.get("base_id")
    return str(row_id) if row_id is not None else None


def get_prompt(row: dict[str, Any]) -> str | None:
    prompt = row.get("prompt") or row.get("text")
    if prompt is None:
        return None
    prompt = str(prompt).strip()
    return prompt if prompt else None


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str | None:
    retry_delays = [2, 4, 8]
    for attempt in range(len(retry_delays) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                messages=messages,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < len(retry_delays):
                delay = retry_delays[attempt]
                print(
                    f"[warn] OpenAI call failed (attempt {attempt + 1}/{len(retry_delays) + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {delay}s."
                )
                time.sleep(delay)
            else:
                print(f"[error] OpenAI call failed after retries: {type(e).__name__}: {e}")
    return None


def load_completed_keys(path: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    done = set()
    for row in load_jsonl(path):
        row_id = row.get("id")
        condition = row.get("condition")
        if row_id is not None and condition is not None:
            done.add((str(row_id), str(condition)))
    return done


def main() -> None:
    args = parse_args()
    load_dotenv_file(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Add it to .env or export it before running.")
    client = OpenAI(api_key=api_key)

    prompts = list(load_jsonl(args.input))
    completed = load_completed_keys(args.output)
    all_conditions = [
        ("baseline", SYSTEM_BASELINE),
        ("constitutional", SYSTEM_CONSTITUTIONAL),
    ]
    conditions = (
        all_conditions
        if args.condition == "both"
        else [c for c in all_conditions if c[0] == args.condition]
    )
    total = len(prompts) * len(conditions)
    processed = 0
    skipped = 0

    idx = 0
    for row in prompts:
        row_id = get_id(row)
        prompt = get_prompt(row)
        if row_id is None or prompt is None:
            for _ in conditions:
                idx += 1
                print(f"[{idx}/{total}] skip malformed row (missing id/prompt)")
                skipped += 1
            continue

        for condition, system_prompt in conditions:
            idx += 1
            key = (row_id, condition)
            if key in completed:
                print(f"[{idx}/{total}] skip {row_id} {condition} (already exists)")
                skipped += 1
                continue

            print(f"[{idx}/{total}] run {row_id} {condition}")
            messages = [
                {"role": "system", "content": system_prompt + "\n\n" + JSON_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
            out = call_openai_with_retry(
                client=client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
            )
            parsed, _ = try_parse_json(out)

            answer: str | None = None
            refused: bool | None = None
            if isinstance(parsed, dict):
                parsed_answer = parsed.get("answer")
                if parsed_answer is not None:
                    answer = str(parsed_answer)
                parsed_refused = parsed.get("refused")
                if isinstance(parsed_refused, bool):
                    refused = parsed_refused
            if answer is None:
                answer = out

            record = {
                "id": row_id,
                "prompt": prompt,
                "category": row.get("category"),
                "severity": row.get("severity"),
                "adversarial_template": row.get("adversarial_template") or row.get("template"),
                "condition": condition,
                "answer": answer,
                "refused": refused,
                "model": args.model,
            }
            append_jsonl(args.output, record)
            completed.add(key)
            processed += 1

    print(
        f"[ok] done. total_input={total} processed={processed} skipped={skipped} output={args.output}"
    )


if __name__ == "__main__":
    main()
