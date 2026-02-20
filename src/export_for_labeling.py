import csv
import glob
import hashlib
import json
import os
from typing import Any


RESULTS_GLOB = "results/**/*.jsonl"
OUTPUT_CSV = "results/labeling/adversarial_to_label.csv"
DATA_GLOB = "data/*.jsonl"

CSV_COLUMNS = [
    "row_id",
    "run_id",
    "condition",
    "category",
    "template",
    "prompt",
    "model_refused",
    "model_refusal_reason",
    "parse_ok",
    "raw_output",
    "behavior_label",
    "actionability",
]


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def build_prompt_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in glob.glob(DATA_GLOB):
        for _, row in load_jsonl(path):
            key = row.get("variant_id") or row.get("prompt_id")
            if not key:
                continue
            lookup[key] = {
                "prompt": row.get("text") or row.get("prompt") or "",
                "template": row.get("template") or row.get("framing") or row.get("shift_type") or "",
                "framing": row.get("framing"),
                "shift_type": row.get("shift_type"),
            }
    return lookup


def infer_parse_ok(row: dict[str, Any]) -> bool:
    if "parse_ok" in row:
        return bool(row["parse_ok"])
    if row.get("parse_error") is not None:
        return False
    return isinstance(row.get("parsed"), dict)


def extract_user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def is_adversarial(row: dict[str, Any], prompt_meta: dict[str, Any] | None) -> bool:
    framing = row.get("framing")
    if framing is None and prompt_meta is not None:
        framing = prompt_meta.get("framing")
    if framing is not None:
        return str(framing).strip().lower() != "base"

    shift_type = row.get("shift_type")
    if shift_type is None and prompt_meta is not None:
        shift_type = prompt_meta.get("shift_type")
    if isinstance(shift_type, str) and "adversarial" in shift_type.lower():
        return True

    template = row.get("template")
    if template is None and prompt_meta is not None:
        template = prompt_meta.get("template")
    if isinstance(template, str) and template and template.lower() not in {"base", "paraphrase"}:
        return True

    prompt_id = row.get("prompt_id")
    if isinstance(prompt_id, str) and "_adv" in prompt_id.lower():
        return True

    return False


def stable_row_id(row: dict[str, Any], source_path: str) -> str:
    run_id = str(row.get("run_id", ""))
    prompt_id = str(row.get("prompt_id", ""))
    condition = str(row.get("condition", ""))
    raw_output = str(row.get("raw_output", ""))
    digest = hashlib.sha1(
        f"{source_path}|{run_id}|{prompt_id}|{condition}|{raw_output}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{run_id}_{digest}" if run_id else digest


def main() -> None:
    prompt_lookup = build_prompt_lookup()
    rows_out: list[dict[str, Any]] = []

    for path in sorted(glob.glob(RESULTS_GLOB, recursive=True)):
        if path.endswith(".csv"):
            continue
        if os.path.abspath(path) == os.path.abspath(OUTPUT_CSV):
            continue

        for _, row in load_jsonl(path):
            prompt_id = row.get("prompt_id")
            prompt_meta = prompt_lookup.get(prompt_id, {}) if isinstance(prompt_id, str) else {}

            if not is_adversarial(row, prompt_meta):
                continue

            parse_ok = infer_parse_ok(row)
            parsed = row.get("parsed") if parse_ok and isinstance(row.get("parsed"), dict) else {}
            model_refused = parsed.get("refused", "") if parse_ok else ""
            model_refusal_reason = parsed.get("refusal_reason", "") if parse_ok else ""

            template = (
                row.get("template")
                or row.get("framing")
                or prompt_meta.get("template")
                or row.get("shift_type")
                or ""
            )
            prompt = (
                row.get("prompt")
                or row.get("text")
                or prompt_meta.get("prompt")
                or extract_user_prompt(row)
                or ""
            )

            rows_out.append(
                {
                    "row_id": stable_row_id(row, path),
                    "run_id": row.get("run_id", ""),
                    "condition": row.get("condition", ""),
                    "category": row.get("category", ""),
                    "template": template,
                    "prompt": prompt,
                    "model_refused": model_refused,
                    "model_refusal_reason": model_refusal_reason,
                    "parse_ok": parse_ok,
                    "raw_output": row.get("raw_output", ""),
                    "behavior_label": "",
                    "actionability": "",
                }
            )

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[ok] wrote {len(rows_out)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
