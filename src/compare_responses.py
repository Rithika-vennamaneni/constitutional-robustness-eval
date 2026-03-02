import json
import os
from typing import Any


LLAMA_BASE = "results/current/llama_baseline_responses.jsonl"
LLAMA_CONST = "results/current/llama_constitutional_responses.jsonl"
GPT_BASE = "results/current/gpt_baseline_responses.jsonl"
GPT_CONST = "results/current/gpt_constitutional_responses.jsonl"
PROMPT_SOURCE = "data/adversarial_prompts_v2.jsonl"


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_id(row: dict[str, Any]) -> str | None:
    v = row.get("id") or row.get("prompt_id") or row.get("variant_id") or row.get("base_id")
    return str(v) if v is not None else None


def parse_llama_row(row: dict[str, Any]) -> tuple[str | None, bool | None]:
    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        refused = parsed.get("refused")
        if isinstance(refused, bool):
            return str(answer) if answer is not None else None, refused
    raw = row.get("raw_output")
    if raw:
        try:
            d = json.loads(raw)
            answer = d.get("answer")
            refused = d.get("refused")
            if isinstance(refused, bool):
                return str(answer) if answer is not None else None, refused
        except Exception:
            pass
    return None, None


def parse_gpt_row(row: dict[str, Any]) -> tuple[str | None, bool | None]:
    answer = row.get("answer")
    refused = row.get("refused")
    return str(answer) if answer is not None else None, refused if isinstance(refused, bool) else None


def build_prompt_meta(path: str) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        rid = get_id(row)
        if rid:
            meta[rid] = {
                "prompt": row.get("text"),
                "category": row.get("category"),
                "severity": row.get("severity"),
                "template": row.get("template"),
            }
    return meta


def run_comparison(
    left_path: str,
    left_fmt: str,
    right_path: str,
    right_fmt: str,
    output_path: str,
    prompt_meta: dict[str, dict[str, Any]],
) -> tuple[int, int, int, int]:
    left_rows = {get_id(r): r for r in load_jsonl(left_path) if get_id(r)}
    right_rows = {get_id(r): r for r in load_jsonl(right_path) if get_id(r)}

    left_parse = parse_llama_row if left_fmt == "llama" else parse_gpt_row
    right_parse = parse_llama_row if right_fmt == "llama" else parse_gpt_row

    common = sorted(set(left_rows) & set(right_rows))
    flagged: list[dict[str, Any]] = []
    refusal_disagree = 0
    both_complied = 0

    for rid in common:
        lrow = left_rows[rid]
        rrow = right_rows[rid]
        l_answer, l_refused = left_parse(lrow)
        r_answer, r_refused = right_parse(rrow)

        if not (isinstance(l_refused, bool) and isinstance(r_refused, bool)):
            continue

        flag_reason = None
        if l_refused != r_refused:
            flag_reason = "refusal_disagreement"
            refusal_disagree += 1
        elif not l_refused and not r_refused:
            flag_reason = "both_complied"
            both_complied += 1

        if flag_reason is None:
            continue

        meta = prompt_meta.get(rid, {})
        prompt = rrow.get("prompt") or lrow.get("prompt") or meta.get("prompt")
        category = rrow.get("category") or lrow.get("category") or meta.get("category")
        severity = rrow.get("severity") or lrow.get("severity") or meta.get("severity")
        adv_template = (
            rrow.get("adversarial_template")
            or lrow.get("adversarial_template")
            or meta.get("template")
        )

        flagged.append({
            "id": rid,
            "prompt": prompt,
            "category": category,
            "severity": severity,
            "adversarial_template": adv_template,
            "left_answer": l_answer,
            "left_refused": l_refused,
            "right_answer": r_answer,
            "right_refused": r_refused,
            "flag_reason": flag_reason,
        })

    write_jsonl(output_path, flagged)
    return len(common), refusal_disagree, both_complied, len(flagged)


def main() -> None:
    prompt_meta = build_prompt_meta(PROMPT_SOURCE)

    comparisons = [
        (
            "llama_baseline vs gpt_baseline",
            LLAMA_BASE, "llama",
            GPT_BASE, "gpt",
            "results/current/disagreements_baseline.jsonl",
        ),
        (
            "llama_constitutional vs gpt_constitutional",
            LLAMA_CONST, "llama",
            GPT_CONST, "gpt",
            "results/current/disagreements_constitutional.jsonl",
        ),
        (
            "llama_baseline vs llama_constitutional",
            LLAMA_BASE, "llama",
            LLAMA_CONST, "llama",
            "results/current/disagreements_llama_condition.jsonl",
        ),
    ]

    col_w = 45
    print(f"{'Comparison':<{col_w}} {'Total':>7} {'Disagree':>10} {'Both Complied':>14} {'Flagged':>9}")
    print("-" * (col_w + 44))

    for label, lp, lf, rp, rf, out in comparisons:
        total, disagree, complied, flagged = run_comparison(lp, lf, rp, rf, out, prompt_meta)
        print(f"{label:<{col_w}} {total:>7} {disagree:>10} {complied:>14} {flagged:>9}")
        print(f"  -> wrote {flagged} flagged rows to {out}")


if __name__ == "__main__":
    main()
