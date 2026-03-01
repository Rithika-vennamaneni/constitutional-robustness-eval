import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any


ALLOWED_GEMINI_LABELS = {
    "clean_refusal",
    "soft_refusal_leakage",
    "harmful_compliance",
    "safe_completion",
    "judge_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Llama-vs-Gemini agreement and gap cases.")
    parser.add_argument(
        "--results-glob",
        default="results/*.jsonl",
        help="Glob for model output JSONL files with Llama self-reported refused field.",
    )
    parser.add_argument(
        "--gemini-path",
        default="results/gemini_judgements.jsonl",
        help="Gemini judgements JSONL path.",
    )
    parser.add_argument(
        "--disagreements-out",
        default="results/disagreements.jsonl",
        help="Output JSONL for structural-behavioral gap cases.",
    )
    return parser.parse_args()


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_llama_rows(results_glob: str):
    excluded = {"gemini_judgements.jsonl", "disagreements.jsonl", "audit_sample.jsonl"}
    for path in sorted(glob.glob(results_glob)):
        if os.path.basename(path) in excluded:
            continue
        for line_no, row in load_jsonl(path):
            if not isinstance(row, dict):
                continue
            base_id = row.get("base_id") or row.get("prompt_id")
            condition = row.get("condition")
            if not isinstance(base_id, str) or not isinstance(condition, str):
                continue
            parsed = row.get("parsed")
            refused = parsed.get("refused") if isinstance(parsed, dict) else None
            if not isinstance(refused, bool):
                continue
            yield {
                "source_file": path,
                "line_no": line_no,
                "base_id": base_id,
                "condition": condition,
                "llama_refused": refused,
                "llama_label": "clean_refusal" if refused else "safe_completion",
                "llama_raw_output": row.get("raw_output"),
            }


def iter_gemini_rows(path: str):
    for line_no, row in load_jsonl(path):
        if not isinstance(row, dict):
            continue
        base_id = row.get("base_id")
        condition = row.get("condition")
        label = row.get("gemini_label")
        if not isinstance(base_id, str) or not isinstance(condition, str) or not isinstance(label, str):
            continue
        label = label.strip()
        if label not in ALLOWED_GEMINI_LABELS:
            continue
        confidence = row.get("gemini_confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reasoning = row.get("gemini_reasoning")
        reasoning = reasoning if isinstance(reasoning, str) else str(reasoning or "")
        yield {
            "line_no": line_no,
            "base_id": base_id,
            "condition": condition,
            "gemini_label": label,
            "gemini_confidence": confidence,
            "gemini_reasoning": " ".join(reasoning.split()).strip(),
        }


def add_occurrence_keys(rows: list[dict[str, Any]], key1: str, key2: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    keyed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        pair = (row[key1], row[key2])
        counters[pair] += 1
        keyed[(pair[0], pair[1], counters[pair])] = row
    return keyed


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("label arrays must be same length")
    n = len(labels_a)
    if n == 0:
        return 0.0

    classes = sorted(set(labels_a) | set(labels_b))
    row_counts = Counter(labels_a)
    col_counts = Counter(labels_b)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    p_e = sum((row_counts[c] / n) * (col_counts[c] / n) for c in classes)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def print_summary_table(
    total: int,
    kappa: float,
    llama_counts: Counter,
    gemini_counts: Counter,
    disagreements: list[dict[str, Any]],
) -> None:
    print("\nSummary")
    print(f"total_pairs: {total}")
    print(f"cohens_kappa: {kappa:.4f}")
    print("llama_label_counts:")
    for label in ["clean_refusal", "safe_completion"]:
        print(f"  {label}: {llama_counts[label]}")
    print("gemini_label_counts:")
    for label in [
        "clean_refusal",
        "soft_refusal_leakage",
        "harmful_compliance",
        "safe_completion",
    ]:
        print(f"  {label}: {gemini_counts[label]}")
    print(f"structural_behavioral_gap_cases: {len(disagreements)}")


def main() -> None:
    args = parse_args()

    llama_rows = list(iter_llama_rows(args.results_glob))
    gemini_rows_all = list(iter_gemini_rows(args.gemini_path))
    excluded_judge_error = sum(1 for row in gemini_rows_all if row["gemini_label"] == "judge_error")
    print(f"[warn] excluded {excluded_judge_error} rows with gemini_label=judge_error")
    gemini_rows = [row for row in gemini_rows_all if row["gemini_label"] != "judge_error"]
    if not llama_rows:
        raise SystemExit(f"No valid Llama rows found for glob: {args.results_glob}")
    if not gemini_rows:
        raise SystemExit(f"No valid Gemini rows found at: {args.gemini_path}")

    llama_keyed = add_occurrence_keys(llama_rows, "base_id", "condition")
    gemini_keyed = add_occurrence_keys(gemini_rows, "base_id", "condition")
    shared_keys = sorted(set(llama_keyed.keys()) & set(gemini_keyed.keys()))
    if not shared_keys:
        raise SystemExit("No overlapping (base_id, condition, occurrence) keys between Llama and Gemini rows.")

    paired = []
    for key in shared_keys:
        l = llama_keyed[key]
        g = gemini_keyed[key]
        paired.append((l, g))

    llama_labels = [l["llama_label"] for l, _ in paired]
    gemini_labels = [g["gemini_label"] for _, g in paired]
    kappa = cohen_kappa(llama_labels, gemini_labels)

    disagreements: list[dict[str, Any]] = []
    for idx, (l, g) in enumerate(paired, start=1):
        if l["llama_refused"] and g["gemini_label"] in {"harmful_compliance", "soft_refusal_leakage"}:
            disagreements.append(
                {
                    "pair_index": idx,
                    "base_id": l["base_id"],
                    "condition": l["condition"],
                    "llama_refused": l["llama_refused"],
                    "llama_label": l["llama_label"],
                    "gemini_label": g["gemini_label"],
                    "gemini_confidence": g["gemini_confidence"],
                    "gemini_reasoning": g["gemini_reasoning"],
                    "source_file": l["source_file"],
                    "source_line_no": l["line_no"],
                    "raw_output": l["llama_raw_output"],
                }
            )

    write_jsonl(args.disagreements_out, disagreements)

    llama_counts = Counter(llama_labels)
    gemini_counts = Counter(gemini_labels)
    print_summary_table(
        total=len(paired),
        kappa=kappa,
        llama_counts=llama_counts,
        gemini_counts=gemini_counts,
        disagreements=disagreements,
    )
    print(f"[ok] wrote {len(disagreements)} disagreement rows -> {args.disagreements_out}")


if __name__ == "__main__":
    main()
