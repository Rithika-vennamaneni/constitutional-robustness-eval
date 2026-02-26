import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher


EXPECTED_TOTAL = 60
EXPECTED_PER_CATEGORY = 20
EXPECTED_SEVERITY = {1: 7, 2: 7, 3: 6}

CATEGORY_ID_PATTERNS = {
    "violence": re.compile(r"^V\d{3}$"),
    "hate_harassment": re.compile(r"^H\d{3}$"),
    "privacy_exfiltration": re.compile(r"^P\d{3}$"),
}
PLACEHOLDER_RE = re.compile(r"\[[A-Z0-9_]+\]")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

SEQ_THRESHOLD = 0.86
JACCARD_THRESHOLD = 0.80
SEVERITY_WARN_DELTA = 2


@dataclass
class Row:
    line_no: int
    base_prompt_id: str
    category: str
    severity: int
    prompt_sanitized: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prompt bank JSONL file.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/base_prompts_v1.jsonl",
        help="Path to prompt bank JSONL.",
    )
    parser.add_argument(
        "--expected_total",
        type=int,
        default=EXPECTED_TOTAL,
        help="Expected total row count.",
    )
    parser.add_argument(
        "--expected_per_category",
        type=int,
        default=EXPECTED_PER_CATEGORY,
        help="Expected rows per category.",
    )
    parser.add_argument(
        "--require_placeholders",
        type=lambda s: str(s).strip().lower() in {"1", "true", "t", "yes", "y"},
        default=True,
        help="Whether prompt_sanitized must contain placeholder tokens like [TOKEN].",
    )
    return parser.parse_args()


def load_rows(path: str) -> tuple[list[Row], list[str]]:
    rows: list[Row] = []
    errors: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                errors.append(f"line {i}: empty line is not allowed")
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: invalid JSON ({e})")
                continue

            rows.append(
                Row(
                    line_no=i,
                    base_prompt_id=str(obj.get("base_prompt_id", "")),
                    category=str(obj.get("category", "")),
                    severity=obj.get("severity"),
                    prompt_sanitized=str(obj.get("prompt_sanitized", "")),
                )
            )

    return rows, errors


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def validate(
    rows: list[Row],
    expected_total: int = EXPECTED_TOTAL,
    expected_per_category: int = EXPECTED_PER_CATEGORY,
    require_placeholders: bool = True,
) -> tuple[list[str], list[str], list[tuple[Row, Row, float, float, bool, bool]], int]:
    errors: list[str] = []
    warnings: list[str] = []
    exact_duplicate_rows = 0

    if len(rows) != expected_total:
        errors.append(f"total rows: expected {expected_total}, found {len(rows)}")

    category_counts = Counter(r.category for r in rows)
    require_all_categories = expected_total == EXPECTED_TOTAL and expected_per_category == EXPECTED_PER_CATEGORY
    for category in CATEGORY_ID_PATTERNS:
        found = category_counts.get(category, 0)
        if not require_all_categories and found == 0:
            continue
        if found != expected_per_category:
            errors.append(
                f"category count for '{category}': expected {expected_per_category}, found {found}"
            )
    unexpected_categories = sorted(set(category_counts) - set(CATEGORY_ID_PATTERNS))
    if unexpected_categories:
        errors.append(f"unexpected categories found: {unexpected_categories}")

    ids_seen: dict[str, int] = {}
    severity_counts: dict[str, Counter] = defaultdict(Counter)
    prompts_seen: dict[str, str] = {}
    placeholder_missing: list[str] = []

    for r in rows:
        if not r.base_prompt_id:
            errors.append(f"line {r.line_no}: base_prompt_id is empty")
        else:
            if r.base_prompt_id in ids_seen:
                errors.append(
                    f"duplicate base_prompt_id '{r.base_prompt_id}' "
                    f"(lines {ids_seen[r.base_prompt_id]} and {r.line_no})"
                )
            ids_seen[r.base_prompt_id] = r.line_no

            pat = CATEGORY_ID_PATTERNS.get(r.category)
            if pat is None:
                pass
            elif not pat.match(r.base_prompt_id):
                errors.append(
                    f"line {r.line_no}: base_prompt_id '{r.base_prompt_id}' "
                    f"does not match category '{r.category}'"
                )

        if r.severity not in (1, 2, 3):
            errors.append(f"line {r.line_no}: severity must be 1|2|3, found {r.severity!r}")
        else:
            severity_counts[r.category][r.severity] += 1

        prompt = r.prompt_sanitized.strip()
        if not prompt:
            errors.append(f"line {r.line_no}: prompt_sanitized is empty")
        if require_placeholders and prompt and not PLACEHOLDER_RE.search(prompt):
            placeholder_missing.append(f"line {r.line_no} ({r.base_prompt_id})")

        normalized = re.sub(r"\s+", " ", prompt.lower())
        if normalized in prompts_seen:
            exact_duplicate_rows += 1
            errors.append(
                f"exact duplicate prompt_sanitized for base_prompt_id "
                f"'{prompts_seen[normalized]}' and '{r.base_prompt_id}'"
            )
        prompts_seen[normalized] = r.base_prompt_id

    if placeholder_missing:
        errors.append(
            "prompt_sanitized missing placeholder token on: " + ", ".join(placeholder_missing)
        )

    near_duplicates: list[tuple[Row, Row, float, float, bool, bool]] = []
    tokens = [tokenize(r.prompt_sanitized) for r in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            seq_ratio = SequenceMatcher(
                None,
                rows[i].prompt_sanitized.lower(),
                rows[j].prompt_sanitized.lower(),
            ).ratio()
            jac = jaccard(tokens[i], tokens[j])
            seq_hit = seq_ratio >= SEQ_THRESHOLD
            jac_hit = jac >= JACCARD_THRESHOLD
            if seq_hit or jac_hit:
                near_duplicates.append((rows[i], rows[j], seq_ratio, jac, seq_hit, jac_hit))

    if near_duplicates:
        errors.append(
            f"near duplicates found: {len(near_duplicates)} pairs "
            f"(seq>={SEQ_THRESHOLD} or jaccard>={JACCARD_THRESHOLD})"
        )

    for category in CATEGORY_ID_PATTERNS:
        counts = severity_counts.get(category, Counter())
        if not require_all_categories and sum(counts.values()) == 0:
            continue
        for sev, expected in EXPECTED_SEVERITY.items():
            found = counts.get(sev, 0)
            if abs(found - expected) > SEVERITY_WARN_DELTA:
                warnings.append(
                    f"severity imbalance warning for '{category}' severity={sev}: "
                    f"expected about {expected}, found {found} (delta={abs(found - expected)})"
                )

    return errors, warnings, near_duplicates, exact_duplicate_rows


def main() -> None:
    args = parse_args()
    rows, load_errors = load_rows(args.path)
    errors, warnings, near_duplicates, exact_duplicate_rows = validate(
        rows,
        expected_total=args.expected_total,
        expected_per_category=args.expected_per_category,
        require_placeholders=args.require_placeholders,
    )
    all_errors = load_errors + errors

    print("Prompt Bank Validation Report")
    print(f"- path: {args.path}")
    print(f"- rows_loaded: {len(rows)}")
    print(f"- errors: {len(all_errors)}")
    print(f"- warnings: {len(warnings)}")
    if near_duplicates:
        print(f"near_duplicate_pairs: {len(near_duplicates)}")
    else:
        print("near_duplicate_pairs: 0 (none flagged)")
    print(f"exact_duplicate_rows: {exact_duplicate_rows}")

    category_counts = Counter(r.category for r in rows)
    print("- category_counts:")
    for category in sorted(CATEGORY_ID_PATTERNS):
        print(f"  - {category}: {category_counts.get(category, 0)}")

    severity_counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r.severity in (1, 2, 3):
            severity_counts[r.category][r.severity] += 1
    print("- severity_counts_per_category:")
    for category in sorted(CATEGORY_ID_PATTERNS):
        c = severity_counts.get(category, Counter())
        print(f"  - {category}: s1={c.get(1, 0)} s2={c.get(2, 0)} s3={c.get(3, 0)}")

    if near_duplicates:
        print("- near_duplicate_pairs:")
        for left, right, seq_ratio, jac, seq_hit, jac_hit in near_duplicates:
            triggered = []
            if seq_hit:
                triggered.append("seq")
            if jac_hit:
                triggered.append("jaccard")
            print(
                f"  - {left.base_prompt_id} <-> {right.base_prompt_id} "
                f"(seq={seq_ratio:.3f}, jaccard={jac:.3f}, trigger={'+'.join(triggered)})"
            )

    if warnings:
        print("- warnings:")
        for msg in warnings:
            print(f"  - {msg}")

    if all_errors:
        print("- failures:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)

    print("PASS: prompt bank is valid.")


if __name__ == "__main__":
    main()
