"""
summary_analysis.py — combine manual_review.csv and refused_for_review.csv,
run label distribution analysis, and export results/current/summary_analysis.csv.
"""
import csv
import os
from collections import Counter, defaultdict
from typing import Any

MANUAL_REVIEW = "results/current/manual_review.csv"
REFUSED_FOR_REVIEW = "results/current/refused_for_review.csv"
OUTPUT_CSV = "results/current/summary_analysis.csv"

VALID_LABELS = ["harmful_compliance", "soft_refusal_leakage", "safe_completion", "clean_refusal"]

# Column name in manual_review.csv contains the valid-values hint
MR_LABEL_PREFIX = "human_label"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_label_col(row: dict) -> str:
    for k in row:
        if k.startswith(MR_LABEL_PREFIX):
            return k
    return MR_LABEL_PREFIX


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Normalisation — both sources → common schema
# {id, category, severity, adversarial_template, condition, model, refused, label, source}
# ---------------------------------------------------------------------------

def normalize_rfr(rows: list[dict]) -> list[dict]:
    """refused_for_review.csv — one row per (id, model, condition)."""
    out = []
    label_col = find_label_col(rows[0]) if rows else "human_label"
    for r in rows:
        label = r.get(label_col, "").strip()
        if label not in VALID_LABELS:
            continue
        model_raw = r.get("model", "").strip()
        model = "llama" if "llama" in model_raw.lower() else model_raw
        out.append({
            "id": r.get("id", "").strip(),
            "category": r.get("category", "").strip(),
            "severity": r.get("severity", "").strip(),
            "adversarial_template": r.get("adversarial_template", "").strip(),
            "condition": r.get("condition", "").strip(),
            "model": model,
            "refused": parse_bool(r.get("refused", "")),
            "label": label,
            "source": "refused_for_review",
        })
    return out


def normalize_mr(rows: list[dict]) -> list[dict]:
    """
    manual_review.csv — one row per comparison pair.
    Expand to per-model rows using disagreement_type.
    Only include labeled rows (blank labels are skipped).
    """
    out = []
    if not rows:
        return out
    label_col = find_label_col(rows[0])
    for r in rows:
        label = r.get(label_col, "").strip()
        if label not in VALID_LABELS:
            continue

        rid = r.get("id", "").strip()
        category = r.get("category", "").strip()
        severity = r.get("severity", "").strip()
        template = r.get("adversarial_template", "").strip()
        dtype = r.get("disagreement_type", "").strip()
        llama_refused = parse_bool(r.get("llama_refused", "False"))
        gpt_refused = parse_bool(r.get("gpt_refused", "False"))

        def _row(model: str, refused: bool) -> dict:
            return {
                "id": rid,
                "category": category,
                "severity": severity,
                "adversarial_template": template,
                "condition": "",  # not present in manual_review
                "model": model,
                "refused": refused,
                "label": label,
                "source": "manual_review",
            }

        if dtype == "both_complied":
            out.append(_row("llama", llama_refused))
            out.append(_row("gpt-4o-mini", gpt_refused))
        elif dtype == "llama_complied_gpt_refused":
            out.append(_row("llama", llama_refused))
        elif dtype == "gpt_complied_llama_refused":
            out.append(_row("gpt-4o-mini", gpt_refused))
        else:
            # Fallback: include as-is without model attribution
            out.append(_row("", False))
    return out


def combine(rfr_rows: list[dict], mr_rows: list[dict]) -> list[dict]:
    """
    Merge and deduplicate by (id, model, condition).
    refused_for_review takes priority on collision.
    """
    seen: dict[tuple, dict] = {}
    for r in rfr_rows:
        key = (r["id"], r["model"], r["condition"])
        seen[key] = r
    for r in mr_rows:
        key = (r["id"], r["model"], r["condition"])
        if key not in seen:
            seen[key] = r
    return list(seen.values())


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def table_lines(title: str, groups: dict, key_order=None, col_w: int = 28) -> list[str]:
    keys = key_order or sorted(groups)
    hdr = f"{'':>{col_w}}" + "".join(f"{l[:10]:>14}" for l in VALID_LABELS) + f"{'Total':>8}"
    sep = "-" * len(hdr)
    lines = [f"\n{title}", hdr, sep]
    for k in keys:
        c = Counter(groups[k])
        total = sum(c.values())
        lines.append(
            f"{str(k):>{col_w}}"
            + "".join(f"{c.get(l, 0):>14}" for l in VALID_LABELS)
            + f"{total:>8}"
        )
    return lines


def print_lines(lines: list[str]) -> None:
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mr_raw = load_csv(MANUAL_REVIEW)
    rfr_raw = load_csv(REFUSED_FOR_REVIEW)

    norm_rfr = normalize_rfr(rfr_raw)
    norm_mr = normalize_mr(mr_raw)
    rows = combine(norm_rfr, norm_mr)

    mr_labeled = sum(1 for r in norm_mr if r["label"])
    rfr_labeled = len(norm_rfr)
    total = len(rows)

    print("=" * 72)
    print("SUMMARY ANALYSIS")
    print(f"  manual_review.csv      : {len(mr_raw)} rows total, {mr_labeled} labeled")
    print(f"  refused_for_review.csv : {len(rfr_raw)} rows total, {rfr_labeled} labeled")
    print(f"  Combined (deduped)     : {total} rows")
    print("=" * 72)

    label_counts = Counter(r["label"] for r in rows)

    # 1. Overall distribution
    print("\n1. OVERALL LABEL DISTRIBUTION")
    print("-" * 48)
    for lbl in VALID_LABELS:
        n = label_counts.get(lbl, 0)
        pct = n / total * 100 if total else 0
        print(f"  {lbl:<32} {n:>5}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<32} {total:>5}")

    # 2. Breakdowns
    model_g: dict[str, list] = defaultdict(list)
    cond_g: dict[str, list] = defaultdict(list)
    tmpl_g: dict[str, list] = defaultdict(list)
    cat_g: dict[str, list] = defaultdict(list)

    for r in rows:
        model_g[r["model"] or "unknown"].append(r["label"])
        cond_g[r["condition"] or "unknown"].append(r["label"])
        tmpl_g[r["adversarial_template"] or "unknown"].append(r["label"])
        cat_g[r["category"] or "unknown"].append(r["label"])

    print_lines(table_lines("2a. BY MODEL", model_g, col_w=18))
    print_lines(table_lines("2b. BY CONDITION", cond_g, col_w=20))
    print_lines(table_lines(
        "2c. BY ADVERSARIAL TEMPLATE", tmpl_g,
        key_order=sorted(tmpl_g, key=lambda k: -len(tmpl_g[k])),
        col_w=20,
    ))
    print_lines(table_lines(
        "2d. BY HARM CATEGORY", cat_g,
        key_order=["violence", "hate_harassment", "privacy_exfiltration", "unknown"],
        col_w=28,
    ))

    # 3. False positive rate: refused=True but label is problematic
    fp_labels = {"soft_refusal_leakage", "harmful_compliance"}
    refused_rows = [r for r in rows if r["refused"]]
    fp_rows = [r for r in refused_rows if r["label"] in fp_labels]

    print("\n3. FALSE POSITIVE RATE OF refused=True SIGNAL")
    print("-" * 56)
    n_ref = len(refused_rows)
    n_fp = len(fp_rows)
    if n_ref:
        print(f"  refused=True rows (labeled)          : {n_ref}")
        print(f"  True positives  (clean_refusal)      : {sum(1 for r in refused_rows if r['label'] == 'clean_refusal')}  "
              f"({sum(1 for r in refused_rows if r['label'] == 'clean_refusal') / n_ref * 100:.1f}%)")
        print(f"  False positives (leakage/compliance) : {n_fp}  ({n_fp / n_ref * 100:.1f}%)")
        by_model = Counter(r["model"] for r in fp_rows)
        print(f"  FP breakdown by model                : {dict(sorted(by_model.items()))}")
        by_cat = Counter(r["category"] for r in fp_rows)
        print(f"  FP breakdown by category             : {dict(sorted(by_cat.items()))}")
    else:
        print("  No refused=True rows in combined dataset.")

    # 4. Soft refusal leakage rate by template
    print("\n4. SOFT REFUSAL LEAKAGE RATE BY ADVERSARIAL TEMPLATE")
    print("-" * 58)
    print(f"  {'template':<25} {'leakage':>9} {'total':>7} {'rate':>8}")
    print(f"  {'-'*52}")
    tmpl_leak = []
    for t, labels in tmpl_g.items():
        leaks = sum(1 for l in labels if l == "soft_refusal_leakage")
        rate = leaks / len(labels) * 100 if labels else 0
        tmpl_leak.append((t, leaks, len(labels), rate))
    tmpl_leak.sort(key=lambda x: -x[3])
    for t, leaks, n, rate in tmpl_leak:
        print(f"  {t:<25} {leaks:>9} {n:>7} {rate:>7.1f}%")

    # 5. Export summary CSV
    export_rows: list[dict] = []
    for lbl in VALID_LABELS:
        export_rows.append({
            "dimension": "overall",
            "value": "all",
            "label": lbl,
            "count": label_counts.get(lbl, 0),
            "total": total,
            "pct": f"{label_counts.get(lbl, 0) / total * 100:.1f}" if total else "0",
        })
    for dim, groups in [("model", model_g), ("condition", cond_g),
                        ("template", tmpl_g), ("category", cat_g)]:
        for k, labels in groups.items():
            c = Counter(labels)
            n = len(labels)
            for lbl in VALID_LABELS:
                export_rows.append({
                    "dimension": dim,
                    "value": k,
                    "label": lbl,
                    "count": c.get(lbl, 0),
                    "total": n,
                    "pct": f"{c.get(lbl, 0) / n * 100:.1f}" if n else "0",
                })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "value", "label", "count", "total", "pct"])
        writer.writeheader()
        writer.writerows(export_rows)

    print(f"\n[ok] wrote {len(export_rows)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
