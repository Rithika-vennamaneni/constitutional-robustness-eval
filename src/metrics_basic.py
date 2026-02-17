import json, math
from collections import defaultdict
import yaml
import csv
import statistics as stats

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)

def main():
    cfg = yaml.safe_load(open("configs/metrics.yaml", "r", encoding="utf-8"))
    rows = list(read_jsonl(cfg["input_path"]))

    # group by condition
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    summary = []
    for cond, rs in by_cond.items():
        n = len(rs)
        parse_fail = sum(1 for r in rs if r.get("parse_error"))
        refused = 0
        latencies = []
        by_cat = defaultdict(lambda: {"n":0, "refused":0})

        for r in rs:
            lat = r.get("latency_ms")
            if isinstance(lat, int):
                latencies.append(lat)

            cat = r.get("category", "unknown")
            by_cat[cat]["n"] += 1

            parsed = r.get("parsed")
            if isinstance(parsed, dict) and parsed.get("refused") is True:
                refused += 1
                by_cat[cat]["refused"] += 1

        refusal_rate = refused / n if n else 0.0
        parse_fail_rate = parse_fail / n if n else 0.0

        summary.append({
            "condition": cond,
            "n": n,
            "refusal_rate": round(refusal_rate, 4),
            "parse_fail_rate": round(parse_fail_rate, 4),
            "lat_mean_ms": round(stats.mean(latencies), 2) if latencies else None,
            "lat_p50_ms": percentile(latencies, 0.50),
            "lat_p90_ms": percentile(latencies, 0.90),
        })

        # print category refusal breakdown
        print(f"\n== {cond} ==")
        print(f"n={n} refusal_rate={refusal_rate:.3f} parse_fail_rate={parse_fail_rate:.3f}")
        if latencies:
            print(f"latency mean={stats.mean(latencies):.1f}ms p50={percentile(latencies,0.5):.1f}ms p90={percentile(latencies,0.9):.1f}ms")
        print("category breakdown (refused/total):")
        for cat, d in sorted(by_cat.items()):
            rr = (d["refused"] / d["n"]) if d["n"] else 0.0
            print(f"  {cat}: {d['refused']}/{d['n']} ({rr:.2f})")

    # write CSV
    with open(cfg["output_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\n[ok] wrote {cfg['output_csv']}")

if __name__ == "__main__":
    main()
