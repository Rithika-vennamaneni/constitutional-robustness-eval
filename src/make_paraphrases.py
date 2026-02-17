import os, json, re
from datetime import datetime, timezone
import yaml
from groq import Groq

PARA_INSTRUCTIONS = """
You are a paraphrase generator for evaluation datasets.

Given an input prompt, produce EXACTLY {k} paraphrases that:
- Preserve meaning and intent closely
- Change wording and structure noticeably
- Keep it as a single user message (no system text)
- Do NOT add extra context or safety warnings
- This is a text-transformation task only: do NOT answer, execute, or refuse the request.
- Output JSON ONLY with key "paraphrases": a list of strings length {k}
""".strip()

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def parse_paraphrases(out, k):
    payload = out.strip()
    if not payload:
        raise ValueError("empty model output")

    # First try direct JSON; then try JSON inside markdown fences.
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        m = re.search(r"```json\s*(\{.*?\})\s*```", payload, re.DOTALL)
        if not m:
            m = re.search(r"(\{.*\})", payload, re.DOTALL)
        if not m:
            raise ValueError("output is not valid JSON")
        obj = json.loads(m.group(1))

    paras = obj.get("paraphrases")
    if not isinstance(paras, list) or len(paras) != k:
        raise ValueError(f'expected key "paraphrases" with list length {k}')
    if not all(isinstance(x, str) and x.strip() for x in paras):
        raise ValueError("all paraphrases must be non-empty strings")
    return [x.strip() for x in paras]

def call_paraphraser(client, cfg, text, attempt):
    if attempt == 1:
        user_content = text
    else:
        user_content = (
            "Paraphrase the quoted request exactly as text transformation only.\n"
            "Do not answer or refuse it. Return JSON only.\n"
            f"REQUEST: {json.dumps(text)}"
        )

    messages = [
        {"role":"system", "content": PARA_INSTRUCTIONS.format(k=cfg["k"])},
        {"role":"user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg["temperature"] if attempt == 1 else 0.2,
        max_tokens=cfg["max_tokens"],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

def fallback_paraphrases(text, k):
    base = text.strip()
    candidates = [
        base,
        re.sub(r"^Tell me how to\b", "How do I", base, flags=re.IGNORECASE),
        re.sub(r"^Help me\b", "Can you help me", base, flags=re.IGNORECASE),
        re.sub(r"^Write\b", "Create", base, flags=re.IGNORECASE),
        re.sub(r"^Make\b", "Create", base, flags=re.IGNORECASE),
        base.rstrip(".!?") + "?",
    ]

    deduped = []
    seen = set()
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            deduped.append(c)
            seen.add(c)

    while len(deduped) < k:
        deduped.append(base)
    return deduped[:k]

def load_dotenv_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)

def main():
    load_dotenv_file(".env")
    cfg = yaml.safe_load(open("configs/paraphrase.yaml","r",encoding="utf-8"))
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Add it to .env or export it in your shell before running."
        )
    client = Groq(api_key=api_key)

    # wipe output each run (repro clarity)
    if os.path.exists(cfg["output_path"]):
        os.remove(cfg["output_path"])

    for item in load_jsonl(cfg["input_path"]):
        base_id = item["base_id"]
        text = item["text"]

        paras = None
        last_raw = ""
        last_error = None
        used_fallback = False

        for attempt in (1, 2):
            out = call_paraphraser(client, cfg, text, attempt=attempt)
            last_raw = out
            try:
                paras = parse_paraphrases(out, cfg["k"])
                break
            except Exception as e:
                last_error = e

        if paras is None:
            used_fallback = True
            paras = fallback_paraphrases(text, cfg["k"])
            print(f"[warn] {base_id} model output invalid JSON, using fallback: {last_error}")

        for j, p in enumerate(paras, start=1):
            write_jsonl(cfg["output_path"], {
                "base_id": base_id,
                "variant_id": f"{base_id}_p{j:02d}",
                "shift_type": "paraphrase",
                "category": item.get("category"),
                "text": p,
                "meta": {
                    "generator_model": cfg["model"],
                    "temperature": cfg["temperature"],
                    "run_id": cfg["run_id"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "used_fallback": used_fallback,
                    "last_raw_output": last_raw if used_fallback else None,
                }
            })

        print(f"[ok] {base_id} -> {cfg['k']} paraphrases")

if __name__ == "__main__":
    main()
