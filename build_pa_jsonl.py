"""
Step 2: After filling in Punjabi translations in translation_sheet.tsv,
run this script to produce prism_eval_pa.jsonl — the final eval set.

Usage:
    python build_pa_jsonl.py --sheet translation_sheet.tsv --source prism_eval_en.jsonl
"""

import json, csv, argparse, os

def load_translations(tsv_path):
    """Returns dict: {id: {field: punjabi_text}}"""
    trans = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rid = row["id"].strip()
            field = row["field"].strip()
            pa = row.get("punjabi_translation", "").strip()
            if rid not in trans:
                trans[rid] = {}
            trans[rid][field] = pa
    return trans

def load_source(jsonl_path):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

# Punjabi task instructions (replace with your reviewed translations)
PA_INSTRUCTIONS = {
    "compression":                "ਹੇਠਾਂ ਦਿੱਤੇ ਪਾਠ ਨੂੰ ਸੰਖੇਪ ਕਰੋ ਅਤੇ ਮੂਲ ਅਰਥ ਬਰਕਰਾਰ ਰੱਖੋ।",
    "text_normalization":         "ਹੇਠਾਂ ਦਿੱਤੇ ਪਾਠ ਨੂੰ ਮਿਆਰੀ ਲਿਖਤੀ ਰੂਪ ਵਿੱਚ ਲਿਖੋ।",
    "noise_robust_normalization": "ਹੇਠਾਂ ਦਿੱਤੇ ਗ਼ਲਤ ਪਾਠ ਨੂੰ ਸਾਫ਼ ਕਰਕੇ ਮਿਆਰੀ ਰੂਪ ਵਿੱਚ ਲਿਖੋ।",
    "simplification":             "ਹੇਠਾਂ ਦਿੱਤੇ ਪਾਠ ਨੂੰ ਸਰਲ ਭਾਸ਼ਾ ਵਿੱਚ ਲਿਖੋ।",
    "style_paraphrase":           "ਹੇਠਾਂ ਦਿੱਤੇ ਪਾਠ ਨੂੰ ਦਿੱਤੀ ਸ਼ੈਲੀ ਅਨੁਸਾਰ ਦੁਬਾਰਾ ਲਿਖੋ।",
    "controlled_rewrite":         "ਹੇਠਾਂ ਦਿੱਤੀ ਸ਼ਰਤ ਅਨੁਸਾਰ ਪਾਠ ਨੂੰ ਦੁਬਾਰਾ ਲਿਖੋ।",
}

def build(sheet_path, source_path, out_path):
    translations = load_translations(sheet_path)
    source_records = load_source(source_path)

    skipped = []
    out_records = []

    for rec in source_records:
        rid = rec["id"]
        task = rec["task"]
        trans = translations.get(rid, {})

        pa_input = trans.get("input", "").strip()
        pa_ref   = trans.get("reference_output", "").strip()

        if not pa_input or not pa_ref:
            skipped.append(rid)
            continue

        instruction = PA_INSTRUCTIONS.get(task, "")
        out_rec = {
            "id":   rid,
            "task": task,
            "text": (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{pa_input}\n\n"
                f"### Response:\n{pa_ref}"
            ),
            "meta": {**rec.get("meta", {}), "lang": "pa"},
        }
        out_records.append(out_rec)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Written: {len(out_records)} records → {out_path}")
    if skipped:
        print(f"Skipped (missing translations): {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet",  default="eval_set/translation_sheet.tsv")
    parser.add_argument("--source", default="eval_set/prism_eval_en.jsonl")
    parser.add_argument("--out",    default="eval_set/prism_eval_pa.jsonl")
    args = parser.parse_args()
    build(args.sheet, args.source, args.out)
