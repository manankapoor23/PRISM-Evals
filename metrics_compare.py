import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import mean

TASK_ORDER = [
    "compression",
    "text_normalization",
    "noise_robust_normalization",
    "simplification",
    "style_paraphrase",
    "controlled_rewrite",
]

TASK_LABELS = {
    "compression": "COMPRESSION",
    "text_normalization": "NORMALISATION",
    "noise_robust_normalization": "NOISE",
    "simplification": "SIMP",
    "style_paraphrase": "STYLE",
    "controlled_rewrite": "CTRL",
}

TASK_KEYWORDS = [
    ("compression", "compression:"),
    ("text_normalization", "text normalization:"),
    ("noise_robust_normalization", "noise robust normalization:"),
    ("simplification", "simplification:"),
    ("style_paraphrase", "style paraphrase:"),
    ("controlled_rewrite", "controlled rewrite:"),
]

TASK_METRICS = {
    "compression": ["bleu", "bertscore", "compression_ratio", "gurmukhi_ratio"],
    "text_normalization": ["bleu", "gurmukhi_ratio"],
    "noise_robust_normalization": ["bleu", "bertscore", "gurmukhi_ratio"],
    "simplification": ["bleu", "bertscore", "avg_word_length", "gurmukhi_ratio"],
    "style_paraphrase": ["bleu", "bertscore", "gurmukhi_ratio"],
    "controlled_rewrite": ["bleu", "edit_similarity", "gurmukhi_ratio"],
}


def _safe_mean(values):
    return mean(values) if values else 0.0


def detect_task(prompt: str | None, instruction: str | None = None) -> str | None:
    text = (instruction or "") + " " + (prompt or "")
    lower = text.lower()
    for task, keyword in TASK_KEYWORDS:
        if keyword in lower:
            return task
    return None


def extract_input_from_prompt(prompt: str | None) -> str:
    if not prompt:
        return ""
    body = prompt
    if "[INST]" in body:
        body = body.split("[INST]", 1)[1]
    if "[/INST]" in body:
        body = body.split("[/INST]", 1)[0]
    body = body.replace("<s>", "").replace("</s>", "")
    parts = body.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return ""


def load_outputs(path: str) -> dict[str, list[dict]]:
    buckets = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            instruction = obj.get("instruction")
            task = obj.get("task") or detect_task(prompt, instruction)
            if not task:
                continue
            output = (
                obj.get("output")
                or obj.get("prediction")
                or obj.get("hypothesis")
                or ""
            )
            reference = obj.get("reference") or obj.get("response") or ""
            input_text = obj.get("input") or extract_input_from_prompt(prompt)
            buckets[task].append(
                {
                    "prompt": prompt or "",
                    "input": input_text,
                    "output": output,
                    "reference": reference,
                }
            )
    return dict(buckets)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise SystemExit("Missing sacrebleu. Install with: pip install sacrebleu") from exc
    bleu = BLEU(tokenize="char").corpus_score(hypotheses, [references]).score
    return bleu


def compute_bertscore(hypotheses: list[str], references: list[str]) -> float:
    try:
        from bert_score import score as bscore
    except ImportError as exc:
        raise SystemExit("Missing bert-score. Install with: pip install bert-score") from exc
    _, _, f1 = bscore(
        hypotheses,
        references,
        model_type="bert-base-multilingual-cased",
        lang="pa",
        verbose=False,
    )
    return f1.mean().item() * 100


def gurmukhi_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    gur = len(re.findall(r"[\u0A00-\u0A7F]", compact))
    return gur / len(compact)


def avg_word_length(text: str) -> float:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def compression_ratio(inp: str, out: str) -> float:
    in_len = len(re.sub(r"\s+", "", inp))
    out_len = len(re.sub(r"\s+", "", out))
    if out_len == 0:
        return 0.0
    return in_len / out_len


def edit_similarity(inp: str, out: str) -> float:
    if not inp and not out:
        return 1.0
    return SequenceMatcher(None, inp, out).ratio()


def compute_task_metrics(task: str, records: list[dict]) -> dict[str, float]:
    outputs = [r["output"] for r in records]
    references = [r["reference"] for r in records]
    inputs = [r["input"] for r in records]

    metrics = {}
    if "bleu" in TASK_METRICS[task]:
        metrics["bleu"] = compute_bleu(outputs, references)
    if "bertscore" in TASK_METRICS[task]:
        metrics["bertscore"] = compute_bertscore(outputs, references)
    if "compression_ratio" in TASK_METRICS[task]:
        metrics["compression_ratio"] = _safe_mean(
            [compression_ratio(i, o) for i, o in zip(inputs, outputs)]
        )
    if "avg_word_length" in TASK_METRICS[task]:
        metrics["avg_word_length"] = _safe_mean([avg_word_length(o) for o in outputs])
    if "edit_similarity" in TASK_METRICS[task]:
        metrics["edit_similarity"] = _safe_mean(
            [edit_similarity(i, o) for i, o in zip(inputs, outputs)]
        )
    if "gurmukhi_ratio" in TASK_METRICS[task]:
        metrics["gurmukhi_ratio"] = _safe_mean([gurmukhi_ratio(o) for o in outputs])

    return metrics


def fmt(value: float) -> str:
    return f"{value:.3f}"


def print_metrics(name: str, metrics: dict[str, float], task: str):
    print(f"{name}:")
    for key in TASK_METRICS[task]:
        label = {
            "bleu": "BLEU",
            "bertscore": "BERTScore",
            "compression_ratio": "CompressionRatio",
            "avg_word_length": "AvgWordLength",
            "edit_similarity": "EditSimilarity",
            "gurmukhi_ratio": "GurmukhiRatio",
        }[key]
        print(f"{label}: {fmt(metrics.get(key, 0.0))}")


def print_delta(a: dict[str, float], b: dict[str, float], task: str, name_a: str, name_b: str):
    print(f"DELTA ({name_a} - {name_b}):")
    for key in TASK_METRICS[task]:
        label = {
            "bleu": "BLEU",
            "bertscore": "BERTScore",
            "compression_ratio": "CompressionRatio",
            "avg_word_length": "AvgWordLength",
            "edit_similarity": "EditSimilarity",
            "gurmukhi_ratio": "GurmukhiRatio",
        }[key]
        print(f"{label}: {fmt(a.get(key, 0.0) - b.get(key, 0.0))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism", required=True, help="Path to PRISM outputs JSONL")
    parser.add_argument("--mistral", required=True, help="Path to Mistral outputs JSONL")
    parser.add_argument("--name-prism", default="PRISM")
    parser.add_argument("--name-mistral", default="MISTRAL")
    args = parser.parse_args()

    prism = load_outputs(args.prism)
    mistral = load_outputs(args.mistral)

    for task in TASK_ORDER:
        prism_records = prism.get(task, [])
        mistral_records = mistral.get(task, [])
        if not prism_records and not mistral_records:
            continue

        print(f"===== {TASK_LABELS.get(task, task.upper())} =====")
        prism_metrics = compute_task_metrics(task, prism_records)
        mistral_metrics = compute_task_metrics(task, mistral_records)

        print_metrics(args.name_prism, prism_metrics, task)
        print("\n" + args.name_mistral + ":")
        print_metrics(args.name_mistral, mistral_metrics, task)
        print()
        print_delta(prism_metrics, mistral_metrics, task, args.name_prism, args.name_mistral)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
