"""
PRISM Unified Evaluator
=======================
Evaluates any model (PRISM or baselines) on the 30-example Punjabi eval set.
Single script — no changes needed between models. All behaviour is config-driven.

Usage:
    # Evaluate PRISM
    python evaluate.py --model prism --mode zeroshot
    python evaluate.py --model prism --mode fewshot

    # Evaluate any baseline
    python evaluate.py --model llama --mode zeroshot
    python evaluate.py --model mistral --mode fewshot
    python evaluate.py --model gemma --mode zeroshot
    python evaluate.py --model qwen --mode fewshot

    # Evaluate ALL models, both modes, in one shot
    python evaluate.py --model all --mode all

    # After all runs, print comparison tables
    python evaluate.py --report
"""

import json, os, re, argparse, time, gc
from collections import defaultdict
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY  —  edit paths here once, never touch again
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "prism": {
        "name":                   "PRISM",
        "path":                   "/path/to/prism/checkpoint",   # ← set this
        "is_finetuned":           True,
        "use_chat_template":      False,   # PRISM uses raw ### format
        "dtype":                  "bfloat16",
        "gpu_memory_utilization": 0.85,
        "max_model_len":          4096,
    },
    "llama": {
        "name":                   "LLaMA-3.1-8B",
        "path":                   "meta-llama/Llama-3.1-8B-Instruct",
        "is_finetuned":           False,
        "use_chat_template":      True,
        "dtype":                  "bfloat16",
        "gpu_memory_utilization": 0.85,
        "max_model_len":          4096,
    },
    "mistral": {
        "name":                   "Mistral-7B-v0.3",
        "path":                   "mistralai/Mistral-7B-Instruct-v0.3",
        "is_finetuned":           False,
        "use_chat_template":      True,
        "dtype":                  "bfloat16",
        "gpu_memory_utilization": 0.85,
        "max_model_len":          4096,
    },
    "gemma": {
        "name":                   "Gemma-2-9B",
        "path":                   "google/gemma-2-9b-it",
        "is_finetuned":           False,
        "use_chat_template":      True,
        "dtype":                  "bfloat16",
        "gpu_memory_utilization": 0.90,
        "max_model_len":          4096,
    },
    "qwen": {
        "name":                   "Qwen-2.5-7B",
        "path":                   "Qwen/Qwen2.5-7B-Instruct",
        "is_finetuned":           False,
        "use_chat_template":      True,
        "dtype":                  "bfloat16",
        "gpu_memory_utilization": 0.85,
        "max_model_len":          4096,
        "trust_remote_code":      True,
    },
}

TASK_ORDER = [
    "compression",
    "text_normalization",
    "noise_robust_normalization",
    "simplification",
    "style_paraphrase",
    "controlled_rewrite",
]

TASK_SHORT = {
    "compression":               "Comp.",
    "text_normalization":        "Norm.",
    "noise_robust_normalization":"Noise",
    "simplification":            "Simp.",
    "style_paraphrase":          "Style",
    "controlled_rewrite":        "CRewrite",
}

SYSTEM_PROMPT = (
    "You are a helpful Punjabi language assistant. "
    "Follow the instruction carefully. "
    "Respond only with the processed Punjabi text. "
    "Do not add explanations, labels, or any text other than the output."
)

BATCH_SIZE     = 8
RESULTS_DIR    = "results"
EVAL_SET_PATH  = "eval_set/prism_eval_pa_final.jsonl"   # final Punjabi eval set

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def parse_record(line: str) -> dict | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    text = obj.get("text", "")
    parts = {"instruction": "", "input": "", "response": ""}
    section = None
    for ln in text.splitlines():
        if ln.startswith("### Instruction:"):
            section = "instruction"
            parts["instruction"] = ln.replace("### Instruction:", "").strip()
        elif ln.startswith("### Input:"):
            section = "input"
            parts["input"] = ln.replace("### Input:", "").strip()
        elif ln.startswith("### Response:"):
            section = "response"
            parts["response"] = ln.replace("### Response:", "").strip()
        elif section:
            parts[section] += " " + ln.strip()
    for k in parts:
        parts[k] = parts[k].strip()
    parts["id"]   = obj.get("id", "")
    parts["task"] = obj.get("task", "")
    parts["meta"] = obj.get("meta", {})
    return parts

def load_eval_set(path: str) -> Dict[str, List[dict]]:
    buckets = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = parse_record(line)
            if rec and rec["task"]:
                buckets[rec["task"]].append(rec)
    total = sum(len(v) for v in buckets.values())
    print(f"Loaded {total} eval records across {len(buckets)} tasks")
    return dict(buckets)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_prism_zeroshot(record: dict) -> str:
    return (
        f"### Instruction:\n{record['instruction']}\n\n"
        f"### Input:\n{record['input']}\n\n"
        f"### Response:\n"
    )

def build_prompt_prism_fewshot(record: dict, examples: List[dict]) -> str:
    prompt = f"### Instruction:\n{record['instruction']}\n\n"
    for ex in examples:
        prompt += (
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['response']}\n\n"
        )
    prompt += f"### Input:\n{record['input']}\n\n### Response:\n"
    return prompt

def build_messages_baseline_zeroshot(record: dict) -> List[dict]:
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"{record['instruction']}\n\n{record['input']}"},
    ]

def build_messages_baseline_fewshot(record: dict, examples: List[dict]) -> List[dict]:
    user_content = (
        "Here are some examples of the task:\n\n"
    )
    for i, ex in enumerate(examples, 1):
        user_content += f"Example {i}:\nInput: {ex['input']}\nOutput: {ex['response']}\n\n"
    user_content += (
        f"Now apply the same task to the following:\n\n"
        f"{record['instruction']}\n\n{record['input']}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

def get_fewshot_examples(task_records: List[dict], current_id: str, k: int = 3) -> List[dict]:
    import random as _r
    _r.seed(42)
    pool = [r for r in task_records if r["id"] != current_id]
    return _r.sample(pool, min(k, len(pool)))

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE (vLLM)
# ─────────────────────────────────────────────────────────────────────────────

class EvalEngine:
    def __init__(self, model_key: str):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        cfg = MODEL_REGISTRY[model_key]
        self.cfg           = cfg
        self.model_key     = model_key
        self.is_finetuned  = cfg["is_finetuned"]
        self.use_chat_tmpl = cfg["use_chat_template"]

        print(f"\nLoading {cfg['name']} from {cfg['path']} ...")
        self.llm = LLM(
            model=cfg["path"],
            dtype=cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
            trust_remote_code=cfg.get("trust_remote_code", False),
            max_model_len=cfg.get("max_model_len", 4096),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"],
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=400,
            stop=["### Input:", "### Instruction:", "<|eot_id|>",
                  "<end_of_turn>", "<|im_end|>", "</s>"],
        )
        print(f"{cfg['name']} loaded.")

    def _apply_chat_template(self, messages: List[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def run_batch(self, raw_inputs) -> List[str]:
        """
        raw_inputs: list of str (for PRISM) or list of list[dict] (for baselines).
        Returns list of generated strings.
        """
        if self.use_chat_tmpl:
            prompts = [self._apply_chat_template(m) for m in raw_inputs]
        else:
            prompts = raw_inputs   # already strings for PRISM

        from vllm import LLM   # already loaded; just use self.llm
        outputs = self.llm.generate(prompts, self.sampling)
        return [o.outputs[0].text.strip() for o in outputs]

    def unload(self):
        del self.llm
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print(f"{self.cfg['name']} unloaded from VRAM.")

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(hypotheses: List[str], references: List[str]) -> dict:
    from sacrebleu.metrics import BLEU, CHRF
    from rouge_score import rouge_scorer as rs
    from bert_score import score as bscore

    bleu_score  = BLEU(tokenize="char").corpus_score(hypotheses, [references]).score
    chrf_score  = CHRF(word_order=2).corpus_score(hypotheses, [references]).score

    scorer      = rs.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l     = sum(scorer.score(r, h)["rougeL"].fmeasure
                      for h, r in zip(hypotheses, references)) / len(references) * 100

    P, R, F1    = bscore(hypotheses, references,
                         model_type="bert-base-multilingual-cased",
                         lang="pa", verbose=False)

    wf1_scores  = []
    for h, r in zip(hypotheses, references):
        h_tok, r_tok = set(h.split()), set(r.split())
        tp = len(h_tok & r_tok)
        if not h_tok or not r_tok:
            wf1_scores.append(0.0); continue
        p, rc = tp / len(h_tok), tp / len(r_tok)
        wf1_scores.append(2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0)

    return {
        "bleu":       round(bleu_score, 4),
        "chrf_pp":    round(chrf_score, 4),
        "rouge_l":    round(rouge_l, 4),
        "bertscore_p":round(P.mean().item() * 100, 4),
        "bertscore_r":round(R.mean().item() * 100, 4),
        "bertscore_f1":round(F1.mean().item() * 100, 4),
        "word_f1":    round(sum(wf1_scores) / len(wf1_scores) * 100, 4),
        "n":          len(hypotheses),
    }

# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_key: str, mode: str, eval_buckets: dict):
    """
    Runs the full evaluation for one model in one mode.
    Saves per-task predictions and metrics to results/.
    """
    assert mode in ("zeroshot", "fewshot"), "mode must be 'zeroshot' or 'fewshot'"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cfg        = MODEL_REGISTRY[model_key]
    model_name = cfg["name"]
    engine     = EvalEngine(model_key)
    all_metrics = {}

    for task in TASK_ORDER:
        records = eval_buckets.get(task, [])
        if not records:
            print(f"  [SKIP] {task} — no records found")
            continue

        print(f"\n  [{model_name}] Task: {task} ({len(records)} samples, mode={mode})")
        raw_inputs, references = [], []

        for rec in records:
            if cfg["is_finetuned"]:
                if mode == "zeroshot":
                    raw_inputs.append(build_prompt_prism_zeroshot(rec))
                else:
                    examples = get_fewshot_examples(records, rec["id"], k=3)
                    raw_inputs.append(build_prompt_prism_fewshot(rec, examples))
            else:
                if mode == "zeroshot":
                    raw_inputs.append(build_messages_baseline_zeroshot(rec))
                else:
                    examples = get_fewshot_examples(records, rec["id"], k=3)
                    raw_inputs.append(build_messages_baseline_fewshot(rec, examples))
            references.append(rec["response"])

        # Batched inference
        hypotheses = []
        for i in range(0, len(raw_inputs), BATCH_SIZE):
            batch = raw_inputs[i:i + BATCH_SIZE]
            hypotheses.extend(engine.run_batch(batch))

        metrics = compute_metrics(hypotheses, references)
        all_metrics[task] = metrics
        print(f"    BLEU={metrics['bleu']:.2f}  chrF++={metrics['chrf_pp']:.2f}  "
              f"ROUGE-L={metrics['rouge_l']:.2f}  BERTScore-F1={metrics['bertscore_f1']:.2f}  "
              f"Word-F1={metrics['word_f1']:.2f}")

        # Save predictions
        pred_file = os.path.join(RESULTS_DIR, f"{model_key}_{task}_{mode}.jsonl")
        with open(pred_file, "w", encoding="utf-8") as pf:
            for rec, hyp in zip(records, hypotheses):
                pf.write(json.dumps({
                    "id":         rec["id"],
                    "task":       task,
                    "input":      rec["input"],
                    "reference":  rec["response"],
                    "hypothesis": hyp,
                    "model":      model_name,
                    "mode":       mode,
                }, ensure_ascii=False) + "\n")

    # Save metrics
    metrics_file = os.path.join(RESULTS_DIR, f"{model_key}_{mode}_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump({"model": model_name, "mode": mode, "tasks": all_metrics}, mf,
                  ensure_ascii=False, indent=2)
    print(f"\n  Metrics saved → {metrics_file}")

    engine.unload()
    return all_metrics

# ─────────────────────────────────────────────────────────────────────────────
# REPORTING — builds paper-ready tables
# ─────────────────────────────────────────────────────────────────────────────

def load_all_results() -> dict:
    """Load all saved metric JSON files from results/."""
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith("_metrics.json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
        key = fname.replace("_metrics.json", "")
        results[key] = data
    return results

def print_table(results: dict, metric: str = "bertscore_f1", mode: str = "zeroshot"):
    try:
        from tabulate import tabulate
    except ImportError:
        print("pip install tabulate for formatted tables")
        return

    header = ["Model"] + [TASK_SHORT[t] for t in TASK_ORDER] + ["Avg"]
    rows   = []

    for model_key in MODEL_REGISTRY:
        run_key = f"{model_key}_{mode}"
        if run_key not in results:
            continue
        task_data = results[run_key].get("tasks", {})
        model_name = MODEL_REGISTRY[model_key]["name"]
        vals = [task_data.get(t, {}).get(metric, 0.0) for t in TASK_ORDER]
        avg  = round(sum(vals) / len([v for v in vals if v > 0]), 2) if any(vals) else 0.0
        rows.append([model_name] + [f"{v:.2f}" for v in vals] + [f"{avg:.2f}"])

    # Sort by avg descending
    rows.sort(key=lambda r: float(r[-1]), reverse=True)

    metric_label = {
        "bertscore_f1": "BERTScore-F1",
        "bleu":         "BLEU",
        "chrf_pp":      "chrF++",
        "rouge_l":      "ROUGE-L",
        "word_f1":      "Word-F1",
    }.get(metric, metric)

    print(f"\n{'─'*80}")
    print(f"  {metric_label}  |  Mode: {mode}")
    print(f"{'─'*80}")
    print(tabulate(rows, headers=header, tablefmt="github"))

def print_full_report():
    results = load_all_results()
    if not results:
        print("No results found in results/ — run evaluations first.")
        return
    for mode in ("zeroshot", "fewshot"):
        for metric in ("bertscore_f1", "bleu", "chrf_pp", "rouge_l", "word_f1"):
            print_table(results, metric=metric, mode=mode)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM Unified Evaluator")
    parser.add_argument("--model",   default="prism",
                        help="Model key: prism | llama | mistral | gemma | qwen | all")
    parser.add_argument("--mode",    default="zeroshot",
                        help="Prompting mode: zeroshot | fewshot | all")
    parser.add_argument("--eval_set",default=EVAL_SET_PATH,
                        help="Path to Punjabi eval JSONL")
    parser.add_argument("--report",  action="store_true",
                        help="Print comparison tables from saved results")
    args = parser.parse_args()

    if args.report:
        print_full_report()
        return

    # Determine which models + modes to run
    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    modes  = ["zeroshot", "fewshot"]     if args.mode  == "all" else [args.mode]

    # Validate
    for m in models:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{m}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    print(f"\nLoading eval set from: {args.eval_set}")
    eval_buckets = load_eval_set(args.eval_set)

    for model_key in models:
        for mode in modes:
            print(f"\n{'='*70}")
            print(f"  Evaluating: {MODEL_REGISTRY[model_key]['name']}  |  Mode: {mode}")
            print(f"{'='*70}")
            t0 = time.time()
            evaluate_model(model_key, mode, eval_buckets)
            elapsed = time.time() - t0
            print(f"\n  Done in {elapsed:.1f}s")

    print("\n\nAll evaluations complete. Run with --report to see comparison tables.")

if __name__ == "__main__":
    main()
