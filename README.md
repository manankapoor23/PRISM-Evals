# Punjabi Text Rewriting Evaluation Toolkit

A compact repo for building and evaluating a Punjabi (Gurmukhi) benchmark across six rewriting tasks.

## Goal

Train-time and eval-time consistency for Punjabi text rewriting by:

- building a Punjabi evaluation JSONL from translated pairs,
- injecting realistic noise for robustness testing,
- evaluating PRISM and baseline LLMs in zero-shot and few-shot modes,
- generating task-wise and model-wise comparison tables.

This repo is designed for fast, repeatable model comparisons on the same fixed evaluation set.

## Tasks Covered

- Compression
- Text Normalization
- Noise Robust Normalization
- Simplification
- Style Paraphrase
- Controlled Rewrite

## Repository Files

- `build_pa_jsonl.py`: Builds Punjabi eval JSONL from translated sheet + English source IDs/tasks.
- `noise_injector.py`: Adds synthetic noise to Noise Robust Normalization inputs.
- `evaluate.py`: Unified evaluator (PRISM + baselines) with metrics + reporting.
- `prism_eval_en.jsonl`: English source evaluation set (IDs/tasks/metadata).
- `translation_sheet.tsv`: Translation sheet used to create Punjabi eval JSONL.
- `translation_sheet.csv`: CSV version of translation data.
- `punjab.txt`: Human-readable Punjabi task data.
- `english_input_reference_output.csv`: Extracted input/reference pairs.
- `requirements.txt`: Python dependencies.

## Setup

### 1) Create and activate environment

```bash
python3 -m venv eval
source eval/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## End-to-End Workflow

### Step 1: Build Punjabi eval JSONL

Use your current root-level files:

```bash
python build_pa_jsonl.py \
  --sheet translation_sheet.tsv \
  --source prism_eval_en.jsonl \
  --out prism_eval_pa.jsonl
```

### Step 2: Inject noise (for robustness task)

```bash
python noise_injector.py \
  --input prism_eval_pa.jsonl \
  --output prism_eval_pa_final.jsonl
```

### Step 3: Configure model paths in evaluator

Open `evaluate.py` and update `MODEL_REGISTRY` (especially `prism.path`) to valid local or hub model paths.

### Step 4: Run evaluations

Single model:

```bash
python evaluate.py --model prism --mode zeroshot --eval_set prism_eval_pa_final.jsonl
python evaluate.py --model prism --mode fewshot  --eval_set prism_eval_pa_final.jsonl
```

Baselines:

```bash
python evaluate.py --model llama   --mode zeroshot --eval_set prism_eval_pa_final.jsonl
python evaluate.py --model mistral --mode fewshot  --eval_set prism_eval_pa_final.jsonl
python evaluate.py --model gemma   --mode zeroshot --eval_set prism_eval_pa_final.jsonl
python evaluate.py --model qwen    --mode fewshot  --eval_set prism_eval_pa_final.jsonl
```

All models and both modes:

```bash
python evaluate.py --model all --mode all --eval_set prism_eval_pa_final.jsonl
```

### Step 5: Print comparison tables

```bash
python evaluate.py --report
```

## Metrics Reported

- BLEU (character tokenization)
- chrF++
- ROUGE-L
- BERTScore (P/R/F1, multilingual)
- Word-level F1 (token overlap)

## Output Artifacts

`evaluate.py` writes results to `results/`:

- per-task predictions: `<model>_<task>_<mode>.jsonl`
- per-run metrics: `<model>_<mode>_metrics.json`

The report mode loads all saved metrics files and prints ranked tables by metric and mode.

## Notes

- `evaluate.py` defaults to `eval_set/prism_eval_pa_final.jsonl`. In this repo, your generated files are currently in the root, so pass `--eval_set prism_eval_pa_final.jsonl` unless you move files into an `eval_set/` folder.
- Few-shot examples are sampled per task with a fixed seed for reproducibility.
- The evaluator uses vLLM and expects compatible GPU resources.

## Minimal Quick Run

```bash
source eval/bin/activate
pip install -r requirements.txt
python build_pa_jsonl.py --sheet translation_sheet.tsv --source prism_eval_en.jsonl --out prism_eval_pa.jsonl
python noise_injector.py --input prism_eval_pa.jsonl --output prism_eval_pa_final.jsonl
python evaluate.py --model prism --mode zeroshot --eval_set prism_eval_pa_final.jsonl
python evaluate.py --report
```

## Future Improvements

- Add a data validation script for missing IDs or empty translations.
- Add a small unit test suite for parsing and metric sanity checks.
- Add a reproducible config file for model paths and runtime settings.
# PRISM-Evals
