# Punjabi Text Rewriting Evaluation Toolkit

This repository presents a focused evaluation project for Punjabi (Gurmukhi) rewriting quality across multiple real-world transformation tasks.

## Project Intent

This is a research-style benchmark repository meant to be read and inspected.

The project demonstrates:

- task-diverse Punjabi rewriting evaluation,
- parallel assessment of a fine-tuned system and strong baseline models,
- robustness testing through controlled noise injection,
- consistent metric-based comparison across task families.

## Task Families

- Compression
- Text Normalization
- Noise Robust Normalization
- Simplification
- Style Paraphrase
- Controlled Rewrite

Together, these tasks represent both meaning-preserving rewriting and form-sensitive transformation.

## Dataset and Pipeline Design

The benchmark is built from a curated English seed set and a Punjabi translation layer, then transformed into a model-ready JSONL format. A dedicated noise pass is applied to robustness examples to emulate realistic orthographic and punctuation corruption.

At evaluation time, each model is tested in both zero-shot and few-shot prompting modes under a shared prompt/evaluation framework to keep comparisons fair.

## Evaluation Lens

Performance is measured using complementary lexical and semantic metrics:

- BLEU
- chrF++
- ROUGE-L
- BERTScore (precision, recall, F1)
- Word-level F1 overlap

This combination is intended to capture both surface alignment and semantic adequacy for Punjabi outputs.

## Repository Contents

- build_pa_jsonl.py: builds Punjabi evaluation records from source IDs and translated fields.
- noise_injector.py: injects synthetic noise for robustness-oriented inputs.
- evaluate.py: unified evaluation engine with multi-model support and report generation.
- prism_eval_en.jsonl: English reference source set.
- translation_sheet.tsv and translation_sheet.csv: translation staging sheets.
- punjab.txt: readable Punjabi task examples and references.
- english_input_reference_output.csv: extracted input/reference pairs.
- requirements.txt: dependency manifest.

## What This Repo Shows

- A compact but complete Punjabi rewriting benchmark workflow.
- A reproducible structure for model-to-model comparison.
- A practical framework for tracking quality across diverse rewrite objectives.

## Current Scope

The repository is intentionally small and focused, prioritizing clarity of benchmark construction and interpretability of results over large-scale infrastructure.

## Next Evolution

- Add explicit data validation checks for missing or malformed records.
- Add lightweight automated tests for parser and metric sanity.
- Externalize model/runtime configuration for cleaner experiment management.
# PRISM-Evals
