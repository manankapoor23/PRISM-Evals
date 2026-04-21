"""
Noise Injector for Punjabi (Gurmukhi) Text
Injects mixed noise into noise_robust_normalization task inputs in prism_eval_pa.jsonl.
Run AFTER build_pa_jsonl.py.

Noise types applied:
  1. Typos — random character substitutions within Gurmukhi Unicode range
  2. Punctuation errors — doubled/missing punctuation, extra spaces
  3. Phonetic romanization — random Gurmukhi words replaced with roman transliteration

Usage:
    python noise_injector.py --input eval_set/prism_eval_pa.jsonl \
                             --output eval_set/prism_eval_pa_noised.jsonl
"""

import json, random, re, argparse, unicodedata
random.seed(42)

# ── Gurmukhi Unicode range: 0x0A00–0x0A7F ──────────────────────────────────
GURMUKHI_CHARS = [chr(c) for c in range(0x0A05, 0x0A73) if unicodedata.category(chr(c)) != "Cn"]

# Common Gurmukhi → rough phonetic roman mappings (word-level)
PHONETIC_MAP = {
    "ਹੈ":    "hai",   "ਹਨ":   "han",   "ਅਤੇ":  "ate",
    "ਵਿੱਚ": "vich",  "ਨੂੰ":  "nu",    "ਦੇ":   "de",
    "ਦੀ":   "di",    "ਦਾ":   "da",    "ਕਿ":   "ki",
    "ਨਹੀਂ": "nahi",  "ਪਰ":   "par",   "ਜੋ":   "jo",
    "ਇਸ":   "is",    "ਉਹ":   "oh",    "ਇੱਕ":  "ik",
    "ਕਰਨ":  "karn",  "ਹੋ":   "ho",    "ਜਦੋਂ": "jadon",
    "ਸਾਡੇ": "saade", "ਤੋਂ":  "ton",   "ਲਈ":   "lai",
}

def inject_typo(word, rate=0.15):
    """Randomly substitute one character in a Gurmukhi word."""
    if not word or random.random() > rate:
        return word
    chars = list(word)
    idx = random.randint(0, len(chars) - 1)
    chars[idx] = random.choice(GURMUKHI_CHARS)
    return "".join(chars)

def inject_punctuation_noise(text, rate=0.3):
    """Double commas/periods, add random extra spaces."""
    # Double punctuation
    text = re.sub(r'([,।])', lambda m: m.group(1) * random.randint(1, 3)
                  if random.random() < rate else m.group(1), text)
    # Extra spaces
    words = text.split()
    noisy = []
    for w in words:
        noisy.append(w)
        if random.random() < 0.12:
            noisy.append("")   # creates double space on join
    return " ".join(noisy)

def inject_phonetic(word):
    """Replace a Gurmukhi word with its phonetic roman equivalent if mapped."""
    return PHONETIC_MAP.get(word, word)

def inject_mixed_noise(text, typo_rate=0.15, phonetic_rate=0.08):
    """Apply all three noise types to a text string."""
    words = text.split()
    noisy_words = []
    for w in words:
        # Phonetic substitution
        if random.random() < phonetic_rate and w in PHONETIC_MAP:
            noisy_words.append(inject_phonetic(w))
        else:
            noisy_words.append(inject_typo(w, rate=typo_rate))
    result = " ".join(noisy_words)
    result = inject_punctuation_noise(result)
    return result

def process(in_path, out_path):
    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    noised_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            if rec["task"] == "noise_robust_normalization":
                # Extract input from text field
                lines = rec["text"].split("\n")
                in_start = False
                input_lines, resp_lines, inst_lines = [], [], []
                section = None
                for line in lines:
                    if line.startswith("### Instruction:"):
                        section = "inst"
                        inst_lines.append(line.replace("### Instruction:", "").strip())
                    elif line.startswith("### Input:"):
                        section = "input"
                        input_lines.append(line.replace("### Input:", "").strip())
                    elif line.startswith("### Response:"):
                        section = "resp"
                        resp_lines.append(line.replace("### Response:", "").strip())
                    elif section == "inst":
                        inst_lines.append(line)
                    elif section == "input":
                        input_lines.append(line)
                    elif section == "resp":
                        resp_lines.append(line)

                clean_input = " ".join(input_lines).strip()
                noisy_input = inject_mixed_noise(clean_input)
                instruction = " ".join(inst_lines).strip()
                response    = " ".join(resp_lines).strip()

                rec["text"] = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{noisy_input}\n\n"
                    f"### Response:\n{response}"
                )
                rec["meta"]["noise_injected"] = True
                noised_count += 1

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Noise injected into {noised_count} records → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="eval_set/prism_eval_pa.jsonl")
    parser.add_argument("--output", default="eval_set/prism_eval_pa_final.jsonl")
    args = parser.parse_args()
    process(args.input, args.output)
