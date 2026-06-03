# datadoubler.py
"""
DataDoubler - Expand a Q&A dataset by generating rephrased question variants
and fresh answers for them.

For each entry the LLM picks one rephrasing style (straight rephrase, simpler,
more detailed, or — if allowed — negative framing) and emits ONLY the variant
question.  A brand-new answer is then generated from that variant alone (no
reference to the original answer), so question/answer consistency is preserved
even when the framing flips.

Each run doubles the working set (exponential): N runs → (2^N)× original size.
Capped at 4 runs = 16× original.
"""

import os
import json
import random
from datetime import datetime

from datacore.llm.client import LLMClient
from datacore.config.loader import load_tool_config
from datacore.progress import ProgressReporter
from datacore.io.json_ops import save_json
from datacore.io.formats import detect_format, apply_output_format
from datacore.io.loaders import load_local_dataset, load_huggingface_dataset
from datacore.personas.loader import get_persona

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "DATASET_NAME":   "doubled-dataset",
    "JOB_ID":         "default-job",
    "IMPORT_PATH":    "import",
    "SOURCE_TYPE":    "local",        # "local" | "huggingface"
    "HF_DATASET":     "",             # used when SOURCE_TYPE == "huggingface"
    "HF_SUBSET":      None,
    "RUNS":           1,
    "ALLOW_NEGATIVE": True,
    "USE_PERSONA":    False,
    "PERSONA_NAME":   "",
    "OUTPUT_FORMAT":  "alpaca",
    "SAVE_INTERVAL":  50,
    "LLM_SETTINGS":   {"base_url": None, "llm_model": None},
}
DEFAULT_CONFIG = load_tool_config(DEFAULT_CONFIG, tool_name="datadoubler")

DATASET_NAME   = DEFAULT_CONFIG["DATASET_NAME"]
IMPORT_PATH    = DEFAULT_CONFIG["IMPORT_PATH"]
SOURCE_TYPE    = DEFAULT_CONFIG.get("SOURCE_TYPE", "local")
HF_DATASET     = DEFAULT_CONFIG.get("HF_DATASET", "")
HF_SUBSET      = DEFAULT_CONFIG.get("HF_SUBSET") or None
RUNS           = max(1, min(4, int(DEFAULT_CONFIG.get("RUNS", 1))))
ALLOW_NEGATIVE = bool(DEFAULT_CONFIG.get("ALLOW_NEGATIVE", True))
USE_PERSONA    = bool(DEFAULT_CONFIG.get("USE_PERSONA", False))
PERSONA_NAME   = DEFAULT_CONFIG.get("PERSONA_NAME", "")
OUTPUT_FORMAT  = DEFAULT_CONFIG.get("OUTPUT_FORMAT", "alpaca")
SAVE_INTERVAL  = int(DEFAULT_CONFIG.get("SAVE_INTERVAL", 50))

OUTPUT_PATH = "."
HF_TOKEN    = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_API_KEY", None))


# ============================================================================
# Tool-specific helpers
# ============================================================================

def extract_qa(entry, fmt):
    """Return (question, answer) from an entry; empty strings if unresolvable."""
    if fmt == "alpaca":
        q = entry.get("instruction", "")
        if entry.get("input"):
            q = f"{q}\n\n{entry['input']}"
        return q, entry.get("output", "")
    if fmt == "qa":
        return entry.get("question", ""), entry.get("answer", "")
    if fmt == "sharegpt":
        convs = entry.get("conversations", [])
        q = next((m.get("value", "") for m in convs if m.get("from") in ("user", "human")), "")
        a = next((m.get("value", "") for m in convs if m.get("from") in ("assistant", "gpt")), "")
        return q, a
    # Best-effort fallback
    for qk, ak in (("question", "answer"), ("instruction", "output"), ("prompt", "response")):
        if qk in entry and ak in entry:
            return entry[qk], entry[ak]
    return "", ""


# ============================================================================
# LLM prompts
# ============================================================================

VARIANT_SYSTEM = (
    "You rephrase user questions to create training-data variants. "
    "You output ONLY the rephrased question — no preamble, no labels, no quotes, "
    "no explanation. Begin with the first word of the question itself."
)

def build_variant_prompt(question, answer, allow_negative):
    styles = [
        "- Straight rephrase: same intent, noticeably different wording and structure",
        "- Simpler/shorter version: the same question asked more plainly or briefly",
        "- More detailed/elaborated version: the same question with extra specificity or context a real user might add",
    ]
    if allow_negative:
        styles.append(
            "- Negatively framed version: asks about the inverse, opposite, or 'what NOT to do' angle — "
            "only choose this when it produces a question a real person might realistically ask"
        )
    style_block = "\n".join(styles)

    return (
        f"Original question:\n{question}\n\n"
        f"Original answer (context only — do NOT reference it in your output):\n{answer}\n\n"
        "Rewrite the question. Pick whichever ONE of the following styles produces the "
        "most useful, realistic variant for this particular question:\n"
        f"{style_block}\n\n"
        "Requirements for your output:\n"
        "- Must sound like something a real person would actually ask\n"
        "- Must be meaningfully different from the original in wording or framing\n"
        "- Must stand alone: no reference to the original question or answer\n"
        "- Output ONLY the variant question text itself. No labels, quotes, or notes.\n"
    )


def build_answer_system(persona_description=None):
    base = "You answer the user's question directly and helpfully."
    if persona_description:
        return f"{base}\n\n{persona_description}"
    return base


# ============================================================================
# Main
# ============================================================================

def _emit_progress(current, total):
    print(f"PROGRESS {current}/{total}", flush=True)


if __name__ == "__main__":
    print("DataDoubler - Dataset Expansion via Question Variants")
    print("=" * 60)
    print(f"Dataset:   {DATASET_NAME}")
    print(f"Source:    {SOURCE_TYPE}" + (f" ({HF_DATASET})" if SOURCE_TYPE == "huggingface" else ""))
    print(f"Runs:      {RUNS}  →  {2 ** RUNS}× original size")
    print(f"Negative:  {'allowed' if ALLOW_NEGATIVE else 'disabled'}")
    print(f"Persona:   {PERSONA_NAME if USE_PERSONA and PERSONA_NAME else '(none)'}")
    print(f"Output fmt: {OUTPUT_FORMAT}")
    print("=" * 60)

    # ── Load source dataset ──────────────────────────────────────────────────
    if SOURCE_TYPE == "huggingface":
        if not HF_DATASET:
            print("Error: SOURCE_TYPE=huggingface but HF_DATASET is empty.")
            exit(1)
        print(f"\nLoading HF dataset: {HF_DATASET}")
        try:
            source_data = load_huggingface_dataset(HF_DATASET, HF_SUBSET, HF_TOKEN)
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            exit(1)
    else:
        if not os.path.isdir(IMPORT_PATH):
            print(f"Error: import directory '{IMPORT_PATH}' not found.")
            exit(1)
        source_files = [f for f in os.listdir(IMPORT_PATH) if f.endswith((".json", ".jsonl"))]
        if not source_files:
            print(f"Error: no .json/.jsonl file in '{IMPORT_PATH}'.")
            exit(1)
        src_path = os.path.join(IMPORT_PATH, source_files[0])
        print(f"\nLoading local file: {src_path}")
        source_data = load_local_dataset(src_path)

    if not source_data:
        print("Error: source dataset is empty.")
        exit(1)

    fmt = detect_format(source_data)
    print(f"Detected format: {fmt}")
    print(f"Entries: {len(source_data)}")

    # Normalise originals into flat {question, answer} form (preserving provenance)
    originals = []
    for entry in source_data:
        q, a = extract_qa(entry, fmt)
        if not q or not a:
            continue
        originals.append({
            "question":  q,
            "answer":    a,
            "_source":   "original",
            "_tool":     "datadoubler",
            "_version":  "2.0",
        })

    if not originals:
        print("Error: no usable question/answer pairs found in source.")
        exit(1)

    print(f"Usable originals: {len(originals)}\n")

    # ── LLM setup ────────────────────────────────────────────────────────────
    llm_settings = DEFAULT_CONFIG.get("LLM_SETTINGS", {}) or {}
    client = LLMClient(
        base_url=llm_settings.get("base_url"),
        default_model=llm_settings.get("llm_model"),
    )

    answer_system = None
    if USE_PERSONA and PERSONA_NAME:
        try:
            persona = get_persona(PERSONA_NAME)
            answer_system = build_answer_system(persona.get("description", ""))
            print(f"Using persona '{PERSONA_NAME}' for fresh answers.\n")
        except Exception as e:
            print(f"Warning: could not load persona '{PERSONA_NAME}' ({e}). Continuing without.\n")
            answer_system = build_answer_system()
    else:
        answer_system = build_answer_system()

    # ── Doublification loop ──────────────────────────────────────────────────
    working = list(originals)      # grows each run
    all_variants = []              # flat list of all generated variants
    total_variants_target = len(originals) * (2 ** RUNS - 1)
    generated_so_far = 0

    checkpoint_path = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}.checkpoint.json")

    for run_idx in range(1, RUNS + 1):
        print(f"── Run {run_idx}/{RUNS} — processing {len(working)} entries ──")
        reporter = ProgressReporter(total=len(working), phase=f"Run {run_idx}/{RUNS}")

        new_variants = []
        for i, entry in enumerate(working, start=1):
            q = entry["question"]
            a = entry["answer"]

            try:
                variant_q = client.call(
                    prompt=build_variant_prompt(q, a, ALLOW_NEGATIVE),
                    system_prompt=VARIANT_SYSTEM,
                    temperature=0.8,
                    max_tokens=300,
                    extra_body={"enable_thinking": False},
                ).strip()
            except Exception as e:
                print(f"  [{i}] variant generation failed: {e}")
                generated_so_far += 1
                _emit_progress(generated_so_far, total_variants_target)
                reporter.update(i)
                continue

            # Strip common wrapper artefacts (quotes, leading labels)
            variant_q = variant_q.strip().strip('"').strip("'").strip()
            if variant_q.lower().startswith(("question:", "variant:", "rephrased:")):
                variant_q = variant_q.split(":", 1)[1].strip()

            if not variant_q:
                generated_so_far += 1
                _emit_progress(generated_so_far, total_variants_target)
                reporter.update(i)
                continue

            try:
                new_answer = client.call(
                    prompt=variant_q,
                    system_prompt=answer_system,
                    temperature=0.7,
                    max_tokens=2000,
                ).strip()
            except Exception as e:
                print(f"  [{i}] answer generation failed: {e}")
                generated_so_far += 1
                _emit_progress(generated_so_far, total_variants_target)
                reporter.update(i)
                continue

            new_variants.append({
                "question":  variant_q,
                "answer":    new_answer,
                "_source":   f"variant_run_{run_idx}",
                "_tool":     "datadoubler",
                "_version":  "2.0",
            })

            generated_so_far += 1
            reporter.update(i)
            _emit_progress(generated_so_far, total_variants_target)

            if generated_so_far % SAVE_INTERVAL == 0:
                save_json(originals + all_variants + new_variants, checkpoint_path)

        all_variants.extend(new_variants)
        working = working + new_variants   # exponential growth for next run
        print(f"Run {run_idx} complete: +{len(new_variants)} variants  (cumulative: {len(originals) + len(all_variants)})\n")
        save_json(originals + all_variants, checkpoint_path)

    # ── Combine + format output ──────────────────────────────────────────────
    final = originals + all_variants
    print(f"Final dataset size: {len(final)}  ({len(originals)} original + {len(all_variants)} variants)")

    data_to_save, fmt_suffix = apply_output_format(
        final, OUTPUT_FORMAT,
        instruction_key="question", output_key="answer", input_key=None,
    )

    timestamp = datetime.now().strftime("%d%m%y")
    size_k    = round(len(final) / 1000, 2)
    filename  = f"{DATASET_NAME}-{fmt_suffix.capitalize()}-{size_k}k-{timestamp}.json"
    filepath  = os.path.join(OUTPUT_PATH, filename)
    save_json(data_to_save, filepath)

    # Clean up checkpoint once the final file is safely written
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass

    usage = client.get_usage_stats()
    print(f"TOKENS {usage['prompt_tokens']}/{usage['completion_tokens']}", flush=True)

    print("\n" + "=" * 60)
    print("DataDoubler complete!")
    print(f"Output:  {filepath}")
    print(f"Tokens:  {usage['total_tokens']:,} total")
    print("=" * 60)
