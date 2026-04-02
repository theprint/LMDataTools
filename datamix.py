# datamix.py
"""
DataMix - Mix and sample datasets from HuggingFace into a unified Alpaca format.

Supports auto-detection of 20+ common dataset field layouts, plus ShareGPT
conversations. Custom format overrides can be specified per dataset source.
"""

import os
import json
import random
from datetime import datetime
from datasets import load_dataset
from datacore.io.json_ops import save_json
from datacore.io.formats import apply_output_format
from datacore.progress import ProgressReporter

# ============================================================================
# FORMAT DEFINITIONS
# ============================================================================
# Each entry maps a dataset layout to (instruction, input, output) fields.
# 'input_key' is optional (None = no context field).
# 'instruction_processor' / 'input_processor' are names of special functions
# applied after extracting the raw field value.
#
# Order matters: auto-detection tries definitions in order and returns the
# first where both instruction_key and output_key are present in the entry.
# More specific layouts (more required keys) should come first.

FORMAT_DEFINITIONS = [
    # Standard Alpaca — instruction + input + output (most specific, check first)
    {"name": "alpaca",                      "instruction_key": "instruction",       "input_key": "input",    "output_key": "output"},
    # Instruction / response variants
    {"name": "instruction_response",        "instruction_key": "instruction",       "input_key": None,       "output_key": "response"},
    {"name": "instruction_output",          "instruction_key": "instruction",       "input_key": None,       "output_key": "output"},
    {"name": "instr_chosen_resp",           "instruction_key": "instruction",       "input_key": None,       "output_key": "chosen_response"},
    {"name": "instr_demonstration",         "instruction_key": "instruction",       "input_key": None,       "output_key": "demonstration"},
    {"name": "info_summary",                "instruction_key": "instruction",       "input_key": "info",     "output_key": "summary",
     "instruction_processor": "random_summary_prompt", "input_processor": "extract_post_info"},
    # Capitalised keys (some datasets use ALL CAPS or Title Case)
    {"name": "cap_instruction_response",    "instruction_key": "INSTRUCTION",       "input_key": None,       "output_key": "RESPONSE"},
    {"name": "cap_context_response",        "instruction_key": "Context",           "input_key": None,       "output_key": "Response"},
    {"name": "cap_human_assistant",         "instruction_key": "Human",             "input_key": None,       "output_key": "Assistant"},
    # Problem / solution variants
    {"name": "problem_answer",              "instruction_key": "problem",           "input_key": None,       "output_key": "answer"},
    {"name": "problem_description_response","instruction_key": "problem-description","input_key": None,      "output_key": "response"},
    {"name": "problem_gold_standard",       "instruction_key": "problem",           "input_key": None,       "output_key": "gold_standard_solution"},
    # Prompt variants
    {"name": "prompt_response",             "instruction_key": "prompt",            "input_key": None,       "output_key": "response"},
    {"name": "prompt_chosen",               "instruction_key": "prompt",            "input_key": None,       "output_key": "chosen"},
    {"name": "prompt_question",             "instruction_key": "prompt",            "input_key": None,       "output_key": "question"},
    # Query / question variants
    {"name": "query_answer",                "instruction_key": "query",             "input_key": None,       "output_key": "answer"},
    {"name": "question_answer",             "instruction_key": "question",          "input_key": None,       "output_key": "answer"},
    {"name": "question_response",           "instruction_key": "question",          "input_key": None,       "output_key": "response"},
    {"name": "question_cot",                "instruction_key": "question",          "input_key": None,       "output_key": "cot"},
    {"name": "question_solution",           "instruction_key": "question",          "input_key": "choices",  "output_key": "solution",
     "input_processor": "join_choices"},
    # Generic input/output
    {"name": "input_output",                "instruction_key": "input",             "input_key": None,       "output_key": "output"},
]

# Build a fast name → definition lookup
FORMAT_BY_NAME = {d["name"]: d for d in FORMAT_DEFINITIONS}
# Add aliases so the webapp dropdown labels work
FORMAT_BY_NAME["qa"] = {"name": "qa", "instruction_key": "question", "input_key": None, "output_key": "answer"}

# ============================================================================
# SPECIAL FIELD PROCESSORS
# ============================================================================

SUMMARY_PROMPTS = [
    "Summarize the following content:",
    "Provide a concise summary of the text below:",
    "Write a brief summary of the following:",
    "Summarize the key points from the content below:",
    "Give a short summary of the following text:",
    "In a few sentences, summarize the following:",
    "Create a summary of the content below:",
    "What are the main points of the following text?",
    "Condense the following into a brief summary:",
    "Summarize the following passage:",
]


def join_choices(choices):
    """Format a list of answer choices as labelled options (A, B, C…)."""
    if isinstance(choices, list):
        return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return str(choices) if choices else ""


def random_summary_prompt(_instruction):
    """Return a random summarisation prompt (ignores the original instruction value)."""
    return random.choice(SUMMARY_PROMPTS)


def extract_post_info(info):
    """Extract a usable text string from a structured post/info field."""
    if isinstance(info, dict):
        for key in ("text", "selftext", "body", "content", "post", "article"):
            if info.get(key):
                return str(info[key])
        parts = [str(v) for v in info.values() if isinstance(v, str) and v]
        return "\n".join(parts[:3])
    return str(info) if info else ""


PROCESSORS = {
    "join_choices":         join_choices,
    "random_summary_prompt": random_summary_prompt,
    "extract_post_info":    extract_post_info,
}

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

if os.path.exists("config.json"):
    with open("config.json", 'r') as f:
        job_config = json.load(f)

    TOTAL_SAMPLES         = job_config.get("total_samples", 10000)
    DATASET_NAME          = job_config.get("dataset_name", "mixed-dataset")
    SEED                  = job_config.get("seed", 310576)
    MIN_INSTRUCTION_LENGTH = job_config.get("min_instruction_length", 10)
    MAX_INSTRUCTION_LENGTH = job_config.get("max_instruction_length", 4000)
    MIN_OUTPUT_LENGTH     = job_config.get("min_output_length", 10)
    MAX_OUTPUT_LENGTH     = job_config.get("max_output_length", 4000)
    OUTPUT_FORMAT         = job_config.get("output_format", "alpaca")

    DATASET_SOURCES = [
        (ds["name"], ds["weight"], ds.get("subset"))
        for ds in job_config.get("dataset_sources", [])
    ]

    # Per-source format overrides: dataset_name → format name
    DATASET_FORMATS = {}
    for ds in job_config.get("dataset_sources", []):
        if ds.get("format") and ds["format"] != "auto":
            DATASET_FORMATS[ds["name"]] = ds["format"]

else:
    # Standalone defaults — edit these for your needs
    TOTAL_SAMPLES          = 10_000
    DATASET_NAME           = "mixed-dataset"
    SEED                   = 310576
    MIN_INSTRUCTION_LENGTH = 10
    MAX_INSTRUCTION_LENGTH = 4000
    MIN_OUTPUT_LENGTH      = 10
    MAX_OUTPUT_LENGTH      = 4000
    OUTPUT_FORMAT          = "alpaca"

    DATASET_SOURCES = [
        ("theprint/databird-sensible",    0.09, None),
        ("theprint/databird-negotiation", 0.10, None),
        ("theprint/databird-power",       0.09, None),
    ]

    DATASET_FORMATS = {}

HF_TOKEN   = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_API_KEY", None))
OUTPUT_PATH = "."

# ============================================================================
# END CONFIGURATION
# ============================================================================


def _is_chatml_list(val):
    """Return True if *val* looks like a ChatML messages list (list of role/content dicts)."""
    return (
        isinstance(val, list)
        and len(val) > 0
        and isinstance(val[0], dict)
        and "role" in val[0]
    )


def detect_format(entry):
    """
    Auto-detect the layout of a dataset entry.
    Returns the format name, or 'unknown' if nothing matches.
    """
    if "conversations" in entry:
        return "sharegpt"
    if "messages" in entry:
        return "messages"
    # ChatML stored as a list in the "input" field, response in "output"
    if _is_chatml_list(entry.get("input")):
        return "chatml_input"
    for defn in FORMAT_DEFINITIONS:
        if defn["instruction_key"] in entry and defn["output_key"] in entry:
            return defn["name"]
    return "unknown"


def _extract_from_definition(entry, defn):
    """Extract (instruction, input_text, output) using a format definition."""
    instruction = entry.get(defn["instruction_key"], "")
    output      = entry.get(defn["output_key"], "")
    input_text  = entry.get(defn.get("input_key") or "", "") if defn.get("input_key") else ""

    # Apply optional field processors
    inst_proc = defn.get("instruction_processor")
    inp_proc  = defn.get("input_processor")

    if inst_proc and inst_proc in PROCESSORS:
        instruction = PROCESSORS[inst_proc](instruction)
    if inp_proc and inp_proc in PROCESSORS:
        input_text = PROCESSORS[inp_proc](input_text)

    return instruction, input_text, output


def extract_qa_from_entry(entry, dataset_name):
    """
    Extract (instruction, input_text, output) from an entry.

    Resolution order:
      1. Per-source format override (DATASET_FORMATS)
      2. ShareGPT conversation structure
      3. Auto-detection via FORMAT_DEFINITIONS
    """
    # 1. User-specified format override
    override = DATASET_FORMATS.get(dataset_name)
    if override and override != "auto":
        defn = FORMAT_BY_NAME.get(override)
        if defn:
            # Conversation/chat formats are handled below; named definitions extracted here
            if override not in ("sharegpt", "messages", "chatml_input"):
                return _extract_from_definition(entry, defn)

    # 2. ShareGPT conversations (from/value keys)
    if "conversations" in entry or override == "sharegpt":
        convs     = entry.get("conversations", [])
        user_msg  = next((m.get("value", "") for m in convs if m.get("from") in ["user", "human"]), "")
        asst_msg  = next((m.get("value", "") for m in convs if m.get("from") in ["assistant", "gpt"]), "")
        return user_msg, "", asst_msg

    # 3. OpenAI/ChatML — messages key (list of role/content dicts)
    if "messages" in entry or override == "messages":
        msgs     = entry.get("messages", [])
        user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        asst_msg = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        return user_msg, "", asst_msg

    # 3b. OpenAI/ChatML — messages list in "input" field, response in "output"
    input_val = entry.get("input")
    if _is_chatml_list(input_val) or override == "chatml_input":
        msgs     = input_val if isinstance(input_val, list) else []
        user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        asst_msg = entry.get("output", "")
        return user_msg, "", asst_msg

    # 4. Auto-detect: first matching definition wins
    for defn in FORMAT_DEFINITIONS:
        if defn["instruction_key"] in entry and defn["output_key"] in entry:
            return _extract_from_definition(entry, defn)

    return "", "", ""


def validate_entry(instruction, output):
    """Check that an entry meets minimum quality/length requirements."""
    if not instruction or not output:
        return False
    if not (MIN_INSTRUCTION_LENGTH <= len(instruction) <= MAX_INSTRUCTION_LENGTH):
        return False
    if not (MIN_OUTPUT_LENGTH <= len(output) <= MAX_OUTPUT_LENGTH):
        return False
    return True


def process_dataset(dataset_name, weight, subset, hf_token, seed, total_samples):
    """Load, shuffle, sample, and normalise a HuggingFace dataset."""
    print(f"\nProcessing: {dataset_name}")

    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split="train", token=hf_token)
        else:
            ds = load_dataset(dataset_name, split="train", token=hf_token)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return []

    ds = ds.shuffle(seed=seed)
    samples_count = min(round(total_samples * weight), len(ds))
    print(f"  Taking {samples_count} of {len(ds)} entries ({weight * 100:.1f}%)")

    # Detect format from first entry for display
    if len(ds) > 0:
        detected = detect_format(dict(ds[0]))
        override = DATASET_FORMATS.get(dataset_name, "auto")
        fmt_label = override if override != "auto" else f"{detected} (auto)"
        print(f"  Format: {fmt_label}")

    processed = []
    for i in range(samples_count):
        entry = dict(ds[i])
        instruction, input_text, output = extract_qa_from_entry(entry, dataset_name)

        if not validate_entry(instruction, output):
            continue

        processed.append({
            "source":      dataset_name,
            "instruction": instruction,
            "input":       input_text,
            "output":      output,
            "_tool":       "datamix",
            "_version":    "2.0",
        })

    print(f"  Kept {len(processed)} valid entries")
    return processed


if __name__ == "__main__":
    print("DataMix - Dataset Mixing Tool")
    print("=" * 60)
    print(f"Target samples: {TOTAL_SAMPLES:,}")
    print(f"Datasets to mix: {len(DATASET_SOURCES)}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    total_weight = sum(w for _, w, _ in DATASET_SOURCES)
    if abs(total_weight - 1.0) > 0.01:
        print(f"\nWarning: Weights sum to {total_weight:.3f}, not 1.0 — sample counts may vary.\n")

    all_data = []
    mix_reporter = ProgressReporter(total=len(DATASET_SOURCES), phase="Loading datasets")
    for i, (dataset_name, weight, subset) in enumerate(DATASET_SOURCES):
        entries = process_dataset(
            dataset_name=dataset_name,
            weight=weight,
            subset=subset,
            hf_token=HF_TOKEN,
            seed=SEED,
            total_samples=TOTAL_SAMPLES,
        )
        all_data.extend(entries)
        mix_reporter.update(i + 1)

    print("\n" + "=" * 60)
    print(f"Total entries collected: {len(all_data)}")

    if not all_data:
        print("No data collected. Exiting.")
        exit(1)

    timestamp = datetime.now().strftime("%d%m%y")
    size_k    = round(len(all_data) / 1000, 2)

    data_to_save, fmt = apply_output_format(
        all_data, OUTPUT_FORMAT,
        instruction_key="instruction", output_key="output", input_key="input",
    )
    filename = f"{DATASET_NAME}-{fmt.capitalize()}-{size_k}k-{timestamp}.json"
    filepath = os.path.join(OUTPUT_PATH, filename)

    save_json(data_to_save, filepath)
    print(f"Saved: {filepath}")
    print("=" * 60)
    print("\nDataMix complete!")
