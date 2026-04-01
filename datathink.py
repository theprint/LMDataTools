# datathink.py
"""
DataThink - Add reasoning blocks to datasets.

Modes
-----
insert_reasoning (default)
    Input:  Q+A dataset (existing answers required)
    Output: <think>...</think>\n\n{original_answer}
    Use when: You have curated/high-quality answers and only want to add
              the reasoning that would lead to them.

regenerate
    Input:  Q+A dataset
    Output: <think>...</think>\n\n{new_answer}
    Use when: You want the model to reason through the question and produce
              a fresh answer informed by that reasoning. Original answer is
              preserved in the 'original_answer' field.

generate_new
    Input:  Questions only (no answers required)
    Output: <think>...</think>\n\n{generated_answer}
    Use when: You have a list of questions and want both reasoning and
              answers generated from scratch.
"""

import os
import json
from datacore.llm.client import LLMClient
from datacore.config.settings import config
from datacore.config.loader import load_tool_config
from datacore.progress import ProgressReporter
from datacore.io.json_ops import save_json, load_json, ResumableProcessor
from datacore.io.formats import detect_format
from datacore.personas.loader import get_persona

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "DATASET_NAME": "my-thinking-dataset",
    "JOB_ID": "default-job",
    "IMPORT_PATH": "import",
    "THINK_MODE": "insert_reasoning",   # insert_reasoning | regenerate | generate_new
    "REASONING_LEVEL": "medium",        # low | medium | high
    "SAVE_INTERVAL": 50,
    "USE_PERSONA": False,
    "PERSONA_NAME": "",
    "THINKING_TEMPERATURE": 0.5,
    "RESPONSE_TEMPERATURE": 0.7,
    "LLM_SETTINGS": {
        "base_url": None,
        "llm_model": None
    }
}

DEFAULT_CONFIG = load_tool_config(DEFAULT_CONFIG, tool_name="datathink")

DATASET_NAME         = DEFAULT_CONFIG["DATASET_NAME"]
JOB_ID               = DEFAULT_CONFIG["JOB_ID"]
IMPORT_PATH          = DEFAULT_CONFIG["IMPORT_PATH"]
THINK_MODE           = DEFAULT_CONFIG.get("THINK_MODE", "insert_reasoning")
REASONING_LEVEL      = DEFAULT_CONFIG["REASONING_LEVEL"]
SAVE_INTERVAL        = DEFAULT_CONFIG["SAVE_INTERVAL"]
USE_PERSONA          = DEFAULT_CONFIG.get("USE_PERSONA", False)
PERSONA_NAME         = DEFAULT_CONFIG.get("PERSONA_NAME", "")
THINKING_TEMPERATURE = DEFAULT_CONFIG.get("THINKING_TEMPERATURE", 0.5)
RESPONSE_TEMPERATURE = DEFAULT_CONFIG.get("RESPONSE_TEMPERATURE", 0.7)

OUTPUT_PATH = "."

# ============================================================================
# END CONFIGURATION
# ============================================================================


def detect_and_load_dataset(file_path):
    """Load dataset and detect its format. Returns (data, format_type)."""
    print(f"  Loading dataset from: {file_path}")
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        data = load_json(file_path)
    format_type = detect_format(data)
    print(f"  Detected format: {format_type}")
    return data, format_type


def extract_query_and_response(entry, format_type):
    """Extract (query, original_response) from an entry. Response may be empty."""
    if format_type == "alpaca":
        query = entry.get("instruction", "")
        if entry.get("input"):
            query = f"{query}\n\n{entry['input']}"
        return query, entry.get("output", "")
    elif format_type == "qa":
        return entry.get("question", ""), entry.get("answer", "")
    elif format_type == "sharegpt":
        convs = entry.get("conversations", [])
        query    = next((m.get("value", "") for m in convs if m.get("from") in ["user", "human"]), "")
        response = next((m.get("value", "") for m in convs if m.get("from") in ["assistant", "gpt"]), "")
        return query, response
    return "", ""


def _level_config(level):
    """Return (prompt_guidance, max_tokens) for a reasoning depth level.

    The guidance string is embedded directly into prompts and tells the model
    both *what* to cover and *how long* to be. max_tokens is a hard ceiling only.
    """
    if level == "low":
        return (
            "Briefly outline the best approach to answering this in 2-3 sentences. "
            "Be concise and direct — do not over-explain.\n\n"
        ), 600
    elif level == "high":
        return (
            "Reason through the following in depth:\n"
            "- What is the best approach to answering this, and why?\n"
            "- What are the key challenges or nuances in this query?\n"
            "- What edge cases or caveats matter, and how should they be handled?\n"
            "Be thorough and comprehensive in your analysis.\n\n"
        ), 3000
    else:  # medium
        return (
            "Reason through the following at a moderate depth:\n"
            "- What is the best approach to answering this, and why?\n"
            "- What key challenges or nuances should be considered?\n"
            "Cover the main points clearly without excessive detail.\n\n"
        ), 1200


def generate_thinking_for_query(client, query, level, temperature):
    """
    Generate reasoning steps for a query WITHOUT producing the answer.
    Used in 'regenerate' and 'generate_new' modes.
    No persona — this is internal LLM-to-LLM reasoning.
    """
    guidance, max_tokens = _level_config(level)
    prompt = (
        "You are about to answer a query, but first think through your approach carefully.\n\n"
        f"{guidance}"
        "IMPORTANT: Do NOT provide the actual answer. Only your reasoning and thought process.\n"
        "This is internal reasoning, not a user-facing response.\n\n"
        f"Query: {query}\n\n"
        "Your reasoning:"
    )
    return client.call(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    ).strip()


def generate_reasoning_for_answer(client, query, original_answer, level, temperature):
    """
    Generate reasoning that explains HOW the given answer was reached.
    Used in 'insert_reasoning' mode — the original answer is not changed.
    No persona — internal reasoning only.
    """
    guidance, max_tokens = _level_config(level)
    prompt = (
        "Below is a question and its correct answer. Work through the reasoning process "
        "that would naturally lead to this answer.\n\n"
        f"{guidance}"
        "IMPORTANT: Do NOT restate or paraphrase the answer itself. "
        "Your reasoning should flow naturally toward the given answer. "
        "This is internal reasoning, not a user-facing response.\n\n"
        f"Question: {query}\n\n"
        f"Answer: {original_answer}\n\n"
        "Your reasoning:"
    )
    return client.call(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    ).strip()


def generate_response_with_thinking(client, query, thinking, temperature, system_prompt=None):
    """
    Generate a user-facing response guided by prior reasoning.
    Persona (system_prompt) is applied here.
    Used in 'regenerate' and 'generate_new' modes.
    """
    prompt = (
        f"The end goal is to answer this query: {query}\n\n"
        "You already thought through the best approach. "
        "Now follow that outline and provide your actual response to the user.\n\n"
        "Your reasoning was:\n"
        f"{thinking}\n\n"
        "Provide a response based on the query and your reasoning above:"
    )
    return client.call(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=8000
    ).strip()


def entry_needs_processing(entry):
    """Check if entry has already been processed (works for all modes)."""
    return "thinking_only" not in entry


if __name__ == "__main__":
    print("DataThink - Reasoning-Enhanced Dataset Generation")
    print("=" * 60)
    mode_labels = {
        "insert_reasoning": "Insert Reasoning  (keep original answer)",
        "regenerate":       "Regenerate        (new answer via reasoning)",
        "generate_new":     "Generate New      (questions only)",
    }
    print(f"Mode:    {mode_labels.get(THINK_MODE, THINK_MODE)}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Depth:   {REASONING_LEVEL}")
    print("=" * 60)

    llm_settings = DEFAULT_CONFIG.get("LLM_SETTINGS", {})
    client = LLMClient(
        base_url=llm_settings.get("base_url"),
        default_model=llm_settings.get("llm_model")
    )

    # Persona only applies to user-facing responses (not insert_reasoning mode)
    system_prompt = None
    if USE_PERSONA and PERSONA_NAME and THINK_MODE in ("regenerate", "generate_new"):
        try:
            persona_data = get_persona(PERSONA_NAME)
            system_prompt = persona_data["description"]
            print(f"Using persona '{PERSONA_NAME}' for user-facing responses.")
            print("Reasoning phase will remain clear and efficient (no persona).")
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load persona ({e}). Continuing without.")

    source_files = [f for f in os.listdir(IMPORT_PATH) if f.endswith(('.json', '.jsonl'))]
    if not source_files:
        print(f"Error: No source file found in '{IMPORT_PATH}'.")
        exit(1)

    source_file_path = os.path.join(IMPORT_PATH, source_files[0])
    source_data, format_type = detect_and_load_dataset(source_file_path)

    if not source_data:
        print("Error: Source dataset is empty.")
        exit(1)

    print(f"\nLoaded {len(source_data)} entries | Format: {format_type}\n")

    output_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}.json")

    with ResumableProcessor(
        input_path=source_file_path,
        output_path=output_file,
        save_interval=SAVE_INTERVAL,
        check_function=entry_needs_processing
    ) as processor:

        reporter = ProgressReporter(total=len(processor.data), phase="Processing entries")

        if processor.start_index >= len(processor.data):
            print("All entries already processed!")
        else:
            print(f"Processing from entry {processor.start_index + 1} of {len(processor.data)}...\n")

        for i in range(processor.start_index, len(processor.data)):
            entry = processor.data[i]

            if not entry_needs_processing(entry):
                continue

            reporter.update(i + 1)
            print(f"Processing entry {i + 1} of {len(processor.data)}...", flush=True)

            query, original_response = extract_query_and_response(entry, format_type)

            if not query:
                print("  Skipping: no query found.\n")
                continue

            # ── INSERT REASONING ──────────────────────────────────────────────
            if THINK_MODE == "insert_reasoning":
                if not original_response:
                    print("  Skipping: no answer found (required for insert_reasoning mode).\n")
                    continue

                print("  Generating reasoning for existing answer...", flush=True)
                thinking = generate_reasoning_for_answer(
                    client, query, original_response, REASONING_LEVEL, THINKING_TEMPERATURE
                )

                entry["question"]        = query
                entry["answer"]          = f"<think>\n{thinking}\n</think>\n\n{original_response}"
                entry["thinking_only"]   = thinking
                entry["original_answer"] = original_response
                print(f"  Done — reasoning: {len(thinking)} chars | original answer preserved.")

            # ── REGENERATE ────────────────────────────────────────────────────
            elif THINK_MODE == "regenerate":
                print("  Generating reasoning...", flush=True)
                thinking = generate_thinking_for_query(
                    client, query, REASONING_LEVEL, THINKING_TEMPERATURE
                )
                print("  Generating new response...", flush=True)
                new_response = generate_response_with_thinking(
                    client, query, thinking, RESPONSE_TEMPERATURE, system_prompt
                )

                entry["question"]        = query
                entry["answer"]          = f"<think>\n{thinking}\n</think>\n\n{new_response}"
                entry["thinking_only"]   = thinking
                entry["original_answer"] = original_response
                print(f"  Done — reasoning: {len(thinking)} chars | response: {len(new_response)} chars.")

            # ── GENERATE NEW ──────────────────────────────────────────────────
            else:  # generate_new
                print("  Generating reasoning...", flush=True)
                thinking = generate_thinking_for_query(
                    client, query, REASONING_LEVEL, THINKING_TEMPERATURE
                )
                print("  Generating answer...", flush=True)
                new_response = generate_response_with_thinking(
                    client, query, thinking, RESPONSE_TEMPERATURE, system_prompt
                )

                entry["question"]      = query
                entry["answer"]        = f"<think>\n{thinking}\n</think>\n\n{new_response}"
                entry["thinking_only"] = thinking
                print(f"  Done — reasoning: {len(thinking)} chars | answer: {len(new_response)} chars.")

            # Clean up old format keys regardless of mode
            for key in ("instruction", "input", "output", "conversations"):
                entry.pop(key, None)

            # Provenance fields (added/updated on every processed entry)
            entry["_tool"] = "datathink"
            entry["_version"] = "2.0"

            processor.checkpoint(i)
            print()

    usage = client.get_usage_stats()
    print(f"TOKENS {usage['prompt_tokens']}/{usage['completion_tokens']}", flush=True)

    print("\n" + "=" * 60)
    print("DataThink Complete!")
    print(f"Output:  {output_file}")
    print(f"Entries: {len(processor.data)}")
    print(f"Tokens:  {usage['total_tokens']:,} total ({usage['prompt_tokens']:,} prompt / {usage['completion_tokens']:,} completion)")
    print("=" * 60)
