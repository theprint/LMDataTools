# datapersona.py
"""
DataPersona - Rewrite dataset responses with specific personas/personalities.
"""

import json
import glob
import os
import numpy as np
from datacore.llm.client import LLMClient
from datacore.config.settings import config, get_tool_output_path
from datacore.io.json_ops import ResumableProcessor, save_json
from datacore.personas.loader import get_persona
from datacore.scoring import calculate_overall_score, flag_for_human_review
from datacore.io.formats import to_alpaca

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "PERSONA": "Devils Advocate",
    "IMPORT_PATH": "import",
    "SOURCE_FILES": [],
    "GENERATE_REPLY_1": True,
    "GENERATE_REPLY_2": True,
    "SAVE_INTERVAL": 250,
    "NUM_BATCHES": -1,
    "EXPORT_ALPACA": False,
    "DATASET_NAME": "my-persona-dataset",
    "JOB_ID": "default-job",
}


# ============================================================================
# END CONFIGURATION
# ============================================================================


def get_source_files():
    """Get files to process based on configuration."""
    if SOURCE_FILES:
        return SOURCE_FILES
    
    # Auto-discover all JSON files
    pattern = os.path.join(IMPORT_PATH, "*.json")
    files = glob.glob(pattern)
    return [os.path.basename(f) for f in files]


def entry_needs_processing(entry):
    """Check if an entry needs any processing."""
    # This function is now simpler: if reply_1 is missing and we need it, process.
    # If reply_2 is missing and we need it, process.
    return (GENERATE_REPLY_1 and "reply_1" not in entry) or \
           (GENERATE_REPLY_2 and "reply_2" not in entry)


def get_question_key(entry):
    """Get the question/instruction key from entry."""
    if "instruction" in entry:
        return "instruction"
    elif "question" in entry:
        return "question"
    else:
        raise KeyError("Entry must contain either 'instruction' or 'question' key")


def get_answer_key(entry):
    """Get the answer/output key from entry."""
    if "output" in entry:
        return "output"
    elif "answer" in entry:
        return "answer"
    else:
        raise KeyError("Entry must contain either 'output' or 'answer' key")


def build_rewrite_prompt(question, original_response):
    """Build the prompt for rewriting a response."""
    prompt = (
        "Below is a query and a response from a different conversation. I want you to rewrite the response"
        " as if given by someone with your background and provided persona. Impart personality in a subtle way that does not clash with the topic"
        " and overall sentiment of the conversation. If you notice any factual errors, you may quietly correct"
        " them in your rewrite but only if you are completely sure. If the source response is poorly written in"
        " general, please do make sure your version is an improvement by comparison. Do not mention that this is"
        " a rewrite or that you are using a persona. Make no mention of the original response. Your reply must be all in character."
        "Do not include comments, acknowledgement or any kind of greeting, instead go straight into the reply as if your persona is part of an already ongoing conversation."
        " Do not include any other text. "
        f"\n\nQuery:\n{question}\n\nOriginal Response:\n{original_response}\n\n"
        "Please respond with your persona-based interpretation the response and no other text."
    )
    return prompt


def main():
    # Apply config overrides from config.json
    config_file = os.path.join(os.getcwd(), "config.json")
    if os.path.exists(config_file):
        print(f"[datapersona] Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            for key, value in user_config.items():
                DEFAULT_CONFIG[key.upper()] = value
        print("[datapersona] Configuration loaded from file")

    # Initialize LLM client
    llm_settings = DEFAULT_CONFIG.get("LLM_SETTINGS", {})
    client = LLMClient(
        base_url=llm_settings.get("base_url"),
        default_model=llm_settings.get("llm_model")
    )

    # Get files to process
    global IMPORT_PATH, SOURCE_FILES
    IMPORT_PATH = DEFAULT_CONFIG["IMPORT_PATH"]
    SOURCE_FILES = DEFAULT_CONFIG["SOURCE_FILES"]
    source_files = get_source_files() # This now uses the globally set IMPORT_PATH

    if not source_files:
        print(f"No files found to process in {IMPORT_PATH}")
        exit(1)

    # Now that we have the files, set the rest of the global vars
    global PERSONA, GENERATE_REPLY_1, GENERATE_REPLY_2, SAVE_INTERVAL, NUM_BATCHES, EXPORT_ALPACA, DATASET_NAME, JOB_ID
    PERSONA = DEFAULT_CONFIG["PERSONA"]
    GENERATE_REPLY_1 = DEFAULT_CONFIG["GENERATE_REPLY_1"]
    GENERATE_REPLY_2 = DEFAULT_CONFIG["GENERATE_REPLY_2"]
    SAVE_INTERVAL = DEFAULT_CONFIG["SAVE_INTERVAL"]
    NUM_BATCHES = DEFAULT_CONFIG["NUM_BATCHES"]
    EXPORT_ALPACA = DEFAULT_CONFIG.get("EXPORT_ALPACA", False)
    DATASET_NAME = DEFAULT_CONFIG["DATASET_NAME"]
    JOB_ID = DEFAULT_CONFIG["JOB_ID"]

    # Load persona data using the correct path
    persona_data = get_persona(PERSONA)
    ai_role = persona_data
    print(f"Using persona: {PERSONA}")
    print(f"Description: {ai_role['description']}\n")

    OUTPUT_PATH = "." # Save files to the current job directory
    print(f"Processing {len(source_files)} file(s)")

    batch = 1
    all_processed_count = 0
    all_data_for_export = []

    for source in source_files:
        print(f"=== Processing {source} ===")

        # Setup file paths
        filepath_in = os.path.join(IMPORT_PATH, source)
        base, ext = os.path.splitext(source)
        filepath_out = os.path.join(OUTPUT_PATH, f"{base}-{DATASET_NAME}-full{ext}")

        # Use resumable processor
        with ResumableProcessor(
            input_path=filepath_in,
            output_path=filepath_out,
            save_interval=SAVE_INTERVAL,
            check_function=entry_needs_processing
        ) as processor:

            # If everything is already complete, skip to next file
            if processor.start_index >= len(processor.data):
                if EXPORT_ALPACA:
                    all_data_for_export.extend(processor.data)
                print(f"File {source} is already complete. Skipping...\n")
                continue
            
            # Process entries starting from resume point
            processed_count = 0
            scores_for_norm = []

            for i in range(processor.start_index, len(processor.data)):
                entry = processor.data[i]

                # Skip if entry doesn't need processing
                if not entry_needs_processing(entry):
                    continue

                all_processed_count += 1
                processed_count += 1
                print(f"Entry {i + 1} of {len(processor.data)} (processing #{all_processed_count}):")

                # Get question and answer
                question_key = get_question_key(entry)
                answer_key = get_answer_key(entry)
                question = entry[question_key]
                org_reply = entry[answer_key]

                # Build prompt
                prompt = build_rewrite_prompt(question, org_reply)

                # Generate reply_1 if needed and enabled
                if GENERATE_REPLY_1 and "reply_1" not in entry:
                    print(" - Generating reply 1 (stream)...", end="", flush=True)
                    result = client.call(
                        prompt=prompt,
                        stream=True,
                        system_prompt=ai_role["description"],
                        temperature=1.0,
                        top_p=0.85,
                        max_tokens=6000,
                        return_dict=True
                    )
                    entry["reply_1"] = result["response"]
                    entry["model_used"] = result["model"] # Use a single model key
                    print(f" done.", flush=True)
                elif "reply_1" in entry:
                    print(f" - Reply 1 already exists, skipping.")

                # Generate reply_2 if needed and enabled
                if GENERATE_REPLY_2 and "reply_2" not in entry:
                    print(" - Generating reply 2 (stream)...", end="", flush=True)
                    result = client.call(
                        prompt=prompt,
                        stream=True,
                        system_prompt=ai_role["description"],
                        temperature=0.4,
                        top_p=0.9,
                        max_tokens=6000,
                        return_dict=True
                    )
                    entry["reply_2"] = result["response"]
                    if not entry.get("model_used"): # Set model if not already set by reply_1
                        entry["model_used"] = result["model"]
                    print(f" done.", flush=True)
                elif "reply_2" in entry:
                    print(f" - Reply 2 already exists, skipping.")

                # Score replies if both exist and scores are missing
                if "reply_1" in entry and "reply_2" in entry:
                    if "reply_1_score" not in entry:
                        entry["reply_1_score"] = calculate_overall_score(client, org_reply, entry["reply_1"], ai_role)
                        scores_for_norm.append(entry["reply_1_score"])
                    if "reply_2_score" not in entry:
                        entry["reply_2_score"] = calculate_overall_score(client, org_reply, entry["reply_2"], ai_role)
                        scores_for_norm.append(entry["reply_2_score"])

                # Auto-checkpoint
                processor.checkpoint(i)

                # Check batch limit
                if NUM_BATCHES > 0 and batch > NUM_BATCHES:
                    print(f"Session halted at batch {batch} per configuration.")
                    break

            if processed_count > 0:
                print(f"Processed {processed_count} entries for {source}\n")
            else:
                print(f"No entries needed processing for {source}\n")
            
            # Normalize scores and select winner for the current file's data
            if scores_for_norm:
                lowest = min(scores_for_norm) if scores_for_norm else 0
                highest = max(scores_for_norm) if scores_for_norm else 1
                score_range = highest - lowest
                if score_range > 0:
                    for entry_to_norm in processor.data:
                        if "reply_1_score" in entry_to_norm:
                            entry_to_norm["reply_1_score"] = round((entry_to_norm["reply_1_score"] - lowest) / score_range, 6)
                        if "reply_2_score" in entry_to_norm:
                            entry_to_norm["reply_2_score"] = round((entry_to_norm["reply_2_score"] - lowest) / score_range, 6)
                        
                        # Flag for human review
                        if "reply_1_score" in entry_to_norm and "reply_2_score" in entry_to_norm:
                            entry_to_norm["outlier"] = flag_for_human_review(entry_to_norm["reply_1_score"], entry_to_norm["reply_2_score"])
                            # Select winning reply
                            if not entry_to_norm["outlier"]:
                                if entry_to_norm["reply_1_score"] >= entry_to_norm["reply_2_score"]:
                                    entry_to_norm["winning_reply"] = "reply_1"
                                else:
                                    entry_to_norm["winning_reply"] = "reply_2"
                                
                                # Replace original output with the winning reply's text
                                winning_reply_key = entry_to_norm["winning_reply"]
                                winning_text = entry_to_norm[winning_reply_key]
                                answer_key = get_answer_key(entry_to_norm)
                                entry_to_norm[answer_key] = winning_text

        if EXPORT_ALPACA:
            all_data_for_export.extend(processor.data)

    print(f"Data processing complete. Full output saved to: {OUTPUT_PATH}")
    # Optional Alpaca Export
    if EXPORT_ALPACA and all_data_for_export:
        print("\nExporting to Alpaca format...")
        # Filter out entries flagged for review
        safe_entries = [e for e in all_data_for_export if not e.get("outlier", False) and e.get("winning_reply")]
        question_key = get_question_key(safe_entries[0]) if safe_entries else "instruction"
        # Use the winning_reply key ("reply_1" or "reply_2") to get the actual response text for alpaca export
        alpaca_dataset = [
            {"instruction": entry[question_key], "input": "", "output": entry[entry["winning_reply"]]} for entry in safe_entries
        ]
        alpaca_filename = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}-alpaca.json")
        save_json(alpaca_dataset, alpaca_filename)
        print(f"Alpaca formatted dataset with {len(alpaca_dataset)} entries saved to: {alpaca_filename}")

if __name__ == '__main__':
    main()