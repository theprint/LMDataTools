# datamix.py
"""
DataMix - Mix and sample datasets from HuggingFace into a unified format.
"""

import os
from datetime import datetime
from datasets import load_dataset
from datacore.config.settings import config, get_tool_output_path
from datacore.io.json_ops import save_json
from datacore.io.formats import to_alpaca

# ============================================================================
# CONFIGURATION
# ============================================================================

# Total samples to generate
TOTAL_SAMPLES = 10_000

# Dataset name for output file
DATASET_NAME = "mixed-dataset"

# HuggingFace token (if needed for private datasets)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Random seed for reproducibility
SEED = 310576

# Output settings
OUTPUT_PATH = "." # Save files to the current job directory
DATASET_SOURCES = [
    # Example datasets - edit these for your needs
    ("theprint/databird-sensible", 0.09, None),
    ("theprint/databird-negotiation", 0.10, None),
    ("theprint/databird-power", 0.09, None),
    # Add more datasets here...
]

# Format mappings for each dataset
# Most common formats are auto-detected, but you can specify if needed
# Format: {dataset_name: format_type}
# Common formats: "alpaca", "sharegpt", "qa", "instruction_response"
DATASET_FORMATS = {
    # Leave empty for auto-detection
    # "dataset_name": "alpaca",
}

# Field mappings for custom formats
# Format: {dataset_name: {"instruction_key": "prompt", "output_key": "response"}}
CUSTOM_MAPPINGS = {
    # Example:
    # "some_dataset": {
    #     "instruction_key": "prompt",
    #     "output_key": "completion",
    #     "input_key": None
    # }
}

# Quality filters
MIN_INSTRUCTION_LENGTH = 10
MAX_INSTRUCTION_LENGTH = 4000
MIN_OUTPUT_LENGTH = 10
MAX_OUTPUT_LENGTH = 4000

# ============================================================================
# END CONFIGURATION
# ============================================================================


def detect_format(entry):
    """Auto-detect dataset format from entry structure."""
    if "conversations" in entry:
        return "sharegpt"
    elif all(k in entry for k in ["instruction", "input", "output"]):
        return "alpaca"
    elif "question" in entry and "answer" in entry:
        return "qa"
    elif "instruction" in entry and "response" in entry:
        return "instruction_response"
    elif "prompt" in entry and "response" in entry:
        return "prompt_response"
    else:
        return "unknown"


def extract_qa_from_entry(entry, dataset_name):
    """Extract instruction/output pair from entry based on format."""
    
    # Check for custom mapping first
    if dataset_name in CUSTOM_MAPPINGS:
        mapping = CUSTOM_MAPPINGS[dataset_name]
        instruction = entry.get(mapping["instruction_key"], "")
        output = entry.get(mapping["output_key"], "")
        input_text = entry.get(mapping.get("input_key", "input"), "")
        return instruction, input_text, output
    
    # Auto-detect format
    format_type = DATASET_FORMATS.get(dataset_name) or detect_format(entry)
    
    if format_type == "alpaca":
        return entry.get("instruction", ""), entry.get("input", ""), entry.get("output", "")
    
    elif format_type == "qa":
        return entry.get("question", ""), "", entry.get("answer", "")
    
    elif format_type == "instruction_response":
        return entry.get("instruction", ""), "", entry.get("response", "")
    
    elif format_type == "prompt_response":
        return entry.get("prompt", ""), "", entry.get("response", "")
    
    elif format_type == "sharegpt":
        # Extract from conversations
        convs = entry.get("conversations", [])
        user_msg = ""
        assistant_msg = ""
        for msg in convs:
            if msg.get("from") in ["user", "human"]:
                user_msg = msg.get("value", "")
            elif msg.get("from") in ["assistant", "gpt"]:
                assistant_msg = msg.get("value", "")
        return user_msg, "", assistant_msg
    
    return "", "", ""


def validate_entry(instruction, output):
    """Check if entry meets quality requirements."""
    if not instruction or not output:
        return False
    
    if len(instruction) < MIN_INSTRUCTION_LENGTH or len(instruction) > MAX_INSTRUCTION_LENGTH:
        return False
    
    if len(output) < MIN_OUTPUT_LENGTH or len(output) > MAX_OUTPUT_LENGTH:
        return False
    
    return True


def get_samples_count(total, weight):
    """Calculate number of samples based on weight."""
    return round(total * weight)


def process_dataset(dataset_name, weight, subset, hf_token, seed, total_samples):
    """
    Load and process a dataset from HuggingFace.
    
    Returns list of processed entries in standard format.
    """
    print(f"\nProcessing: {dataset_name}")
    
    # Load dataset
    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split="train", token=hf_token)
        else:
            ds = load_dataset(dataset_name, split="train", token=hf_token)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return []
    
    # Shuffle
    ds = ds.shuffle(seed=seed)
    
    # Calculate samples needed
    samples_count = min(get_samples_count(total_samples, weight), len(ds))
    print(f"  Taking {samples_count} of {len(ds)} entries ({weight*100:.1f}%)")
    
    # Process entries
    processed = []
    for i in range(samples_count):
        entry = ds[i]
        
        # Extract instruction/output
        instruction, input_text, output = extract_qa_from_entry(entry, dataset_name)
        
        # Validate
        if not validate_entry(instruction, output):
            continue
        
        # Store in standard format
        processed_entry = {
            "source": dataset_name,
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
        processed.append(processed_entry)
    
    print(f"  Kept {len(processed)} valid entries")
    return processed


if __name__ == "__main__":
    print("DataMix - Dataset Mixing Tool")
    print("=" * 60)
    print(f"Target samples: {TOTAL_SAMPLES:,}")
    print(f"Datasets to mix: {len(DATASET_SOURCES)}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Validate weights sum to ~1.0
    total_weight = sum(weight for _, weight, _ in DATASET_SOURCES)
    if abs(total_weight - 1.0) > 0.01:
        print(f"\nWarning: Weights sum to {total_weight:.2f}, not 1.0")
        print("Results may not match expected sample counts.\n")
    
    # Process each dataset
    all_data = []
    for dataset_name, weight, subset in DATASET_SOURCES:
        entries = process_dataset(
            dataset_name=dataset_name,
            weight=weight,
            subset=subset,
            hf_token=HF_TOKEN,
            seed=SEED,
            total_samples=TOTAL_SAMPLES
        )
        all_data.extend(entries)
    
    print("\n" + "=" * 60)
    print(f"Total entries collected: {len(all_data)}")
    
    if len(all_data) == 0:
        print("No data collected. Exiting.")
        exit(1)
    
    # Generate filename
    timestamp = datetime.now().strftime("%d%m%y")
    size_k = round(len(all_data) / 1000, 2)
    filename = f"{DATASET_NAME}-Alpaca-{size_k}k-{timestamp}.json"
    filepath = os.path.join(OUTPUT_PATH, filename)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True) if OUTPUT_PATH != "." else None
    # Save
    save_json(all_data, filepath)
    
    print(f"Saved: {filepath}")
    print("=" * 60)
    print("\nDataMix complete!")