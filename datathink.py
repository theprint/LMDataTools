# datathink.py
"""
DataThink - Enhance datasets with reasoning steps before generating responses.

Pipeline:
1. Load existing dataset (supports qa, alpaca, sharegpt formats)
2. For each entry:
   - Generate reasoning/thinking steps (without persona - clear internal reasoning)
   - Generate final response using that reasoning (persona applied here if enabled)
3. Export enhanced dataset with:
   - question: original query
   - answer: enhanced response with <think> block
   - original_answer: original response from source file
   - thinking_only: just the reasoning steps

Note: Persona is ONLY applied to the user-facing response, not the internal
reasoning phase. The thinking block should be clear and efficient LLM-to-LLM
communication.
"""

import os
import json
from datacore.llm.client import LLMClient
from datacore.config.settings import config
from datacore.io.json_ops import save_json, load_json, ResumableProcessor
from datacore.io.formats import detect_format
from datacore.personas.loader import get_persona
from datacore.personas.prompt_manager import inject_persona_into_prompt

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "DATASET_NAME": "my-thinking-dataset",
    "JOB_ID": "default-job",
    "IMPORT_PATH": "import",
    "REASONING_LEVEL": "medium",
    "SAVE_INTERVAL": 50,
    "USE_PERSONA": False,
    "PERSONA_NAME": "",
    "THINKING_TEMPERATURE": 0.7,
    "RESPONSE_TEMPERATURE": 0.7,
    "LLM_SETTINGS": {
        "base_url": None,
        "llm_model": None
    }
}

# Check for config.json override
config_file = os.path.join(os.getcwd(), "config.json")
if os.path.exists(config_file):
    print(f"[datathink] Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        user_config = json.load(f)
        for key, value in user_config.items():
            DEFAULT_CONFIG[key.upper()] = value

# Apply configuration
DATASET_NAME = DEFAULT_CONFIG["DATASET_NAME"]
JOB_ID = DEFAULT_CONFIG["JOB_ID"]
IMPORT_PATH = DEFAULT_CONFIG["IMPORT_PATH"]
REASONING_LEVEL = DEFAULT_CONFIG["REASONING_LEVEL"]
SAVE_INTERVAL = DEFAULT_CONFIG["SAVE_INTERVAL"]
USE_PERSONA = DEFAULT_CONFIG.get("USE_PERSONA", False)
PERSONA_NAME = DEFAULT_CONFIG.get("PERSONA_NAME", "")
THINKING_TEMPERATURE = DEFAULT_CONFIG.get("THINKING_TEMPERATURE", 0.7)
RESPONSE_TEMPERATURE = DEFAULT_CONFIG.get("RESPONSE_TEMPERATURE", 0.7)

OUTPUT_PATH = "."  # Save files to the current job directory

# ============================================================================
# END CONFIGURATION
# ============================================================================


def detect_and_load_dataset(file_path):
    """
    Load dataset and detect its format.
    
    Returns:
        Tuple of (data, format_type)
    """
    print(f"  Loading dataset from: {file_path}")
    
    # Load the data
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        data = load_json(file_path)
    
    # Detect format
    format_type = detect_format(data)
    print(f"  Detected format: {format_type}")
    
    return data, format_type


def extract_query_and_response(entry, format_type):
    """
    Extract the query and original response from an entry based on format.
    
    Returns:
        Tuple of (query, original_response)
    """
    if format_type == "alpaca":
        query = entry.get("instruction", "")
        if entry.get("input"):
            query = f"{query}\n\n{entry['input']}"
        return query, entry.get("output", "")
    
    elif format_type == "qa":
        return entry.get("question", ""), entry.get("answer", "")
    
    elif format_type == "sharegpt":
        conversations = entry.get("conversations", [])
        query = ""
        original_response = ""
        
        # Get first user message as query
        for msg in conversations:
            if msg.get("from") in ["user", "human"]:
                query = msg.get("value", "")
                break
        
        # Get first assistant message as original response
        for msg in conversations:
            if msg.get("from") in ["assistant", "gpt"]:
                original_response = msg.get("value", "")
                break
        
        return query, original_response
    
    return "", ""


def generate_thinking(client, query, level="medium"):
    """
    Generate reasoning steps for a query without providing the answer.
    No persona is used here - this is internal LLM reasoning that should be
    clear and efficient.
    
    Args:
        client: LLM client
        query: The user's query
        level: Level of detail for reasoning ("low", "medium", "high")
    
    Returns:
        Thinking/reasoning text
    """
    thinking_steps = "- Briefly outline the best approach to consider, and why?\n"
    thinking_tokens = 600
    if level == "medium":
        thinking_steps += "- If any, what are the major challenges or complexities of this query?\n"
        thinking_tokens = 1200
    elif level == "high":
        thinking_steps += "- If any, what are the major challenges or complexities of this query?\n- What caveats, edge cases, or important considerations should be kept in mind? For each such consideration, explain why it matters and how it should be handled.\n"
        thinking_tokens = 3000

    thinking_prompt = (
        "You are about to help answer a query, but first you need to think through your approach.\n\n"
        "Analyze the following query carefully and think through:\n"
        f"{thinking_steps}\n"
        "IMPORTANT: Do NOT provide the actual answer. Only provide your reasoning and thought process.\n"
        "Be concise and efficient, include only the instructions you need - this is internal reasoning, not a user-facing response.\n\n"
        f"Query: {query}\n\n"
        "Your reasoning (again, don't answer):"
    )
    
    # No system prompt - keep thinking clear and efficient
    thinking = client.call(
        prompt=thinking_prompt,
        temperature=THINKING_TEMPERATURE,
        max_tokens=thinking_tokens,
    )
    
    return thinking.strip()


def generate_response_with_thinking(client, query, thinking, system_prompt=None):
    """
    Generate final response using the reasoning steps.
    
    Args:
        client: LLM client
        query: The user's query
        thinking: The reasoning steps
        system_prompt: Optional system prompt (for persona)
    
    Returns:
        Final response including <think> block
    """
    response_prompt = (
        f"The end goal is to answer this query: {query}\n\n"
        "You already thought through how best to approach this query. "
        "Now you must follow the outline and provide your actual response to the user.\n\n"
        "Your reasoning was:\n"
        f"{thinking}\n\n"
        "Provide a response based on the query and your reasoning above:"
    )
    
    response = client.call(
        prompt=response_prompt,
        system_prompt=system_prompt,
        temperature=RESPONSE_TEMPERATURE,
        max_tokens=8000
    )
    
    # Combine thinking and response
    full_response = f"<think>\n{thinking}\n</think>\n\n{response.strip()}"
    
    return full_response


def entry_needs_processing(entry):
    """Check if entry has already been processed."""
    return "answer" not in entry or "thinking_only" not in entry


if __name__ == "__main__":
    print("DataThink - Reasoning-Enhanced Dataset Generation")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Initialize LLM client
    llm_settings = DEFAULT_CONFIG.get("LLM_SETTINGS", {})
    client = LLMClient(
        base_url=llm_settings.get("base_url"),
        default_model=llm_settings.get("llm_model")
    )
    
    # Setup persona if enabled
    system_prompt = None
    if USE_PERSONA and PERSONA_NAME:
        try:
            persona_data = get_persona(PERSONA_NAME)
            system_prompt = persona_data["description"]
            print(f"[datathink] Using persona '{PERSONA_NAME}' for user-facing responses.")
            print(f"[datathink] Reasoning phase will remain clear and efficient (no persona).")
        except (ValueError, FileNotFoundError) as e:
            print(f"[datathink] Warning: Could not load persona. {e}. Proceeding without persona.")
    
    # Find and load source file
    source_files = [f for f in os.listdir(IMPORT_PATH) if f.endswith(('.json', '.jsonl'))]
    if not source_files:
        print(f"Error: No source file found in the '{IMPORT_PATH}' directory.")
        exit(1)
    
    source_file_path = os.path.join(IMPORT_PATH, source_files[0])
    source_data, format_type = detect_and_load_dataset(source_file_path)
    
    if not source_data:
        print("Error: Source dataset is empty.")
        exit(1)
    
    print(f"Loaded {len(source_data)} entries from source dataset.")
    print(f"Format: {format_type}")
    print()
    
    # Setup output path
    output_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}.json")
    
    # Process entries with resumable processor
    with ResumableProcessor(
        input_path=source_file_path,
        output_path=output_file,
        save_interval=SAVE_INTERVAL,
        check_function=entry_needs_processing
    ) as processor:
        
        if processor.start_index >= len(processor.data):
            print("All entries already processed!")
        else:
            print(f"Processing from entry {processor.start_index + 1} of {len(processor.data)}...\n")
        
        for i in range(processor.start_index, len(processor.data)):
            entry = processor.data[i]
            
            if not entry_needs_processing(entry):
                continue
            
            print(f"Processing entry {i + 1}/{len(processor.data)}...")
            
            # Extract query and original response
            query, original_response = extract_query_and_response(entry, format_type)
            
            if not query:
                print(f"  Skipping entry {i + 1}: No query found")
                continue
            
            print(f"  Query: {query[:100]}...")
            
            # Step 1: Generate thinking (no persona - internal reasoning)
            print("  - Generating reasoning...")
            thinking = generate_thinking(client, query, level=REASONING_LEVEL)
            print(f"    Reasoning: {len(thinking)} chars")
            
            # Step 2: Generate response with thinking (persona applied here if enabled)
            print("  - Generating enhanced response...")
            enhanced_response = generate_response_with_thinking(
                client, query, thinking, system_prompt
            )
            print(f"    Response: {len(enhanced_response)} chars")
            
            # Store results in clean Q&A format
            entry["question"] = query
            entry["answer"] = enhanced_response
            entry["original_answer"] = original_response
            entry["thinking_only"] = thinking
            
            # Remove old format keys if they exist
            entry.pop("instruction", None)
            entry.pop("input", None)
            entry.pop("output", None)
            entry.pop("conversations", None)
            
            # Checkpoint
            processor.checkpoint(i)
            print()
    
    print("\n" + "=" * 60)
    print("DataThink Complete!")
    print("=" * 60)
    print(f"Output saved to: {output_file}")
    print(f"Total entries processed: {len(processor.data)}")