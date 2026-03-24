# dataconvo.py
"""
DataConvo - Expands existing single-turn conversations into multi-turn dialogues.
"""
import json
import random
import os
from datacore.llm.client import LLMClient
from datacore.config.settings import config
from datacore.config.loader import load_tool_config
from datacore.progress import ProgressReporter
from datacore.io.json_ops import save_json
from datacore.personas.loader import get_persona
from datacore.personas.prompt_manager import inject_persona_into_prompt

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "DATASET_NAME": "my-convo-dataset",
    "IMPORT_PATH": "import",
    "SAVE_INTERVAL": 100,
    "ROUND_WEIGHTS": {"rounds_1": 25, "rounds_2": 50, "rounds_3": 25},
    "USE_PERSONA": False,
    "PERSONA_NAME": "Enthusiastic Organizer",
    "LLM_SETTINGS": {
        "base_url": None,
        "llm_model": None
    }
}

DEFAULT_CONFIG = load_tool_config(DEFAULT_CONFIG, tool_name="dataconvo")

# ============================================================================
# PROMPTS
# ============================================================================

USER_SYSTEM_PROMPT = "You are a typical user asking for help. Ask questions, provide context, and react naturally."
ASSISTANT_SYSTEM_PROMPT = "You are a helpful AI assistant. Continue the conversation naturally."

# ============================================================================
# MAIN LOGIC
# ============================================================================

def main():
    """
    Main function to generate the conversation dataset.
    """
    cfg = DEFAULT_CONFIG
    llm_settings = cfg.get("LLM_SETTINGS", {})
    client = LLMClient(
        base_url=llm_settings.get("base_url"),
        default_model=llm_settings.get("llm_model")
    )

    # Find the uploaded file
    import_path = cfg.get("IMPORT_PATH", "import")
    source_files = [f for f in os.listdir(import_path) if f.endswith(('.json', '.jsonl'))]
    if not source_files:
        print(f"Error: No source file found in the '{import_path}' directory.")
        return

    source_file_path = os.path.join(import_path, source_files[0])
    print(f"[dataconvo] Loading source data from: {source_file_path}")
    with open(source_file_path, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f] if source_file_path.endswith('.jsonl') else json.load(f)

    print(f"[dataconvo] Found {len(source_data)} conversations to expand.")
    reporter = ProgressReporter(total=len(source_data), phase="Expanding conversations")

    # Prepare the assistant's system prompt (with optional persona)
    assistant_system_prompt = ASSISTANT_SYSTEM_PROMPT
    if cfg["USE_PERSONA"] and cfg["PERSONA_NAME"]:
        try:
            persona_data = get_persona(cfg["PERSONA_NAME"])
            assistant_system_prompt = inject_persona_into_prompt(
                base_prompt=ASSISTANT_SYSTEM_PROMPT,
                persona_name=persona_data["persona"],
                persona_description=persona_data["description"]
            )
            print(f"[dataconvo] Using persona '{cfg['PERSONA_NAME']}' for assistant replies.")
        except (ValueError, FileNotFoundError) as e:
            print(f"[dataconvo] Warning: Could not load persona. {e}. Proceeding without persona.")

    # Determine number of rounds for each entry
    round_weights = cfg.get("ROUND_WEIGHTS", {"rounds_1": 25, "rounds_2": 50, "rounds_3": 25})
    num_rounds_choices = [1] * round_weights["rounds_1"] + [2] * round_weights["rounds_2"] + [3] * round_weights["rounds_3"]

    expanded_conversations = []
    for i, entry in enumerate(source_data):
        reporter.update(i + 1)
        print(f"  Processing entry {i + 1} of {len(source_data)}...")
        
        # Accept ShareGPT (conversations), Alpaca (instruction/output), or Q&A (question/answer) format
        if "conversations" in entry:
            turns = entry["conversations"]
            human_turn = next((t for t in turns if t.get("from") == "human"), None)
            gpt_turn = next((t for t in turns if t.get("from") in ("gpt", "assistant")), None)
            initial_user_message = human_turn["value"] if human_turn else None
            initial_assistant_message = gpt_turn["value"] if gpt_turn else None
        else:
            initial_user_message = entry.get("instruction") or entry.get("question")
            initial_assistant_message = entry.get("output") or entry.get("answer")

        if not initial_user_message or not initial_assistant_message:
            print(f"    - Skipping entry {i+1} due to missing 'instruction'/'question' or 'output'/'answer'.")
            continue

        conversation_history = [
            {"role": "user", "content": initial_user_message},
            {"role": "assistant", "content": initial_assistant_message}
        ]

        num_additional_rounds = random.choice(num_rounds_choices)

        for round_num in range(num_additional_rounds):
            # Generate user follow-up
            user_followup_prompt = (
                "You are playing the role of a typical user. Your task is to generate ONLY the user's next reply. "
                "It is critical that you do NOT respond as an assistant. Do not be helpful, analytical, or use a formal tone. Do not offer solutions or summarize the conversation. "
                "Instead, ask a simple follow-up question, express confusion, or react naturally and informally to what the assistant just said. "
                f"\n\n## Conversation History:\n{json.dumps(conversation_history, indent=2)}\n\n"
                "## Your Reply (as the user):"
            )
            user_followup = client.call(prompt=user_followup_prompt, system_prompt=USER_SYSTEM_PROMPT).strip()
            conversation_history.append({"role": "user", "content": user_followup})

            # Generate assistant response
            assistant_response_prompt = f"You are the assistant. Based on the last user message and the full conversation history, generate your next helpful response.\n\n## Conversation History:\n{json.dumps(conversation_history, indent=2)}"
            assistant_response = client.call(prompt=assistant_response_prompt, system_prompt=assistant_system_prompt).strip()
            conversation_history.append({"role": "assistant", "content": assistant_response})

        # Convert to ShareGPT format with "from" and "value" keys
        sharegpt_conversation = [
            {"from": "human" if turn["role"] == "user" else "gpt", "value": turn["content"]}
            for turn in conversation_history
        ]
        expanded_conversations.append({"conversations": sharegpt_conversation})

        # Save checkpoint
        if (i + 1) % cfg.get("SAVE_INTERVAL", 100) == 0:
            output_path = os.path.join(os.getcwd(), f"{cfg['DATASET_NAME']}-checkpoint.json")
            save_json(expanded_conversations, output_path)
            print(f"    - Checkpoint saved with {len(expanded_conversations)} conversations.")

    # Save the final dataset
    output_path = os.path.join(os.getcwd(), f"{cfg['DATASET_NAME']}.json")
    save_json(expanded_conversations, output_path)
    
    # Clean up checkpoint file
    checkpoint_path = os.path.join(os.getcwd(), f"{cfg['DATASET_NAME']}-checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\n[dataconvo] Conversation generation complete.")
    print(f"[dataconvo] Dataset saved to {output_path}")

if __name__ == '__main__':
    main()