# datawriter.py
"""
DataWriter - Generate non-conversational text content with procedural personas.
"""

import os
import json
from datetime import datetime
from datacore.llm.client import LLMClient
from datacore.config.settings import config, get_tool_output_path
from datacore.io.json_ops import save_json
from datacore.personas.generator import PersonaGenerator
from datacore.topics import get_random_topic

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default configuration
DEFAULT_CONFIG = {
    "DOCUMENT_COUNT": 500,
    "MIN_TOKENS": 200, # New: Minimum token length
    "MAX_TOKENS": 10000,
    "TEMPERATURE": 0.8,
    "DATASET_NAME": "my-writer-dataset",
    "JOB_ID": "default-job",
    "ADD_SUMMARY": False, # New: Option to add summary
}

# Check for config.json override from web UI
config_file = os.path.join(os.getcwd(), "config.json")
if os.path.exists(config_file):
    print(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        user_config = json.load(f)
        # Case-insensitive update
        for key, value in user_config.items():
            DEFAULT_CONFIG[key.upper()] = value

DOCUMENT_COUNT = DEFAULT_CONFIG["DOCUMENT_COUNT"]
MIN_TOKENS = DEFAULT_CONFIG["MIN_TOKENS"]
MAX_TOKENS = DEFAULT_CONFIG["MAX_TOKENS"]
TEMPERATURE = DEFAULT_CONFIG["TEMPERATURE"]
DATASET_NAME = DEFAULT_CONFIG["DATASET_NAME"]
JOB_ID = DEFAULT_CONFIG["JOB_ID"]
ADD_SUMMARY = DEFAULT_CONFIG["ADD_SUMMARY"]
 
OUTPUT_PATH = "." # Save files to the current job directory

# Topic tier weights (must sum to 1.0)
TIER_WEIGHTS = [
    (1, 0.20),  # Tier 1: 20%
    (2, 0.35),  # Tier 2: 35%
    (3, 0.10),  # Tier 3: 10%
    (4, 0.15),  # Tier 4: 15%
    (5, 0.10),  # Tier 5: 10%
    (6, 0.10),  # Tier 6: 10%
]

# ============================================================================
# END CONFIGURATION
# ============================================================================


def generate_unique_id(length=8):
    """Generate a simple unique ID using timestamp."""
    from random import choice
    import string
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    suffix = ''.join(choice(string.ascii_letters + string.digits) for _ in range(length - len(timestamp)))
    return timestamp + suffix


if __name__ == "__main__":
    # Initialize
    client = LLMClient(base_url=config.LLM_BASE_URL, default_model=os.getenv("LLM_MODEL_NAME", config.LLM_MODEL)) 
    persona_gen = PersonaGenerator(client=client)
    
    all_entries = []
    
    # Validate token range
    if MIN_TOKENS >= MAX_TOKENS:
        raise ValueError(f"MIN_TOKENS ({MIN_TOKENS}) must be less than MAX_TOKENS ({MAX_TOKENS}).")

    print(f"Generating {DOCUMENT_COUNT} documents...")
    print(f"Output directory: {OUTPUT_PATH}\n")
    
    for i in range(DOCUMENT_COUNT):
        print(f"=== Generating document {i+1} of {DOCUMENT_COUNT} ===", flush=True)
        
        # Generate persona and document type
        system_prompt, doc_type = persona_gen.generate_writer_persona()
        
        # Generate writing style (no flush needed here as it returns a string)
        style = persona_gen.generate_writing_style()
        
        # Select topic with tier weighting
        topic_string, tier = get_random_topic(tier_weights=TIER_WEIGHTS)
        
        # Build system prompt with style
        full_system_prompt = f"{system_prompt} Always respond as that persona, in a {style} style. Incorprate these contextual elements in a natural way, without forced mentions."
        
        # Build user prompt with token length guidance
        token_range_guidance = f"Your response should be between {MIN_TOKENS} and {MAX_TOKENS} tokens in length. "

        user_prompt = (
            f"Write a detailed {doc_type} on the topic: {topic_string} "
            f"{token_range_guidance}"
            f"Give it your own unique perspective, matching your persona. "
            f"You can include anecdotes, examples, and insights to make it engaging. "
            f"Or you can keep it more straightforward if that fits your style better. "
            f"You should not quote your persona description directly; it is there for stylistic context "
            f"so you can model your response according to the age, gender and linguistic style of the persona. "
            f"The current date is {datetime.now().strftime('%m/%d/%Y')}. "
            f"You do not need to cite sources, but ensure the information is accurate and well-researched. "
            f"Use markdown formatting where appropriate. "
            f"Do not acknowledge the task, include any notes or comments, only respond with the in-character text."
        )

        # Generate content (remove min_tokens parameter)
        response = client.call(
            prompt=user_prompt,
            system_prompt=full_system_prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Create entry
        entry = {
            "persona": {
                "description": system_prompt,
                "style": style,
            },
            "doc_type": doc_type,
            "topic": topic_string,
            "tier": tier,
            "prompt": user_prompt,
            "text": response
        }
        
        print(f"Generated a tier {tier} {doc_type}:")
        print(f"  {response[:100]}...\n", flush=True)
        
        all_entries.append(entry)
    
    # Generate summaries if requested
if ADD_SUMMARY:
    print(f"=== Generating summaries for {len(all_entries)} documents ===", flush=True)
    for i, entry in enumerate(all_entries):
        print(f"  Summarizing document {i+1} of {len(all_entries)}...", flush=True)
        
        # Truncate very long texts (keep first ~3000 tokens worth of chars as safety measure)
        text_to_summarize = entry['text']
        if len(text_to_summarize) > 12000:  # Rough char estimate for ~3k tokens
            text_to_summarize = text_to_summarize[:12000] + "\n\n[Text truncated for summarization]"
            print(f"  Warning: Document truncated for summarization (original length: {len(entry['text'])} chars)", flush=True)
        
        summary_user_prompt = (
            f"Summarize the following text concisely, ensuring key points and sentiments are carried over. "
            f"A full page should be summarized in a paragraph, and individual paragraphs in a single sentence. "
            f"Do not acknowledge the task, include any notes or comments, only respond with the summary.\n\n"
            f"Text to summarize:\n\n{text_to_summarize}"
        )
        
        try:
            summary_response = client.call(
                prompt=summary_user_prompt,
                system_prompt="You are a helpful assistant tasked with summarizing text concisely.",
                temperature=0.5,
                max_tokens=600  # Increased from 250 to allow more room
            )
            
            # Validate response
            if summary_response and summary_response.strip():
                entry["summary"] = summary_response.strip()
                print(f"  Summary: {summary_response[:100]}...\n", flush=True)
            else:
                entry["summary"] = "[Summary generation failed - empty response]"
                print(f"  Warning: Empty summary response for document {i+1}\n", flush=True)
                
        except Exception as e:
            entry["summary"] = f"[Summary generation failed - error: {str(e)}]"
            print(f"  Error summarizing document {i+1}: {e}\n", flush=True)

    # Save results
    unique_id = generate_unique_id()
    filename = f"{DATASET_NAME}_{unique_id}_{DOCUMENT_COUNT}.json"
    filepath = os.path.join(OUTPUT_PATH, filename)
    
    save_json(all_entries, filepath)
    
    print(f"\n=== Saved {DOCUMENT_COUNT} generated documents ===")
    print(f"File: {filepath}")