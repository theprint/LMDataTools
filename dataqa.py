# ============================================================================
# DataQA - Generate Q&A datasets from web content with perspective-based questioning.
# ============================================================================

import os
import json
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from datacore.llm.client import LLMClient
from datacore.config.settings import config
from datacore.config.loader import load_tool_config
from datacore.progress import ProgressReporter
from datacore.io.json_ops import save_json, load_json
from datacore.io.formats import apply_output_format
from datacore.personas.loader import get_persona
from datacore.personas.prompt_manager import inject_persona_into_prompt
from datacore.cleaning.html import clean_html
from datacore.cleaning.text import clean_answer, normalize_whitespace
from datacore.cleaning.validation import (
    is_complete_answer, starts_with_capital, contains_cyrillic, has_empty_code_blocks
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_url(source: str) -> bool:
    """Check if source is a URL or file path."""
    return source.strip().startswith(('http://', 'https://'))


def read_file_content(file_path: str) -> str:
    """Read content from a local file."""
    from pathlib import Path
    
    # Strip quotes if present
    file_path = file_path.strip().strip('"').strip("'")
    
    print(f"  Reading file: {file_path}")

    try:
        path = Path(file_path)

        if not path.exists():
            print(f"  Error: File not found: {file_path}")
            return None
        
        # Read file based on extension
        if path.suffix.lower() in ['.html', '.htm']:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            cleaned = clean_html(content)
        elif path.suffix.lower() in ['.txt', '.md']:
            with open(path, 'r', encoding='utf-8') as f:
                cleaned = f.read()
        else:
            # Try as text file
            with open(path, 'r', encoding='utf-8') as f:
                cleaned = f.read()
        
        print(f"  Extracted {len(cleaned.split())} words")
        return cleaned
        
    except Exception as e:
        print(f"  Error reading file: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


def get_content(source: str) -> str:
    """Get content from either URL or file path."""
    if is_url(source):
        return scrape_url(source)
    else:
        return read_file_content(source)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "DATASET_NAME": "my-qa-dataset",
    "SOURCES": [],
    "MANUAL_PERSPECTIVES": [
        "someone trying to understand the basics",
        "an expert looking for advanced insights",
        "a beginner just getting started",
    ],
    "AUTO_PERSPECTIVES": True,
    "AUTO_PERSPECTIVE_COUNT": 10,
    "TARGET_AUDIENCE": "people interested in learning new things",
    "ANSWER_STYLE": (
        "Include enough detail to go beyond the very basics but stick to one concept at a time. "
        "Use markdown to format responses as needed. Include examples when appropriate and always go step by step."
    ),
    "CHUNK_SIZE": 1024,
    "MIN_ANSWER_LENGTH": 8,
    "MAX_ANSWER_LENGTH": 6000,
    "MAX_QUESTION_LENGTH": 2000,
    "CONFIDENCE_THRESHOLD": 0.68,
    "SAVE_INTERVAL": 10,
    "USE_PERSONA": False,
    "PERSONA_NAME": "Socratic Tutor",
    "OUTPUT_FORMAT": "alpaca",
}

DEFAULT_CONFIG = load_tool_config(DEFAULT_CONFIG, tool_name="dataqa")

JOB_ID = DEFAULT_CONFIG.get("JOB_ID", "local-run")
OUTPUT_PATH = "."

# ============================================================================
# END CONFIGURATION
# ============================================================================


def load_or_create_dataset():
    """Load existing dataset or create new one."""
    dataset_file = os.path.join(OUTPUT_PATH, f"{DEFAULT_CONFIG['DATASET_NAME']}-working.json")
    os.makedirs(OUTPUT_PATH, exist_ok=True) if OUTPUT_PATH != "." else None
    
    if os.path.exists(dataset_file):
        print(f"  Loading existing dataset: {dataset_file}")
        data = load_json(dataset_file)
        print(f"  Found {len(data)} existing Q&A pairs")
        return data, dataset_file
    else:
        print(f"  Starting new dataset: {dataset_file}")
        return [], dataset_file


def save_checkpoint(data, dataset_file):
    """Save current progress."""
    save_json(data, dataset_file)
    print(f"  Checkpoint saved: {len(data)} total Q&A pairs")


def scrape_url(url):
    """Scrape and clean content from a URL."""
    print(f"  Scraping: {url}")
    
    try:
        user_agent = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5756.197 Safari/537.36'
        }
        response = requests.get(url, headers=user_agent, timeout=30)
        response.raise_for_status()
        
        cleaned = clean_html(response.content)
        print(f"  Extracted {len(cleaned.split())} words")
        
        return cleaned
    
    except Exception as e:
        print(f"  Error scraping URL: {e}")
        return None


def chunk_text(text, chunk_size=1024):
    """Split text into manageable chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def generate_auto_perspectives(client, topic, count):
    """Generate perspectives automatically for a topic."""
    print(f"  Generating {count} perspectives for: {topic}")
    
    prompt = (
        f"Generate {count} different types of people who would have questions about {topic}. "
        "For each person, describe who they are and what they're trying to do. "
        "Format each as: (\"who\", \"doing what\") "
        "For example: (\"a student\", \"learning the basics for an exam\") "
        f"Generate exactly {count} perspectives, one per line, in that exact format. "
        "No other text."
    )
    
    response = client.call(
        prompt=prompt,
        temperature=0.8,
        max_tokens=1024
    )
    
    perspectives = []
    lines = response.strip().split('\n')
    
    for line in lines:
        match = re.search(r'\(["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\)', line)
        if match:
            perspectives.append(f"{match.group(1)} who is {match.group(2)}")
    
    if not perspectives:
        print(f"  ⚠️  Failed to parse perspectives. LLM response was:")
        print(f"  {response[:500]}")
        print(f"  Using default perspectives instead.")
        return DEFAULT_CONFIG.get("MANUAL_PERSPECTIVES", [])
    
    print(f"  ✓ Parsed {len(perspectives)} perspectives")
    return perspectives[:count]


def generate_qa_for_chunk(client, content, perspectives):
    """Generate Q&A pairs from content using perspectives."""
    qa_pairs = []

    base_formatting_instructions = (
        "Do not promote specific individuals or product titles, but generalize the topic instead. " 
        "Use Markdown formatting in the answer-text (but not the question) for headlines, bullet points, "
        "lists, links, etc. Format replies to enhance readability as needed. "
        "Never reference other questions and answers in this reply, only address the specific question. "
        "Ensure each question and answer is formatted exactly with just 'Q:' and 'A:' "
        "and do not number questions or answers. It is crucial that the question and answer be worded to stand alone. "
        f"Questions and answers must be tailored to the intended target audience: {DEFAULT_CONFIG['TARGET_AUDIENCE']}. "
        f"{DEFAULT_CONFIG['ANSWER_STYLE']}"
    )

    # *** THIS IS THE ONLY ADDED LOGIC: PERSONA INFUSION ***
    formatting_instructions = base_formatting_instructions
    if DEFAULT_CONFIG.get("USE_PERSONA") and DEFAULT_CONFIG.get("PERSONA_NAME"):
        try:
            persona_data = get_persona(DEFAULT_CONFIG["PERSONA_NAME"])
            formatting_instructions = inject_persona_into_prompt(
                base_prompt=base_formatting_instructions,
                persona_name=persona_data["persona"],
                persona_description=persona_data["description"]
            )
            print(f"      - [Persona Enabled: {DEFAULT_CONFIG['PERSONA_NAME']}]")
        except (ValueError, FileNotFoundError) as e:
            print(f"      - [Persona Warning] Could not load persona: {e}. Using default prompt.")

    # Step 1: Generate High-Level General Questions
    print("      - Generating high-level Q&A...")
    high_level_prompt = (
        f"Generate standalone sets of very high-level, general questions and their detailed answers. "
        f"Focus on questions that would likely be asked by {DEFAULT_CONFIG['TARGET_AUDIENCE']}, "
        f"and base your responses on the following content:\n\n{content}\n\n"
        f"{formatting_instructions} "
        "Remember, you MUST precede all questions with Q: and precede answers with A: "
        "or the entire response will be discarded."
    )
    high_level_response = client.call(prompt=high_level_prompt, temperature=0.6, max_tokens=8000)
    high_level_qa_pairs = extract_qa_pairs(high_level_response)
    qa_pairs.extend(high_level_qa_pairs)
    print(f"        - Added {len(high_level_qa_pairs)} high-level pairs.")

    # Step 2: Generate Follow-Up Questions
    print("      - Generating follow-up Q&A...")
    for question, answer in high_level_qa_pairs:
        follow_up_prompt = (
            f"Generate more detailed follow-up question and answer pairs to this question: {question}.\n"
            f"Both question and answer must be worded to stand alone and written with "
            f"{DEFAULT_CONFIG['TARGET_AUDIENCE']} as the target audience. Do NOT number or group questions and answers."
            f" Go deep into details, techniques and specifics, and provide in-depth, detailed answers"
            f" to each generated question. Add step-by-step instructions if necessary." 
            f"{formatting_instructions}\n"
            f"To re-iterate, we need standalone follow-up Q&A sets that dig into this original question and answer:\n"
            f" Q: {question}\n"
            f" A: {answer}"
        )
        follow_up_response = client.call(prompt=follow_up_prompt, temperature=0.7, max_tokens=8000)
        follow_up_qa_pairs = extract_qa_pairs(follow_up_response)
        qa_pairs.extend(follow_up_qa_pairs)
        if follow_up_qa_pairs:
            print(f"        - Added {len(follow_up_qa_pairs)} follow-up pairs for: '{question[:50]}...'")

    # Step 3: Generate Perspective-Based Questions
    print("      - Generating perspective-based Q&A...")
    for perspective in perspectives:
        prompt = (
            f"Generate standalone sets of questions and their detailed answers. "
            f"Focus on questions that would likely be asked by {perspective}, "
            f"and base your responses on the following content:\n\n{content}\n\n" 
            f"{formatting_instructions} "
            "Remember, you MUST precede all questions with Q: and precede answers with A: "
            "or the entire response will be discarded."
        )
        
        response = client.call(
            prompt=prompt,
            temperature=0.6,
            max_tokens=8000
        )
        
        pairs = extract_qa_pairs(response)
        qa_pairs.extend(pairs)
        
        if pairs:
            print(f"        - Generated {len(pairs)} pairs from perspective: {perspective}")
    
    return qa_pairs


def extract_qa_pairs(text):
    """Extract Q&A pairs from generated text."""
    qa_pairs = []
    seen = set()
    
    pattern = re.compile(r'Q:\s*(.*?)\s*A:\s*(.*?)(?=(?:\s*Q:|$))', re.DOTALL)
    matches = pattern.finditer(text)
    
    for match in matches:
        question = normalize_whitespace(match.group(1).strip())
        answer = clean_answer(match.group(2).strip())
        
        if question and answer:
            pair_key = (question.lower(), answer.lower())
            if pair_key not in seen:
                qa_pairs.append((question, answer))
                seen.add(pair_key)
    
    return qa_pairs


def validate_qa_pair(question, answer):
    """Validate Q&A pair quality."""
    
    # Length checks
    if len(question) > DEFAULT_CONFIG.get("MAX_QUESTION_LENGTH", 2000):
        return False, "question too long"
    
    if len(answer) < DEFAULT_CONFIG.get("MIN_ANSWER_LENGTH", 8):
        return False, "answer too short"
    
    if len(answer) > DEFAULT_CONFIG.get("MAX_ANSWER_LENGTH", 6000):
        return False, "answer too long"
    
    # Answer quality checks
    if not is_complete_answer(answer):
        return False, "incomplete answer"
    
    has_letter, is_capital = starts_with_capital(answer)
    if not has_letter or not is_capital:
        return False, "answer doesn't start with capital"
    
    if contains_cyrillic(answer):
        return False, "contains non-latin characters"
    
    if has_empty_code_blocks(answer):
        return False, "has empty code blocks"
    
    return True, "valid"


def calculate_confidence(client, question, answer):
    """Calculate confidence score for Q&A pair."""
    prompt = (
        f"Rate the quality of this Q&A pair on a scale of 0.0 to 1.0:\n\n"
        f"Q: {question}\n\nA: {answer}\n\n"
        "Consider:\n"
        "- Is the answer accurate and complete?\n"
        "- Does it directly address the question?\n"
        "- Is it well-formatted and clear?\n\n"
        "Respond with ONLY a number between 0.0 and 1.0, nothing else."
    )
    
    try:
        response = client.call(
            prompt=prompt,
            temperature=0.3,
            max_tokens=150
        )

        score_str = re.search(r'0?\.\d+|[01]\.?\d*', response)
        if not score_str:
            print(f"    [confidence] Could not parse score from response: {response[:100]!r}")
            return 0.0

        score = float(score_str.group())
        return round(score, 4)
    except Exception as e:
        print(f"    [confidence] API call failed: {e}")
        return 0.0


def extract_keywords(text, top_n=5):
    """Simple keyword extraction from text."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very'
    }
    
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def generate_readme(dataset_name, total_entries, all_keywords):
    """Generate a simple README for the dataset."""
    
    keyword_freq = {}
    for keywords in all_keywords:
        for kw in keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
    
    top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    readme = f"""# {dataset_name}

A Q&A dataset generated from web content with {total_entries:,} question-answer pairs.

## Top Keywords

"""
    
    for kw, count in top_keywords:
        readme += f"- {kw} ({count} occurrences)\n"
    
    readme += f"\n## Format\n\nEach entry contains:\n- `question`: The question asked\n- `answer`: The detailed answer\n- `source`: Source URL\n- `confidence`: Quality confidence score (0.0-1.0)\n- `keywords`: Relevant keywords extracted from the Q&A\n"
    
    return readme


if __name__ == "__main__":
    print("DataQA - Web Content to Q&A Dataset")
    print("=" * 60)
    
    # Initialize LLMClient inside main to ensure config is loaded
    llm_settings = DEFAULT_CONFIG.get("LLM_SETTINGS", {})
    client = LLMClient(
        base_url=llm_settings.get("base_url"),
        api_key=llm_settings.get("api_key"),
        default_model=llm_settings.get("llm_model")
    )

    # Load/resume dataset
    all_qa_data, dataset_file = load_or_create_dataset()
    all_keywords = [entry.get("keywords", []) for entry in all_qa_data]

    # Load sources
    sources = DEFAULT_CONFIG.get("SOURCES", [])
    if not sources:
        print("No sources provided in the configuration. Exiting.")
        exit(1)

    print(f"Dataset: {DEFAULT_CONFIG['DATASET_NAME']}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    print(f"\nFound {len(sources)} total URLs to process.\n")

    total_sources = len(sources)
    qa_count_since_save = 0
    source_reporter = ProgressReporter(total=total_sources, phase="Processing sources")
    
    # Tracking stats for verification
    total_chunks_processed = 0
    total_pairs_generated = 0
    total_pairs_validated = 0
    total_pairs_added = 0
    
    try:
        for source_idx, source in enumerate(sources):  
            print(f"\n{'='*60}")
            print(f"Processing: {source}")  
            print('='*60)
            
            # Get content (handles both URLs and files)
            content = get_content(source)
            if not content:
                print("  Skipping due to error")
                continue
            
            # Chunk content
            chunks = chunk_text(content, chunk_size=DEFAULT_CONFIG.get("CHUNK_SIZE", 1024))
            print(f"  Split into {len(chunks)} chunks")
            
            # Get perspectives
            if DEFAULT_CONFIG.get("AUTO_PERSPECTIVES", True):
                # Extract topic from source (URL or file path)
                topic = None
                
                if is_url(source):
                    # Extract topic from URL
                    from urllib.parse import urlparse
                    parsed = urlparse(source)
                    
                    # Try path segments first (excluding empty strings)
                    path_parts = [p for p in parsed.path.split('/') if p]
                    
                    if path_parts:
                        # Use last meaningful path segment
                        topic = path_parts[-1]
                        # Remove common file extensions
                        topic = re.sub(r'\.(html?|php|asp|jsp|txt|md)$', '', topic, flags=re.IGNORECASE)
                        # Convert dashes/underscores to spaces
                        topic = topic.replace('-', ' ').replace('_', ' ')
                    else:
                        # Fallback to domain name if path is empty
                        topic = parsed.netloc.replace('www.', '')
                else:
                    # Extract topic from file path
                    from pathlib import Path
                    path = Path(source)
                    # Use filename without extension
                    topic = path.stem
                    # Convert dashes/underscores to spaces
                    topic = topic.replace('-', ' ').replace('_', ' ')
                
                # Final cleanup
                topic = topic.strip() if topic else None
                
                if not topic:
                    print(f"  Warning: Could not extract topic from source, using generic 'the content'")
                    topic = "the content"
                
                print(f"  Generating perspectives for: {topic}")
                perspectives = generate_auto_perspectives(client, topic, DEFAULT_CONFIG.get("AUTO_PERSPECTIVE_COUNT", 5))
            else:
                perspectives = DEFAULT_CONFIG.get("MANUAL_PERSPECTIVES", [])
            
            print(f"  Using {len(perspectives)} perspectives")
            
            # Accumulate Q&A pairs for all chunks of a single source
            all_pairs_for_source = []
            pairs_count_before_source = len(all_qa_data)

            # Generate Q&A for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Calculate progress for webapp monitoring
                progress_for_this_chunk = ((chunk_idx + 1) / len(chunks)) * (1 / total_sources) * 100
                progress_from_previous_sources = (source_idx / total_sources) * 100
                total_progress = round(progress_from_previous_sources + progress_for_this_chunk)
                
                print(f"    Processing chunk {chunk_idx + 1}/{len(chunks)}... Progress: {total_progress}/100", flush=True)
                
                qa_pairs = generate_qa_for_chunk(client, chunk, perspectives)
                print(f"      → Generated {len(qa_pairs)} pairs from this chunk")
                
                all_pairs_for_source.extend(qa_pairs)
                total_chunks_processed += 1
                total_pairs_generated += len(qa_pairs)
                
                print(f"      → Accumulated total for source: {len(all_pairs_for_source)} pairs from {chunk_idx + 1} chunks")
            # Validate and score all pairs for the URL
            print(f"\n  Validating {len(all_pairs_for_source)} pairs from {len(chunks)} chunks for {source}...")
            
            for question, answer in all_pairs_for_source:
                total_pairs_validated += 1
                
                is_valid, reason = validate_qa_pair(question, answer)
                
                if not is_valid:
                    print(f"    Rejected: {reason}")
                    continue
                
                # Calculate confidence
                confidence = calculate_confidence(client, question, answer)
                
                if confidence < DEFAULT_CONFIG.get("CONFIDENCE_THRESHOLD", 0.68):
                    print(f"    Rejected: low confidence ({confidence})")
                    continue
                
                # Extract keywords
                combined_text = f"{question} {answer}"
                keywords = extract_keywords(combined_text)
                
                # Create entry
                entry = {
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "confidence": confidence,
                    "keywords": keywords,
                    "_tool": "dataqa",
                    "_version": "2.0",
                }
                
                all_qa_data.append(entry)
                all_keywords.append(keywords)
                qa_count_since_save += 1
                total_pairs_added += 1
                print(f"    [ADDED] (confidence: {confidence})")
                
                # Checkpoint if needed
                if DEFAULT_CONFIG.get("SAVE_INTERVAL", 10) > 0 and qa_count_since_save >= DEFAULT_CONFIG.get("SAVE_INTERVAL", 10):
                    save_checkpoint(all_qa_data, dataset_file)
                    qa_count_since_save = 0
            
            pairs_added_for_source = len(all_qa_data) - pairs_count_before_source
            source_reporter.update(source_idx + 1)
            print(f"\n  [SOURCE COMPLETE] Added {pairs_added_for_source} pairs from {source} ({source_idx + 1}/{total_sources})")

            # Save after each URL
            save_checkpoint(all_qa_data, dataset_file)
            qa_count_since_save = 0
        
        # Final save and README generation
        print("\n" + "=" * 60)
        print("Finalizing dataset...")
        print("=" * 60)
        
        # Print comprehensive stats
        print(f"\n📊 GENERATION STATISTICS:")
        print(f"  • Total chunks processed: {total_chunks_processed}")
        print(f"  • Total pairs generated: {total_pairs_generated}")
        print(f"  • Total pairs validated: {total_pairs_validated}")
        print(f"  • Total pairs added to dataset: {total_pairs_added}")
        print(f"  • Final dataset size: {len(all_qa_data)}")
        
        if not all_qa_data:
            print("\n⚠️  No Q&A pairs generated. Exiting.")
            exit(1)
        
        # Save final version with proper naming
        size_k = round(len(all_qa_data) / 1000, 2)
        data_to_save, fmt = apply_output_format(
            all_qa_data, DEFAULT_CONFIG.get("OUTPUT_FORMAT", "alpaca"),
            instruction_key="question", output_key="answer",
        )
        final_file = os.path.join(OUTPUT_PATH, f"{DEFAULT_CONFIG['DATASET_NAME']}-{size_k}k-{fmt}.json")
        save_json(data_to_save, final_file)
        
        # Generate and save README
        readme_content = generate_readme(DEFAULT_CONFIG['DATASET_NAME'], len(all_qa_data), all_keywords)
        readme_file = os.path.join(OUTPUT_PATH, "README.md")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Clean up temporary working file
        if os.path.exists(dataset_file):
            os.remove(dataset_file)
        
        usage = client.get_usage_stats()
        print(f"TOKENS {usage['prompt_tokens']}/{usage['completion_tokens']}", flush=True)

        print(f"\nFinal dataset: {final_file}")
        print(f"README: {readme_file}")
        print(f"Total Q&A pairs: {len(all_qa_data)}")
        print(f"Token usage: {usage['total_tokens']:,} total ({usage['prompt_tokens']:,} prompt / {usage['completion_tokens']:,} completion)")
        print("\n" + "=" * 60)
        print("DataQA Complete!")
        print("=" * 60)
        
    except Exception as e:
        # Final save before exiting on error
        if 'all_qa_data' in locals() and 'dataset_file' in locals():
            print("\nAttempting to save progress before exiting due to error...")
            save_checkpoint(all_qa_data, dataset_file)
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)