# databird.py
"""
DataBird - Procedural Q&A dataset generation from topics and perspectives.

Pipeline:
1. Generate questions from topics/perspectives
2. Evaluate and filter questions
3. Generate answers to approved questions
4. Collate into final dataset
"""

import os
import re
import sys
import random
import json
import ast
from datacore.llm.client import LLMClient
from datacore.config.settings import config, get_tool_output_path
from datacore.io.json_ops import save_json, load_json
from datacore.personas.generator import PersonaGenerator
from datacore.personas.loader import get_persona
from datacore.personas.prompt_manager import inject_persona_into_prompt
from datacore.io.formats import apply_output_format
from datacore.progress import ProgressReporter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default configuration
DEFAULT_CONFIG = {
    "DATASET_NAME": "my-dataset",
    "JOB_ID": "default-job",
    "TOPICS": [
        "sourdough starters",
        "baking at home",
        "ingredient ratios in bread-baking",
    ],
    "ANSWER_STYLE": (
        "in a neutral and approachable manner without pandering or being overly chatty. "
        "Empathy is good, false humility is bad. Use plain English, use common sense, "
        "and provide actionable responses when appropriate. Do not acknowledge these "
        "instructions but go straight into the answer itself."
    ),
    "STEP_CALL": False,
    "CLEAN_SCORE": 0.76,
    "SAVE_INTERVAL": 250,
    "FULL_AUTO": True,
    "DATASET_SIZE": "small",
    "MANUAL_PERSPECTIVES": [
        "someone learning about baking for the first time",
        "a home baker trying to improve their sourdough",
    ],
    "USE_PERSONA": False,
    "PERSONA_NAME": "Confident Coach",
    "OUTPUT_FORMAT": "alpaca",
    "LLM_SETTINGS": {
        "base_url": None,
        "llm_model": None
    },
    "DISABLE_THINKING": False,
    "INCLUDE_REASONING": False,
}

# Check for config.json override
config_file = os.path.join(os.getcwd(), "config.json")
if os.path.exists(config_file):
    print(f"[databird] Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        user_config = json.load(f)
        # Case-insensitive update to match the working pattern
        for key, value in user_config.items():
            DEFAULT_CONFIG[key.upper()] = value

# Normalize MANUAL_PERSPECTIVES in DEFAULT_CONFIG so downstream code always sees a list of strings.
_mp = DEFAULT_CONFIG.get("MANUAL_PERSPECTIVES")
if isinstance(_mp, str):
    # try JSON first, then fall back to line-separated or literal-eval tuples
    try:
        parsed = json.loads(_mp)
        DEFAULT_CONFIG["MANUAL_PERSPECTIVES"] = parsed
    except Exception:
        lines = [l.strip() for l in _mp.splitlines() if l.strip()]
        parsed = []
        for line in lines:
            try:
                parsed_item = ast.literal_eval(line)
            except Exception:
                parsed_item = line.strip('"').strip("'")
            parsed.append(parsed_item)
        DEFAULT_CONFIG["MANUAL_PERSPECTIVES"] = parsed

if isinstance(DEFAULT_CONFIG.get("MANUAL_PERSPECTIVES"), (list, tuple)):
    normalized = []
    for item in DEFAULT_CONFIG["MANUAL_PERSPECTIVES"]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append(f"{item[0]} {item[1]}")
        else:
            normalized.append(str(item))
    DEFAULT_CONFIG["MANUAL_PERSPECTIVES"] = normalized

# Apply configuration
DATASET_NAME = DEFAULT_CONFIG["DATASET_NAME"]
JOB_ID = DEFAULT_CONFIG["JOB_ID"]
TOPICS = DEFAULT_CONFIG["TOPICS"]
ANSWER_STYLE = DEFAULT_CONFIG["ANSWER_STYLE"]
STEP_CALL = DEFAULT_CONFIG["STEP_CALL"]
CLEAN_SCORE = DEFAULT_CONFIG["CLEAN_SCORE"]
SAVE_INTERVAL = DEFAULT_CONFIG["SAVE_INTERVAL"]
FULL_AUTO = DEFAULT_CONFIG["FULL_AUTO"]
DATASET_SIZE = DEFAULT_CONFIG["DATASET_SIZE"]
MANUAL_PERSPECTIVES = DEFAULT_CONFIG["MANUAL_PERSPECTIVES"]
USE_PERSONA = DEFAULT_CONFIG.get("USE_PERSONA", False)
PERSONA_NAME = DEFAULT_CONFIG.get("PERSONA_NAME", "")
DISABLE_THINKING = DEFAULT_CONFIG.get("DISABLE_THINKING", False)
INCLUDE_REASONING = DEFAULT_CONFIG.get("INCLUDE_REASONING", False)
OUTPUT_FORMAT = DEFAULT_CONFIG.get("OUTPUT_FORMAT", "alpaca")

# When thinking is disabled, pass this to client.call() as extra_body.
# Supported by LM Studio for Qwen3, DeepSeek-R1, and similar reasoning models.
_no_think = {"enable_thinking": False} if DISABLE_THINKING else None
 
OUTPUT_PATH = "." # Save files to the current job directory
# ============================================================================
# END CONFIGURATION
# ============================================================================


# Question type descriptors
DESCRIPTORS = [
    "most relevant", "toughest", "hardest", "most common", "most misunderstood",
    "most interesting", "most inspiring", "most technical", "most deep-cut",
    "best how-to", "most challenging", "wisest", "most important", "most critical",
    "most advanced", "beginner-friendly", "funniest", "most realistic",
    "most applicable", "most exciting"
]

if DATASET_SIZE == 'medium' or DATASET_SIZE == 'large':
    DESCRIPTORS.extend([
        "most pointed", "most cost-conscious", "most astute", "cleverest",
        "multi-faceted", "most imaginative", "exploratory", "most detail-oriented",
        "most capable", "most evocative", "grandest", "most debateable",
        "most objective", "most subjective", "most well-educated",
        "likeliest open-ended", "most overlooked, fundamental", "most investigative",
        "most soul searching", "most inquisitive", "most observant",
        "most gut-wrenching", "scariest", "most empathetic", "kindest"
    ])

if DATASET_SIZE == 'large':
    DESCRIPTORS.extend([
        "most beautiful", "most timely", "keenest", "most intriguing", "multi-step",
        "visionary", "most alarming", "most specific", "most creative",
        "most practical", "greatest", "typical kind of", "most clarifying",
        "most suggestive", "most loaded", "trickiest open-ended", "most ignored",
        "specific research", "most curious", "most banal", "attention-grabbing",
        "most skeptical", "most frantic", "trouble-shooting", "brainstorming"
    ])


def strip_quotation_marks(s):
    """Remove quotation marks from start and end of string."""
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    return s


def generate_perspectives(client, topics):
    """Generate perspectives automatically using LLM."""
    asker_types = [
        "someone being introduced to",
        "someone with advanced knowledge in",
        "someone working on a project related to",
        "someone struggling with an aspect of",
        "an amateur in"
    ]
    
    if DATASET_SIZE == 'medium' or DATASET_SIZE == 'large':
        asker_types.extend([
            "an intermediate learner in",
            "a person trying to grasp",
            "a novice in",
            "an authority on",
            "someone solving a specific issue related to"
        ])
    
    if DATASET_SIZE == 'large':
        asker_types.extend([
            "a deep-diving explorer in",
            "someone who has misunderstood",
            "a fan of",
            "a problem fixer in",
            "a theorist of",
        ])
    
    perspectives = []
    print(f"Generating {len(asker_types) * len(topics)} unique perspectives...")
    
    for asker_type in asker_types:
        for topic in topics:
            ask_prompt = (
                "You are developing a realistic persona as part of synthesizing a data set. "
                "Your precision and creativity is required. "
                "Please create and return a partial string (for later concatenation) that briefly "
                "describes 1 specific type of person (use a job title or similar descriptor) "
                f"doing something. Both the character and the action must in some way be related to {topic}, "
                "but keep it believable and a little open-ended. The character should be someone with a "
                "related question or problem to solve. "
                "It's crucial that the string you create follows the format shown in the 3 examples here "
                "(just the words, not the 'example x:' part), otherwise the result may not concatenate properly:\n"
                "\nexample 1: a seasoned TTRPG game master who brainstorming a campaign"
                "\nexample 2: an amateur home cook who is struggling to learn a new technique"
                "\nexample 3: a budding author working out an issue with their novel"
                f"\n\nFor this particular string, think of {asker_type} the topic of {topic} as basis for your response. "
                "Please respond with your partial string and no other text."
            )
            
            persona = client.call(
                prompt=ask_prompt,
                temperature=0.8,
                max_tokens=4096,
                extra_body=_no_think,
            )
            
            persona = strip_quotation_marks(persona)
            print(f"  - {persona}")
            perspectives.append(persona)
    
    return perspectives


def generate_questions(client, topics, perspectives):
    """Generate questions from topics and perspectives."""
    questions = []

    per_topic = min(10, len(DESCRIPTORS))
    total_expected = len(perspectives) * len(topics) * per_topic if perspectives and topics else 0
    reporter = ProgressReporter(total=max(1, total_expected), phase="Generating questions")

    for asker in perspectives:
        print(f"\nGenerating questions from perspective: {asker}")

        for topic in topics:
            ten_chosen = random.sample(DESCRIPTORS, min(10, len(DESCRIPTORS)))
            print(f"  10 questions about {topic}:")

            for descriptor in ten_chosen:
                pre_prompt = (
                    f"Your task is to create a straightforward question that a user might ask a large language model. "
                    f"Begin your question with one of: where, why, when, who, what, how or please - and with that in mind: "
                    f"I want you to think of the *{descriptor}* question about {topic}, that only "
                    f"{asker} would ask? Do not answer the question. Do not put your response in quotation marks. "
                    f"\nDo NOT confirm, repeat or comment on the given task in any way - doing so will invalidate your response. "
                    "Let me also repeat this: DO NOT ANSWER THE QUESTION THAT YOU COME UP WITH! "
                    f"You MUST respond in plain, conversational English with the correctly formatted query and no other text!"
                )

                question = client.call(
                    prompt=pre_prompt,
                    temperature=0.9,
                    max_tokens=4096,
                    extra_body=_no_think,
                )

                if question:
                    question = strip_quotation_marks(question)
                    item = {
                        "asker": asker,
                        "topic": topic,
                        "question": question
                    }
                    questions.append(item)
                    reporter.update(len(questions))
                    print(f"    {len(questions)}: {question}")

    reporter.done()
    return questions


def evaluate_question(client, question, asker, topic):
    """Evaluate a question's quality."""
    eval_prompt = (
        "You must evaluate the following question assign a score from 0.00 to 1.00. "
        "This score should be based on how well-suited you think this question is for inclusion "
        "in a high quality data set. This is crucial for setting the quality bar of the final data sets. "
        f"For context, the question here is about {topic}, and it is asked by {asker}. "
        "Keep this context in mind when evaluating the question. "
        "A good questions will either be relevant to a lot of people or be useful AND so specific "
        "that finding it elsewhere would be challenging. A good question is well-written and clear, "
        "where a bad question is vague, verbose or poorly formatted. Good questions can be open to "
        "different opinions and replies where bad questions are leading. A good question makes sense "
        "logically and is not a leading question, where bad questions might mix concepts or include "
        "incorrect assumptions. Refusal to ask or generate a question is a complete failure. "
        "Good questions can be complex or simple, but a good question is never convoluted or verbose. "
        "Take all of these factors into account when evaluating the question here. "
        "You MUST respond with 1 single score as a 4-decimal floating point number and no other text. "
        "\nExample 1 represents a very low score: 0.1103 "
        "\nExample 2 represents a low score: 0.3782 "
        "\nExample 3 represents a medium score: 0.5952 "
        "\nExample 4 represents a high score: 0.7867 "
        "\nExample 5 represents a very high score: 0.9738"
        f"\n\nQuestion: {question}"
        "\n\nRespond with your score now: "
    )
    
    tries = 0
    while tries < 3:
        try:
            raw_score = client.call(
                prompt=eval_prompt,
                temperature=0.5,
                max_tokens=4096,
                extra_body={"enable_thinking": False},  # scoring never benefits from thinking
            )
        except Exception as e:
            tries += 1
            print(f"  - API error on attempt {tries}/3: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue

        # Strip thinking blocks from reasoning models (e.g. <think>...</think>)
        clean_score = re.sub(r'<think>.*?</think>', '', raw_score, flags=re.DOTALL).strip()

        score = re.search(r'\b([01]\.\d+)\b', clean_score)
        if score:
            return float(score.group(1))

        tries += 1
        print(f"  - Error on attempt {tries}/3: '{clean_score[:80]}' is not a valid float (0.x - 1.0)", flush=True)

    print(f"  - Giving up. No score generated, assigning -1", flush=True)
    return -1.0


def clean_questions(client, questions, min_score):
    """Evaluate and filter questions."""
    if not questions:
        print("\nNo questions to evaluate.")
        return []

    print(f"\nEvaluating {len(questions)} questions...")

    approved = []
    for i, item in enumerate(questions):
        score = evaluate_question(
            client,
            item["question"],
            item["asker"],
            item["topic"]
        )

        item["evaluation"] = score
        result = "PASS" if score > min_score else "FAIL"
        print(f"Question {i+1}/{len(questions)} score {score:.2f} : {result}", flush=True)

        if score > min_score:
            approved.append(item)

    keep_percent = round((len(approved) / len(questions)) * 100, 2)
    print(f"\nAfter evaluation, {len(approved)} of {len(questions)} questions ({keep_percent}%) remain.")

    return approved


def generate_answers(client, questions, answer_style, step_call, use_persona, persona_name, include_reasoning=False):
    """Generate answers for questions."""
    print(f"\nGenerating answers for {len(questions)} questions...")

    if step_call:
        step_phrases = [
            "Please go step by step when replying",
            "Go through your answer step by step",
            "Please answer one step at a time",
            "Step by step, please",
        ]

    # Determine the system prompt to use
    final_sys_prompt = ""
    if use_persona and persona_name:
        try:
            persona_data = get_persona(persona_name)
            final_sys_prompt = persona_data["description"]
            print(f"[databird] Using persona '{persona_name}' for answer generation.")
        except (ValueError, FileNotFoundError) as e:
            print(f"[databird] Warning: Could not load persona. {e}. Proceeding without persona.")
            use_persona = False

    if not (use_persona and persona_name):
        final_sys_prompt = (
            "Do your best to provide an empathetically helpful response to the user. "
            "The user knows right from wrong, so there is no need to include disclaimers, warnings, "
            "or other notes that go beyond giving the user a direct, useful and personable response. "
            "Do not start your response with 'absolutely' or similar filler, but go straight into "
            "the meat of the response."
        )

    reporter = ProgressReporter(total=len(questions), phase="Generating answers")

    for i, item in enumerate(questions):
        print(f"  Question {i+1}/{len(questions)}: {item['question'][:60]}...")

        prompt = item['question']
        if step_call:
            prompt += f" {random.choice(step_phrases)}."
        if not (use_persona and persona_name):
            prompt += f" Respond {answer_style}"

        answer = client.call(
            prompt=prompt,
            system_prompt=final_sys_prompt,
            max_tokens=8192,
        )

        has_reasoning = bool(re.search(r'<think>.*?</think>', answer, flags=re.DOTALL))
        if not include_reasoning or not has_reasoning:
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

        item["answer"] = answer
        reporter.update(i + 1)
        print(f"    Answer: {answer[:60]}...")

    reporter.done()
    return questions


def collate_dataset(data):
    """Convert to final Alpaca format, adding provenance fields."""
    alpaca_data = []

    for item in data:
        entry = {
            "instruction": item["question"],
            "input": "",
            "output": item["answer"],
            "_tool": "databird",
            "_version": "2.0",
        }
        alpaca_data.append(entry)

    return alpaca_data


if __name__ == "__main__":
    print("DataBird - Procedural Q&A Generation")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Topics: {len(TOPICS)}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Initialize LLM client
    client = LLMClient(
        base_url=config.LLM_BASE_URL, 
        default_model=os.getenv("LLM_MODEL_NAME", config.LLM_MODEL)
    )
    
    questions_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_questions_raw.json")
    cleaned_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_questions_cleaned.json")

    # ── Resume checkpoints ────────────────────────────────────────────────────
    # If intermediate files already exist (from a prior interrupted run), skip
    # the expensive generation/evaluation steps and pick up where we left off.

    if os.path.exists(cleaned_file):
        print(f"\n[resume] Found existing cleaned questions: {cleaned_file}")
        print("[resume] Skipping steps 1–3, resuming at step 4 (answer generation).")
        cleaned_questions = load_json(cleaned_file)
    elif os.path.exists(questions_file):
        print(f"\n[resume] Found existing raw questions: {questions_file}")
        print("[resume] Skipping steps 1–2, resuming at step 3 (evaluation).")
        questions = load_json(questions_file)
        cleaned_questions = None  # will be set in step 3 below
    else:
        questions = None
        cleaned_questions = None

    if cleaned_questions is None and questions is None:
        # Step 1: Generate or use perspectives
        print("\n" + "=" * 60)
        print("STEP 1: Generate Perspectives")
        print("=" * 60)

        if FULL_AUTO:
            perspectives = generate_perspectives(client, TOPICS)
        else:
            perspectives = MANUAL_PERSPECTIVES
            print(f"Using {len(perspectives)} manual perspectives")

        # Step 2: Generate questions
        print("\n" + "=" * 60)
        print("STEP 2: Generate Questions")
        print("=" * 60)

        questions = generate_questions(client, TOPICS, perspectives)

        if not questions:
            print("\n[ERROR] Step 2 produced no questions. Check your LLM connection and config.")
            sys.exit(1)

        save_json(questions, questions_file)
        print(f"\nSaved raw questions: {questions_file}")

    if cleaned_questions is None:
        # Step 3: Clean questions
        print("\n" + "=" * 60)
        print("STEP 3: Evaluate and Filter Questions")
        print("=" * 60)

        cleaned_questions = clean_questions(client, questions, CLEAN_SCORE)

        save_json(cleaned_questions, cleaned_file)
        print(f"\nSaved cleaned questions: {cleaned_file}")
    
    # Step 4: Generate answers
    print("\n" + "=" * 60)
    print("STEP 4: Generate Answers")
    print("=" * 60)
    
    answered_data = generate_answers(client, cleaned_questions, ANSWER_STYLE, STEP_CALL, USE_PERSONA, PERSONA_NAME, INCLUDE_REASONING)
    
    # Step 5: Collate final dataset
    print("\n" + "=" * 60)
    print("STEP 5: Collate Final Dataset")
    print("=" * 60)
    
    final_dataset = collate_dataset(answered_data)

    data_to_save, fmt = apply_output_format(
        final_dataset, OUTPUT_FORMAT,
        instruction_key="instruction", output_key="output", input_key="input",
    )

    # Save final dataset
    size_k = round(len(data_to_save) / 1000, 2)
    final_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}-{size_k}k-{fmt}.json")
    save_json(data_to_save, final_file)
    
    # Clean up temporary files
    if os.path.exists(questions_file):
        os.remove(questions_file)
    if os.path.exists(cleaned_file):
        os.remove(cleaned_file)
    
    # Emit token-usage summary so the webapp can persist it in job metadata.
    usage = client.get_usage_stats()
    print(f"TOKENS {usage['prompt_tokens']}/{usage['completion_tokens']}", flush=True)

    print("\n" + "=" * 60)
    print("DataBird Complete!")
    print("=" * 60)
    print(f"Final dataset: {final_file}")
    print(f"Total entries: {len(final_dataset)}")
    print(f"Token usage  : {usage['total_tokens']:,} total ({usage['prompt_tokens']:,} prompt / {usage['completion_tokens']:,} completion)")