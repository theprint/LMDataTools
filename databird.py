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
import math
import time
import random
import json
from datacore.llm.client import LLMClient
from datacore.config.settings import config, get_tool_output_path
from datacore.io.json_ops import save_json, load_json
from datacore.personas.generator import PersonaGenerator
from datacore.personas.loader import get_persona
from datacore.personas.prompt_manager import inject_persona_into_prompt

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
        ("someone", "learning about baking for the first time"),
        ("a home baker", "trying to improve their sourdough"),
    ],
    "USE_PERSONA": False,
    "PERSONA_NAME": "Confident Coach",
    "LLM_SETTINGS": {
        "base_url": None,
        "llm_model": None
    }
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
                max_tokens=1024,
            )
            
            persona = strip_quotation_marks(persona)
            print(f"  - {persona}")
            perspectives.append(persona)
    
    return perspectives


def generate_questions(client, topics, perspectives):
    """Generate questions from topics and perspectives."""
    questions = []
    
    for asker in perspectives:
        print(f"\nGenerating questions from perspective: {asker}")
        
        for topic in topics:
            # Select 10 random descriptors for this topic
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
                    max_tokens=512,
                )
                
                if question:
                    question = strip_quotation_marks(question)
                    item = {
                        "asker": asker,
                        "topic": topic,
                        "question": question
                    }
                    questions.append(item)
                    print(f"    {len(questions)}: {question}")
    
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
    )
    
    tries = 0
    while tries < 3:
        raw_score = client.call(
            prompt=eval_prompt,
            temperature=0.5,
            max_tokens=200,
        )
        
        try:
            score = float(raw_score.strip())
            return score
        except:
            tries += 1
            print(f"  - Error on attempt {tries}/3: '{raw_score}' is not a float")
    
    print(f"  - Giving up. No score generated, assigning -1")
    return -1.0


def clean_questions(client, questions, min_score):
    """Evaluate and filter questions."""
    print(f"\nEvaluating {len(questions)} questions...")
    
    approved = []
    for i, item in enumerate(questions):
        print(f"  Question {i+1}/{len(questions)}")
        
        score = evaluate_question(
            client,
            item["question"],
            item["asker"],
            item["topic"]
        )
        
        item["evaluation"] = score
        
        if score > min_score:
            approved.append(item)
    
    keep_percent = round((len(approved) / len(questions)) * 100, 2)
    print(f"\nAfter evaluation, {len(approved)} of {len(questions)} questions ({keep_percent}%) remain.")
    
    return approved


def generate_answers(client, questions, answer_style, step_call, use_persona, persona_name):
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
            # The persona description REPLACES the base system prompt entirely.
            final_sys_prompt = persona_data["description"]
            print(f"[databird] Using persona '{persona_name}' for answer generation.")
        except (ValueError, FileNotFoundError) as e:
            print(f"[databird] Warning: Could not load persona. {e}. Proceeding without persona.")
            # Fallback to default prompt if persona fails to load
            use_persona = False 

    if not (use_persona and persona_name):
        # Use the default base prompt ONLY if a persona is not being used.
        final_sys_prompt = (
            "Do your best to provide an empathetically helpful response to the user. "
            "The user knows right from wrong, so there is no need to include disclaimers, warnings, "
            "or other notes that go beyond giving the user a direct, useful and personable response. "
            "Do not start your response with 'absolutely' or similar filler, but go straight into "
            "the meat of the response."
        )

    for i, item in enumerate(questions):
        print(f"  Question {i+1}/{len(questions)}: {item['question'][:60]}...")
        
        prompt = item['question']
        if step_call:
            prompt += f" {random.choice(step_phrases)}."
        # Only add the default answer style if a persona is NOT being used
        if not (use_persona and persona_name):
            prompt += f" Respond {answer_style}"

        answer = client.call(
            prompt=prompt,
            system_prompt=final_sys_prompt,
            max_tokens=8192,
        )
        
        item["answer"] = answer
        print(f"    Answer: {answer[:60]}...")
    
    return questions


def collate_dataset(data):
    """Convert to final Alpaca format."""
    alpaca_data = []
    
    for item in data:
        entry = {
            "instruction": item["question"],
            "input": "",
            "output": item["answer"]
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
    
    # DEBUG: Print raw LLM_MODEL environment variable
    print(f"DEBUG: os.getenv('LLM_MODEL') = \"{os.getenv('LLM_MODEL')}\"")

    # Initialize LLM client using the original, working method
    client = LLMClient(
        base_url=config.LLM_BASE_URL, 
        default_model=os.getenv("LLM_MODEL_NAME", config.LLM_MODEL)
    )
    
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
    
    # Save raw questions
    questions_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_questions_raw.json")
    save_json(questions, questions_file)
    print(f"\nSaved raw questions: {questions_file}")
    
    # Step 3: Clean questions
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate and Filter Questions")
    print("=" * 60)
    
    cleaned_questions = clean_questions(client, questions, CLEAN_SCORE)
    
    # Save cleaned questions
    cleaned_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_questions_cleaned.json")
    save_json(cleaned_questions, cleaned_file)
    print(f"\nSaved cleaned questions: {cleaned_file}")
    
    # Step 4: Generate answers
    print("\n" + "=" * 60)
    print("STEP 4: Generate Answers")
    print("=" * 60)
    
    answered_data = generate_answers(client, cleaned_questions, ANSWER_STYLE, STEP_CALL, USE_PERSONA, PERSONA_NAME)
    
    # Step 5: Collate final dataset
    print("\n" + "=" * 60)
    print("STEP 5: Collate Final Dataset")
    print("=" * 60)
    
    final_dataset = collate_dataset(answered_data)
    
    # Save final dataset
    size_k = round(len(final_dataset) / 1000, 2)
    final_file = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}-{size_k}k-alpaca.json")
    save_json(final_dataset, final_file)
    
    print("\n" + "=" * 60)
    print("DataBird Complete!")
    print("=" * 60)
    print(f"Final dataset: {final_file}")
    print(f"Total entries: {len(final_dataset)}")