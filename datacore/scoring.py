# datacore/scoring.py
import re
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .llm.client import LLMClient

"""Utilities for scoring and evaluating model outputs."""
def calculate_cosine_similarity(text1, text2):
    """Calculates cosine similarity between two texts, handling short or empty documents."""
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        vectors = vectorizer.fit_transform([text1, text2])
        if vectors.shape[0] < 2:
            return 0.0
        similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        if 0.75 <= similarity_score <= 0.95:
            return 1 - abs(similarity_score - 0.85) / 0.13
        else:
            return 0.0
    except ValueError:
        return 0.0

def calculate_length_difference_score(original_length, rewritten_length):
    """Calculates length difference score based on specified rules."""
    if original_length == 0:
        return 0.0
    ratio = rewritten_length / original_length
    if 0.85 <= ratio <= 1.15:
        return 1.5
    elif 0.65 <= ratio < 0.85:
        return 1 - ((0.85 - ratio) / 0.1)
    elif 1.15 < ratio <= 1.35:
        return 1 - ((ratio - 1.15) / 0.1)
    else:
        return 0.0

def calculate_repetition_score(text):
    """Calculates repetition score based on repeated phrases."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 3:
        return 1.0
    phrases = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    repeated_phrases = set(phrase for phrase in phrases if phrases.count(phrase) > 1)
    penalty = len(repeated_phrases) * 0.05
    return max(0.0, 1.0 - penalty)

def calculate_persona_score(text, original_text):
    """Calculates persona score based on specific phrases and words."""
    score = 0
    lazy_openers = ["here's how", "okay,", "sure thing", "certainly,", "absolutely,", "alright,", "hey,", "absolutely!", "certainly!","sure!"]
    if text.lower().startswith(tuple(lazy_openers)):
        score -= 0.125
    else:
        score += 0.5
    if text.lower().startswith(tuple(lazy_openers)) and not original_text.lower().startswith(tuple(lazy_openers)):
        score -= 0.25
    return score

def check_for_non_latin(input_string):
    """Checks if a string contains any non-Latin characters and returns a score."""
    non_latin_count = 0
    for char in input_string:
        if not (
                (ord(char) <= 0x024F) or
                (char.isspace()) or
                (char.isdigit()) or
                (ord(char) in range(0x2000, 0x218F))
        ):
            non_latin_count += 1
    if non_latin_count == 0:
        return 1.0
    else:
        return max(-0.1 * non_latin_count, -2)

def calc_bad_words_score(rewritten_text, ai_role):
    """Calculates the score based on specified bad words."""
    bad_word_patterns = []
    if ai_role and "bad_words" in ai_role and ai_role["bad_words"]:
        for word in ai_role["bad_words"]:
            escaped_word = re.escape(word)
            bad_word_patterns.append(r"\b" + escaped_word + r"\b")
    else:
        default_bad_words = ["blockchain", "NFT", "MAGA", "\u2018em", "covid", "vr/ar"]
        for word in default_bad_words:
             escaped_word = re.escape(word)
             bad_word_patterns.append(r"\b" + escaped_word + r"\b")

    if not bad_word_patterns:
        return 0.0

    try:
        pattern = r'(' + '|'.join(bad_word_patterns) + r')'
        matches = re.findall(pattern, rewritten_text, flags=re.IGNORECASE)
    except re.error as e:
        print(f"Regex error in calc_bad_words_score: {e}")
        return 0.0

    penalty_per_match = -0.25
    return len(matches) * penalty_per_match

def calculate_number_consistency_score(original_text, rewritten_text):
    """Compares the presence and values of numbers between texts."""
    def extract_numbers(text):
        numbers = []
        pattern = r'\b(\d+\.?\d*|\.\d+)\b'
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                num = float(m)
                if '.' not in m:
                    num = int(num)
                numbers.append(num)
            except ValueError:
                continue
        return numbers

    original_numbers = extract_numbers(original_text)
    rewritten_numbers = extract_numbers(rewritten_text)

    if not original_numbers and not rewritten_numbers:
        return 0.0

    penalty = 0.0
    if len(original_numbers) != len(rewritten_numbers):
        penalty += -0.125 * abs(len(original_numbers) - len(rewritten_numbers))
        return 1.0 + penalty

    for o_num, r_num in zip(original_numbers, rewritten_numbers):
        if isinstance(o_num, int) and isinstance(r_num, float) and r_num.is_integer():
            r_num = int(r_num)
        elif isinstance(o_num, float) and isinstance(r_num, int):
            o_num = int(o_num)
        try:
            if not np.isclose(o_num, r_num, atol=1e-3):
                penalty += -0.125
        except TypeError:
            penalty += -0.125
    return round(1.0 + penalty, 6)

def llm_scoring(client: LLMClient, original_text, rewritten_text, ai_role):
    """Scores the rewrite using an LLM call."""
    eval_prompt = (f"Below are two versions of the same text. The second version is "
                   f"a rewrite of the first version, but with a specific personality infused into it. Please"
                   f" look carefully at both versions and compare the rewrite to both the original "
                   f"and the provided persona to determine how good the rewrite is from a scale of "
                   f"0.0000 to 1.0000. You shall base your score on the following: Is the provided "
                   f"persona integrated into the rewritten response? Does the response contain comments "
                   f"or notes that are not part of responding to the original query? Is the quality of the rewrite "
                   f"on par with or better than the original? Did the rewrite include corrections to the original? "
                   f"Does the rewrite have mistakes that were not present in the original? Is the "
                   f"rewrite complete without mentioning that it is a rewrite of the original? Take all these factors into "
                   f"consideration and respond with a score between 0.0000 and 1.0000.\n\n"
                   f"Original: {original_text}\n\nRewritten: {rewritten_text}\n\nPersona: {ai_role}"
                   f"\n\nYou MUST include a score as indicated, and your final rating MUST be between 0.0000 and 1.0000,"
                   f" or I will not be able to process it and it will be deleted.")
    
    response = client.call(prompt=eval_prompt, temperature=0.5)
    
    try:
        rating = float(response.strip())
        return min(rating, 1.0)
    except (ValueError, AttributeError):
        float_pattern = r'[-+]?\d*\.?\d+'
        matches = re.findall(float_pattern, str(response))
        valid_matches = [float(m) for m in matches if float(m) <= 1.0]
        if valid_matches:
            rating = valid_matches[0]
            print(f"LLM response contained extra text. Extracted float: {rating}")
            return rating
        warnings.warn(f"No valid float found in LLM response. Returning 0.")
        return 0.0

def flag_for_human_review(score1, score2):
    """Flags entries for human review based on score outliers or equal scores."""
    if score1 > 1.01 or score2 > 1.01:
        return True
    if score1 < -0.01 or score2 < -0.01:
        return True
    if abs(score1 - score2) < 0.001:
        return True
    return False

def calculate_overall_score(client: LLMClient, original_text, rewritten_text, ai_role):
    """Calculates the overall score based on all components."""
    cosine_similarity_score = calculate_cosine_similarity(original_text, rewritten_text)
    latin_character_score = check_for_non_latin(rewritten_text)
    original_length = len(original_text.split())
    rewritten_length = len(rewritten_text.split())
    length_score = calculate_length_difference_score(original_length, rewritten_length)
    repetition_score = calculate_repetition_score(rewritten_text)
    persona_score = calculate_persona_score(rewritten_text, original_text)
    number_score = calculate_number_consistency_score(original_text, rewritten_text)
    bad_words_score = calc_bad_words_score(rewritten_text, ai_role)
    llm_score_val = llm_scoring(client, original_text, rewritten_text, ai_role)

    weights = {
        "cosine": 1.0, "latin": 1.0, "length": 0.45, "repetition": 1.55,
        "number": 0.55, "bad_words": 1.35, "persona": 1.25, "llm": 2.15
    }

    overall_score = round(
        (cosine_similarity_score * weights["cosine"]) +
        (latin_character_score * weights["latin"]) +
        (length_score * weights["length"]) +
        (repetition_score * weights["repetition"]) +
        (number_score * weights["number"]) +
        (bad_words_score * weights["bad_words"]) +
        (persona_score * weights["persona"]) +
        (llm_score_val * weights["llm"]),
        6)
    return overall_score