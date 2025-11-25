# datacore/cleaning/validation.py

import re
import emoji
from typing import Tuple


def is_complete_answer(text: str) -> bool:
    """
    Check if an answer appears complete (ends properly).
    
    Args:
        text: Answer text
        
    Returns:
        True if answer appears complete
    """
    valid_endings = ('.', '!', '?', '```', '>')
    return any(text.rstrip().endswith(ending) for ending in valid_endings)


def starts_with_capital(text: str, allow_formatting: bool = True) -> Tuple[bool, bool]:
    """
    Check if text starts with a capital letter.
    
    Args:
        text: Input text
        allow_formatting: Allow * and # as valid starts (for markdown)
        
    Returns:
        Tuple of (starts_with_letter, is_capital)
    """
    if not text:
        return False, False
    
    first_char = text[0]
    
    if first_char.isalpha():
        return True, first_char.isupper()
    
    if allow_formatting and first_char in ['*', '#']:
        return True, True
    
    return False, False


def contains_non_latin(text: str, max_count: int = 10) -> Tuple[bool, int]:
    """
    Check for non-Latin characters.
    
    Args:
        text: Input text
        max_count: Maximum non-Latin chars before returning early
        
    Returns:
        Tuple of (has_non_latin, count)
    """
    non_latin_count = 0
    
    for char in text:
        # Check if character is NOT in Latin ranges
        if not (
            (ord(char) <= 0x024F) or  # Latin ranges
            char.isspace() or
            char.isdigit() or
            (0x2000 <= ord(char) <= 0x218F)  # Common punctuation
        ):
            non_latin_count += 1
            if non_latin_count >= max_count:
                return True, non_latin_count
    
    return non_latin_count > 0, non_latin_count


def contains_cyrillic(text: str) -> bool:
    """
    Check if text contains Cyrillic characters.
    
    Args:
        text: Input text
        
    Returns:
        True if Cyrillic characters found
    """
    return bool(re.search('[а-яА-Я]', text))


def has_empty_code_blocks(text: str) -> bool:
    """
    Check for empty markdown code blocks.
    
    Args:
        text: Input text
        
    Returns:
        True if empty code blocks found
    """
    pattern = re.compile(r'```[\s\n]*```')
    return bool(pattern.search(text))


def is_emoji(character: str) -> bool:
    """
    Check if character is an emoji.
    
    Args:
        character: Single character
        
    Returns:
        True if emoji
    """
    return character in emoji.EMOJI_DATA


def has_quality_markers(text: str, require_count: int = 1) -> bool:
    """
    Check if text contains explanatory/quality markers.
    
    Args:
        text: Input text
        require_count: Minimum number of markers required
        
    Returns:
        True if enough quality markers found
    """
    markers = [
        ' because ', ' how ', ' why ', ' what ', ' which means ',
        ' for example ', ' such as ', ' this means '
    ]
    
    text_lower = text.lower()
    count = sum(1 for marker in markers if marker in text_lower)
    
    return count >= require_count


def has_bad_patterns(text: str) -> bool:
    """
    Check for low-quality patterns.
    
    Args:
        text: Input text
        
    Returns:
        True if bad patterns found
    """
    bad_patterns = [
        'click here', 'subscribe now', 'cookie policy', 'cookies policy',
        '© copyright', 'terms of service', 'privacy policy', 'all rights reserved',
        'sign up', 'log in', 'follow us', 'share this'
    ]
    
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in bad_patterns)