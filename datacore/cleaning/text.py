# datacore/cleaning/text.py

import re
from typing import Tuple, List, Optional


def trim_after_eos(text: str, eos_token: str = "</s>") -> str:
    """
    Trim everything after an end-of-sequence token.
    
    Args:
        text: Input text
        eos_token: End of sequence token
        
    Returns:
        Trimmed text
    """
    if eos_token in text:
        return text[:text.index(eos_token)]
    return text


def remove_bad_starts(text: str, bad_starts: Optional[List[str]] = None) -> str:
    """
    Remove common unwanted prefixes from text.
    
    Args:
        text: Input text
        bad_starts: List of bad prefixes (uses defaults if None)
        
    Returns:
        Cleaned text
    """
    if bad_starts is None:
        bad_starts = [
            "combined answer:", "answer:", "answer 3:", "your answer:", "your reply:",
            "unaltered text:", "altered text:", "answer 1 and 2 combined:", ": ", 
            "edited answer:", "the edited answer:", "combining both answers,", 
            "combined and edited:", "combining both responses:", "here's how",
            "okay,", "sure thing", "certainly,", "absolutely,", "alright,", "hey,",
            "absolutely!", "certainly!", "sure!", "revised answer:", "revision:", 
            "revised text:", "edited text:", "here's the revised text:",
            "here is the revised text:", "\n---\n", "\n", "===", "- reply:",
            "merging both responses:", "here's the revised text based on the given rules:"
        ]
    
    text = text.strip()
    
    # Keep removing bad starts until none found
    changed = True
    while changed:
        changed = False
        for bad_start in bad_starts:
            if text.lower().startswith(bad_start.lower()):
                text = text[len(bad_start):].strip()
                changed = True
                break
    
    return text


def remove_trailing_junk(text: str) -> str:
    """
    Remove trailing unwanted characters.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove trailing single characters
    while text and text[-1] in ['\n', '-', '|']:
        text = text[:-1]
    
    # Remove trailing HTML tags
    while text.endswith('<hr>') or text.endswith('<br>'):
        text = text[:-4]
    
    # Remove trailing special characters (but preserve valid endings)
    valid_endings = ('.', '!', '?', '```', '>')
    if text and not text.endswith(valid_endings):
        text = re.sub(r'[\s.,;:!?#&%@]+$', '', text)
    
    return text


def remove_xml_tags(text: str, preserve_code_blocks: bool = True) -> str:
    """
    Remove XML-style tags from text.
    
    Args:
        text: Input text
        preserve_code_blocks: If True, don't remove tags inside markdown code blocks
        
    Returns:
        Text with XML tags removed
    """
    if not preserve_code_blocks:
        return re.sub(r'<[^>]+>', '', text)
    
    # Identify and temporarily replace markdown code blocks
    code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
    placeholder = '\0'
    
    for i, block in enumerate(code_blocks):
        text = text.replace(block, f"{placeholder}{i}{placeholder}")
    
    # Remove XML tags from remaining text
    text = re.sub(r'<[^>]+>', '', text)
    
    # Restore code blocks
    for i, block in enumerate(code_blocks):
        text = text.replace(f"{placeholder}{i}{placeholder}", block)
    
    return text


def clean_answer(text: str, remove_eos: bool = True) -> str:
    """
    Apply all common cleaning operations to answer text.
    
    Args:
        text: Input text
        remove_eos: Whether to trim after </s> token
        
    Returns:
        Cleaned text
    """
    if remove_eos:
        text = trim_after_eos(text)
    
    text = remove_bad_starts(text)
    text = remove_trailing_junk(text)
    text = remove_xml_tags(text)
    
    return text.strip()


def strip_quotation_marks(text: str) -> str:
    """
    Remove quotation marks from start and end of string.
    
    Args:
        text: Input text
        
    Returns:
        Text without surrounding quotes
    """
    text = text.strip()
    if len(text) >= 2:
        if (text[0] in ['"', "'"] and text[-1] in ['"', "'"]):
            return text[1:-1]
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()


def strip_non_numeric(text: str) -> str:
    """
    Remove all non-numeric characters.
    
    Args:
        text: Input text
        
    Returns:
        Only numeric characters and decimal points
    """
    return re.sub(r'[^0-9.]', '', text)