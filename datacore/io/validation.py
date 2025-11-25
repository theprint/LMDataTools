# datacore/io/validation.py

from typing import List, Dict, Any


def validate_entry(entry: Dict[str, Any], 
                   required_keys: List[str],
                   min_length: int = 0,
                   max_length: int = None) -> tuple[bool, str]:
    """
    Validate a data entry.
    
    Args:
        entry: Entry to validate
        required_keys: Keys that must be present
        min_length: Minimum text length for string values
        max_length: Maximum text length for string values
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required keys
    for key in required_keys:
        if key not in entry:
            return False, f"Missing required key: {key}"
        
        value = entry[key]
        
        # Check if value is string and not empty
        if not isinstance(value, str) or not value.strip():
            return False, f"Key '{key}' is empty or not a string"
        
        # Check length constraints
        if min_length and len(value) < min_length:
            return False, f"Key '{key}' is too short (min: {min_length})"
        
        if max_length and len(value) > max_length:
            return False, f"Key '{key}' is too long (max: {max_length})"
    
    return True, ""


def filter_valid_entries(data: List[Dict[str, Any]],
                        required_keys: List[str],
                        min_length: int = 0,
                        max_length: int = None,
                        verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Filter dataset to only valid entries.
    
    Args:
        data: Dataset to filter
        required_keys: Keys that must be present
        min_length: Minimum text length
        max_length: Maximum text length
        verbose: Print filtering stats
        
    Returns:
        Filtered dataset
    """
    valid_entries = []
    rejected = 0
    
    for entry in data:
        is_valid, error = validate_entry(entry, required_keys, min_length, max_length)
        if is_valid:
            valid_entries.append(entry)
        else:
            rejected += 1
            if verbose:
                print(f"Rejected entry: {error}")
    
    if verbose:
        print(f"Kept {len(valid_entries)} entries, rejected {rejected}")
    
    return valid_entries