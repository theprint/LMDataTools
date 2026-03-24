# datacore/io/json_ops.py

import json
import os
from typing import Any, List, Dict, Optional, Literal
from datacore.config.settings import config
from pathlib import Path


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 4) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save to
        indent: JSON indentation level
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_or_create(
    input_path: str, 
    output_path: str,
    create_from_input: bool = True
) -> tuple[Any, int]:
    """
    Load existing output file or create from input file.
    Useful for resumable operations.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        create_from_input: If output doesn't exist, copy from input
        
    Returns:
        Tuple of (data, start_index) where start_index indicates where to resume
    """
    if os.path.exists(output_path):
        print(f"Found existing output file: {output_path}")
        try:
            data = load_json(output_path)
            print(f"Loaded {len(data)} entries from existing output file")
            return data, len(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading output file: {e}")
            print("Starting fresh from input file...")
    
    if create_from_input and os.path.exists(input_path):
        print(f"Loading from input file: {input_path}")
        data = load_json(input_path)
        print(f"Loaded {len(data)} entries from input file")
        return data, 0
    
    # Return empty list if nothing exists
    return [], 0


def ensure_directory(file_path: str) -> None:
    """Ensure the directory for a file path exists."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def find_resume_point(
    data: List[Dict],
    required_keys: List[str],
    check_function: callable = None
) -> int:
    """
    Find the index where processing should resume.
    
    Args:
        data: List of entries to check
        required_keys: Keys that must be present for entry to be complete
        check_function: Optional custom function to determine if entry needs processing
                       Should take an entry and return True if processing needed
        
    Returns:
        Index of first entry that needs processing, or len(data) if all complete
    """
    for i, entry in enumerate(data):
        if check_function:
            if check_function(entry):
                return i
        else:
            # Check if any required key is missing
            if any(key not in entry for key in required_keys):
                return i
    
    return len(data)


def needs_processing(entry: Dict, config: Dict) -> bool:
    """
    Generic check if an entry needs processing based on config flags.
    Useful for joe.py style multi-field processing.
    
    Args:
        entry: Entry to check
        config: Dict with keys like {"do_field_1": True, "do_field_2": False}
                and corresponding field names
        
    Returns:
        True if entry needs any processing
    """
    for do_flag, field_name in config.items():
        if do_flag and field_name not in entry:
            return True
    return False


def save_checkpoint(
    data: List[Dict],
    output_path: str,
    index: int,
    total: int,
    metadata: Dict = None
) -> None:
    """
    Save a checkpoint with optional metadata.
    
    Args:
        data: Data to save
        output_path: Where to save
        index: Current index being processed
        total: Total items
        metadata: Optional metadata to include
    """
    save_json(data, output_path)
    
    print(f"Checkpoint saved: {index + 1}/{total} entries processed")
    
    if metadata:
        metadata_path = output_path.replace('.json', '_metadata.json')
        save_json(metadata, metadata_path)

SUPPORTED_FORMATS = Literal["alpaca", "sharegpt", "qa"]

class DatasetFormatter:
    """
    A tool to reformat JSON datasets between supported formats.
    
    Supported formats:
    - 'alpaca': [{"instruction": str, "input": str, "output": str}]
    - 'sharegpt': [{"conversations": [{"from": "human"|"gpt", "value": str}]}]
    - 'qa': [{"question": str, "answer": str}]
    """

    def __init__(self, from_format: SUPPORTED_FORMATS, to_format: SUPPORTED_FORMATS):
        """
        Initializes the formatter.
        
        Args:
            from_format: The source format of the dataset.
            to_format: The target format for the dataset.
        """
        if from_format == to_format:
            raise ValueError("Source and target formats cannot be the same.")
            
        self.from_format = from_format
        self.to_format = to_format

        # Get the appropriate conversion function
        converter_name = f"_from_{from_format}_to_{to_format}"
        if not hasattr(self, converter_name):
            raise NotImplementedError(f"Conversion from '{from_format}' to '{to_format}' is not supported.")
        self.converter = getattr(self, converter_name)

    def needs_reformatting(self, entry: Dict) -> bool:
        """
        Checks if an entry has been reformatted by looking for a marker.
        An entry needs processing if the marker is not present.
        """
        return not entry.get("_reformatted", False)

    def reformat_entry(self, entry: Dict) -> Dict:
        """
        Reformats a single data entry from the source to the target format.
        """
        if self.needs_reformatting(entry):
            new_entry = self.converter(entry)
            new_entry["_reformatted"] = True
            return new_entry
        return entry

    def _from_alpaca_to_qa(self, entry: Dict) -> Dict:
        """Converts a single entry from Alpaca to QA format."""
        instruction = entry.get("instruction", "")
        inp = entry.get("input", "")
        question = f"{instruction}\n{inp}".strip() if inp else instruction
        return {"question": question, "answer": entry.get("output", "")}

    def _from_qa_to_alpaca(self, entry: Dict) -> Dict:
        """Converts a single entry from QA to Alpaca format."""
        return {
            "instruction": entry.get("question", ""),
            "input": "",
            "output": entry.get("answer", "")
        }

    def _from_sharegpt_to_qa(self, entry: Dict) -> Dict:
        """Converts a single ShareGPT conversation to a single QA pair."""
        question = ""
        answer = ""
        conversations = entry.get("conversations", [])
        if conversations:
            # Use the first human turn as the question
            first_human_turn = next((turn for turn in conversations if turn.get("from") == "human"), None)
            if first_human_turn:
                question = first_human_turn.get("value", "")
            
            # Concatenate all assistant turns as the answer
            answer = "\n".join(
                turn.get("value", "") for turn in conversations if turn.get("from") == "gpt"
            ).strip()
        return {"question": question, "answer": answer}

    def _from_qa_to_sharegpt(self, entry: Dict) -> Dict:
        """Converts a single QA pair to a ShareGPT conversation."""
        return {
            "conversations": [
                {"from": "human", "value": entry.get("question", "")},
                {"from": "gpt", "value": entry.get("answer", "")}
            ]
        }

    def _from_alpaca_to_sharegpt(self, entry: Dict) -> Dict:
        """Converts a single Alpaca entry to a ShareGPT conversation."""
        qa_entry = self._from_alpaca_to_qa(entry)
        return self._from_qa_to_sharegpt(qa_entry)

    def _from_sharegpt_to_alpaca(self, entry: Dict) -> Dict:
        """Converts a single ShareGPT conversation to an Alpaca entry."""
        qa_entry = self._from_sharegpt_to_qa(entry)
        # This conversion is lossy as ShareGPT can have multiple turns
        # We simplify to the first question and combined answers.
        return self._from_qa_to_alpaca(qa_entry)


class ResumableProcessor:
    """
    Context manager for resumable batch processing.
    Handles loading, checkpointing, and final save automatically.
    """
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        save_interval: int = config.DEFAULT_SAVE_INTERVAL,
        check_function: callable = None,
        required_keys: List[str] = None
    ):
        """
        Initialize resumable processor.
        
        Args:
            input_path: Path to input data
            output_path: Path to output data
            save_interval: Save checkpoint every N entries (-1 to disable)
            check_function: Function to check if entry needs processing
            required_keys: Keys that must be present for entry to be complete
        """
        self.input_path = input_path
        self.output_path = output_path
        self.save_interval = save_interval
        self.check_function = check_function
        self.required_keys = required_keys or []
        
        self.data = None
        self.start_index = 0
        self.processed_count = 0
    
    def __enter__(self):
        """Load data and find resume point."""
        # Load existing output or create from input
        self.data, _ = load_or_create(self.input_path, self.output_path)
        
        # Find exact resume point
        self.start_index = find_resume_point(
            self.data,
            self.required_keys,
            self.check_function
        )
        
        if self.start_index >= len(self.data):
            print(f"All {len(self.data)} entries already complete!")
        else:
            print(f"Resuming from entry {self.start_index + 1} of {len(self.data)}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save final data on exit."""
        if exc_type is not None:
            # Exception occurred - save what we have
            print(f"Exception occurred, saving progress...")
            save_json(self.data, self.output_path)
            print(f"Saved {len(self.data)} entries before crash")
            return False  # Re-raise exception
        
        # Normal completion - final save
        save_json(self.data, self.output_path)
        print(f"Processing complete. Saved {len(self.data)} entries.")
        return True
    
    def checkpoint(self, current_index: int):
        """
        Save checkpoint if interval reached.
        
        Args:
            current_index: Current processing index
        """
        self.processed_count += 1
        
        if self.save_interval > 0 and self.processed_count % self.save_interval == 0:
            save_json(self.data, self.output_path)
            print(f"Checkpoint: {current_index + 1}/{len(self.data)} entries processed")
    
    def should_process(self, index: int) -> bool:
        """
        Check if an entry at index should be processed.
        
        Args:
            index: Index to check
            
        Returns:
            True if entry needs processing
        """
        if index < self.start_index:
            return False
        
        if index >= len(self.data):
            return False
        
        entry = self.data[index]
        
        if self.check_function:
            return self.check_function(entry)
        
        # Check required keys
        return any(key not in entry for key in self.required_keys)