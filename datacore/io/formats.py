# datacore/io/formats.py

from typing import List, Dict, Any


def to_alpaca(data: List[Dict[str, Any]], 
              instruction_key: str = "question",
              output_key: str = "answer",
              input_key: str = None) -> List[Dict[str, str]]:
    """
    Convert data to Alpaca format.
    
    Args:
        data: Source data
        instruction_key: Key containing the instruction/question
        output_key: Key containing the output/answer
        input_key: Optional key containing additional input
        
    Returns:
        Data in Alpaca format
    """
    alpaca_data = []
    for entry in data:
        alpaca_entry = {
            "instruction": entry.get(instruction_key, ""),
            "input": entry.get(input_key, "") if input_key else "",
            "output": entry.get(output_key, "")
        }
        # Preserve provenance/metadata fields (underscore-prefixed)
        for k, v in entry.items():
            if k.startswith("_"):
                alpaca_entry[k] = v
        alpaca_data.append(alpaca_entry)
    return alpaca_data


def to_sharegpt(data: List[Dict[str, Any]],
                instruction_key: str = "question",
                output_key: str = "answer",
                input_key: str = None,
                system_prompt: str = None) -> List[Dict[str, Any]]:
    """
    Convert data to ShareGPT format.

    When *input_key* is provided and the entry contains a non-empty value for
    that key, it is appended to the instruction with a blank line separator so
    no context is lost in the conversion.

    Args:
        data: Source data
        instruction_key: Key containing the user message / instruction
        output_key: Key containing the assistant response
        input_key: Optional key containing supplementary context (e.g. Alpaca
                   "input").  Combined with the instruction into one user turn.
        system_prompt: Optional system prompt to include

    Returns:
        Data in ShareGPT format
    """
    sharegpt_data = []
    for entry in data:
        conversation = {"conversations": []}

        if system_prompt:
            conversation["conversations"].append({
                "from": "system",
                "value": system_prompt
            })

        instruction = entry.get(instruction_key, "")
        context     = entry.get(input_key, "") if input_key else ""
        user_value  = f"{instruction}\n\n{context}" if context else instruction

        conversation["conversations"].append({
            "from": "user",
            "value": user_value
        })
        
        conversation["conversations"].append({
            "from": "assistant",
            "value": entry.get(output_key, "")
        })
        
        # Preserve provenance/metadata fields (underscore-prefixed)
        for k, v in entry.items():
            if k.startswith("_"):
                conversation[k] = v

        sharegpt_data.append(conversation)
    
    return sharegpt_data


def from_alpaca(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert from Alpaca format to standard Q&A format.

    Utility function used by tools that need to re-ingest Alpaca-formatted
    data (e.g. datamix when mixing datasets of different formats).

    Args:
        data: Alpaca formatted data

    Returns:
        Standard Q&A format
    """
    qa_data = []
    for entry in data:
        qa_entry = {
            "question": entry["instruction"],
            "answer": entry["output"]
        }
        if entry.get("input"):
            qa_entry["input"] = entry["input"]
        qa_data.append(qa_entry)
    return qa_data


def apply_output_format(
    data: List[Dict[str, Any]],
    output_format: str,
    instruction_key: str = "instruction",
    output_key: str = "output",
    input_key: str = None,
) -> tuple:
    """
    Convert data to the requested output format.

    Returns (converted_data, format_suffix) where format_suffix is the
    lowercase format name suitable for use in filenames.
    Falls back to Alpaca for any unrecognised format value.
    """
    if output_format == "sharegpt":
        return (
            to_sharegpt(data, instruction_key=instruction_key, output_key=output_key, input_key=input_key),
            "sharegpt",
        )
    return (
        to_alpaca(data, instruction_key=instruction_key, output_key=output_key, input_key=input_key),
        "alpaca",
    )


def detect_format(data: List[Dict[str, Any]]) -> str:
    """
    Detect the format of a dataset.
    
    Returns:
        Format name: "alpaca", "sharegpt", "qa", or "unknown"
    """
    if not data:
        return "unknown"
    
    first_entry = data[0]
    
    # Check for Alpaca format
    if all(key in first_entry for key in ["instruction", "input", "output"]):
        return "alpaca"
    
    # Check for ShareGPT format
    if "conversations" in first_entry:
        return "sharegpt"
    
    # Check for Q&A format
    if "question" in first_entry and "answer" in first_entry:
        return "qa"
    
    return "unknown"