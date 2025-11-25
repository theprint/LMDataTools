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
        alpaca_data.append(alpaca_entry)
    return alpaca_data


def to_sharegpt(data: List[Dict[str, Any]],
                instruction_key: str = "question",
                output_key: str = "answer",
                system_prompt: str = None) -> List[Dict[str, Any]]:
    """
    Convert data to ShareGPT format.
    
    Args:
        data: Source data
        instruction_key: Key containing the user message
        output_key: Key containing the assistant response
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
        
        conversation["conversations"].append({
            "from": "user",
            "value": entry.get(instruction_key, "")
        })
        
        conversation["conversations"].append({
            "from": "assistant",
            "value": entry.get(output_key, "")
        })
        
        # Preserve any metadata
        if "response_model" in entry:
            conversation["response_model"] = entry["response_model"]
        
        sharegpt_data.append(conversation)
    
    return sharegpt_data


def from_alpaca(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert from Alpaca format to standard Q&A format.
    
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