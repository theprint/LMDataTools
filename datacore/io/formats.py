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
        # Fast path: full multi-turn conversation preserved from a conversation-type source.
        # Use it directly so system prompts and all turns are retained as-is.
        if "_conversations" in entry:
            conversation = {"conversations": list(entry["_conversations"])}
            for k, v in entry.items():
                if k.startswith("_") and k != "_conversations":
                    conversation[k] = v
            sharegpt_data.append(conversation)
            continue

        # Slow path: build a single-turn conversation from flat instruction/output fields.
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
            "from": "human",
            "value": user_value
        })

        conversation["conversations"].append({
            "from": "gpt",
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


def to_chatml(data: List[Dict[str, Any]],
              instruction_key: str = "question",
              output_key: str = "answer",
              input_key: str = None,
              system_prompt: str = None) -> List[Dict[str, Any]]:
    """
    Convert data to ChatML format (OpenAI messages schema).

    Each entry becomes {"messages": [{"role": "...", "content": "..."}]}.
    System messages are included when *system_prompt* is provided or when the
    source entry already carries a system turn via the ShareGPT fast-path.
    """
    chatml_data = []
    for entry in data:
        # Fast path: full multi-turn conversation from a conversation-type source.
        if "_conversations" in entry:
            messages = []
            for turn in entry["_conversations"]:
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                role = role_map.get(turn.get("from", ""), turn.get("from", ""))
                messages.append({"role": role, "content": turn.get("value", "")})
            record = {"messages": messages}
            for k, v in entry.items():
                if k.startswith("_") and k != "_conversations":
                    record[k] = v
            chatml_data.append(record)
            continue

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        instruction = entry.get(instruction_key, "")
        context     = entry.get(input_key, "") if input_key else ""
        user_content = f"{instruction}\n\n{context}" if context else instruction
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": entry.get(output_key, "")})

        record = {"messages": messages}
        for k, v in entry.items():
            if k.startswith("_"):
                record[k] = v
        chatml_data.append(record)

    return chatml_data


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
    if output_format == "chatml":
        return (
            to_chatml(data, instruction_key=instruction_key, output_key=output_key, input_key=input_key),
            "chatml",
        )
    if output_format == "qa":
        # Pass-through when source already uses question/answer keys; otherwise
        # remap from instruction/output to question/answer while preserving
        # underscore-prefixed metadata.
        qa_data = []
        for entry in data:
            qa_entry = {
                "question": entry.get(instruction_key, entry.get("question", "")),
                "answer": entry.get(output_key, entry.get("answer", "")),
            }
            ctx = entry.get(input_key, "") if input_key else ""
            if ctx:
                qa_entry["input"] = ctx
            for k, v in entry.items():
                if k.startswith("_"):
                    qa_entry[k] = v
            qa_data.append(qa_entry)
        return (qa_data, "qa")
    return (
        to_alpaca(data, instruction_key=instruction_key, output_key=output_key, input_key=input_key),
        "alpaca",
    )


def extract_first_turn(entry: Dict[str, Any]) -> tuple:
    """
    Extract the first user/assistant message pair from an entry in any supported format.

    Handles ShareGPT (conversations[].from/value), ChatML (messages[].role/content),
    Alpaca (instruction/output), and Q&A (question/answer).

    Returns:
        (user_message, assistant_message) — either may be None if not found.
    """
    if "conversations" in entry:
        turns = entry["conversations"]
        human = next((t for t in turns if t.get("from") in ("human", "user")), None)
        gpt   = next((t for t in turns if t.get("from") in ("gpt", "assistant")), None)
        return (human["value"] if human else None), (gpt["value"] if gpt else None)

    if "messages" in entry:
        msgs     = entry["messages"]
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
        return (user_msg["content"] if user_msg else None), (asst_msg["content"] if asst_msg else None)

    user      = entry.get("instruction") or entry.get("question")
    assistant = entry.get("output") or entry.get("answer")
    return user, assistant


def detect_format(data: List[Dict[str, Any]]) -> str:
    """
    Detect the format of a dataset.

    Returns:
        Format name: "alpaca", "sharegpt", "chatml", "qa", or "unknown"
    """
    if not data:
        return "unknown"

    first_entry = data[0]

    # Check for Alpaca format
    if all(key in first_entry for key in ["instruction", "input", "output"]):
        return "alpaca"

    # Check for ShareGPT format
    if "conversations" in first_entry and isinstance(first_entry["conversations"], list):
        return "sharegpt"

    # Check for ChatML format (OpenAI messages schema)
    if "messages" in first_entry and isinstance(first_entry["messages"], list):
        msgs = first_entry["messages"]
        if msgs and "role" in msgs[0] and "content" in msgs[0]:
            return "chatml"

    # Check for Q&A format
    if "question" in first_entry and "answer" in first_entry:
        return "qa"

    return "unknown"