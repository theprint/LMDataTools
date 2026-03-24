# datacore/io/readme.py
"""
README / dataset-card generator for LM Data Tools output.

Produces a standardised markdown dataset card that can be uploaded alongside
the dataset to Hugging Face or included in a ZIP archive.

Usage
-----
    from datacore.io.readme import generate_readme

    card = generate_readme(
        tool="DataBird",
        dataset_name="cooking-qa-1k",
        entry_count=1200,
        output_format="alpaca",
        model="Mistral-7B-Instruct",
        notes="Generated from sourdough and home-baking topics.",
    )
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(card)
"""

from datetime import datetime

# ── Per-format field descriptions ────────────────────────────────────────────

_FORMAT_DESCRIPTIONS: dict[str, str] = {
    "alpaca": (
        "Each entry is a JSON object with three fields:\n"
        "- `instruction` — the question or task prompt\n"
        "- `input` — optional additional context (may be empty)\n"
        "- `output` — the model response"
    ),
    "sharegpt": (
        "Each entry has a `conversations` list of turn objects:\n"
        "- `from` — `\"user\"` or `\"assistant\"` (or `\"system\"`)\n"
        "- `value` — the message content"
    ),
    "qa": (
        "Each entry is a JSON object with:\n"
        "- `question` — the question\n"
        "- `answer` — the answer (may contain `<think>…</think>` reasoning blocks)"
    ),
}

_FORMAT_DESCRIPTIONS_DEFAULT = "Custom or unknown format — see the data files for the schema."

# ── Per-tool default descriptions ────────────────────────────────────────────

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "DataBird": (
        "Procedural Q&A pairs generated from a configurable set of topics and "
        "perspectives using an LLM. Questions are scored and filtered for quality "
        "before answers are generated."
    ),
    "DataPersona": (
        "Dataset responses rewritten through a specific persona or personality "
        "using an LLM. Two candidate rewrites are generated per entry and scored; "
        "the better one is selected automatically."
    ),
    "DataThink": (
        "Dataset enhanced with `<think>…</think>` chain-of-thought reasoning blocks "
        "prepended to each answer, making it suitable for reasoning-model fine-tuning."
    ),
    "DataConvo": (
        "Single-turn Q&A pairs expanded into multi-turn conversations using an LLM, "
        "with configurable round-weighting."
    ),
    "DataWriter": (
        "Long-form documents generated from topic prompts using an LLM."
    ),
    "DataQA": (
        "Q&A pairs extracted from web pages or uploaded documents using an LLM "
        "to generate questions from the source content."
    ),
    "DataMix": (
        "Samples drawn from multiple HuggingFace datasets, auto-detected and "
        "normalised to a unified Alpaca format with configurable per-source weights."
    ),
    "Reformat": (
        "Dataset converted from one format to another (e.g. ShareGPT → Alpaca)."
    ),
}

# ── Card template ─────────────────────────────────────────────────────────────

_TEMPLATE = """\
# {dataset_name}

## Dataset Card

| Field | Value |
|---|---|
| **Tool** | {tool} |
| **Format** | `{output_format}` |
| **Entries** | {entry_count:,} |
| **Model** | {model} |
| **Created** | {created} |

## Description

{description}

## Format

{format_description}
{notes_section}"""


# ── Public API ────────────────────────────────────────────────────────────────

def generate_readme(
    tool: str,
    dataset_name: str,
    entry_count: int,
    output_format: str = "alpaca",
    model: str = "unknown",
    description: str = "",
    notes: str = "",
    created: str = None,
) -> str:
    """
    Generate a markdown dataset card.

    Parameters
    ----------
    tool:
        Name of the LM Data Tool that created the dataset (e.g. ``"DataBird"``).
    dataset_name:
        Human-readable name of the dataset.
    entry_count:
        Number of entries in the final dataset.
    output_format:
        Output schema — ``"alpaca"``, ``"sharegpt"``, or ``"qa"``.
    model:
        LLM model name used during generation.
    description:
        Custom description paragraph.  Falls back to a tool-specific default.
    notes:
        Extra markdown text appended at the end of the card (optional).
    created:
        ISO date string (``YYYY-MM-DD``).  Defaults to today's date.

    Returns
    -------
    str
        Markdown dataset card, ready to write to ``README.md``.
    """
    if not created:
        created = datetime.now().strftime("%Y-%m-%d")

    desc = description or _TOOL_DESCRIPTIONS.get(tool, f"Dataset created with {tool}.")
    fmt_desc = _FORMAT_DESCRIPTIONS.get(output_format, _FORMAT_DESCRIPTIONS_DEFAULT)
    notes_section = f"\n## Notes\n\n{notes}" if notes else ""

    return _TEMPLATE.format(
        dataset_name=dataset_name,
        tool=tool,
        output_format=output_format,
        entry_count=entry_count,
        model=model,
        created=created,
        description=desc,
        format_description=fmt_desc,
        notes_section=notes_section,
    ).rstrip() + "\n"
