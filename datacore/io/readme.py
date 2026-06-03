# datacore/io/readme.py
"""
Unified README / dataset-card generation for LM Data Tools output.

All tools' output archives use the same Hugging Face-compatible dataset card
produced by `generate_standard_readme()`.  An optional LLM-written
single-paragraph summary, grounded in real samples, can be built with
`generate_dataset_summary()`.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Optional


# ── Tool metadata ────────────────────────────────────────────────────────────

TOOL_DISPLAY_NAMES: dict[str, str] = {
    "datawriter":  "DataWriter",
    "databird":    "DataBird",
    "datapersona": "DataPersona",
    "dataconvo":   "DataConvo",
    "datathink":   "DataThink",
    "dataqa":      "DataQA",
    "datamix":     "DataMix",
    "reformat":    "Reformat",
    "datadoubler": "DataDoubler",
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "datawriter":  "Generates long-form documents from topics and tiers.",
    "databird":    "Generates high-quality Q&A pairs from topic lists with quality scoring.",
    "datapersona": "Re-styles existing datasets using AI personas.",
    "dataconvo":   "Expands short conversations into multi-turn dialogues.",
    "datathink":   "Enhances datasets with chain-of-thought reasoning steps.",
    "dataqa":      "Generates Q&A datasets from web content or local files.",
    "datamix":     "Samples and combines datasets from Hugging Face.",
    "reformat":    "Converts datasets between Alpaca, ShareGPT, and Q&A formats.",
    "datadoubler": "Expands datasets by generating rephrased question variants with fresh answers.",
}

TOOL_TASK_CATEGORIES: dict[str, list[str]] = {
    "databird":    ["question-answering", "text-generation"],
    "dataqa":      ["question-answering"],
    "datapersona": ["conversational", "text-generation"],
    "dataconvo":   ["conversational"],
    "datawriter":  ["text-generation"],
    "datathink":   ["text-generation"],
    "datamix":     ["text-generation"],
    "reformat":    ["text-generation"],
    "datadoubler": ["question-answering", "text-generation"],
}

TOOL_EXTRA_TAGS: dict[str, list[str]] = {
    "databird":    ["instruction-tuning", "qa-pairs"],
    "dataqa":      ["instruction-tuning", "qa-pairs", "retrieval-augmented"],
    "datapersona": ["instruction-tuning", "persona"],
    "dataconvo":   ["instruction-tuning", "multi-turn"],
    "datawriter":  ["long-form", "documents"],
    "datathink":   ["chain-of-thought", "reasoning"],
    "datamix":     ["mixed"],
    "reformat":    [],
    "datadoubler": ["augmented", "question-variants"],
}

TOOL_FIELD_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "databird": {
        "question":   "A generated user question about the topic",
        "asker":      "The perspective or persona of the person asking",
        "topic":      "The subject area the question is about",
        "evaluation": "Automated quality score (0–1)",
        "answer":     "Model-generated answer to the question",
    },
    "dataqa": {
        "question":   "A generated user question",
        "answer":     "Answer derived from source documents",
        "source":     "URL or filename the answer was sourced from",
        "confidence": "Confidence score for the answer quality (0–1)",
    },
    "datapersona": {
        "question": "The original user question",
        "answer":   "Response re-styled through the selected AI persona",
        "persona":  "The persona name applied to the response",
    },
    "dataconvo": {
        "conversation": "Multi-turn dialogue expanded from the source exchange",
    },
    "datawriter": {
        "title":   "Document title",
        "content": "Full generated document text",
        "topic":   "Topic the document covers",
        "tier":    "Writing complexity / target-audience tier",
    },
    "datathink": {
        "question": "The original question",
        "thinking": "Step-by-step chain-of-thought reasoning",
        "answer":   "Final answer produced after reasoning",
    },
    "datadoubler": {
        "question": "Original or LLM-rephrased question variant",
        "answer":   "Answer (original, or freshly generated for a variant)",
        "_source":  "Provenance: 'original' or 'variant_run_N'",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _size_category(n: int) -> str:
    if n < 1_000:     return "n<1K"
    if n < 10_000:    return "1K<n<10K"
    if n < 100_000:   return "10K<n<100K"
    if n < 1_000_000: return "100K<n<1M"
    return "1M<n<10M"


def _select_sample_entries(data: list, budget_chars: int = 1500) -> list:
    """Return 2–4 random entries whose combined JSON length fits within budget_chars.

    Long string values are truncated so a single verbose entry doesn't dominate
    the prompt.  Always returns at least 2 entries (or all of them if fewer exist).
    """
    if not data:
        return []
    pool = random.sample(data, min(4, len(data)))
    selected = []
    used = 0
    for entry in pool:
        trimmed = {
            k: (v[:400] + "…") if isinstance(v, str) and len(v) > 400 else v
            for k, v in entry.items()
        }
        snippet = json.dumps(trimmed, ensure_ascii=False)
        if used + len(snippet) > budget_chars and len(selected) >= 2:
            break
        selected.append(trimmed)
        used += len(snippet)
    return selected


# ── Public API ───────────────────────────────────────────────────────────────

def generate_dataset_summary(
    data_file_contents: dict,
    tool_name: str,
    dataset_name: str,
    llm_settings: Optional[dict] = None,
) -> str:
    """Produce a single-paragraph human-readable dataset description via the LLM.

    Uses 2–4 random samples so the paragraph is grounded in the actual data.
    Returns an empty string on any error so README generation always succeeds
    even if the LLM is unavailable or unconfigured.
    """
    llm_settings = llm_settings or {}
    try:
        all_data = next(
            (data for data in data_file_contents.values() if isinstance(data, list) and data),
            None,
        )
        if not all_data:
            return ""

        samples = _select_sample_entries(all_data)
        if not samples:
            return ""

        samples_text = json.dumps(samples, indent=2, ensure_ascii=False)
        display = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)

        system_prompt = (
            "You are a concise technical writer. "
            "Output ONLY the paragraph you are asked to write — no analysis, no numbered "
            "steps, no preamble, no self-reflection, no headings. "
            "Begin writing the paragraph immediately with its first word."
        )
        prompt = (
            f"Write a single paragraph (3–5 sentences) for a machine learning dataset card.\n\n"
            f"Dataset name: {dataset_name}\n"
            f"Generated by: {display}\n\n"
            f"Sample entries:\n{samples_text}\n\n"
            f"Requirements:\n"
            f"- Describe what the dataset contains, its topics/domains, and who it is most useful for\n"
            f"- Be specific and concrete — mention actual subjects visible in the samples\n"
            f"- Do not start with 'This dataset'"
        )

        from datacore.llm.client import LLMClient

        base_url = llm_settings.get("base_url") or None
        api_key = llm_settings.get("api_key") or ""
        # The OpenAI SDK refuses to initialize without *some* api_key, but many
        # OpenAI-compatible servers (LM Studio, Ollama, vLLM, local or LAN) don't
        # require one.  Supply a harmless stub whenever the user hasn't set a key;
        # if the endpoint actually needs auth, the call itself will fail cleanly.
        # The base_url from Settings is always passed through untouched.
        if not api_key:
            api_key = "lm-studio"

        client = LLMClient(
            base_url=base_url,
            api_key=api_key,
            default_model=llm_settings.get("llm_model"),
        )
        summary = client.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=300,
            # Suppresses visible chain-of-thought on Qwen3 / DeepSeek-R1 via vLLM.
            # Providers that don't recognise this field silently ignore it.
            extra_body={"enable_thinking": False},
        )
        return summary.strip()

    except Exception as exc:
        print(f"[readme] Could not generate dataset summary: {exc}")
        return ""


def generate_standard_readme(
    tool_name: str,
    dataset_name: str,
    data_file_contents: dict,
    metadata: dict,
    summary: str = "",
) -> str:
    """Generate a Hugging Face-compatible dataset card (README.md) for any tool's output.

    data_file_contents: dict mapping filename -> parsed JSON data (or None on error)
    """
    display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    description  = TOOL_DESCRIPTIONS.get(tool_name, "")
    task_cats    = TOOL_TASK_CATEGORIES.get(tool_name, ["text-generation"])
    extra_tags   = TOOL_EXTRA_TAGS.get(tool_name, [])
    field_descs  = TOOL_FIELD_DESCRIPTIONS.get(tool_name, {})

    created_at = metadata.get("created_at", "")
    if created_at:
        try:
            created_at = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass

    # Introspect entry keys, count, and filename from the first data file
    entry_keys    = []
    entry_count   = 0
    data_filename = ""
    for filename, data in data_file_contents.items():
        if isinstance(data, list) and data:
            entry_keys    = list(data[0].keys())
            entry_count   = len(data)
            data_filename = filename
            break

    all_tags = ["synthetic", "llm-generated", "lmdatatools"] + extra_tags

    # ── YAML front matter (Hugging Face dataset card metadata) ────────────────
    yaml_lines = ["---"]
    yaml_lines.append("language:")
    yaml_lines.append("- en")
    yaml_lines.append("task_categories:")
    for cat in task_cats:
        yaml_lines.append(f"- {cat}")
    yaml_lines.append("tags:")
    for tag in all_tags:
        yaml_lines.append(f"- {tag}")
    yaml_lines.append(f"pretty_name: {dataset_name}")
    yaml_lines.append("size_categories:")
    yaml_lines.append(f"- {_size_category(entry_count)}")
    yaml_lines.append("---")
    yaml_lines.append("")

    # ── Body ──────────────────────────────────────────────────────────────────
    body = []
    body += [f"# {dataset_name}", ""]
    body += [
        f"> Generated with [LMDataTools](https://github.com/theprint/LMDataTools)"
        f" using **{display_name}**.",
        "",
    ]
    if description:
        body += [description, ""]

    if summary:
        body += [summary, ""]

    # Dataset Details table
    body += [
        "## Dataset Details",
        "",
        "| | |",
        "|---|---|",
        f"| **Entries** | {entry_count:,} |",
        f"| **Created** | {created_at} |",
        f"| **Format**  | JSON |",
        f"| **Tool**    | {display_name} |",
        "",
    ]

    # Dataset Structure
    if entry_keys:
        body += ["## Dataset Structure", ""]
        body.append("Each entry contains the following fields:")
        body.append("")
        body.append("| Field | Description |")
        body.append("|-------|-------------|")
        for key in entry_keys:
            desc = field_descs.get(key, "")
            body.append(f"| `{key}` | {desc} |")
        body.append("")

    # Configuration
    config = metadata.get("config", {})
    if config:
        skip_keys = {"job_id", "output_path", "llm_settings", "uploaded_filenames"}
        # Don't show persona_name when persona is disabled — it's misleading to list
        # a persona that was not actually applied to the generated data.
        if not config.get("use_persona"):
            skip_keys = skip_keys | {"persona_name"}
        visible = {k: v for k, v in config.items()
                   if k not in skip_keys and v not in (None, "", [], {})}
        if visible:
            body += ["## Configuration", "", "| Setting | Value |", "|---------|-------|"]
            for k, v in visible.items():
                body.append(f"| `{k}` | `{v}` |")
            body.append("")

    # Usage snippet
    if data_filename:
        body += [
            "## Usage",
            "",
            "```python",
            "import json",
            "",
            f'with open("{data_filename}") as f:',
            "    data = json.load(f)",
            "",
            'print(f"Loaded {len(data)} entries")',
            "print(data[0])",
            "```",
            "",
        ]

    body += [
        "---",
        "_Created with [LMDataTools](https://github.com/theprint/LMDataTools)_",
    ]

    return "\n".join(yaml_lines + body)
