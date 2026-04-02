# webapp.py
"""
Web interface for DataSynthesis Suite
"""
from datapersona import DEFAULT_CONFIG as DATAPERSONA_DEFAULTS
from openai import OpenAI # Import OpenAI client for model listing
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, File, UploadFile, Form
from starlette.websockets import WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import subprocess
import json
import os
import random
import shutil
import zipfile
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import asyncio
from pathlib import Path

import re
app = FastAPI(title="LM Data Tools")
app.mount("/static", StaticFiles(directory="webapp"), name="static")

@app.on_event("startup")
async def recover_stale_jobs():
    """Mark any jobs left in 'running' state as failed — they were interrupted by a server restart."""
    if not os.path.isdir(JOBS_DIR):
        return
    for job_id in os.listdir(JOBS_DIR):
        metadata_file = os.path.join(JOBS_DIR, job_id, "metadata.json")
        if not os.path.exists(metadata_file):
            continue
        try:
            with open(metadata_file) as f:
                meta = json.load(f)
            if meta.get("status") == "running":
                meta["status"] = "cancelled"
                meta["updated_at"] = datetime.now().isoformat()
                meta["error"] = "Job interrupted: server was restarted while this job was running."
                with open(metadata_file, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"[startup] Marked stale job {job_id} as failed (was 'running' at restart)")
        except Exception as e:
            print(f"[startup] Could not recover job {job_id}: {e}")

# Job tracking
active_jobs: Dict[str, dict] = {}
# Process handles for running jobs — kept separate because they are not JSON-serialisable.
# Keyed by job_id; cleaned up when the subprocess exits or is cancelled.
active_processes: Dict[str, asyncio.subprocess.Process] = {}
JOBS_DIR = "./jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# User settings storage
SETTINGS_FILE = "./user_settings.json"

# ============================================================================
# Data Models
# ============================================================================

class LLMSettings(BaseModel):
    llm_provider: str = "openai"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    llm_model: Optional[str] = None
    hugging_face_api_key: Optional[str] = None

class DataPersonaConfig(BaseModel):
    persona: str
    generate_reply_1: bool = True
    generate_reply_2: bool = True
    save_interval: int = 250
    dataset_name: str = "my-persona-dataset"
    llm_settings: Optional[LLMSettings] = None

class DataBirdConfig(BaseModel):
    dataset_name: str
    topics: List[str]
    full_auto: bool = True
    dataset_size: str = "small"
    clean_score: float = 0.76
    manual_perspectives: Optional[List] = None
    include_reasoning: bool = False
    llm_settings: Optional[LLMSettings] = None

class DataWriterConfig(BaseModel):
    document_count: int = 500
    min_tokens: int = 200
    max_tokens: int = 10000
    temperature: float = 0.8
    add_summary: bool = False
    dataset_name: str = "my-writer-dataset"
    llm_settings: Optional[LLMSettings] = None

class DataQAConfig(BaseModel):
    dataset_name: str
    sources: List[str] = []
    auto_perspectives: bool = True
    confidence_threshold: float = 0.68
    manual_perspectives: Optional[List] = None
    llm_settings: Optional[LLMSettings] = None

class DatasetSource(BaseModel):
    name: str
    weight: float
    subset: Optional[str] = None
    format: Optional[str] = None

class DataMixConfig(BaseModel):
    dataset_name: str
    total_samples: int = 10000
    seed: int = 310576
    dataset_sources: List[DatasetSource]
    min_instruction_length: int = 10
    max_instruction_length: int = 4000
    min_output_length: int = 10
    max_output_length: int = 4000
    llm_settings: Optional[LLMSettings] = None

# ============================================================================
# Utilities
# ============================================================================

def load_user_settings():
    """Load user settings from file."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_settings(tool_name: str, settings: dict):
    """Save user settings for a specific tool."""
    all_settings = load_user_settings()
    all_settings[tool_name] = settings
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(all_settings, f, indent=2)

def get_global_pref(key: str, default=None):
    """Read a single value from the stored global user preferences."""
    return load_user_settings().get("_global_prefs", {}).get(key, default)

def generate_job_id():
    """Generate unique job ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_job_workspace(job_id: str, tool_name: str, dataset_name: str, config: dict = None):
    """Create workspace for a job."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    metadata = {
        "job_id": job_id,
        "tool": tool_name,
        "dataset_name": dataset_name,
        "status": "starting",
        "created_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Save full config for recreation (excluding sensitive data and file contents)
    if config:
        # Deep copy via JSON round-trip so popping api_key below doesn't mutate
        # the caller's config dict (shallow copy shares the nested llm_settings object).
        config_for_metadata = json.loads(json.dumps(config, default=str))

        # Remove sensitive data
        if 'llm_settings' in config_for_metadata and config_for_metadata['llm_settings']:
            if isinstance(config_for_metadata['llm_settings'], str):
                config_for_metadata['llm_settings'] = json.loads(config_for_metadata['llm_settings'])
            config_for_metadata['llm_settings'].pop('api_key', None)
            config_for_metadata['llm_settings'].pop('hugging_face_api_key', None)
        
        # Extract filenames from file upload fields
        uploaded_filenames = []
        if 'files' in config:
            uploaded_filenames = config['files']
            config_for_metadata.pop('files', None)
        elif 'file' in config:
            uploaded_filenames = [config['file']]
            config_for_metadata.pop('file', None)
        
        if uploaded_filenames:
            config_for_metadata['uploaded_filenames'] = uploaded_filenames
        
        metadata['config'] = config_for_metadata
    
    with open(os.path.join(job_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return job_dir


TOOL_DISPLAY_NAMES = {
    "datawriter": "DataWriter",
    "databird": "DataBird",
    "datapersona": "DataPersona",
    "dataconvo": "DataConvo",
    "datathink": "DataThink",
    "dataqa": "DataQA",
    "datamix": "DataMix",
    "reformat": "Reformat",
}

TOOL_DESCRIPTIONS = {
    "datawriter":  "Generates long-form documents from topics and tiers.",
    "databird":    "Generates high-quality Q&A pairs from topic lists with quality scoring.",
    "datapersona": "Re-styles existing datasets using AI personas.",
    "dataconvo":   "Expands short conversations into multi-turn dialogues.",
    "datathink":   "Enhances datasets with chain-of-thought reasoning steps.",
    "dataqa":      "Generates Q&A datasets from web content or local files.",
    "datamix":     "Samples and combines datasets from Hugging Face.",
    "reformat":    "Converts datasets between Alpaca, ShareGPT, and Q&A formats.",
}

TOOL_TASK_CATEGORIES = {
    "databird":    ["question-answering", "text-generation"],
    "dataqa":      ["question-answering"],
    "datapersona": ["conversational", "text-generation"],
    "dataconvo":   ["conversational"],
    "datawriter":  ["text-generation"],
    "datathink":   ["text-generation"],
    "datamix":     ["text-generation"],
    "reformat":    ["text-generation"],
}

TOOL_EXTRA_TAGS = {
    "databird":    ["instruction-tuning", "qa-pairs"],
    "dataqa":      ["instruction-tuning", "qa-pairs", "retrieval-augmented"],
    "datapersona": ["instruction-tuning", "persona"],
    "dataconvo":   ["instruction-tuning", "multi-turn"],
    "datawriter":  ["long-form", "documents"],
    "datathink":   ["chain-of-thought", "reasoning"],
    "datamix":     ["mixed"],
    "reformat":    [],
}

# Known field descriptions per tool. Fields not listed fall back to an empty description.
TOOL_FIELD_DESCRIPTIONS = {
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
}


def _size_category(n: int) -> str:
    if n < 1_000:     return "n<1K"
    if n < 10_000:    return "1K<n<10K"
    if n < 100_000:   return "10K<n<100K"
    if n < 1_000_000: return "100K<n<1M"
    return "1M<n<10M"


def _select_sample_entries(data: list, budget_chars: int = 1500) -> list:
    """Return 2-4 random entries whose combined JSON length fits within budget_chars.

    Long string values are truncated so a single verbose entry doesn't dominate
    the prompt.  Always returns at least 2 entries (or all of them if fewer exist).
    """
    if not data:
        return []
    pool = random.sample(data, min(4, len(data)))
    selected = []
    used = 0
    for entry in pool:
        # Truncate long string values to keep the prompt readable
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


def _generate_dataset_summary(
    data_file_contents: dict,
    tool_name: str,
    dataset_name: str,
    llm_settings: dict,
) -> str:
    """Call the LLM to produce a single-paragraph human-readable dataset description.

    Uses 2-4 random samples so the paragraph is grounded in the actual data rather
    than just repeating the config settings.  Returns an empty string on any error
    so README generation always succeeds even if the LLM is unavailable.
    """
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
        client = LLMClient(
            base_url=llm_settings.get("base_url"),
            api_key=llm_settings.get("api_key") or "",
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


def generate_standard_readme(tool_name: str, dataset_name: str, data_file_contents: dict, metadata: dict, summary: str = "") -> str:
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


def _patch_metadata_tokens(job_id: str, prompt_tokens: int, completion_tokens: int):
    """Persist token-usage counts into a job's metadata.json (best-effort)."""
    try:
        metadata_file = os.path.join(JOBS_DIR, job_id, "metadata.json")
        if not os.path.exists(metadata_file):
            return
        with open(metadata_file) as f:
            meta = json.load(f)
        meta["token_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        meta["updated_at"] = datetime.now().isoformat()
        with open(metadata_file, "w") as f:
            json.dump(meta, f, indent=2)
        if job_id in active_jobs:
            active_jobs[job_id]["token_usage"] = meta["token_usage"]
    except Exception as e:
        print(f"[tokens] Could not save token usage for {job_id}: {e}")


def update_job_status(job_id: str, status: str, progress: int = None):
    """Update job status."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    metadata_file = os.path.join(job_dir, "metadata.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata["status"] = status
        if progress is not None:
            metadata["progress"] = progress
        metadata["updated_at"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        active_jobs[job_id] = metadata

async def run_tool_subprocess(tool_name: str, job_id: str, config: dict):
    """Run a tool as subprocess with progress monitoring."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    
    try:
        update_job_status(job_id, "running", 0)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        if 'llm_settings' in config and config['llm_settings']:
            llm_settings = config['llm_settings']
            
            if llm_settings.get('llm_provider'):
                env["LLM_PROVIDER"] = llm_settings['llm_provider']
            
            provider_key = llm_settings.get('llm_provider', 'openai').upper()

            # Strip whitespace from key/url — a leading space is invisible in logs
            # but causes providers like OpenRouter to return 401 "User not found".
            raw_key = (llm_settings.get('api_key') or '').strip()
            key_was_trimmed = raw_key != (llm_settings.get('api_key') or '')
            if raw_key:
                env[f"LLM_API_KEY_{provider_key}"] = raw_key
                env["LLM_API_KEY"] = raw_key

            if llm_settings.get('base_url'):
                env[f"LLM_BASE_URL_{provider_key}"] = llm_settings['base_url'].strip()

            # Diagnostic: show what LLM env vars were injected for this job.
            # Show first-4 + last-8 so any whitespace-corruption is immediately visible.
            key_src = "ui" if raw_key else "env/.env"
            key_var = f"LLM_API_KEY_{provider_key}"
            resolved_key = env.get(key_var) or env.get("LLM_API_KEY") or ""
            if len(resolved_key) >= 12:
                key_display = f"{resolved_key[:4]}...{resolved_key[-8:]}"
            elif resolved_key:
                key_display = f"({len(resolved_key)} chars)"
            else:
                key_display = "(none)"
            diag_msg = (
                f"[LLM] provider={env.get('LLM_PROVIDER')} key_source={key_src} "
                f"key_var={key_var} key={key_display} key_trimmed={key_was_trimmed} "
                f"base_url={env.get(f'LLM_BASE_URL_{provider_key}') or env.get('LLM_BASE_URL', '(fallback)')}"
            )
            print(diag_msg)
            # Also write to debug.log in the job directory — readable without Docker logs
            try:
                with open(os.path.join(job_dir, "debug.log"), 'w') as _dbg:
                    _dbg.write(diag_msg + "\n")
                    _dbg.write(f"received_api_key_empty={not bool(raw_key)}\n")
                    _dbg.write(f"received_api_key_length={len(raw_key)}\n")
                    _dbg.write(f"key_was_trimmed={key_was_trimmed}\n")
            except Exception:
                pass

            selected_model = llm_settings.get('llm_model')
            llm_provider = llm_settings.get('llm_provider', 'openai')

            if selected_model:
                env["LLM_MODEL_NAME"] = selected_model
            elif llm_provider == "local":
                env["LLM_MODEL_NAME"] = "Local LLM"

            if llm_settings.get('hugging_face_api_key'):
                env["HUGGING_FACE_API_KEY"] = llm_settings['hugging_face_api_key']

        config_for_file = json.loads(json.dumps(config))

        if 'llm_settings' in config_for_file and config_for_file['llm_settings']:
            config_for_file['llm_settings'].pop('api_key', None)
            config_for_file['llm_settings'].pop('hugging_face_api_key', None)

        config_file = os.path.join(job_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_for_file, f, indent=2)
        
        stdout_log = os.path.join(job_dir, "stdout.log")
        stderr_log = os.path.join(job_dir, "stderr.log")
        
        script_path = os.path.join(os.getcwd(), f"{tool_name}.py")

        process = await asyncio.create_subprocess_exec(
            "python", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=job_dir,
            env=env
        )

        # Register the process so it can be cancelled via the cancel endpoint.
        active_processes[job_id] = process

        stdout_data = []
        stderr_data = []

        # Legacy per-tool progress patterns kept as fallbacks alongside the
        # unified "PROGRESS {current}/{total}" protocol.
        _legacy_progress_patterns = {
            'datapersona': r"Entry (\d+) of (\d+)",
            'datawriter':  r"Generating document (\d+) of (\d+)",
            'dataqa':      r"Progress: (\d+)/(\d+)",
            'datamix':     r"Taking (\d+) of (\d+) entries",
            'dataconvo':   r"Processing entry (\d+) of (\d+)",
            'datathink':   r"Processing entry (\d+) of (\d+)",
            'reformat':    r"Reformatted entry (\d+) of (\d+)",
            # databird is handled by "PROGRESS X/Y" now
        }

        async def read_stdout():
            while not process.stdout.at_eof():
                line = await process.stdout.readline()
                if not line:
                    break
                line_text = line.decode(errors='ignore').strip()

                # Only update progress while the job is still running
                current_status = active_jobs.get(job_id, {}).get("status", "running")
                if current_status == "running":
                    # ── Unified progress protocol (all tools emit this going forward) ──
                    # Format:  PROGRESS {current}/{total} [optional phase label]
                    unified_match = re.match(r"PROGRESS (\d+)/(\d+)", line_text)
                    if unified_match:
                        current, total = int(unified_match.group(1)), int(unified_match.group(2))
                        if total > 0:
                            progress = int((current / total) * 100)
                            update_job_status(job_id, "running", progress)
                    else:
                        # ── Legacy patterns (backward-compatible fallback) ──────────
                        pattern = _legacy_progress_patterns.get(tool_name)
                        if pattern:
                            m = re.search(pattern, line_text)
                            if m:
                                current, total = int(m.group(1)), int(m.group(2))
                                if total > 0:
                                    progress = int((current / total) * 100)
                                    update_job_status(job_id, "running", progress)

                    # ── Token-usage summary line emitted by tools at completion ──
                    # Format:  TOKENS {prompt}/{completion}
                    token_match = re.match(r"TOKENS (\d+)/(\d+)", line_text)
                    if token_match:
                        prompt_tok = int(token_match.group(1))
                        completion_tok = int(token_match.group(2))
                        # Persist in metadata so the UI and README can display it
                        _patch_metadata_tokens(job_id, prompt_tok, completion_tok)

                stdout_data.append(line_text)
                print(f"[{tool_name}] {line_text}")

        async def read_stderr():
            while not process.stderr.at_eof():
                line = await process.stderr.readline()
                if not line:
                    break
                line_text = line.decode(errors='ignore').strip()
                stderr_data.append(line_text)
                print(f"[{tool_name} ERROR] {line_text}")
        
        await asyncio.gather(read_stdout(), read_stderr())

        await process.wait()

        # Deregister process handle now that the subprocess has exited.
        active_processes.pop(job_id, None)
        
        with open(stdout_log, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stdout_data))
        
        with open(stderr_log, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stderr_data))
        
        if os.path.exists(config_file):
            os.remove(config_file)
        
        if process.returncode == 0:
            try:
                actual_output_path = job_dir

                zip_path = os.path.join(job_dir, f"{tool_name}_output.zip")

                # Collect data files (read before deleting so README can introspect them)
                data_file_contents = {}  # filename -> parsed JSON data
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in os.listdir(actual_output_path):
                        file_path = os.path.join(actual_output_path, file)
                        if file.endswith('.json') and file not in ['config.json', 'metadata.json']:
                            try:
                                with open(file_path, encoding='utf-8') as fh:
                                    data_file_contents[file] = json.load(fh)
                            except Exception:
                                data_file_contents[file] = None
                            zipf.write(file_path, arcname=file)
                            os.remove(file_path)
                        elif file == 'README.md':
                            # Discard tool-generated README; we'll add our own below
                            os.remove(file_path)

                print(f"[{tool_name}] Job {job_id} completed successfully, updating status...")
                update_job_status(job_id, "completed", 100)
                print(f"[{tool_name}] Job {job_id} status updated to completed")

                # Load updated metadata and generate standard README
                metadata_file = os.path.join(job_dir, "metadata.json")
                job_metadata = {}
                if os.path.exists(metadata_file):
                    with open(metadata_file) as f:
                        job_metadata = json.load(f)

                dataset_name = job_metadata.get("dataset_name", tool_name)
                dataset_summary = _generate_dataset_summary(
                    data_file_contents, tool_name, dataset_name,
                    config.get("llm_settings") or {}
                )
                readme_content = generate_standard_readme(tool_name, dataset_name, data_file_contents, job_metadata, summary=dataset_summary)
                readme_path = os.path.join(job_dir, "README.md")
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)

                # Append README.md and metadata.json to zip
                with zipfile.ZipFile(zip_path, 'a') as zipf:
                    zipf.write(readme_path, arcname="README.md")
                    if os.path.exists(metadata_file):
                        zipf.write(metadata_file, arcname="metadata.json")
            except Exception as zip_error:
                print(f"[{tool_name}] Error creating zip for job {job_id}: {zip_error}")
                # Still mark as completed even if zip fails - data is there
                update_job_status(job_id, "completed", 100)
        else:
            print(f"[{tool_name}] Job {job_id} failed with return code {process.returncode}")
            update_job_status(job_id, "failed", 0)
            
            with open(os.path.join(job_dir, "error.log"), 'w', encoding='utf-8') as f:
                f.write(f"Process exited with code: {process.returncode}\n\n")
                f.write(f"STDOUT:\n{chr(10).join(stdout_data)}\n\n")
                f.write(f"STDERR:\n{chr(10).join(stderr_data)}")
    
    except Exception as e:
        active_processes.pop(job_id, None)
        update_job_status(job_id, "failed", 0)
        with open(os.path.join(job_dir, "error.log"), 'w', encoding='utf-8') as f:
            f.write(f"Exception: {str(e)}\n")
            f.write(f"Type: {type(e).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def home():
    """Serve main page."""
    return HTMLResponse(content=open("webapp/index.html", encoding="utf-8").read())

@app.get("/api/tools")
async def list_tools():
    """List available tools."""
    return {
        "tools": [
            {"id": "datapersona", "name": "DataPersona", "description": "Rewrite with personas"},
            {"id": "databird", "name": "DataBird", "description": "Procedural Q&A generation"},
            {"id": "datawriter", "name": "DataWriter", "description": "Generate documents"},
            {"id": "dataqa", "name": "DataQA", "description": "Web/Document scraping to Q&A"},
            {"id": "datamix", "name": "DataMix", "description": "Mix HuggingFace datasets"},
            {"id": "dataconvo", "name": "DataConvo", "description": "Expand conversations"},
            {"id": "reformat", "name": "Reformat", "description": "Convert dataset formats"},
            {"id": "datathink", "name": "DataThink", "description": "Enhance datasets with reasoning steps"}
        ]
    }

@app.get("/api/personas")
async def list_personas(full: bool = False):
    """List available personas from personas.json.

    Args:
        full: If True, returns full persona objects. If False (default), returns only names.
    """
    personas_file = "personas.json"
    if not os.path.exists(personas_file):
        raise HTTPException(status_code=404, detail="personas.json not found")

    with open(personas_file, 'r', encoding='utf-8') as f:
        personas_data = json.load(f)

    if full:
        return {"personas": [{"name": k, **v} for k, v in personas_data.items()]}
    else:
        return {"personas": list(personas_data.keys())}

@app.get("/api/persona/{persona_name}")
async def get_persona(persona_name: str):
    """Get details for a specific persona."""
    personas_file = "personas.json"
    if not os.path.exists(personas_file):
        raise HTTPException(status_code=404, detail="personas.json not found")

    with open(personas_file, 'r', encoding='utf-8') as f:
        personas_data = json.load(f)
    
    if persona_name not in personas_data:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_name}' not found")
    
    return personas_data[persona_name]


@app.post("/api/personas")
async def create_persona(payload: dict):
    """Create a new persona."""
    name = payload.get("name", "").strip()
    description = payload.get("description", "").strip()
    if not name or not description:
        raise HTTPException(status_code=400, detail="name and description are required")
    personas_file = "personas.json"
    personas_data = {}
    if os.path.exists(personas_file):
        with open(personas_file, 'r', encoding='utf-8') as f:
            personas_data = json.load(f)
    if name in personas_data:
        raise HTTPException(status_code=409, detail=f"Persona '{name}' already exists")
    personas_data[name] = {"persona": name, "description": description, "bad_words": []}
    with open(personas_file, 'w', encoding='utf-8') as f:
        json.dump(personas_data, f, indent=2, ensure_ascii=False)
    return {"status": "created", "name": name}


@app.put("/api/personas/{persona_name}")
async def update_persona(persona_name: str, payload: dict):
    """Update an existing persona (rename and/or change description)."""
    personas_file = "personas.json"
    if not os.path.exists(personas_file):
        raise HTTPException(status_code=404, detail="personas.json not found")
    with open(personas_file, 'r', encoding='utf-8') as f:
        personas_data = json.load(f)
    if persona_name not in personas_data:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_name}' not found")
    new_name = payload.get("new_name", persona_name).strip() or persona_name
    description = payload.get("description", personas_data[persona_name].get("description", "")).strip()
    entry = personas_data.pop(persona_name)
    entry["persona"] = new_name
    entry["description"] = description
    personas_data[new_name] = entry
    with open(personas_file, 'w', encoding='utf-8') as f:
        json.dump(personas_data, f, indent=2, ensure_ascii=False)
    return {"status": "updated", "name": new_name}


@app.delete("/api/personas/{persona_name}")
async def delete_persona(persona_name: str):
    """Delete a persona."""
    personas_file = "personas.json"
    if not os.path.exists(personas_file):
        raise HTTPException(status_code=404, detail="personas.json not found")
    with open(personas_file, 'r', encoding='utf-8') as f:
        personas_data = json.load(f)
    if persona_name not in personas_data:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_name}' not found")
    del personas_data[persona_name]
    with open(personas_file, 'w', encoding='utf-8') as f:
        json.dump(personas_data, f, indent=2, ensure_ascii=False)
    return {"status": "deleted", "name": persona_name}


@app.get("/api/defaults/datapersona")
async def get_datapersona_defaults():
    """Get default configuration for DataPersona."""
    return DATAPERSONA_DEFAULTS

@app.post("/api/jobs/datapersona")
async def run_datapersona(
    background_tasks: BackgroundTasks,
    persona: str = Form(...),
    generate_reply_1: bool = Form(False),
    generate_reply_2: bool = Form(False),
    export_alpaca: bool = Form(False),
    save_interval: int = Form(...),
    dataset_name: str = Form(...),
    llm_settings: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Start DataPersona job."""
    job_id = generate_job_id()
    # Before calling create_job_workspace, build the config
    config_dict = {
        "persona": persona,
        "generate_reply_1": generate_reply_1,
        "generate_reply_2": generate_reply_2,
        "save_interval": save_interval,
        "export_alpaca": export_alpaca,
        "dataset_name": dataset_name,
        "files": [file.filename for file in files],
        "llm_settings": json.loads(llm_settings)
    }
    job_dir = create_job_workspace(job_id, "datapersona", dataset_name, config_dict)
    import_dir = os.path.join(job_dir, "import")
    os.makedirs(import_dir, exist_ok=True)

    for file in files:
        with open(os.path.join(import_dir, file.filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Save user settings for next time (exclude file-specific data)
    settings_to_save = {
        "persona": persona,
        "save_interval": save_interval,
        "generate_reply_1": generate_reply_1,
        "generate_reply_2": generate_reply_2,
        "export_alpaca": export_alpaca
    }
    save_user_settings("datapersona", settings_to_save)

    background_tasks.add_task(run_tool_subprocess, "datapersona", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.post("/api/jobs/databird")
async def run_databird(config: DataBirdConfig, background_tasks: BackgroundTasks):
    """Start DataBird job."""
    job_id = generate_job_id()

    output_path = os.path.abspath(os.path.join(os.getcwd(), "output", "databird", f"{job_id}_{config.dataset_name}"))

    config_dict = config.dict()
    config_dict["job_id"] = job_id
    config_dict["output_path"] = output_path
    config_dict["output_format"] = get_global_pref("preferred_output_format", "alpaca")

    job_dir = create_job_workspace(job_id, "databird", config.dataset_name, config_dict)

    # Save user settings for next time
    settings_to_save = {
        "dataset_size": config.dataset_size,
        "clean_score": config.clean_score,
        "full_auto": config.full_auto,
        "manual_perspectives": config.manual_perspectives,
        "include_reasoning": config.include_reasoning
    }
    save_user_settings("databird", settings_to_save)

    background_tasks.add_task(run_tool_subprocess, "databird", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.post("/api/jobs/datawriter")
async def run_datawriter(config: DataWriterConfig, background_tasks: BackgroundTasks):
    """Start DataWriter job."""
    job_id = generate_job_id()
    config_dict = config.dict()
    config_dict["job_id"] = job_id

    job_dir = create_job_workspace(job_id, "datawriter", config.dataset_name, config_dict)

    background_tasks.add_task(run_tool_subprocess, "datawriter", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.post("/api/jobs/dataqa")
async def run_dataqa(
    background_tasks: BackgroundTasks,
    dataset_name: str = Form(...),
    sources: str = Form(""),
    auto_perspectives: bool = Form(True),
    confidence_threshold: float = Form(0.68),
    manual_perspectives: str = Form(""),
    use_persona: bool = Form(False),
    persona_name: Optional[str] = Form(None),
    llm_settings: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    """Start DataQA job."""
    job_id = generate_job_id()
    
    config_dict = {
        "dataset_name": dataset_name,
        "sources": sources,
        "auto_perspectives": auto_perspectives,
        "confidence_threshold": confidence_threshold,
        "use_persona": use_persona,
        "persona_name": persona_name,
        "files": [file.filename for file in files] if files else [],
        "llm_settings": json.loads(llm_settings)
    }

    job_dir = create_job_workspace(job_id, "dataqa", dataset_name, config_dict)

    import_dir = os.path.join(job_dir, "import")
    os.makedirs(import_dir, exist_ok=True)

    # Save uploaded files
    uploaded_file_paths = []
    for file in files:
        file_path = os.path.join(import_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Add relative path for the config
        uploaded_file_paths.append(f"import/{file.filename}")

    # Parse sources from textarea
    source_list = [s.strip() for s in sources.split('\n') if s.strip()]
    
    # Combine URLs and uploaded file paths
    all_sources = source_list + uploaded_file_paths

    config_dict = {
        "dataset_name": dataset_name,
        "sources": all_sources,
        "auto_perspectives": auto_perspectives,
        "confidence_threshold": confidence_threshold,
        "use_persona": use_persona,
        "persona_name": persona_name,
        "job_id": job_id,
        "output_format": get_global_pref("preferred_output_format", "alpaca"),
        "llm_settings": json.loads(llm_settings)
    }
    
    # Parse manual perspectives if provided
    if manual_perspectives.strip():
        try:
            config_dict["manual_perspectives"] = json.loads(manual_perspectives)
            config_dict["auto_perspectives"] = False
        except:
            pass

    # Save user settings for next time
    settings_to_save = {
        "auto_perspectives": auto_perspectives,
        "confidence_threshold": confidence_threshold,
        "use_persona": use_persona,
        "persona_name": persona_name,
        "manual_perspectives": manual_perspectives
    }
    save_user_settings("dataqa", settings_to_save)

    background_tasks.add_task(run_tool_subprocess, "dataqa", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.post("/api/jobs/datathink")
async def run_datathink(
    background_tasks: BackgroundTasks,
    dataset_name: str = Form(...),
    save_interval: int = Form(...),
    thinking_temperature: float = Form(...),
    response_temperature: float = Form(...),
    reasoning_level: str = Form("medium"),
    think_mode: str = Form("insert_reasoning"),
    use_persona: bool = Form(False),
    persona_name: Optional[str] = Form(None),
    llm_settings: str = Form(...),
    file: UploadFile = File(...)
):
    """Start DataThink job."""
    job_id = generate_job_id()

    config_dict = {
        "dataset_name": dataset_name,
        "save_interval": save_interval,
        "thinking_temperature": thinking_temperature,
        "response_temperature": response_temperature,
        "reasoning_level": reasoning_level,
        "think_mode": think_mode,
        "use_persona": use_persona,
        "persona_name": persona_name,
        "import_path": "import",
        "job_id": job_id,
        "llm_settings": json.loads(llm_settings)
    }

    job_dir = create_job_workspace(job_id, "datathink", dataset_name, config_dict)

    import_dir = os.path.join(job_dir, "import")
    os.makedirs(import_dir, exist_ok=True)

    # Save uploaded file to import directory
    with open(os.path.join(import_dir, file.filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save user settings for next time
    settings_to_save = {
        "save_interval": save_interval,
        "thinking_temperature": thinking_temperature,
        "response_temperature": response_temperature,
        "reasoning_level": reasoning_level,
        "think_mode": think_mode,
        "use_persona": use_persona,
        "persona_name": persona_name
    }
    save_user_settings("datathink", settings_to_save)

    background_tasks.add_task(run_tool_subprocess, "datathink", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.post("/api/jobs/dataconvo")
async def run_dataconvo(
    background_tasks: BackgroundTasks,
    dataset_name: str = Form(...),
    save_interval: int = Form(...),
    round_weights: str = Form(...),
    use_persona: bool = Form(False),
    persona_name: Optional[str] = Form(None),
    llm_settings: str = Form(...),
    file: UploadFile = File(...)
):
    """Start DataConvo job."""
    job_id = generate_job_id()
    config_dict = {
        "dataset_name": dataset_name,
        "save_interval": save_interval,
        "round_weights": json.loads(round_weights),
        "use_persona": use_persona,
        "persona_name": persona_name,
        "file": file.filename,
        "llm_settings": json.loads(llm_settings)
    }

    job_dir = create_job_workspace(job_id, "dataconvo", dataset_name, config_dict)

    import_dir = os.path.join(job_dir, "import")
    os.makedirs(import_dir, exist_ok=True)

    with open(os.path.join(import_dir, file.filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save user settings for next time
    settings_to_save = {
        "save_interval": save_interval,
        "round_weights": round_weights,
        "use_persona": use_persona,
        "persona_name": persona_name
    }
    save_user_settings("dataconvo", settings_to_save)

    background_tasks.add_task(run_tool_subprocess, "dataconvo", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.post("/api/jobs/reformat")
async def run_reformat(
    background_tasks: BackgroundTasks,
    dataset_name: str = Form(...),
    target_format: str = Form(...),
    llm_settings: str = Form(...),
    file: UploadFile = File(...)
):
    """Start Reformat job."""
    job_id = generate_job_id()
    config_dict = {
        "dataset_name": dataset_name,
        "target_format": target_format,
        "file": file.filename,
        "llm_settings": json.loads(llm_settings)
    }

    job_dir = create_job_workspace(job_id, "reformat", dataset_name, config_dict)

    import_dir = os.path.join(job_dir, "import")
    os.makedirs(import_dir, exist_ok=True)

    with open(os.path.join(import_dir, file.filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_tool_subprocess, "reformat", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.post("/api/jobs/datamix")
async def run_datamix(config: DataMixConfig, background_tasks: BackgroundTasks):
    """Start DataMix job."""
    job_id = generate_job_id()
    config_dict = config.dict()
    config_dict["job_id"] = job_id
    config_dict["output_format"] = get_global_pref("preferred_output_format", "alpaca")

    job_dir = create_job_workspace(job_id, "datamix", config.dataset_name, config_dict)

    background_tasks.add_task(run_tool_subprocess, "datamix", job_id, config_dict)
    return {"job_id": job_id, "status": "starting"}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id in active_jobs:
        return active_jobs[job_id]
    
    metadata_file = os.path.join(JOBS_DIR, job_id, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    raise HTTPException(status_code=404, detail="Job not found")

@app.get("/api/jobs/{job_id}/download")
async def download_job_output(job_id: str):
    """Download job output as zip."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    
    zip_files = list(Path(job_dir).glob("*.zip"))
    if not zip_files:
        raise HTTPException(status_code=404, detail="Output not found")
    
    return FileResponse(
        str(zip_files[0]),
        media_type="application/zip",
        filename=zip_files[0].name
    )

@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str, body: dict, background_tasks: BackgroundTasks):
    """Re-run a failed/interrupted job using its original config and fresh LLM credentials."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    metadata_file = os.path.join(job_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        raise HTTPException(status_code=404, detail="Job not found")

    with open(metadata_file) as f:
        metadata = json.load(f)

    if metadata.get("status") == "running":
        raise HTTPException(status_code=409, detail="Job is already running")

    tool_name = metadata.get("tool")
    config = dict(metadata.get("config", {}))

    # Inject fresh LLM settings from the resume request (carries the current API key)
    if body.get("llm_settings"):
        config["llm_settings"] = body["llm_settings"]

    # Mark as running again so the UI updates immediately
    update_job_status(job_id, "running", metadata.get("progress", 0))

    background_tasks.add_task(run_tool_subprocess, tool_name, job_id, config)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job by terminating its subprocess.

    The job directory and any partial output are preserved so the user can
    inspect or resume from the last checkpoint.
    """
    process = active_processes.get(job_id)
    if process is None:
        # Check whether the job even exists before returning 404
        metadata_file = os.path.join(JOBS_DIR, job_id, "metadata.json")
        if not os.path.exists(metadata_file):
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=409, detail="No running process found for this job")

    try:
        process.terminate()
    except ProcessLookupError:
        # Process already exited between the lookup and the terminate call
        pass

    active_processes.pop(job_id, None)
    update_job_status(job_id, "cancelled", active_jobs.get(job_id, {}).get("progress", 0))
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = []
    for job_id in os.listdir(JOBS_DIR):
        metadata_file = os.path.join(JOBS_DIR, job_id, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    jobs.append(json.load(f))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Skipping corrupted metadata file for job {job_id}: {e}")
                continue
            except Exception as e:
                print(f"Error reading metadata file for job {job_id}: {e}")
                continue

    return {"jobs": sorted(jobs, key=lambda x: x["created_at"], reverse=True)}

@app.get("/api/settings/{tool_name}")
async def get_tool_settings(tool_name: str):
    """Get saved settings for a specific tool."""
    settings = load_user_settings()
    return settings.get(tool_name, {})

@app.post("/api/settings/{tool_name}")
async def save_tool_settings(tool_name: str, settings: dict):
    """Save settings for a specific tool."""
    save_user_settings(tool_name, settings)
    return {"status": "success"}

@app.get("/api/settings/global/prefs")
async def get_global_prefs():
    """Get global user preferences."""
    settings = load_user_settings()
    defaults = {
        "preferred_output_format": "alpaca",
        "default_persona": "",
        "include_reasoning_output": False,
        "default_temperature": 0.7,
        "default_save_interval": 250,
    }
    stored = settings.get("_global_prefs", {})
    return {**defaults, **stored}

@app.post("/api/settings/global/prefs")
async def save_global_prefs(prefs: dict):
    """Save global user preferences."""
    save_user_settings("_global_prefs", prefs)
    return {"status": "success"}

@app.get("/api/debug/llm")
async def debug_llm_env():
    """Show current server-side LLM environment (key suffixes only, no full values)."""
    provider    = os.environ.get("LLM_PROVIDER", "(not set)")
    model       = os.environ.get("LLM_MODEL_NAME", "(not set)")
    base_url    = os.environ.get("LLM_BASE_URL", "(not set)")
    direct_key  = os.environ.get("LLM_API_KEY", "")
    # Collect all provider-specific keys present in env
    provider_keys = {
        k: (v[-6:] if v else "(empty)")
        for k, v in os.environ.items()
        if k.startswith("LLM_API_KEY_")
    }
    return {
        "LLM_PROVIDER": provider,
        "LLM_MODEL_NAME": model,
        "LLM_BASE_URL": base_url,
        "LLM_API_KEY_suffix": ("..." + direct_key[-6:]) if direct_key else "(not set)",
        "provider_specific_keys": {k: "..." + suffix for k, suffix in provider_keys.items()},
        "note": "Visit /api/jobs/<job_id> then check debug.log in the job folder for per-job diagnostics."
    }

@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket for real-time progress updates."""
    await websocket.accept()

    try:
        while True:
            job_data = active_jobs.get(job_id)
            if job_data:
                await websocket.send_json(job_data)

            if job_data and job_data.get("status") in ["completed", "failed"]:
                await asyncio.sleep(1)
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        # Normal client disconnect - no error logging needed
        pass
    except Exception as e:
        print(f"WebSocket unexpected error: {e}")
    finally:
        # Only close if not already closed
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except Exception:
            pass  # Already closed or in invalid state

@app.delete("/api/jobs/clear_failed")
async def clear_failed_jobs():
    """Delete all failed job runs and their associated output directories."""
    deleted_count = 0
    for job_id in os.listdir(JOBS_DIR):
        job_dir = os.path.join(JOBS_DIR, job_id)
        metadata_file = os.path.join(job_dir, "metadata.json")

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get("status") == "failed":
                try:
                    shutil.rmtree(job_dir)
                    if job_id in active_jobs:
                        del active_jobs[job_id]
                    deleted_count += 1
                except Exception as e:
                    print(f"ERROR: Failed to delete job {job_id}: {e}")
        
    return {"message": f"Deleted {deleted_count} failed jobs."}

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job's workspace."""
    job_dir = os.path.join(JOBS_DIR, job_id)

    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        shutil.rmtree(job_dir)
        if job_id in active_jobs:
            del active_jobs[job_id]
        return {"message": f"Job {job_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")

@app.get("/api/llm/models")
async def get_llm_models(base_url: str, api_key: Optional[str] = None):
    """Fetches a list of available models from the given LLM provider's /models endpoint."""
    try:
        if not api_key and ("localhost" in base_url or "127.0.0.1" in base_url):
            api_key = "lm-studio"

        client = OpenAI(base_url=base_url, api_key=api_key)
        models_response = client.models.list()
        models = [model.id for model in models_response.data]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8910)