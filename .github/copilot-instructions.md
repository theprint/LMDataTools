# Copilot instructions for LMDataTools (DataToolbox)

This repository is a small suite of data-synthesis tools built around a FastAPI web UI and a thin LLM client layer. The goal of these instructions is to help an AI coding assistant be immediately productive by calling out the repo's architecture, conventions, key files, and common workflows.

High-level architecture
- Web server: `webapp.py` (FastAPI). Static frontend files live in `webapp/` and the server exposes the UI on port `8910` by default.
- Tool scripts: small CLI-style scripts at repo root (e.g., `datapersona.py`, `databird.py`, `datawriter.py`, `dataqa.py`, `datamix.py`) that wire the UI to shared logic in `datacore/`.
- Core library: `datacore/` contains reusable modules (cleaning, io, scoring, topics, personas, llm client, config). Prefer changes here for cross-tool fixes.
- LLM integration: `datacore/llm/client.py` is the single unified client used by tools. It wraps an OpenAI-compatible SDK and exposes `LLMClient` and convenience `call_llm`.
- Jobs & outputs: `jobs/` is the working directory for all generated data, each job has a `metadata.json` and output files.

Configuration and environment conventions
- Central config: `datacore/config/settings.py` holds `Config` and a `config` instance. Read and update values through this file when changing defaults.
- Provider selection: set `LLM_PROVIDER` (e.g., `openai`, `local`, `openrouter`, `gemini`). `config` exposes `LLM_API_KEY` and `LLM_BASE_URL` derived from provider.
- Environment variables used by project (examples):
  - `LLM_PROVIDER`, `LLM_MODEL` (or `LLM_MODEL_NAME` used in `datacore/llm/client.py`),
  - `LLM_API_KEY_OPENAI`, `LLM_API_KEY_OPENROUTER`, `HUGGING_FACE_API_KEY`,
  - `LLM_BASE_URL_OPENAI`, `LLM_BASE_URL_LOCAL`, etc.
- Docker: `Dockerfile` and `docker-compose.yml` are provided; `docker-compose up --build` builds and runs the app on port `8910`. NOTE: the example `docker-compose.yml` contains placeholder API keys—do not commit secrets; prefer a `.env` or CI secret store.

Coding patterns & conventions to follow
- Small scripts at repo root are thin: they call into `datacore.*`. Make most logic changes in `datacore/` to avoid duplication.
- LLM calls: use `datacore.llm.client.LLMClient` or `call_llm`. The client will set `model` only if non-null (see `client.py`) — be careful when changing default-model logic.
- Streaming vs non-streaming: `LLMClient.call(..., stream=True)` yields partial output; streaming code prints dots while accumulating text. Keep this behavior in mind when changing the I/O surface.
- Output paths: use `datacore.config.settings.get_tool_output_path(tool_name, job_id, dataset_name)` to build job-specific output directories; do not hardcode paths.
- Persona generation: `datacore/personas/generator.py` shows examples of prompt structure and deterministic post-processing (stripping quotes). If you modify persona prompts, keep the expected format (see examples in the file) so concatenation logic remains valid.

Debugging and developer workflows
- Run locally (dev):
  - `pip install -r requirements.txt`
  - `uvicorn webapp:app --host 0.0.0.0 --port 8910 --reload` or `python webapp.py`.
- Docker: `docker-compose up --build` (exposes `8910:8910`, mounts `./jobs` into the container).
- Inspect models: `datacore/llm/client.py` has `llm_client.list_models()` to retrieve models from the endpoint — useful to validate connectivity.
- Logs & prints: several modules use `print()` for lightweight debugging (e.g., `LLMClient` prints initialization info). When adding logging, prefer a single logger across `datacore`.

Files to inspect when making changes
- `webapp.py` — request routing and background job orchestration.
- `datacore/config/settings.py` — central place for environment defaults and output paths.
- `datacore/llm/client.py` — all LLM integration; modify here for provider-specific changes.
- `datacore/personas/generator.py` — persona prompt templates and deterministic formatting.
- `jobs/` — real job metadata (look at recent timestamped folders to see produced JSON/metadata format).

Examples (copy-pasteable patterns)
- Create LLM client with explicit base URL: `client = LLMClient(base_url=config.LLM_BASE_URL)`
- Generate a persona programmatically:
  ```py
  from datacore.personas.generator import PersonaGenerator
  gen = PersonaGenerator()
  text = gen.generate_asker_persona('someone being introduced to', 'TTRPG')
  ```
- Get tool output path:
  ```py
  from datacore.config.settings import get_tool_output_path
  out = get_tool_output_path('datapersona', job_id='20251124_0001', dataset_name='myset')
  ```

What *not* to change lightly
- Do not change `LLMClient` call signature without updating all callers (multiple tools call it directly or via `call_llm`).
- Avoid renaming environment variables in `settings.py` without adding compatibility aliases — these names are used in `docker-compose.yml` and by users' shells.

If something is unclear
- Tell me which file or workflow you want clarified and I'll add a focused snippet or update this file.

---
Small, actionable edits welcome — ask for a PR or a targeted change.
