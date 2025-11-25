# LM Data Tools

Small suite of data-synthesis tools for generating training and evaluation datasets with LLMs. The project exposes a simple FastAPI web UI and several thin CLI scripts that call into a shared `datacore` library.
Core features

- Web UI: `webapp.py` serves a lightweight web frontend (port `8910` by default) to configure and run jobs.
- Background jobs: tools run asynchronously and store outputs under `jobs/` with `metadata.json` for each run.
- Modular tools: small CLI scripts drive the reproducible logic in `datacore/`.
- Unified LLM client: `datacore/llm/client.py` wraps an OpenAI-compatible SDK and provides `LLMClient` + `call_llm` helpers.
Tools included (top-level scripts)

- `datapersona.py` — persona-based dataset transformations (uses `datacore/personas`).
- `databird.py` — topic-driven Q&A generation.
- `datawriter.py` — long-form document generation for pre-training or content synthesis.
- `dataqa.py` — scrape+Q&A generation from source URLs.
- `datamix.py` — combine/weight multiple datasets (Hugging Face sources) into a new dataset.
Important code areas

- `webapp.py`: FastAPI routing, job orchestration, and upload/download endpoints.
- `datacore/config/settings.py`: Central configuration (provider, base URLs, API key lookup, output paths).
- `datacore/llm/client.py`: All LLM integration — changing call signatures here affects all tools.
- `datacore/personas/*`: Persona prompt templates and local generation helpers used by `datapersona`.
Quickstart (local)

1. Install dependencies:

```powershell
pip install -r requirements.txt
```
2. Start the web UI:

```powershell
uvicorn webapp:app --host 0.0.0.0 --port 8910 --reload
# or: python webapp.py
```
3. Open `http://127.0.0.1:8910` in your browser.

Docker

Build and run with docker-compose (binds `./jobs` into the container):

```powershell
docker-compose up --build
```
Configuration and secrets

- Use `datacore/config/settings.py` to inspect defaults and environment variable names. Common env vars: `LLM_PROVIDER`, `LLM_MODEL` (or `LLM_MODEL_NAME`), `LLM_API_KEY_OPENAI`, `LLM_API_KEY_OPENROUTER`, `HUGGING_FACE_API_KEY`.
- For local development use an `.env` file referenced from `docker-compose.yml` or host environment variables. Do NOT commit secrets — add `.env` to `.gitignore` and provide a `.env.example` with placeholders.
- For production, prefer Docker secrets or an external secret manager (Vault, cloud provider secrets) and avoid committing secret material.
Developer notes & conventions

- Keep tool logic in `datacore/` and use top-level scripts as thin wrappers.
- When modifying LLM behavior, update `datacore/llm/client.py` and verify all tools that call `call_llm` still work.
- Persona outputs are post-processed (e.g., quote stripping) in `datacore/personas/generator.py`; maintain the example formats there if you change prompts.
Where outputs land

- Jobs and outputs are stored under `jobs/<timestamp>*/` with a `metadata.json` file describing inputs and config used for the run. Inspect these folders for generated artifacts.

Contributing

Open an issue or PR with small, targeted changes. If you change environment variable names or `LLMClient` signatures, include compatibility notes in the PR description.

---
Created by Rasmus Rasmussen — see project repo for author links.
# LM Data Tools

This is a collection of data synthesis tools for generating training data for fine-tuning Large Language Models (LLMs). It provides an intuitive interface to run and manage various data generation tasks, from creating persona-based datasets to scraping web content for Q&A pairs.

## Features

*   **Web-Based UI**: A clean and simple interface built with FastAPI to configure and run data synthesis jobs.
*   **Asynchronous Job Processing**: Handles multiple jobs in the background without blocking the UI.
*   **Real-Time Progress Tracking**: Monitor the status and progress of running jobs in real-time.
*   **Modular Toolset**: A suite of distinct tools for different synthetic data needs.
*   **Flexible LLM Configuration**: Supports various LLM providers, including OpenAI, Hugging Face, and local models (e.g., via LM Studio, Ollama).
*   **Secure Credential Management**: API keys and other sensitive information are handled securely and not exposed in configuration files.

## The Tools

The suite includes the following tools, each tailored for a specific data generation task:

*   **DataPersona**: Rewrites an existing dataset from the perspective of a defined "persona." This is great for creating instruction-following or role-playing datasets.
*   **DataBird**: Procedurally generates question-and-answer datasets based on a list of topics.
*   **DataWriter**: Generates a specified number of long-form documents on various subjects, useful for creating pre-training or document-based datasets.
*   **DataQA**: Scrapes content from a list of source URLs and generates question-and-answer pairs based on the scraped information.
*   **DataMix**: Combines multiple datasets from Hugging Face into a new dataset, allowing you to specify weights and subsets for each source.

## Getting Started

Follow these steps to get the DataSynthesis Suite up and running on your local machine.

### 1. Prerequisites

*   Python 3.8+
*   Pip for package management

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/theprint/LMDataTools
    cd DataToolbox
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Running the Application

1.  **Start the web server:**
    ```bash
    python webapp.py
    ```
    Or using Uvicorn for more control:
    ```bash
    uvicorn webapp:app --host 0.0.0.0 --port 8910 --reload
    ```

2.  **Access the web interface:**
    Open your web browser and navigate to `http://127.0.0.1:8910`.

## How to Use

1.  **Select a Tool**: From the main page, choose one of the available data synthesis tools.
2.  **Configure LLM Settings**: In the "LLM Settings" section, select your provider (e.g., OpenAI, Local), enter the Base URL, and provide an API key if required. You can fetch available models directly from your endpoint.
3.  **Set Job Parameters**: Fill in the specific parameters for the selected tool, such as the dataset name, topics, or source URLs.
4.  **Start the Job**: Click the "Run" button to start the data generation process.
5.  **Monitor Progress**: You will be redirected to the "Jobs" page, where you can see the real-time status and progress of your job.
6.  **Download Output**: Once a job is completed, you can download the generated dataset as a `.zip` file directly from the jobs list.

## Project Structure

```
DataToolbox/
├── jobs/                 # Workspace for all jobs, logs, and outputs
├── webapp/               # Frontend static files (HTML, CSS, JS)
├── datapersona.py        # Script for the DataPersona tool
├── databird.py           # Script for the DataBird tool
├── datawriter.py         # Script for the DataWriter tool
├── dataqa.py             # Script for the DataQA tool
├── datamix.py            # Script for the DataMix tool
├── webapp.py             # Main FastAPI application and backend logic
├── personas.json         # Pre-defined personas for the DataPersona tool
├── README.md             # This file
└── requirements.txt      # Python dependencies
└── topics.json           # Tiered list of topics for the DataWriter tool
```

---

This project was created by Rasmus Rasmussen: [Blog](https://rasmusrasmussen.com) | [LinkedIn](https://www.linkedin.com/in/theprint/) | [GitHub](https://github.com/theprint) | [Huggingface](https://huggingface.co/theprint) | [Bluesky](https://bsky.app/rasmusrasmussen.com)