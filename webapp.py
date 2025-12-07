# webapp.py
"""
Web interface for DataSynthesis Suite
"""
from datapersona import DEFAULT_CONFIG as DATAPERSONA_DEFAULTS
from openai import OpenAI # Import OpenAI client for model listing
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import subprocess
import json
import os
import shutil
import zipfile
from datetime import datetime
from typing import Optional, Dict, List
import asyncio
from pathlib import Path

import re
app = FastAPI(title="LM Data Tools")

# Job tracking
active_jobs: Dict[str, dict] = {}
JOBS_DIR = "./jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# ============================================================================
# Data Models
# ============================================================================

class LLMSettings(BaseModel):
    llm_provider: str = "openai" # Add llm_provider here with a default
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    llm_model: Optional[str] = None
    hugging_face_api_key: Optional[str] = None

class DataPersonaConfig(BaseModel):
    persona: str
    generate_reply_1: bool = True
    generate_reply_2: bool = True
    save_interval: int = 250
    dataset_name: str = "my-persona-dataset" # Add dataset_name here
    llm_settings: Optional[LLMSettings] = None


class DataBirdConfig(BaseModel):
    dataset_name: str
    topics: List[str]
    full_auto: bool = True
    dataset_size: str = "small"
    clean_score: float = 0.76
    llm_settings: Optional[LLMSettings] = None


class DataWriterConfig(BaseModel):
    document_count: int = 500
    temperature: float = 0.8
    dataset_name: str = "my-writer-dataset" # Add dataset_name here
    llm_settings: Optional[LLMSettings] = None


class DataQAConfig(BaseModel):
    dataset_name: str
    source_urls: List[str] = []
    auto_perspectives: bool = True
    confidence_threshold: float = 0.68
    manual_perspectives: Optional[List[tuple]] = None
    llm_settings: Optional[LLMSettings] = None


class DataMixConfig(BaseModel):
    dataset_name: str
    total_samples: int = 10000
    dataset_sources: List[tuple]  # (name, weight, subset)
    llm_settings: Optional[LLMSettings] = None


# ============================================================================
# Utilities
# ============================================================================

def generate_job_id():
    """Generate unique job ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_job_workspace(job_id: str, tool_name: str, dataset_name: str):
    """Create workspace for a job."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Create job metadata
    metadata = {
        "job_id": job_id,
        "tool": tool_name,
        "dataset_name": dataset_name,
        "status": "starting",
        "created_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    with open(os.path.join(job_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return job_dir


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


def write_config_file(tool_name: str, config_dict: dict, job_dir: str):
    """Write a temporary config file for a tool."""
    config_file = os.path.join(job_dir, f"{tool_name}_config.py")
    
    config_lines = ["# Auto-generated configuration\n"]
    for key, value in config_dict.items():
        if isinstance(value, str):
            config_lines.append(f'{key.upper()} = "{value}"\n')
        elif isinstance(value, list):
            config_lines.append(f'{key.upper()} = {value}\n')
        else:
            config_lines.append(f'{key.upper()} = {value}\n')
    
    with open(config_file, 'w') as f:
        f.writelines(config_lines)
    
    return config_file


async def run_tool_subprocess(tool_name: str, job_id: str, config: dict):
    """Run a tool as subprocess with progress monitoring."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    
    try:
        update_job_status(job_id, "running", 0)

        # Prepare environment variables for the subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if 'llm_settings' in config and config['llm_settings']:
            llm_settings = config['llm_settings']
            print(f"DEBUG: llm_settings.llm_model received from UI: {llm_settings.get('llm_model')}")
            # Set LLM_PROVIDER environment variable
            if llm_settings.get('llm_provider'):
                env["LLM_PROVIDER"] = llm_settings['llm_provider']
            
            # Set API key environment variable based on provider
            if llm_settings.get('api_key'):
                provider_key = llm_settings.get('llm_provider', 'openai').upper()
                env[f"LLM_API_KEY_{provider_key}"] = llm_settings['api_key']
            
            # Set Base URL environment variable based on provider
            if llm_settings.get('base_url'):
                provider_key = llm_settings.get('llm_provider', 'openai').upper()
                env[f"LLM_BASE_URL_{provider_key}"] = llm_settings['base_url']

            # Ensure LLM_MODEL is set correctly based on selection and provider
            selected_model = llm_settings.get('llm_model')
            llm_provider = llm_settings.get('llm_provider', 'openai') # Default to openai if not specified

            if selected_model: # If a model is explicitly selected (not empty string)
                env["LLM_MODEL_NAME"] = selected_model
            elif llm_provider == "local": # If no model selected, but provider is local
                env["LLM_MODEL_NAME"] = "Local LLM" # Use local placeholder
            # If no model selected and provider is remote, LLM_MODEL_NAME is not set.
            # This allows the remote API to use its own default or raise an error if a model is required.

            if llm_settings.get('hugging_face_api_key'):
                env["HUGGING_FACE_API_KEY"] = llm_settings['hugging_face_api_key']

        # Create a deep copy of the config to modify it before writing to disk
        config_for_file = json.loads(json.dumps(config))

        # Remove sensitive keys from the config that will be written to a file.
        # These are passed securely via environment variables.
        if 'llm_settings' in config_for_file and config_for_file['llm_settings']:
            config_for_file['llm_settings'].pop('api_key', None)
            config_for_file['llm_settings'].pop('hugging_face_api_key', None)

        # Write config.json for the tool to load
        config_file = os.path.join(job_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_for_file, f, indent=2)
        
        # Create log files
        stdout_log = os.path.join(job_dir, "stdout.log")
        stderr_log = os.path.join(job_dir, "stderr.log")
        
        # Run the tool
        script_path = os.path.join(os.getcwd(), f"{tool_name}.py")

        process = await asyncio.create_subprocess_exec(
            "python", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=job_dir,
            env=env
        )
        
        # Capture output
        stdout_data = []
        stderr_data = []
        
        async def read_stdout():
            while not process.stdout.at_eof():
                line = await process.stdout.readline()
                if not line:
                    break
                line_text = line.decode(errors='ignore').strip()

                # --- Progress Parsing ---
                progress_patterns = {
                    # For datapersona: "Entry 123 of 456"
                    'datapersona': r"Entry (\d+) of (\d+)",
                    # For databird: "Question 123/456"
                    'databird': r"Question (\d+)/(\d+)",
                    # For datawriter: "Generating document 123 of 500"
                    'datawriter': r"Generating document (\d+) of (\d+)",
                    # For dataqa: "Progress: 23/100"
                    'dataqa': r"Progress: (\d+)/(\d+)",
                    # For datamix: "Taking 123 of 456 entries"
                    'datamix': r"Taking (\d+) of (\d+) entries",
                }

                if tool_name in progress_patterns:
                    pattern = progress_patterns[tool_name]
                    match = re.search(pattern, line_text)
                    if match:
                        current, total = int(match.group(1)), int(match.group(2))
                        if total > 0:
                            progress = int((current / total) * 100)
                            update_job_status(job_id, "running", progress)
                
                # --- Progress Parsing for streaming ---
                if tool_name == 'datapersona' and line_text.strip().endswith('.'):
                     # This is a simple way to show activity. We don't calculate percentage here.
                     update_job_status(job_id, "running", active_jobs[job_id].get('progress', 0))
                # --- End Progress Parsing ---

                stdout_data.append(line_text)
                print(f"[{tool_name}] {line_text}")  # Console logging
        
        async def read_stderr():
            while not process.stderr.at_eof():
                line = await process.stderr.readline()
                if not line:
                    break
                line_text = line.decode(errors='ignore').strip()
                stderr_data.append(line_text)
                print(f"[{tool_name} ERROR] {line_text}")  # Console logging
        
        # Read both streams concurrently
        await asyncio.gather(read_stdout(), read_stderr())
        
        await process.wait()
        
        # Save logs
        with open(stdout_log, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stdout_data))
        
        with open(stderr_log, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stderr_data))
        
        # Clean up config file
        if os.path.exists(config_file):
            os.remove(config_file)
        
        if process.returncode == 0:
            # Determine the actual output path used by the tool
            # This assumes the tool writes to output/{tool_name}/{job_id}_{dataset_name}
            # The tool scripts now save their output directly into the job_dir.
            actual_output_path = job_dir
            
            zip_path = os.path.join(job_dir, f"{tool_name}_output.zip")
            output_files_found = False
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Find JSON files generated by the tool in the job directory
                for file in os.listdir(actual_output_path):
                    if file.endswith('.json') and file not in ['config.json', 'metadata.json']:
                        file_path = os.path.join(actual_output_path, file)
                        zipf.write(file_path, arcname=file)
                        os.remove(file_path) # Clean up the file after adding to zip
                        output_files_found = True
            
            update_job_status(job_id, "completed", 100)
        else:
            update_job_status(job_id, "failed", 0)
            
            # Save detailed error log
            with open(os.path.join(job_dir, "error.log"), 'w', encoding='utf-8') as f:
                f.write(f"Process exited with code: {process.returncode}\n\n")
                f.write(f"STDOUT:\n{chr(10).join(stdout_data)}\n\n")
                f.write(f"STDERR:\n{chr(10).join(stderr_data)}")
    
    except Exception as e:
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
            {"id": "dataqa", "name": "DataQA", "description": "Web scraping to Q&A"},
            {"id": "datamix", "name": "DataMix", "description": "Mix HuggingFace datasets"}
        ]
    }


@app.get("/api/personas")
async def list_personas():
    """List available personas from personas.json."""
    personas_file = "personas.json"
    if not os.path.exists(personas_file):
        raise HTTPException(status_code=404, detail="personas.json not found")

    with open(personas_file, 'r', encoding='utf-8') as f:
        personas_data = json.load(f)

    return {"personas": personas_data}


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
    llm_settings: str = Form(...), # JSON string
    files: List[UploadFile] = File(...)
):
    """Start DataPersona job."""
    job_id = generate_job_id()
    job_dir = create_job_workspace(job_id, "datapersona", dataset_name)
    import_dir = os.path.join(job_dir, "import")
    os.makedirs(import_dir, exist_ok=True)

    for file in files:
        with open(os.path.join(import_dir, file.filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    config_dict = {
        "persona": persona, "generate_reply_1": generate_reply_1,
        "generate_reply_2": generate_reply_2, "save_interval": save_interval, "export_alpaca": export_alpaca,
        "dataset_name": dataset_name, "job_id": job_id,
        "import_path": "import", # Pass only the subdirectory name,
        "llm_settings": json.loads(llm_settings)
    }

    background_tasks.add_task(
        run_tool_subprocess, "datapersona", job_id, config_dict
    )

    return {"job_id": job_id, "status": "starting"}


@app.post("/api/jobs/databird")
async def run_databird(config: DataBirdConfig, background_tasks: BackgroundTasks):
    """Start DataBird job."""
    job_id = generate_job_id()
    job_dir = create_job_workspace(job_id, "databird", config.dataset_name)
    
    # Define the absolute output path for the tool
    output_path = os.path.abspath(os.path.join(os.getcwd(), "output", "databird", f"{job_id}_{config.dataset_name}"))

    # Convert config to dict and add job_id
    config_dict = config.dict()
    config_dict["job_id"] = job_id # Add job_id here
    config_dict["output_path"] = output_path # Explicitly add output_path
    
    background_tasks.add_task(
        run_tool_subprocess,
        "databird",
        job_id,
        config_dict # Pass the modified config_dict
    )
    
    return {"job_id": job_id, "status": "starting"}


@app.post("/api/jobs/datawriter")
async def run_datawriter(config: DataWriterConfig, background_tasks: BackgroundTasks):
    """Start DataWriter job."""
    job_id = generate_job_id()
    job_dir = create_job_workspace(job_id, "datawriter", config.dataset_name)

    # Convert config to dict and add job_id
    config_dict = config.dict()
    config_dict["job_id"] = job_id

    background_tasks.add_task(
        run_tool_subprocess,
        "datawriter",
        job_id,
        config_dict # Pass the modified config_dict
    )

    return {"job_id": job_id, "status": "starting"}


@app.post("/api/jobs/dataqa")
async def run_dataqa(config: DataQAConfig, background_tasks: BackgroundTasks):
    """Start DataQA job."""
    job_id = generate_job_id()
    job_dir = create_job_workspace(job_id, "dataqa", config.dataset_name)

    config_dict = config.dict()
    config_dict["job_id"] = job_id

    background_tasks.add_task(
        run_tool_subprocess,
        "dataqa",
        job_id,
        config_dict
    )

    return {"job_id": job_id, "status": "starting"}


@app.post("/api/jobs/datamix")
async def run_datamix(config: DataMixConfig, background_tasks: BackgroundTasks):
    """Start DataMix job."""
    job_id = generate_job_id()
    job_dir = create_job_workspace(job_id, "datamix", config.dataset_name)

    config_dict = config.dict()
    config_dict["job_id"] = job_id

    background_tasks.add_task(
        run_tool_subprocess, "datamix", job_id, config_dict
    )
    return {"job_id": job_id, "status": "starting"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id in active_jobs:
        return active_jobs[job_id]
    
    # Try loading from disk
    metadata_file = os.path.join(JOBS_DIR, job_id, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/jobs/{job_id}/download")
async def download_job_output(job_id: str):
    """Download job output as zip."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    
    # Find zip file
    zip_files = list(Path(job_dir).glob("*.zip"))
    if not zip_files:
        raise HTTPException(status_code=404, detail="Output not found")
    
    return FileResponse(
        str(zip_files[0]),
        media_type="application/zip",
        filename=zip_files[0].name
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = []
    for job_id in os.listdir(JOBS_DIR):
        metadata_file = os.path.join(JOBS_DIR, job_id, "metadata.json")
        try:
            with open(metadata_file, 'r') as f:
                jobs.append(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            # Skip corrupted or empty job files
            print(f"Warning: Skipping corrupted metadata file for job {job_id}: {e}")
            continue
        except Exception as e:
            # Log other errors but continue
            print(f"Error reading metadata file for job {job_id}: {e}")
            continue
        try:
            with open(metadata_file, 'r') as f:
                jobs.append(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            # Skip corrupted or empty job files
            print(f"Warning: Skipping corrupted metadata file for job {job_id}: {e}")
            continue
        except Exception as e:
            # Log other errors but continue
            print(f"Error reading metadata file for job {job_id}: {e}")
            continue
        try:
            with open(metadata_file, 'r') as f:
                jobs.append(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            # Skip corrupted or empty job files
            print(f"Warning: Skipping corrupted metadata file for job {job_id}: {e}")
            continue
        except Exception as e:
            # Log other errors but continue
            print(f"Error reading metadata file for job {job_id}: {e}")
            continue
    
    return {"jobs": sorted(jobs, key=lambda x: x["created_at"], reverse=True)}


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket for real-time progress updates."""
    await websocket.accept()
    
    try:
        while True:
            # Check job status
            job_data = active_jobs.get(job_id)
            if job_data:
                await websocket.send_json(job_data)
            
            # Stop if job completed or failed
            if job_data and job_data.get("status") in ["completed", "failed"]:
                await asyncio.sleep(1)  # Give client time to receive final message
                break
            
            await asyncio.sleep(1)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.delete("/api/jobs/clear_failed")
async def clear_failed_jobs():
    """Delete all failed job runs and their associated output directories."""
    deleted_count = 0
    print(f"DEBUG: Starting clear_failed_jobs. JOBS_DIR: {JOBS_DIR}")
    for job_id in os.listdir(JOBS_DIR):
        job_dir = os.path.join(JOBS_DIR, job_id)
        metadata_file = os.path.join(job_dir, "metadata.json")

        print(f"DEBUG: Checking job_id: {job_id}")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            job_status = metadata.get("status")
            print(f"DEBUG: Job {job_id} status: {job_status}")
            if job_status == "failed":
                print(f"DEBUG: Attempting to delete failed job: {job_id}")
                try:
                    shutil.rmtree(job_dir)
                    if job_id in active_jobs:
                        del active_jobs[job_id]
                    deleted_count += 1
                    print(f"DEBUG: Successfully deleted job: {job_id}")
                except Exception as e:
                    print(f"ERROR: Failed to delete job {job_id}: {e}")
        else:
            print(f"DEBUG: metadata.json not found for job_id: {job_id}")
        
    print(f"DEBUG: Finished clear_failed_jobs. Deleted {deleted_count} jobs.")
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
    """
    Fetches a list of available models from the given LLM provider's /models endpoint.
    """
    try:
        # Ensure a placeholder API key is used if none is provided for local/other
        if not api_key and ("localhost" in base_url or "127.0.0.1" in base_url):
            api_key = "lm-studio" # Default placeholder for local LLMs

        client = OpenAI(base_url=base_url, api_key=api_key)
        models_response = client.models.list()
        models = [model.id for model in models_response.data]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")
# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8910)