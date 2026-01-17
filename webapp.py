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
import shutil
import zipfile
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import asyncio
from pathlib import Path

import re
app = FastAPI(title="LM Data Tools")
app.mount("/static", StaticFiles(directory="webapp"), name="static")

# Job tracking
active_jobs: Dict[str, dict] = {}
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
    llm_settings: Optional[LLMSettings] = None

class DataWriterConfig(BaseModel):
    document_count: int = 500
    temperature: float = 0.8
    dataset_name: str = "my-writer-dataset"
    llm_settings: Optional[LLMSettings] = None

class DataQAConfig(BaseModel):
    dataset_name: str
    sources: List[str] = []
    auto_perspectives: bool = True
    confidence_threshold: float = 0.68
    manual_perspectives: Optional[List[tuple]] = None
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
        config_for_metadata = config.copy()
        
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
        if 'llm_settings' in config and config['llm_settings']:
            llm_settings = config['llm_settings']
            
            if llm_settings.get('llm_provider'):
                env["LLM_PROVIDER"] = llm_settings['llm_provider']
            
            if llm_settings.get('api_key'):
                provider_key = llm_settings.get('llm_provider', 'openai').upper()
                env[f"LLM_API_KEY_{provider_key}"] = llm_settings['api_key']
            
            if llm_settings.get('base_url'):
                provider_key = llm_settings.get('llm_provider', 'openai').upper()
                env[f"LLM_BASE_URL_{provider_key}"] = llm_settings['base_url']

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
        
        stdout_data = []
        stderr_data = []

        async def read_stdout():
            while not process.stdout.at_eof():
                line = await process.stdout.readline()
                if not line:
                    break
                line_text = line.decode(errors='ignore').strip()

                progress_patterns = {
                    # match either Question X/Y or Answer X/Y for databird
                    'datapersona': r"Entry (\d+) of (\d+)",
                    'databird': r"(?:Question|Answer) (\d+)/(\d+)",
                    'datawriter': r"Generating document (\d+) of (\d+)",
                    'dataqa': r"Progress: (\d+)/(\d+)",
                    'datamix': r"Taking (\d+) of (\d+) entries",
                    'dataconvo': r"Processing entry (\d+) of (\d+)",
                    'datathink': r"Processing entry (\d+) of (\d+)",
                    'reformat': r"Reformatted entry (\d+) of (\d+)",
                }

                # Only update progress if still running (not completed/failed)
                current_status = active_jobs.get(job_id, {}).get("status", "running")
                if current_status == "running":
                    if tool_name in progress_patterns:
                        pattern = progress_patterns[tool_name]
                        match = re.search(pattern, line_text)
                        if match:
                            current, total = int(match.group(1)), int(match.group(2))
                            if total > 0:
                                progress = int((current / total) * 100)
                                update_job_status(job_id, "running", progress)

                    if tool_name == 'datapersona' and line_text.strip().endswith('.'):
                         update_job_status(job_id, "running", active_jobs[job_id].get('progress', 0))

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

                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in os.listdir(actual_output_path):
                        if file.endswith('.json') and file != 'config.json':
                            file_path = os.path.join(actual_output_path, file)
                            zipf.write(file_path, arcname=file)
                            os.remove(file_path)

                print(f"[{tool_name}] Job {job_id} completed successfully, updating status...")
                update_job_status(job_id, "completed", 100)
                print(f"[{tool_name}] Job {job_id} status updated to completed")
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
            {"id": "reformat", "name": "Reformat", "description": "Convert dataset formats"}
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
        return {"personas": personas_data}
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

    job_dir = create_job_workspace(job_id, "databird", config.dataset_name, config_dict)

    # Save user settings for next time
    settings_to_save = {
        "dataset_size": config.dataset_size,
        "clean_score": config.clean_score,
        "full_auto": config.full_auto,
        "manual_perspectives": config.manual_perspectives
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
        "round_weights": round_weights,
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