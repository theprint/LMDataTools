# datacore/config/settings.py

import os


class Config:
    """Global configuration for data synthesis tools."""
    
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
    # Removed LLM_DEFAULT_MODEL
    LLM_DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "6000"))
    HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY", "not-set")
    
    # New: Dictionary to store API keys for different providers
    API_KEYS = {
        "openai": os.getenv("LLM_API_KEY_OPENAI", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
        "gemini": os.getenv("LLM_API_KEY_GEMINI", "AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
        "openrouter": os.getenv("LLM_API_KEY_OPENROUTER", "sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"), # Added OpenRouter default
        "local": os.getenv("LLM_API_KEY_LOCAL", "lm-studio"), # Default for local
        "other": os.getenv("LLM_API_KEY_OTHER", "lm-studio"), # Default for other
    }

    # New: Dictionary to store base URLs for different providers
    BASE_URLS = {
        "openai": os.getenv("LLM_BASE_URL_OPENAI", "https://api.openai.com/v1"),
        "gemini": os.getenv("LLM_BASE_URL_GEMINI", "https://generativelanguage.googleapis.com/v1beta/models"),
        "openrouter": os.getenv("LLM_BASE_URL_OPENROUTER", "https://openrouter.ai/api/v1"), # Added OpenRouter default
        "local": os.getenv("LLM_BASE_URL_LOCAL", "http://localhost:1234/v1"), # Corrected for LM Studio
        "other": os.getenv("LLM_BASE_URL_OTHER", "http://localhost:1234/v1"), # Corrected for LM Studio
    }

    @property
    def LLM_API_KEY(self):
        """Returns the API key: direct LLM_API_KEY env var takes priority, then provider-specific."""
        direct = os.getenv("LLM_API_KEY")
        if direct:
            return direct
        provider = self.LLM_PROVIDER.lower()
        dynamic = os.getenv(f"LLM_API_KEY_{provider.upper()}")
        if dynamic:
            return dynamic
        return self.API_KEYS.get(provider, "not-needed")

    @property
    def LLM_BASE_URL(self):
        """Returns the base URL: direct LLM_BASE_URL env var takes priority, then provider-specific."""
        direct = os.getenv("LLM_BASE_URL")
        if direct:
            return direct
        provider = self.LLM_PROVIDER.lower()
        dynamic = os.getenv(f"LLM_BASE_URL_{provider.upper()}")
        if dynamic:
            return dynamic
        return self.BASE_URLS.get(provider, "http://localhost:1234/v1")
    
    @property
    def LLM_MODEL(self): # New property to get the selected model
        """Returns the selected LLM model."""
        return os.getenv("LLM_MODEL", "Local LLM") # Default to "Local LLM" if not set
    
    # File paths
    PERSONAS_FILE = "personas.json"
    TOPICS_FILE = "topics.json"
    FORMATS_FILE = "formats.yaml"
    
    # Output structure
    OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH", "./output")
    
    # Other shared settings
    DEFAULT_SAVE_INTERVAL = 250


# Global config instance
config = Config()


def get_tool_output_path(tool_name: str, job_id: str = None, dataset_name: str = None) -> str:
    """
    Get the output path for a specific tool, optionally creating a job-specific subdirectory.
    
    Args:
        tool_name: Name of the tool (e.g., 'datapersona')
        job_id: Optional job ID to create a job-specific subdirectory.
        dataset_name: Optional dataset name to include in the job-specific subdirectory.
        
    Returns:
        Full path to tool's output directory
    """
    path = os.path.join(config.OUTPUT_BASE_PATH, tool_name)
    if job_id and dataset_name:
        path = os.path.join(path, f"{job_id}_{dataset_name}")
    os.makedirs(path, exist_ok=True)
    return path