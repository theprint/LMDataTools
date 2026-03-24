# datacore/personas/loader.py
"""
Handles loading persona data from the personas.json file.
"""
import json
import os

def _find_project_root(start_path):
    """Find the project root by looking for a known file, e.g., 'webapp.py'."""
    path = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(path, 'webapp.py')):
            return path
        parent_path = os.path.dirname(path)
        if parent_path == path: # Reached the filesystem root
            return None
        path = parent_path

def _load_personas(file_path=None):
    """Loads the personas from the JSON file."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    project_root = _find_project_root(os.getcwd()) or _find_project_root(__file__)
    if not project_root:
        raise FileNotFoundError("Could not find project root to locate personas.json.")

    personas_path = os.path.join(project_root, 'personas.json')
    if not os.path.exists(personas_path):
        raise FileNotFoundError(f"personas.json not found at the project root: {personas_path}")

    with open(personas_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_persona(name, file_path=None):
    """
    Retrieve a single persona by name.
    """
    personas = _load_personas(file_path)
    if name not in personas:
        raise ValueError(f"Persona '{name}' not found in personas file.")
    return personas[name]

def get_all_personas(file_path=None):
    """
    Returns a list of all available persona names.
    """
    personas = _load_personas(file_path)
    return list(personas.keys())

def get_persona_description(name, file_path=None):
    """
    Fetches just the description for a given persona.
    """
    persona_data = get_persona(name, file_path)
    return persona_data.get("description", "")