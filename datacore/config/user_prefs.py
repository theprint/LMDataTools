# datacore/config/user_prefs.py
"""
UserPreferences - Global user preference store for LM Data Tools.

Persists to ``user_prefs.json`` in the application root directory (detected
automatically by searching parent directories for ``webapp.py``).

The preferences here act as low-priority defaults that flow into tool configs
when specific values have not been set by the user in the job config.  Tools
are free to ignore them; the webapp injects them when building job configs.

Usage
-----
    from datacore.config.user_prefs import prefs

    # Read a preference (with optional fallback)
    temp = prefs.get("default_temperature", 0.7)

    # Write / update
    prefs.set("default_persona", "Helpful Assistant")
    prefs.update({"default_temperature": 0.8, "default_model": "gpt-4o"})

    # Get everything
    all_prefs = prefs.all()
"""

import os
import json

# ── Default values ──────────────────────────────────────────────────────────

_DEFAULT_PREFS: dict = {
    # Output format preferred by this user across tools
    "preferred_output_format": "alpaca",   # "alpaca" | "sharegpt" | "qa"

    # Persona applied by default when a tool supports USE_PERSONA
    "default_persona": "",

    # Whether to include <think> reasoning blocks in DataThink output
    "include_reasoning_output": True,

    # LLM generation defaults (tools may override per-job)
    "default_temperature": 0.7,
    "default_save_interval": 250,

    # LLM provider defaults
    "default_provider": "local",
    "default_model": "",
}


# ── Path resolution ─────────────────────────────────────────────────────────

def _find_prefs_path() -> str:
    """
    Locate the app root by walking up from cwd until ``webapp.py`` is found.
    Falls back to cwd if nothing is found within 4 levels.
    """
    path = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.exists(os.path.join(path, "webapp.py")):
            return os.path.join(path, "user_prefs.json")
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.path.join(os.getcwd(), "user_prefs.json")


# ── Main class ───────────────────────────────────────────────────────────────

class UserPreferences:
    """Simple JSON-backed key-value preference store."""

    def __init__(self, prefs_path: str = None):
        self._path: str = prefs_path or _find_prefs_path()
        self._data: dict = dict(_DEFAULT_PREFS)
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._data.update(saved)
            except (json.JSONDecodeError, OSError):
                pass  # Silently ignore corrupt prefs file

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass  # Best-effort — never crash a tool run over prefs I/O

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, key: str, default=None):
        """Return the preference value for *key*, or *default* if not set."""
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        """Set a single preference and persist immediately."""
        self._data[key] = value
        self._save()

    def update(self, updates: dict) -> None:
        """Merge *updates* into preferences and persist."""
        self._data.update(updates)
        self._save()

    def all(self) -> dict:
        """Return a copy of all preferences."""
        return dict(self._data)

    def defaults(self) -> dict:
        """Return the built-in default preferences (not including saved overrides)."""
        return dict(_DEFAULT_PREFS)


# ── Singleton ────────────────────────────────────────────────────────────────
# Import this from tool scripts and webapp.py for shared access.
prefs = UserPreferences()
