# datacore/personas/prompt_manager.py
"""
Handles the logic of combining a tool's base system prompt with a persona.
"""

def inject_persona_into_prompt(base_prompt, persona_name, persona_description):
    """
    Injects a persona description into a base system prompt using a standard template.

    Args:
        base_prompt (str): The original system prompt of the tool.
        persona_name (str): The name of the persona (e.g., "Confident Coach").
        persona_description (str): The detailed description of the persona.

    Returns:
        str: A new, combined system prompt.
    """
    injection_template = f"""{base_prompt}

---

Your personality and style should be that of a {persona_name}.

**Persona Description:**
{persona_description}

---

Adopt this persona subtly. Do not mention you are playing a role. Respond directly to the user's query in character."""
    return injection_template.strip()