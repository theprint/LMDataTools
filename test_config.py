"""
Tests for config/settings.py and llm/client.py fixes.
Verifies that LLM_BASE_URL and LLM_API_KEY env vars are honoured correctly
and that nothing defaults to OpenAI when not configured.
"""

import os
import importlib
import sys


def reload_modules():
    """Force fresh import of config and client after env var changes."""
    for mod in list(sys.modules.keys()):
        if "datacore.config" in mod or "datacore.llm" in mod:
            del sys.modules[mod]
    import datacore.config.settings as settings_mod
    import datacore.llm.client as client_mod
    return settings_mod, client_mod


def clear_llm_env():
    for key in [
        "LLM_BASE_URL", "LLM_API_KEY", "LLM_PROVIDER",
        "LLM_MODEL", "LLM_MODEL_NAME",
        "LLM_BASE_URL_LOCAL", "LLM_API_KEY_LOCAL",
        "LLM_BASE_URL_OPENAI", "LLM_API_KEY_OPENAI",
    ]:
        os.environ.pop(key, None)


def run(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
    except AssertionError as e:
        print(f"  FAIL  {label}")
        print(f"        {e}")


# ---------------------------------------------------------------------------
# settings.py tests
# ---------------------------------------------------------------------------

def test_direct_base_url_overrides_provider():
    """LLM_BASE_URL env var must take priority over provider lookup."""
    clear_llm_env()
    os.environ["LLM_BASE_URL"] = "http://my-server:9000/v1"
    os.environ["LLM_PROVIDER"] = "openai"  # would normally return api.openai.com
    s, _ = reload_modules()
    result = s.config.LLM_BASE_URL
    assert result == "http://my-server:9000/v1", \
        f"Expected http://my-server:9000/v1, got {result!r}"


def test_direct_api_key_overrides_provider():
    """LLM_API_KEY env var must take priority over provider lookup."""
    clear_llm_env()
    os.environ["LLM_API_KEY"] = "my-real-key"
    os.environ["LLM_PROVIDER"] = "openai"  # would normally return placeholder
    s, _ = reload_modules()
    result = s.config.LLM_API_KEY
    assert result == "my-real-key", f"Expected my-real-key, got {result!r}"


def test_provider_local_default():
    """Default provider must be 'local', not 'openai'."""
    clear_llm_env()
    s, _ = reload_modules()
    assert s.config.LLM_PROVIDER == "local", \
        f"Default provider should be 'local', got {s.config.LLM_PROVIDER!r}"


def test_provider_local_uses_local_url():
    """With LLM_PROVIDER=local and no LLM_BASE_URL, must use local URL not OpenAI."""
    clear_llm_env()
    os.environ["LLM_PROVIDER"] = "local"
    s, _ = reload_modules()
    result = s.config.LLM_BASE_URL
    assert "openai.com" not in result, \
        f"Local provider must not resolve to openai.com, got {result!r}"


def test_no_env_does_not_use_openai_url():
    """With no env vars set at all, URL must not point to api.openai.com."""
    clear_llm_env()
    s, _ = reload_modules()
    result = s.config.LLM_BASE_URL
    assert "openai.com" not in result, \
        f"Default must not point to openai.com, got {result!r}"


# ---------------------------------------------------------------------------
# client.py tests
# ---------------------------------------------------------------------------

def test_global_client_no_gpt_fallback():
    """Global llm_client must not default to gpt-3.5-turbo."""
    clear_llm_env()
    _, c = reload_modules()
    model = c.llm_client.default_model
    assert model != "gpt-3.5-turbo", \
        f"Global client must not default to gpt-3.5-turbo, got {model!r}"


def test_global_client_respects_llm_model_name():
    """LLM_MODEL_NAME env var must set the default model on the global client."""
    clear_llm_env()
    os.environ["LLM_MODEL_NAME"] = "mistral-7b"
    _, c = reload_modules()
    assert c.llm_client.default_model == "mistral-7b", \
        f"Expected mistral-7b, got {c.llm_client.default_model!r}"


def test_global_client_respects_llm_model():
    """LLM_MODEL env var must set the default model on the global client."""
    clear_llm_env()
    os.environ["LLM_MODEL"] = "llama3"
    _, c = reload_modules()
    assert c.llm_client.default_model == "llama3", \
        f"Expected llama3, got {c.llm_client.default_model!r}"


def test_client_uses_base_url_from_config():
    """LLMClient must pick up LLM_BASE_URL from the config property."""
    clear_llm_env()
    os.environ["LLM_BASE_URL"] = "http://localhost:8001/v1"
    s, c = reload_modules()
    client = c.LLMClient()
    assert "localhost:8001" in str(client.base_url), \
        f"Client base_url should contain localhost:8001, got {client.base_url!r}"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\ndatacore config/client tests\n" + "-" * 40)

    run("LLM_BASE_URL env var overrides provider lookup", test_direct_base_url_overrides_provider)
    run("LLM_API_KEY env var overrides provider lookup",  test_direct_api_key_overrides_provider)
    run("Default provider is 'local' not 'openai'",       test_provider_local_default)
    run("LLM_PROVIDER=local does not resolve to openai.com", test_provider_local_uses_local_url)
    run("No env vars set does not point to openai.com",   test_no_env_does_not_use_openai_url)
    run("Global client does not default to gpt-3.5-turbo", test_global_client_no_gpt_fallback)
    run("LLM_MODEL_NAME sets global client default model", test_global_client_respects_llm_model_name)
    run("LLM_MODEL sets global client default model",      test_global_client_respects_llm_model)
    run("LLMClient picks up LLM_BASE_URL from config",    test_client_uses_base_url_from_config)

    print()
    clear_llm_env()
