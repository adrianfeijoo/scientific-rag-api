"""Settings parsing: values from .env / environment must behave sensibly."""

from app.config import Settings


def test_empty_string_falls_back_to_default():
    # `.env.example` ships blank keys; copying it as-is must be safe.
    settings = Settings(_env_file=None, OLLAMA_BASE_URL="", OPENAI_API_KEY="")
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.openai_api_key is None


def test_explicit_values_are_honoured():
    settings = Settings(_env_file=None, LLM_PROVIDER="ollama", CHUNK_SIZE="900")
    assert settings.llm_provider == "ollama"
    assert settings.chunk_size == 900


def test_defaults_match_documentation():
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 750
    assert settings.chunk_overlap == 140
    assert settings.llm_provider == "auto"
