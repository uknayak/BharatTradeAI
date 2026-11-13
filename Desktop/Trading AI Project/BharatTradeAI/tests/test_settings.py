from config.settings import settings
from config.settings import Settings


def test_settings_loads():
    assert settings.app_name == "BharatTradeAI"


def test_ollama_model_setting_accepts_ollama_model_and_legacy_name():
    preferred_name = Settings(_env_file=None, OLLAMA_MODEL="gemma3:latest")
    legacy_name = Settings(_env_file=None, MODEL_NAME="gemma3:latest")

    assert preferred_name.model_name == "gemma3:latest"
    assert legacy_name.model_name == "gemma3:latest"
