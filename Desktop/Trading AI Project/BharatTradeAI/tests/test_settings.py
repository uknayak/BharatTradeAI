from config.settings import settings


def test_settings_loads():
    assert settings.app_name == "BharatTradeAI"
