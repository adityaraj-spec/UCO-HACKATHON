import pytest


@pytest.mark.asyncio
async def test_health_reports_layer1_liveness_dev_bypass_warning(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main.settings, "LAYER1_LIVENESS_CHECK_ENABLED", False)
    payload = await main.health()

    assert payload["layer1_liveness_check_enabled"] is False
    assert payload["warning"] == (
        "LAYER1 LIVENESS CHECK DISABLED — DEV/DEMO MODE ONLY, DO NOT USE IN PRODUCTION."
    )
