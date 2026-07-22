"""
app/utils/telemetry.py

OpenTelemetry distributed tracing initialization and context propagation helper.
"""

from fastapi import FastAPI
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


def setup_opentelemetry(app: FastAPI) -> None:
    """Instrument FastAPI application with OpenTelemetry distributed tracing."""
    if not settings.ENABLE_OPENTELEMETRY:
        log.info("OpenTelemetry disabled by config")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        log.info("OpenTelemetry distributed tracing initialized successfully")
    except Exception as exc:
        log.warning("Could not initialize OpenTelemetry instrumentation: %s", exc)
