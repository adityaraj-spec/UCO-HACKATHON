"""
app/main.py

FastAPI application entrypoint for PhaseGuard - Layer 2
(Speaker Verification Service).

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from app.utils.windows_symlink_patch import apply_windows_symlink_fallback

# Must run before speechbrain (or anything that creates symlinks) is
# imported anywhere in the process. On Windows, regular user accounts
# without Administrator rights or Developer Mode enabled cannot create
# symlinks, which causes SpeechBrain's model-fetching code to fail with
# "OSError: [WinError 1314] A required privilege is not held by the
# client" when it tries to symlink cached model files into
# ECAPA_MODEL_SAVE_DIR. This patches Path.symlink_to so that if symlink
# creation fails due to a permissions error, it transparently falls back
# to copying the file instead — slightly more disk usage, but works on
# every machine with zero special permissions or admin rights required.
apply_windows_symlink_fallback()

# Fix for newer torchaudio versions breaking SpeechBrain 1.0.0
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, 'io'):
    class DummyIO:
        StreamReader = object
    torchaudio.io = DummyIO



from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ml.ecapa_service import get_ecapa_service
from app.utils.exceptions import PhaseGuardError

settings = get_settings()
configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup: pre-load the ECAPA-TDNN model so the first incoming request
    doesn't pay the (large) model-loading cost. Loading is wrapped in a
    try/except so the API can still start (and report errors via /health)
    even if the model download fails in a constrained environment.
    """
    log.info(f"Starting {settings.APP_NAME} (env={settings.APP_ENV})")
    try:
        get_ecapa_service().load_model()
    except Exception as exc:  # noqa: BLE001
        log.error(
            f"ECAPA-TDNN model failed to preload at startup: {exc}. "
            "It will be lazily loaded on first request."
        )

    yield

    log.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title="PhaseGuard - Layer 2 (Speaker Verification)",
    description=(
        "Production backend for PhaseGuard's Layer 2 identity-verification "
        "engine. Enrolls customer voiceprints using SpeechBrain's "
        "ECAPA-TDNN model and verifies live audio against stored "
        "voiceprints using cosine similarity, feeding results into the "
        "PhaseGuard Risk Engine."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - permissive by default for hackathon/demo use; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PhaseGuardError)
async def phaseguard_exception_handler(request: Request, exc: PhaseGuardError):
    """
    Fallback handler for any PhaseGuardError that isn't already converted to
    an HTTPException within an endpoint. Returns a structured 400 response.
    """
    log.warning(f"Unhandled PhaseGuardError on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message},
    )


@app.get("/", tags=["Health"], summary="Root health check")
async def root() -> dict:
    return {
        "service": settings.APP_NAME,
        "status": "ok",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"], summary="Detailed health check")
async def health() -> dict:
    """
    Reports basic service health, including whether the ECAPA-TDNN model is
    currently loaded in memory.
    """
    ecapa = get_ecapa_service()
    model_loaded = ecapa._model is not None  # noqa: SLF001 - internal status check

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "embedding_dim": settings.EMBEDDING_DIM,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "layer1_fraud_threshold": settings.LAYER1_FRAUD_THRESHOLD,
    }


app.include_router(api_router, prefix="/api/v1")

from fastapi.staticfiles import StaticFiles
import os

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/ui")
async def serve_ui():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_dir, "index.html"))
