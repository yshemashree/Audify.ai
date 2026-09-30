import os
import time
import uuid
import logging
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import run_agent
from database import warm_up

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("audify")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FILES = 100          # keep at most this many generated clips on disk
MAX_AGE_SECONDS = 3600   # and none older than an hour
MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav"}

app = FastAPI(title="Audify.ai", description="Text-to-Sound Multi-Agent System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    # Load the embedding model in the background so the first request isn't
    # stuck behind an 80 MB download.
    threading.Thread(target=warm_up, daemon=True).start()


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


class PromptRequest(BaseModel):
    prompt: str


class AudioResponse(BaseModel):
    description: str
    source: str
    engine: str
    audio_url: str
    download_url: str


def _cleanup_outputs():
    try:
        files = sorted(
            (os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)),
            key=os.path.getmtime,
            reverse=True,
        )
        now = time.time()
        for i, path in enumerate(files):
            if i >= MAX_FILES or now - os.path.getmtime(path) > MAX_AGE_SECONDS:
                os.remove(path)
    except OSError:
        pass


def _resolve(audio_id: str) -> str:
    name = os.path.basename(audio_id)
    path = os.path.join(OUTPUT_DIR, name)
    if name != audio_id or os.path.splitext(name)[1] not in MIME or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Audio not found — it may have expired, generate it again")
    return path


# Plain `def` so FastAPI runs it in a worker thread: the pipeline makes blocking
# network calls, and running them on the event loop froze the whole server.
@app.post("/generate", response_model=AudioResponse)
def generate_sound(request: PromptRequest):
    prompt = (request.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(prompt) > 300:
        raise HTTPException(status_code=400, detail="Prompt is too long (max 300 characters)")

    _cleanup_outputs()
    out_base = os.path.join(OUTPUT_DIR, uuid.uuid4().hex)
    started = time.time()
    result = run_agent(prompt, out_base)
    audio_id = os.path.basename(result["path"])
    log.info("'%s' -> %s via %s in %.1fs", prompt, audio_id, result["engine"], time.time() - started)

    return AudioResponse(
        description=result["description"],
        source=result["source"],
        engine=result["engine"],
        audio_url=f"/audio/{audio_id}",
        download_url=f"/download/{audio_id}",
    )


@app.get("/audio/{audio_id}")
def serve_audio(audio_id: str):
    path = _resolve(audio_id)
    return FileResponse(path, media_type=MIME[os.path.splitext(path)[1]])


@app.get("/download/{audio_id}")
def download_audio(audio_id: str):
    path = _resolve(audio_id)
    ext = os.path.splitext(path)[1]
    return FileResponse(path, media_type=MIME[ext], filename=f"audify_output{ext}")


@app.get("/health")
def health():
    from tools import ELEVENLABS_KEY, FREESOUND_KEY, HF_TOKEN
    return {
        "status": "ok",
        "service": "Audify.ai",
        "engines": {
            "freesound": bool(FREESOUND_KEY),
            "elevenlabs": bool(ELEVENLABS_KEY),
            "llm": bool(HF_TOKEN),
            "procedural": True,
        },
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
