import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="Audify.ai", description="Text-to-Sound Multi-Agent System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

class AudioResponse(BaseModel):
    description: str
    source: str
    audio: str

@app.post("/generate", response_model=AudioResponse)
async def generate_sound(request: PromptRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    result = run_agent(request.prompt.strip())

    return AudioResponse(
        description=result.get("description", f"Sound of {request.prompt}"),
        source=result.get("source", "generated"),
        audio=result.get("audio", ""),
    )

@app.get("/audio")
async def serve_audio():
    for path, mime in [("generated_audio.mp3", "audio/mpeg"), ("generated_audio.wav", "audio/wav")]:
        if os.path.exists(path):
            return FileResponse(path, media_type=mime)
    raise HTTPException(status_code=404, detail="No generated audio file found")

@app.get("/download")
async def download_audio():
    for path, mime, name in [
        ("generated_audio.mp3", "audio/mpeg", "audify_output.mp3"),
        ("generated_audio.wav", "audio/wav", "audify_output.wav"),
    ]:
        if os.path.exists(path):
            return FileResponse(path, media_type=mime, filename=name, headers={"Content-Disposition": f"attachment; filename={name}"})
    raise HTTPException(status_code=404, detail="No generated audio file found")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Audify.ai"}

if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
