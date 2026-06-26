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
    path = "generated_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No generated audio file found")
    return FileResponse(path, media_type="audio/wav")

@app.get("/download")
async def download_audio():
    path = "generated_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No generated audio file found")
    return FileResponse(path, media_type="audio/wav", filename="audify_output.wav", headers={"Content-Disposition": "attachment; filename=audify_output.wav"})

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Audify.ai"}

if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
