<div align="center">

# Audify.ai

**Type a word. Hear it.**

Describe any sound — thunder, a robot booting up, a wolf howling at 3am — and Audify synthesises it instantly using AI.

[![Railway](https://img.shields.io/badge/deployed%20on-Railway-0B0D0E?logo=railway&logoColor=white)](https://railway.app)
[![ElevenLabs](https://img.shields.io/badge/powered%20by-ElevenLabs-6C3AED?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIGZpbGw9IndoaXRlIi8+PC9zdmc+)](https://elevenlabs.io)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

</div>

---

## What it does

You type any sound in plain English. Audify expands your input, searches a library, and if nothing matches — generates a brand-new audio clip using ElevenLabs AI. In seconds.

---

## Architecture

```mermaid
graph TB
    subgraph Browser["🌐 Browser"]
        UI["Three.js UI\nInput + 3D Orb"]
    end

    subgraph Server["⚙️ FastAPI Server"]
        API["/generate\n/audio\n/download"]
        AGENT["agent.py\nOrchestrator"]
    end

    subgraph Pipeline["🔧 Audio Pipeline"]
        EXPAND["1. elaborate_prompt()\nKeyword expansion"]
        SEARCH["2. search_audio()\nLibrary match"]
        GEN["3. generate_audio()\nAI synthesis"]
    end

    subgraph External["☁️ External"]
        EL["ElevenLabs\nSound Generation API"]
        LOCAL["Procedural Fallback\nPure Python synthesis"]
    end

    UI -->|"POST /generate"| API
    API --> AGENT
    AGENT --> EXPAND
    EXPAND --> SEARCH
    SEARCH -->|"NO_MATCH"| GEN
    SEARCH -->|"match found"| API
    GEN -->|"API key present"| EL
    GEN -->|"no key / error"| LOCAL
    EL --> API
    LOCAL --> API
    API -->|"WAV file"| UI
```

---

## Audio Generation Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Browser UI
    participant API as FastAPI
    participant Agent as agent.py
    participant Tools as tools.py
    participant EL as ElevenLabs API

    User->>UI: Types "dog barking"
    UI->>API: POST /generate { prompt }
    API->>Agent: run_agent("dog barking")
    Agent->>Tools: elaborate_prompt("dog barking")
    Tools-->>Agent: "a sharp bark, medium-sized dog, outdoor reverb..."
    Agent->>Tools: search_audio(description)
    Tools-->>Agent: NO_MATCH
    Agent->>Tools: generate_audio(description)
    Tools->>EL: POST /v1/sound-generation
    EL-->>Tools: audio bytes (WAV)
    Tools-->>Agent: "generated_audio.wav"
    Agent-->>API: { description, source, audio }
    API-->>UI: JSON response
    UI->>API: GET /audio
    API-->>UI: WAV stream
    UI->>User: 🔊 Plays audio + shows download
```

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS, Three.js r128, CSS 3D |
| Backend | Python 3, FastAPI, Uvicorn |
| AI Sound Generation | ElevenLabs Sound Generation API |
| Fallback Synthesis | Pure Python (math, struct, random) |
| Deployment | Railway |

---

## How to run locally

```bash
# 1. Clone
git clone https://github.com/yshemashree/Audify.ai
cd Audify.ai

# 2. Install deps
pip install -r requirements.txt

# 3. Add your ElevenLabs key (optional — app works without it via fallback)
echo "ELEVENLABS_API_KEY=your_key_here" > .env

# 4. Start
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`.

---

## Project structure

```mermaid
graph LR
    subgraph Root
        MAIN["main.py\nFastAPI app + routes"]
        AGENT["agent.py\nOrchestrator"]
        TOOLS["tools.py\nAll audio logic"]
        REQ["requirements.txt"]
    end

    subgraph Frontend["frontend/"]
        HTML["index.html\nFull UI — Three.js, CSS 3D"]
    end

    MAIN --> AGENT
    AGENT --> TOOLS
    MAIN --> HTML
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | No | Enables ElevenLabs AI generation. Without it, the procedural fallback runs. |

---

## Fallback sounds

When no API key is present, Audify generates audio procedurally in Python — no external calls, zero latency:

`rain` · `thunder` · `fire` · `wind` · `ocean` · `heartbeat` · `cat` · `dog` · `bird` · `keyboard` · `clock` · `footsteps` · `car` · `water` · `crowd` · `noise`

Anything else defaults to pink noise.

---

<div align="center">
  Built with ElevenLabs · FastAPI · Railway
</div>
