<div align="center">

# Audify.ai

**Type a word. Hear it.**

An agentic AI pipeline that takes any text description and synthesises it into audio — powered by a live LLM, ElevenLabs sound generation, and a Three.js frontend.

[![Railway](https://img.shields.io/badge/deployed%20on-Railway-0B0D0E?logo=railway&logoColor=white)](https://railway.app)
[![ElevenLabs](https://img.shields.io/badge/audio-ElevenLabs-6C3AED)](https://elevenlabs.io)
[![Qwen](https://img.shields.io/badge/LLM-Qwen%202.5%2072B-FF6B00)](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

</div>

---

## What makes it agentic

Most text-to-audio tools send your input directly to a generation model. Audify runs a **3-stage agentic pipeline** — an LLM reasons about your input first, then decides whether to retrieve or generate audio, then calls the appropriate tool.

```mermaid
flowchart LR
    INPUT["🗣️ User Input\n'dog barking'"]
    LLM["🧠 Qwen 2.5 72B\nSound Design Agent"]
    SEARCH["🔍 search_audio()\nLibrary lookup"]
    GEN["⚡ generate_audio()\nElevenLabs AI"]
    OUT["🔊 Audio Output"]

    INPUT --> LLM
    LLM -->|"sharp medium-dog bark,\noutdoor echo, three bursts..."| SEARCH
    SEARCH -->|NO_MATCH| GEN
    SEARCH -->|match| OUT
    GEN --> OUT
```

The LLM doesn't just paraphrase your input — it acts as an expert sound designer, translating human descriptions into precise acoustic language (texture, pitch, reverb, distance, material) that ElevenLabs can synthesise accurately.

---

## Agentic pipeline — deep dive

```mermaid
graph TB
    subgraph Agent["🤖 Agentic Orchestrator — agent.py"]
        direction TB
        T1["Tool 1\nelaborate_prompt()"]
        T2["Tool 2\nsearch_audio()"]
        T3["Tool 3\ngenerate_audio()"]
        T1 --> T2
        T2 -->|NO_MATCH| T3
    end

    subgraph LLM["☁️ LLM — Qwen 2.5 72B via HuggingFace"]
        SP["System prompt:\nExpert sound designer\nOutputs acoustic descriptions"]
    end

    subgraph Audio["☁️ Audio Generation — ElevenLabs"]
        EL["Sound Generation API\nprompt_influence: 0.5\nduration: 5s"]
    end

    subgraph Fallback["🛡️ Local Fallback"]
        PROC["Procedural synthesis\nPure Python — math + struct\n16 sound recipes"]
    end

    T1 <-->|"chat completion"| LLM
    T3 -->|"API key present"| Audio
    T3 -->|"no key / error"| Fallback
```

### Stage 1 — elaborate_prompt()

The LLM receives a system prompt written as an expert sound designer. It outputs a precise acoustic description — never generic, always specific to what ElevenLabs needs to produce a great result.

> Input: `"wolf at night"`
> Output: `"lone wolf howl, long sustained note with vibrato, open mountain valley, night wind, distant echo tail, eerie silence before and after"`

### Stage 2 — search_audio()

The elaborated description is compared against a library of known clips using keyword similarity. A high-confidence match returns instantly without any generation.

### Stage 3 — generate_audio()

On `NO_MATCH`, the elaborated prompt is sent to ElevenLabs. If ElevenLabs is unavailable, a pure-Python procedural synthesiser generates audio locally with zero external calls.

---

## Full system architecture

```mermaid
graph TB
    subgraph Browser["🌐 Browser"]
        UI["Three.js UI\n3D orb + input\nCSS 3D card deck"]
    end

    subgraph Server["⚙️ FastAPI — main.py"]
        R1["POST /generate"]
        R2["GET /audio"]
        R3["GET /download"]
    end

    subgraph Pipeline["🔧 Pipeline — tools.py"]
        EP["elaborate_prompt()"]
        SA["search_audio()"]
        GA["generate_audio()"]
    end

    subgraph External["☁️ External APIs"]
        HF["HuggingFace Router\nQwen 2.5 72B"]
        EL["ElevenLabs\nSound Generation"]
    end

    subgraph Local["🛡️ Local Fallback"]
        SYNTH["Procedural WAV synthesis\nPure Python"]
    end

    UI -->|POST prompt| R1
    R1 --> EP
    EP <-->|chat completion| HF
    EP --> SA
    SA -->|NO_MATCH| GA
    GA -->|ELEVENLABS_API_KEY| EL
    GA -->|fallback| SYNTH
    EL --> R2
    SYNTH --> R2
    R2 -->|WAV stream| UI
    R3 -->|WAV download| UI
```

---

## Request lifecycle

```mermaid
sequenceDiagram
    actor User
    participant UI as Browser
    participant API as FastAPI
    participant Agent as agent.py
    participant Qwen as Qwen 2.5 72B
    participant EL as ElevenLabs

    User->>UI: "wolf howling at night"
    UI->>API: POST /generate
    API->>Agent: run_agent(prompt)

    Agent->>Qwen: elaborate_prompt("wolf howling at night")
    Note over Qwen: Expert sound design system prompt
    Qwen-->>Agent: "lone wolf howl, long note with vibrato,\nmountain valley, night wind, echo tail..."

    Agent->>Agent: search_audio(description)
    Agent-->>Agent: NO_MATCH

    Agent->>EL: generate_audio(description)
    Note over EL: prompt_influence=0.5, duration=5s
    EL-->>Agent: audio bytes

    Agent-->>API: { description, source: "generated", audio }
    API-->>UI: JSON response
    UI->>API: GET /audio
    API-->>UI: WAV stream
    UI->>User: 🔊 Plays + shows download
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM Agent | Qwen 2.5 72B via HuggingFace Inference Router |
| Sound Generation | ElevenLabs Sound Generation API |
| Fallback Synthesis | Pure Python (math, struct, random) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS + Three.js r128 + CSS 3D |
| Deployment | Railway |

---

## Run locally

```bash
git clone https://github.com/yshemashree/Audify.ai
cd Audify.ai
pip install -r requirements.txt

# Add keys — both optional, app works without them via fallback
echo "ELEVENLABS_API_KEY=your_key" > .env
echo "HUGGINGFACE_API_TOKEN=your_token" >> .env

uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`.

---

## Environment variables

| Variable | Required | Effect |
|---|---|---|
| `ELEVENLABS_API_KEY` | No | Enables AI sound generation. Without it, procedural synthesis runs. |
| `HUGGINGFACE_API_TOKEN` | No | Enables LLM prompt expansion via Qwen 72B. Without it, keyword expansion runs. |

The app is fully functional with no API keys — just less accurate sound descriptions and procedural audio.

---

## Procedural fallback sounds

When no ElevenLabs key is present, Audify synthesises audio locally in pure Python:

`rain` · `thunder` · `fire` · `wind` · `ocean` · `heartbeat` · `cat` · `dog` · `bird` · `keyboard` · `clock` · `footsteps` · `car` · `water` · `crowd` · `noise`

---

<div align="center">
  Qwen 2.5 · ElevenLabs · FastAPI · Railway
</div>
