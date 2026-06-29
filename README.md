<div align="center">

# Audify.ai

**Type a word. Hear it.**

An agentic AI pipeline that takes any text description and synthesises it into audio — powered by a live LLM, a ChromaDB vector database, ElevenLabs sound generation, and a Three.js frontend.

[![Railway](https://img.shields.io/badge/deployed%20on-Railway-0B0D0E?logo=railway&logoColor=white)](https://railway.app)
[![ElevenLabs](https://img.shields.io/badge/audio-ElevenLabs-6C3AED)](https://elevenlabs.io)
[![Qwen](https://img.shields.io/badge/LLM-Qwen%202.5%2072B-FF6B00)](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
[![ChromaDB](https://img.shields.io/badge/vector%20db-ChromaDB-7C3AED)](https://www.trychroma.com)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

</div>

---

## What makes it agentic

Most text-to-audio tools send your input directly to a generation model. Audify runs a **3-stage agentic pipeline** — an LLM reasons about your input first, then a ChromaDB vector search retrieves the best expert description, then the audio is generated from it.

```mermaid
flowchart LR
    INPUT["User Input\n'wolf howling'"]
    LLM["Qwen 2.5 72B\nSound Design Agent"]
    CHROMA["ChromaDB\nVector Search\n60 expert descriptions"]
    MATCH{Cosine\ndistance\n< 0.45?}
    GEN["ElevenLabs\nSound Generation"]
    OUT["Audio Output"]

    INPUT --> LLM
    LLM -->|acoustic description| CHROMA
    CHROMA --> MATCH
    MATCH -->|YES — use matched description| GEN
    MATCH -->|NO — use LLM description| GEN
    GEN --> OUT
```

The LLM acts as an expert sound designer, translating human descriptions into precise acoustic language. ChromaDB then checks if a curated expert description already exists for that sound — if so, that higher-quality description goes to ElevenLabs instead.

---

## Agentic pipeline — deep dive

```mermaid
graph TB
    subgraph Agent["Agentic Orchestrator — agent.py"]
        direction TB
        T1["Stage 1\nelaborate_prompt()"]
        T2["Stage 2\nsearch_audio() — ChromaDB"]
        T3["Stage 3\ngenerate_audio()"]
        T1 --> T2
        T2 -->|matched description| T3
        T2 -->|NO_MATCH — LLM description| T3
    end

    subgraph LLM["LLM — Qwen 2.5 72B via HuggingFace"]
        SP["System prompt: Expert sound designer\nOutputs acoustic descriptions"]
    end

    subgraph VectorDB["Vector Database — ChromaDB"]
        EMB["Sentence embeddings\nHNSW cosine index"]
        DS["60 curated expert\nacoustic descriptions"]
        EMB --- DS
    end

    subgraph Audio["Audio Generation — ElevenLabs"]
        EL["Sound Generation API\nprompt_influence: 0.5\nduration: 5s"]
    end

    subgraph Fallback["Local Fallback"]
        PROC["Procedural synthesis\nPure Python — math + struct\n16 sound recipes"]
    end

    T1 <-->|chat completion| LLM
    T2 <-->|embed + query| VectorDB
    T3 -->|API key present| Audio
    T3 -->|no key / error| Fallback
```

### Stage 1 — elaborate_prompt()

The LLM receives a system prompt written as an expert sound designer. It outputs a precise acoustic description — never generic, always specific to what ElevenLabs needs.

> Input: `"wolf at night"`
> Output: `"lone wolf howl, long sustained note with vibrato, open mountain valley, night wind, distant echo tail, eerie silence before and after"`

### Stage 2 — search_audio() with ChromaDB

The elaborated description is embedded into a vector and compared against 60 hand-curated expert acoustic descriptions stored in ChromaDB using cosine similarity.

- **Distance < 0.45** — semantically close match found. Return the curated expert description (higher quality than LLM output).
- **Distance ≥ 0.45** — no close match. Fall through with the LLM description.

This is a **RAG (Retrieval-Augmented Generation)** pattern: retrieve a better prompt before generating.

### Stage 3 — generate_audio()

The winning description (curated or LLM-generated) is sent to ElevenLabs. If ElevenLabs is unavailable, a pure-Python procedural synthesiser generates audio locally with zero external calls.

---

## ChromaDB vector database

`database.py` builds and queries the sound collection:

```python
# 60 expert descriptions are embedded and indexed on startup
collection.add(
    ids=[...],
    documents=["steady rainfall on leaves, soft patter rhythm, outdoor reverb", ...],
    metadatas=[{"label": "rain"}, ...]
)

# At query time — cosine similarity search
results = collection.query(query_texts=[elaborated_prompt], n_results=1)
distance = results["distances"][0][0]  # < 0.45 = match
```

ChromaDB uses an HNSW index with cosine distance. The dataset was curated with acoustic language (pitch, texture, reverb, distance, material) that matches how ElevenLabs expects prompts — generic descriptions give poor results, so the dataset is small but domain-specific.

---

## Full system architecture

```mermaid
graph TB
    subgraph Browser["Browser"]
        UI["Three.js UI\n3D orb + input\nCSS 3D card deck"]
    end

    subgraph Server["FastAPI — main.py"]
        R1["POST /generate"]
        R2["GET /audio"]
        R3["GET /download"]
    end

    subgraph Pipeline["Pipeline — tools.py + database.py"]
        EP["elaborate_prompt()"]
        VS["vector_search() — ChromaDB"]
        GA["generate_audio()"]
    end

    subgraph External["External APIs"]
        HF["HuggingFace Router\nQwen 2.5 72B"]
        EL["ElevenLabs\nSound Generation"]
    end

    subgraph Local["Local Fallback"]
        SYNTH["Procedural WAV synthesis\nPure Python"]
    end

    UI -->|POST prompt| R1
    R1 --> EP
    EP <-->|chat completion| HF
    EP --> VS
    VS -->|matched description or NO_MATCH| GA
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
    participant DB as ChromaDB
    participant EL as ElevenLabs

    User->>UI: "wolf howling at night"
    UI->>API: POST /generate
    API->>Agent: run_agent(prompt)

    Agent->>Qwen: elaborate_prompt("wolf howling at night")
    Note over Qwen: Expert sound design system prompt
    Qwen-->>Agent: "lone wolf howl, long note with vibrato,\nmountain valley, night wind, echo tail..."

    Agent->>DB: vector_search(description)
    Note over DB: Embed query → cosine similarity\nagainst 60 expert descriptions
    DB-->>Agent: match: "lone wolf howl..." (distance=0.21)

    Agent->>EL: generate_audio(matched_description)
    Note over EL: prompt_influence=0.5, duration=5s
    EL-->>Agent: audio bytes

    Agent-->>API: { description, source: "retrieved", audio }
    API-->>UI: JSON response
    UI->>API: GET /audio
    API-->>UI: WAV stream
    UI->>User: Plays + shows download
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM Agent | Qwen 2.5 72B via HuggingFace Inference Router |
| Vector Database | ChromaDB with cosine similarity (HNSW index) |
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
  Qwen 2.5 · ChromaDB · ElevenLabs · FastAPI · Railway
</div>
