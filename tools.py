import os
import json
import time
import requests
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from smolagents import tool

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

SOUND_DB = [
    {"description": "a cat meowing softly indoors", "url": "https://www.soundjay.com/animals/cat-meow-1.wav"},
    {"description": "heavy rain falling on a rooftop", "url": "https://www.soundjay.com/nature/rain-01.wav"},
    {"description": "thunder rumbling in a storm", "url": "https://www.soundjay.com/nature/thunder-1.wav"},
    {"description": "ocean waves crashing on a beach", "url": "https://www.soundjay.com/nature/waves-1.wav"},
    {"description": "a dog barking loudly outside", "url": "https://www.soundjay.com/animals/dog-bark-1.wav"},
    {"description": "birds chirping in a forest at dawn", "url": "https://www.soundjay.com/nature/birds-1.wav"},
    {"description": "fire crackling in a fireplace", "url": "https://www.soundjay.com/nature/fire-1.wav"},
    {"description": "a clock ticking quietly", "url": "https://www.soundjay.com/clock/clock-ticking-1.wav"},
    {"description": "wind blowing through trees", "url": "https://www.soundjay.com/nature/wind-1.wav"},
    {"description": "keyboard typing rapidly", "url": "https://www.soundjay.com/office/keyboard-1.wav"},
]

def get_embedding(text: str) -> list:
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    for attempt in range(3):
        response = requests.post(api_url, headers=HEADERS, json={"inputs": text})
        if response.status_code == 503:
            time.sleep(20)
            continue
        if response.status_code == 200:
            data = response.json()
            if isinstance(data[0], list):
                return data[0]
            return data
    return []

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    if a.ndim > 1:
        a = a.mean(axis=0)
    if b.ndim > 1:
        b = b.mean(axis=0)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

@tool
def elaborate_prompt(prompt: str) -> str:
    """
    Takes a short user prompt like 'cat' or 'thunder' and expands it into
    a detailed, descriptive sentence for better audio search and generation accuracy.

    Args:
        prompt: The short user input describing a sound (e.g. 'cat', 'rain', 'thunder').

    Returns:
        A detailed description of the sound as a string.
    """
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    instruction = (
        f"[INST] You are a sound design expert. Expand this into one vivid, descriptive sentence "
        f"describing the sound for audio generation. Be specific about texture, environment, and intensity. "
        f"Only output the description, nothing else.\n\nSound: {prompt} [/INST]"
    )
    for attempt in range(3):
        response = requests.post(
            api_url,
            headers=HEADERS,
            json={"inputs": instruction, "parameters": {"max_new_tokens": 80, "temperature": 0.7}},
        )
        if response.status_code == 503:
            time.sleep(25)
            continue
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                text = result[0]["generated_text"]
                if "[/INST]" in text:
                    text = text.split("[/INST]")[-1].strip()
                return text.strip()
    return f"a realistic sound of {prompt}"

@tool
def search_audio(elaborated_prompt: str) -> str:
    """
    Searches a database of existing audio clips using semantic similarity.
    Returns a URL to a matching audio clip if similarity is above threshold, otherwise returns 'NO_MATCH'.

    Args:
        elaborated_prompt: A detailed description of the sound to search for.

    Returns:
        A URL string if a match is found, or 'NO_MATCH' if no close match exists.
    """
    query_embedding = get_embedding(elaborated_prompt)
    if not query_embedding:
        return "NO_MATCH"

    best_score = 0.0
    best_url = None

    for item in SOUND_DB:
        db_embedding = get_embedding(item["description"])
        if not db_embedding:
            continue
        score = cosine_similarity(query_embedding, db_embedding)
        if score > best_score:
            best_score = score
            best_url = item["url"]

    if best_score >= 0.75:
        return best_url
    return "NO_MATCH"

@tool
def generate_audio(elaborated_prompt: str) -> str:
    """
    Generates a new audio clip from scratch using the AudioLDM2 model via HuggingFace Inference API.
    Saves the result as a .wav file and returns the file path.

    Args:
        elaborated_prompt: A detailed description of the sound to generate.

    Returns:
        The file path to the generated .wav audio file as a string.
    """
    api_url = "https://api-inference.huggingface.co/models/cvssp/audioldm2"
    for attempt in range(4):
        response = requests.post(
            api_url,
            headers=HEADERS,
            json={"inputs": elaborated_prompt},
            timeout=120,
        )
        if response.status_code == 503:
            wait = 30 * (attempt + 1)
            time.sleep(wait)
            continue
        if response.status_code == 200:
            output_path = "generated_audio.wav"
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
    return "ERROR: Audio generation failed after retries"
