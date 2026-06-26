import os
import json
import time
import requests
from dotenv import load_dotenv
from smolagents import tool

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

SOUND_DB = []

def keyword_similarity(query: str, description: str) -> float:
    query_words = set(query.lower().split())
    desc_words = set(description.lower().split())
    if not query_words or not desc_words:
        return 0.0
    intersection = query_words & desc_words
    return len(intersection) / len(query_words | desc_words)

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
    expansions = {
        "cat": "a cat meowing softly indoors",
        "rain": "heavy rain falling on a rooftop",
        "thunder": "thunder rumbling in a distant storm",
        "ocean": "ocean waves crashing gently on a sandy beach",
        "dog": "a dog barking loudly outside",
        "birds": "birds chirping in a forest at dawn",
        "fire": "fire crackling in a cozy fireplace",
        "clock": "a clock ticking quietly in a silent room",
        "wind": "wind blowing softly through trees",
        "keyboard": "keyboard typing rapidly",
        "footsteps": "footsteps walking on a wooden floor",
        "car": "a car engine revving and driving away",
        "water": "water flowing gently in a stream",
        "crowd": "a crowd of people talking in a busy place",
        "music": "ambient background music playing softly",
    }
    prompt_lower = prompt.lower().strip()
    for key, description in expansions.items():
        if key in prompt_lower:
            return description
    return f"a realistic, high-quality sound of {prompt}"

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
    best_score = 0.0
    best_url = None

    for item in SOUND_DB:
        score = keyword_similarity(elaborated_prompt, item["description"])
        if score > best_score:
            best_score = score
            best_url = item["url"]

    if best_score >= 0.3:
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
    api_url = "https://router.huggingface.co/hf-inference/models/cvssp/audioldm2"
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
