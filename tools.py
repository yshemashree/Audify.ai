import os
import time
import struct
import math
import random
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

def write_wav(path: str, samples: list, sample_rate: int = 44100):
    num_samples = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + num_samples * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", 1))   # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))   # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", num_samples * 2))
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            f.write(struct.pack("<h", int(clamped * 32767)))

def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def generate_rain(duration=5, sr=44100):
    n = duration * sr
    # White noise with slight low-pass via running average
    samples = []
    prev = 0.0
    for _ in range(n):
        w = random.uniform(-1.0, 1.0)
        s = 0.3 * w + 0.7 * prev
        prev = s
        samples.append(clamp(s * 0.8))
    return samples

def generate_thunder(duration=5, sr=44100):
    n = duration * sr
    samples = []
    # Low rumble: sum of low-frequency sine waves + decaying noise burst
    for i in range(n):
        t = i / sr
        rumble = (math.sin(2 * math.pi * 40 * t) * 0.3 +
                  math.sin(2 * math.pi * 60 * t) * 0.2 +
                  math.sin(2 * math.pi * 80 * t) * 0.1)
        noise = random.uniform(-1.0, 1.0) * 0.4
        envelope = math.exp(-t * 0.8)
        samples.append(clamp((rumble + noise) * envelope))
    return samples

def generate_fire(duration=5, sr=44100):
    n = duration * sr
    samples = []
    # Brown noise (integrated white noise)
    prev = 0.0
    for _ in range(n):
        w = random.uniform(-0.02, 0.02)
        prev = clamp(prev + w, -1.0, 1.0)
        samples.append(prev * 0.7)
    return samples

def generate_wind(duration=5, sr=44100):
    n = duration * sr
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / sr
        w = random.uniform(-1.0, 1.0)
        s = 0.05 * w + 0.95 * prev
        prev = s
        # Slow amplitude modulation for gusting
        mod = 0.5 + 0.5 * math.sin(2 * math.pi * 0.3 * t)
        samples.append(clamp(s * mod * 0.9))
    return samples

def generate_ocean(duration=5, sr=44100):
    n = duration * sr
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / sr
        w = random.uniform(-1.0, 1.0)
        s = 0.1 * w + 0.9 * prev
        prev = s
        # Wave envelope
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * 0.15 * t)
        samples.append(clamp(s * wave * 0.8))
    return samples

def generate_heartbeat(duration=5, sr=44100):
    samples = [0.0] * (duration * sr)
    bpm = 72
    beat_interval = int(sr * 60 / bpm)
    for beat_start in range(0, len(samples), beat_interval):
        for i, offset in enumerate([0, int(sr * 0.15)]):
            pos = beat_start + offset
            for j in range(int(sr * 0.08)):
                if pos + j < len(samples):
                    t = j / sr
                    env = math.exp(-t * 40)
                    samples[pos + j] += clamp(math.sin(2 * math.pi * 80 * t) * env * 0.9)
    return [clamp(s) for s in samples]

def generate_tone(duration=5, sr=44100, freq=440):
    n = duration * sr
    return [math.sin(2 * math.pi * freq * i / sr) * 0.6 for i in range(n)]

def generate_noise(duration=5, sr=44100):
    return [random.uniform(-0.5, 0.5) for _ in range(duration * sr)]

SOUND_RECIPES = {
    "rain": generate_rain,
    "thunder": generate_thunder,
    "storm": generate_thunder,
    "fire": generate_fire,
    "campfire": generate_fire,
    "wind": generate_wind,
    "ocean": generate_ocean,
    "wave": generate_ocean,
    "sea": generate_ocean,
    "heartbeat": generate_heartbeat,
    "heart": generate_heartbeat,
    "pulse": generate_heartbeat,
}

def generate_local_audio(prompt: str, path: str = "generated_audio.wav") -> str:
    prompt_lower = prompt.lower()
    fn = None
    for keyword, recipe in SOUND_RECIPES.items():
        if keyword in prompt_lower:
            fn = recipe
            break
    if fn is None:
        fn = generate_noise
    samples = fn()
    write_wav(path, samples)
    return path

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
        "storm": "a raging thunderstorm with heavy rain and thunder",
        "ocean": "ocean waves crashing gently on a sandy beach",
        "dog": "a dog barking loudly outside",
        "birds": "birds chirping in a forest at dawn",
        "fire": "fire crackling in a cozy fireplace",
        "campfire": "a campfire crackling under a night sky",
        "clock": "a clock ticking quietly in a silent room",
        "wind": "wind blowing softly through trees",
        "keyboard": "keyboard typing rapidly",
        "footsteps": "footsteps walking on a wooden floor",
        "car": "a car engine revving and driving away",
        "water": "water flowing gently in a stream",
        "crowd": "a crowd of people talking in a busy place",
        "heartbeat": "a steady human heartbeat",
        "wave": "waves rolling onto a sandy shore",
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
    Generates a new audio clip from scratch using procedural synthesis.
    Saves the result as a .wav file and returns the file path.

    Args:
        elaborated_prompt: A detailed description of the sound to generate.

    Returns:
        The file path to the generated .wav audio file as a string.
    """
    output_path = "generated_audio.wav"
    generate_local_audio(elaborated_prompt, output_path)
    return output_path
