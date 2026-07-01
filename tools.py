import os
import time
import struct
import math
import random
import requests
from dotenv import load_dotenv
from database import vector_search

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")

HF_API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

SOUND_PROMPT_SYSTEM = """You are an expert sound designer and audio engineer writing prompts for ElevenLabs sound generation AI.

Given a user's sound request, output ONLY a single vivid prompt string — no explanation, no quotes, no extra text.

Rules:
- Describe the sound in concrete acoustic terms: texture, pitch, rhythm, space, distance, material, intensity
- Include acoustic environment (indoors, outdoors, reverb, echo, close-mic, distant)
- Mention layered elements if relevant (e.g. background ambience + foreground event)
- Keep it under 40 words
- Never say "sound of" — describe it directly
- Make it specific enough that an AI can synthesize it accurately

Examples:
User: cat → soft indoor meow, slight upward pitch inflection, close-mic, domestic shorthair, light room reverb
User: thunderstorm → deep rolling thunder crack overhead, heavy rain on leaves, distant rumble tail, wide outdoor reverb
User: robot booting up → rising electronic hum, servo whir, three ascending beep tones, industrial hiss, metallic resonance"""

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

def generate_cat(duration=4, sr=44100):
    samples = [0.0] * (duration * sr)
    # Two meow calls: rising then falling frequency sweep
    for meow_start in [int(sr * 0.3), int(sr * 2.0)]:
        meow_len = int(sr * 0.8)
        for i in range(meow_len):
            t = i / sr
            # Frequency sweep 600Hz -> 1200Hz -> 800Hz
            progress = i / meow_len
            if progress < 0.5:
                freq = 600 + 1200 * progress
            else:
                freq = 1800 - 1000 * progress
            env = math.sin(math.pi * progress) ** 0.5
            phase = 2 * math.pi * freq * t
            s = (math.sin(phase) * 0.5 + math.sin(2 * phase) * 0.2 + math.sin(3 * phase) * 0.1)
            pos = meow_start + i
            if pos < len(samples):
                samples[pos] += clamp(s * env * 0.7)
    return [clamp(s) for s in samples]

def generate_dog(duration=4, sr=44100):
    samples = [0.0] * (duration * sr)
    bark_times = [int(sr * 0.2), int(sr * 1.0), int(sr * 1.8)]
    for bark_start in bark_times:
        bark_len = int(sr * 0.3)
        for i in range(bark_len):
            t = i / sr
            progress = i / bark_len
            freq = 280 - 80 * progress
            env = math.exp(-progress * 6) * (1 - math.exp(-progress * 30))
            noise = random.uniform(-0.3, 0.3)
            s = math.sin(2 * math.pi * freq * t) * 0.6 + noise
            pos = bark_start + i
            if pos < len(samples):
                samples[pos] += clamp(s * env * 0.8)
    return [clamp(s) for s in samples]

def generate_bird(duration=4, sr=44100):
    samples = [0.0] * (duration * sr)
    chirp_times = [int(sr * t) for t in [0.1, 0.5, 0.9, 1.4, 1.8, 2.3, 2.7, 3.1, 3.5]]
    for start in chirp_times:
        chirp_len = int(sr * 0.12)
        base_freq = random.choice([2800, 3200, 3600, 4000])
        for i in range(chirp_len):
            t = i / sr
            progress = i / chirp_len
            freq = base_freq + 800 * math.sin(math.pi * progress)
            env = math.sin(math.pi * progress) ** 2
            s = math.sin(2 * math.pi * freq * t)
            pos = start + i
            if pos < len(samples):
                samples[pos] += clamp(s * env * 0.6)
    return [clamp(s) for s in samples]

def generate_keyboard(duration=4, sr=44100):
    samples = [0.0] * (duration * sr)
    # Random keystrokes at ~5 per second
    click_interval = sr // 5
    for start in range(0, len(samples) - sr, click_interval + random.randint(-sr//20, sr//20)):
        click_len = int(sr * 0.015)
        for i in range(click_len):
            progress = i / click_len
            env = math.exp(-progress * 80)
            freq = random.choice([3000, 4000, 5000])
            s = math.sin(2 * math.pi * freq * i / sr) + random.uniform(-0.2, 0.2)
            pos = start + i
            if pos < len(samples):
                samples[pos] += clamp(s * env * 0.4)
    return [clamp(s) for s in samples]

def generate_clock(duration=5, sr=44100):
    samples = [0.0] * (duration * sr)
    tick_interval = sr  # 1 tick per second
    for start in range(0, len(samples), tick_interval):
        tick_len = int(sr * 0.02)
        for i in range(tick_len):
            progress = i / tick_len
            env = math.exp(-progress * 100)
            s = math.sin(2 * math.pi * 2000 * i / sr)
            pos = start + i
            if pos < len(samples):
                samples[pos] += clamp(s * env * 0.7)
    return [clamp(s) for s in samples]

def generate_footsteps(duration=5, sr=44100):
    samples = [0.0] * (duration * sr)
    step_interval = int(sr * 0.55)
    for start in range(0, len(samples) - sr, step_interval):
        step_len = int(sr * 0.08)
        for i in range(step_len):
            progress = i / step_len
            env = math.exp(-progress * 30)
            noise = random.uniform(-1.0, 1.0)
            thud = math.sin(2 * math.pi * 120 * i / sr)
            s = thud * 0.6 + noise * 0.4
            pos = start + i
            if pos < len(samples):
                samples[pos] += clamp(s * env * 0.8)
    return [clamp(s) for s in samples]

def generate_car(duration=5, sr=44100):
    samples = []
    for i in range(duration * sr):
        t = i / sr
        # Engine: rising RPM then settling
        rpm_freq = 80 + 120 * min(t / 2.0, 1.0) * math.exp(-t * 0.3)
        engine = (math.sin(2 * math.pi * rpm_freq * t) * 0.4 +
                  math.sin(2 * math.pi * rpm_freq * 2 * t) * 0.2 +
                  random.uniform(-0.1, 0.1))
        samples.append(clamp(engine * 0.7))
    return samples

def generate_water(duration=5, sr=44100):
    n = duration * sr
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / sr
        w = random.uniform(-1.0, 1.0)
        s = 0.2 * w + 0.8 * prev
        prev = s
        ripple = 0.7 + 0.3 * math.sin(2 * math.pi * 0.8 * t)
        samples.append(clamp(s * ripple * 0.75))
    return samples

def generate_crowd(duration=5, sr=44100):
    n = duration * sr
    samples = []
    # Multiple noise sources at speech frequencies
    prev1, prev2, prev3 = 0.0, 0.0, 0.0
    for i in range(n):
        t = i / sr
        w = random.uniform(-1.0, 1.0)
        prev1 = 0.4 * w + 0.6 * prev1
        prev2 = 0.35 * w + 0.65 * prev2
        prev3 = 0.3 * w + 0.7 * prev3
        s = (prev1 * 0.4 + prev2 * 0.3 + prev3 * 0.3)
        mod = 0.6 + 0.4 * math.sin(2 * math.pi * 0.2 * t + random.uniform(0, 0.1))
        samples.append(clamp(s * mod * 0.7))
    return samples

def generate_noise(duration=5, sr=44100):
    return [random.uniform(-0.4, 0.4) for _ in range(duration * sr)]

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
    "cat": generate_cat,
    "meow": generate_cat,
    "dog": generate_dog,
    "bark": generate_dog,
    "bird": generate_bird,
    "chirp": generate_bird,
    "keyboard": generate_keyboard,
    "typing": generate_keyboard,
    "clock": generate_clock,
    "tick": generate_clock,
    "footstep": generate_footsteps,
    "walking": generate_footsteps,
    "step": generate_footsteps,
    "car": generate_car,
    "engine": generate_car,
    "water": generate_water,
    "stream": generate_water,
    "crowd": generate_crowd,
    "people": generate_crowd,
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
    samples = fn(duration=15)
    write_wav(path, samples)
    return path

def _llm_elaborate(prompt: str) -> str:
    """Call Qwen via HF router to generate a precise ElevenLabs sound prompt."""
    if not HF_TOKEN:
        return None
    try:
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={
                "model": HF_MODEL,
                "messages": [
                    {"role": "system", "content": SOUND_PROMPT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 80,
                "temperature": 0.7,
            },
            timeout=15,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None

def elaborate_prompt(prompt: str) -> str:
    """
    Takes a short user prompt and expands it into a detailed acoustic description
    optimised for ElevenLabs sound generation.

    Args:
        prompt: The short user input describing a sound (e.g. 'cat', 'rain', 'thunder').

    Returns:
        A detailed acoustic description of the sound as a string.
    """
    # Try LLM first
    result = _llm_elaborate(prompt)
    if result:
        return result

    # Keyword fallback if no HF token or API failure
    expansions = {
        "cat": "soft indoor meow, slight upward pitch, close-mic, light reverb",
        "rain": "heavy rain on a rooftop, steady white-noise wash, occasional drip",
        "thunder": "deep rolling thunder crack, heavy rain on leaves, wide outdoor reverb",
        "storm": "violent thunderstorm, lightning crack, torrential rain, howling wind",
        "ocean": "slow waves breaking on sand, foam hiss, seagull distant, open air",
        "dog": "sharp medium-dog bark, outdoor echo, three bursts with short pauses",
        "birds": "dawn chorus, multiple bird species chirping, forest reverb, gentle",
        "fire": "dry wood crackling, occasional pop, warm low rumble, close-mic",
        "campfire": "campfire crackling under open sky, wood pop, night ambience",
        "clock": "mechanical clock ticking, quiet room, steady 1Hz rhythm, close-mic",
        "wind": "gusty wind through tree leaves, low whistle, outdoor open field",
        "keyboard": "rapid mechanical keyboard typing, clicks and clacks, office room",
        "footsteps": "footsteps on hardwood floor, steady walking pace, slight echo",
        "car": "car engine cold start, rev, drive away fade, outdoor reverb",
        "water": "water stream over rocks, gentle gurgle, outdoor natural reverb",
        "crowd": "busy indoor crowd murmur, overlapping voices, restaurant ambience",
        "heartbeat": "steady human heartbeat 72bpm, close-mic, two-thud rhythm",
        "wave": "ocean wave rolling onto shore, foam hiss, receding water pull",
    }
    prompt_lower = prompt.lower().strip()
    for key, description in expansions.items():
        if key in prompt_lower:
            return description
    return f"realistic high-fidelity sound of {prompt}, natural acoustic environment, clear recording"

def search_audio(elaborated_prompt: str) -> str:
    """
    Searches the ChromaDB vector database for a semantically similar expert sound description.
    Returns the matched description if cosine distance is below threshold, else 'NO_MATCH'.
    On match, the matched expert description is used for generation instead of the raw elaboration.
    """
    try:
        match = vector_search(elaborated_prompt)
        if match:
            description, label, distance = match
            print(f"[ChromaDB] Match: '{label}' (distance={distance:.3f})")
            return description
    except Exception as e:
        print(f"[ChromaDB] Search error: {e}")
    return "NO_MATCH"

def generate_audio(elaborated_prompt: str) -> str:
    """
    Generates a new audio clip from scratch. Tries ElevenLabs sound generation
    first (any sound, AI quality), falls back to local procedural synthesis.
    Saves the result as a .wav file and returns the file path.

    Args:
        elaborated_prompt: A detailed description of the sound to generate.

    Returns:
        The file path to the generated .wav audio file as a string.
    """
    output_path = "generated_audio.wav"

    if ELEVENLABS_KEY:
        try:
            response = requests.post(
                "https://api.elevenlabs.io/v1/sound-generation",
                headers={
                    "xi-api-key": ELEVENLABS_KEY,
                    "Content-Type": "application/json",
                },
                json={"text": elaborated_prompt, "duration_seconds": 22.0, "prompt_influence": 0.5},
                timeout=30,
            )
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return output_path
        except Exception:
            pass

    generate_local_audio(elaborated_prompt, output_path)
    return output_path
