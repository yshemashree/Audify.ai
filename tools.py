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
FREESOUND_KEY = os.getenv("FREESOUND_API_KEY")

HF_API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

SOUND_PROMPT_SYSTEM = """You are a Hollywood sound effects designer writing prompts for ElevenLabs sound generation AI.

Given a user's sound request, output ONLY a single vivid prompt string — no explanation, no quotes, no extra text.

Rules:
- This is a CINEMATIC SOUND EFFECT — exaggerated, over-the-top, not a subtle field recording
- NONSTOP and CONTINUOUS — zero silence, zero gaps, zero pauses for the entire duration
- EXTREMELY LOUD and IN-YOUR-FACE — as if the sound is happening right next to you
- Layer multiple simultaneous elements so there is never a quiet moment
- Use extreme intensity words: roaring, blasting, thundering, hammering, screaming, pounding, crashing
- Keep it under 50 words
- Never say "sound of" — describe it directly

Examples:
User: cat → nonstop aggressive cat screaming and yowling, continuous overlapping loud meows, extreme close-mic, zero pauses, intense throughout entire duration
User: thunderstorm → nonstop roaring thunder blasting every second, torrential rain hammering all surfaces simultaneously, lightning crack repeating, extreme volume, zero silence throughout
User: dog → nonstop ferocious dog barking and growling continuously, overlapping loud aggressive barks, extreme close-mic, no gaps whatsoever, intense throughout
User: fire → nonstop roaring inferno, continuous wood exploding and crackling, extreme heat roar, overlapping pops and blasts, zero silence throughout
User: rain → nonstop torrential rain hammering hard surface, continuous loud impact overlapping with rushing water, extreme volume, zero gaps throughout"""

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
                    {"role": "user", "content": f"{prompt} — make it NONSTOP, EXTREMELY LOUD, CONTINUOUS, ZERO SILENCE throughout entire duration"},
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

CINEMATIC_SOUNDS = {
    "cat": "aggressive domestic cat screaming and yowling in distress, sharp high-pitched meows layered continuously, raspy throat vocalisation, close-mic dry indoor acoustic, fur bristling tension in every call, relentless overlapping cries with no pause",
    "kitten": "tiny kitten crying and mewing desperately, fragile high-pitched squeaky calls layered back to back, close-mic indoor, soft nasal texture with sharp upward pitch, continuous overlapping with zero gaps",
    "dog": "large aggressive dog barking and snarling ferociously, deep chest resonance on each bark, saliva-wet growl underneath, sharp explosive attack on every burst, close-mic outdoor dry air, overlapping continuous barks with zero silence",
    "puppy": "small puppy yelping and whining frantically, high-pitched sharp yaps layered continuously, soft wet nose texture in each call, close-mic indoor, overlapping cries with no breaks",
    "wolf": "lone wolf howling mournfully into open night air, long sustained vibrato note rising and falling, secondary wolves answering in distance, wind rustling underneath, deep chest resonance, howls layering continuously with haunting echo tail",
    "rain": "torrential rain hammering concrete and metal surfaces simultaneously, millions of dense high-frequency droplet impacts overlapping, deep rushing water pooling below, thunder rumbling underneath continuously, roaring white noise texture filling every frequency band",
    "thunder": "massive thunderclap cracking directly overhead with bone-shaking low frequency shockwave, heavy rain hammering all surfaces beneath, rolling deep rumble sustaining for seconds, lightning electrical sizzle preceding each crack, layered continuously with zero silence",
    "storm": "violent tropical thunderstorm at full intensity, torrential rain hammering every surface, repeated thunderclaps blasting overhead, howling gale-force wind screaming through trees, lightning electrical crack repeating, all layers simultaneously with overwhelming volume",
    "lightning": "sharp electrical lightning crack splitting air with instantaneous sizzle and boom, thunder shockwave rolling deep and low immediately after, rain intensifying beneath, repeated strikes every few seconds, electromagnetic crackle between strikes",
    "fire": "raging wood fire roaring with intense heat, dry timber exploding and crackling continuously, high-frequency spark pops layering over deep combustion roar, air rushing into flame creating wind-like whoosh, overlapping crackle and hiss with zero silence",
    "campfire": "crackling campfire with dry logs popping rhythmically, warm mid-frequency crackle texture layered over low combustion roar, occasional sharp wood snap, embers hissing, close-mic outdoor night air, continuous crackling with no gaps",
    "explosion": "massive detonation with instantaneous low-frequency pressure blast, shockwave rattling everything around it, debris and glass cascading, secondary smaller explosions overlapping, deep rolling reverb tail sustaining, dust and air movement hiss beneath",
    "wind": "powerful gale-force wind roaring through open landscape, deep low-frequency pressure wave alternating with high-pitched whistle through gaps, branches whipping and snapping, dust and debris rushing, sustained howling with gusts intensifying in waves",
    "ocean": "powerful ocean waves surging and crashing on rocky shore, deep low-frequency water mass impact, white foam hissing as wave retreats pulling pebbles, next wave building underneath, continuous rhythmic cycle of crash and retreat, salt spray mist sound",
    "wave": "enormous ocean wave rising and crashing with thunderous deep impact, tons of water colliding with shore, foam and turbulence roaring, undertow pulling back with gravelly texture, waves layering continuously one after another",
    "water": "powerful river rapids roaring over boulders, turbulent churning white water crashing and splashing, deep rushing flow underneath high-frequency splash, water impact on rocks creating staccato bursts layered over continuous roar",
    "waterfall": "massive waterfall thundering into pool below, enormous continuous water volume creating deep low-frequency roar, mist and spray hissing around impact zone, surrounding echo of water off rock walls, overwhelming continuous volume",
    "heartbeat": "powerful human heartbeat close-mic on chest, deep resonant double-thud of lub-dub rhythm at 72bpm, low-frequency cardiac muscle contraction, slight breath sound between beats, intimate body-close recording, continuous steady pounding throughout",
    "bird": "dawn chorus of dozens of bird species singing simultaneously, high-frequency melodic warbles and sharp chirps layering, near and distant birds creating depth, forest reverb underneath, continuous overlapping calls with no silence",
    "forest": "dense forest alive with sound, overlapping bird calls, wind through high canopy, insect drone, distant woodpecker tapping, leaves rustling, all layers simultaneously creating rich natural texture continuously",
    "keyboard": "rapid mechanical keyboard typing with satisfying clicky switches, crisp sharp plastic-on-spring click on downstroke, metallic ping on upstroke, fingers flying across keys creating dense rhythmic pattern, close-mic desk surface, continuous typing throughout",
    "clock": "antique mechanical clock ticking with precise sharp tick-tock, metal escapement clicking, spring tension in each beat, quiet room amplifying each strike, steady metronomic rhythm, close-mic dry acoustic, continuous throughout",
    "footstep": "heavy boots striking hardwood floor with sharp impact, hollow resonance beneath each step, heel-to-toe weight transfer sound, floor creaking slightly, rhythmic walking pace, close-mic dry indoor, continuous footsteps throughout",
    "car": "powerful car engine running at idle then revving hard, mechanical piston rhythm, exhaust rumble from tailpipe, turbo whine building under acceleration, metal vibration through chassis, close-mic engine bay, continuous mechanical sound",
    "engine": "heavy diesel engine running at full power, deep rhythmic cylinder firing, metal components vibrating, air intake roar, exhaust blasting, mechanical stress sounds layering continuously over deep low-frequency rumble",
    "train": "heavy freight train passing at speed, rhythmic metal wheels pounding track joints rapidly, diesel horn blasting long and loud, steel chassis vibrating, air brake hiss, Doppler pitch shift as train passes, continuous mechanical roar",
    "helicopter": "military helicopter at close range, powerful rotor blades chopping air with rhythmic thwop-thwop, turbine engine screaming, rotor wash creating wind pressure, metal fuselage vibrating, continuous overwhelming mechanical presence",
    "gun": "high-calibre pistol firing rapidly, sharp explosive crack with muzzle blast pressure wave, brass casing ejecting and hitting floor, slide racking mechanically, overlapping shots fired in quick succession, close-mic indoor with sharp acoustic snap",
    "glass": "thick glass pane shattering dramatically, instantaneous high-frequency crystal impact, thousands of shards cascading onto hard floor, secondary tinkling of smaller fragments settling, crisp sharp texture throughout, close-mic hard surface",
    "bell": "large bronze church bell struck hard, deep fundamental tone with rich harmonic overtones blooming outward, long sustained resonance with slow decay, vibrating metal texture, repeated strikes layering before previous ring fades",
    "piano": "grand piano played forcefully, heavy felt hammer striking steel strings, warm woody resonance from soundboard, sustain pedal held allowing harmonics to bloom, notes layering continuously, close-mic studio with full dynamic range",
    "guitar": "acoustic guitar strummed aggressively with pick, bright steel string attack, woody body resonance, palm muting creating percussive thud between chords, overlapping chord rings, close-mic with full string texture and fret noise",
    "drum": "full drum kit played at full intensity, tight snare crack, deep kick drum thud, hi-hat rapid sixteenth notes, crash cymbal washing over, all elements layering simultaneously, close-mic studio with full dynamic punch",
    "robot": "complex robotic machine in operation, hydraulic servo motors whirring and clicking, pneumatic hiss between movements, electronic processing beeps, metal joints articulating, cooling fan underneath, all mechanical sounds layering continuously",
    "siren": "emergency vehicle siren at full volume close range, alternating high-low wail sweeping rapidly, electronic horn cutting through, Doppler pitch variation as it approaches and passes, piercing and unavoidable",
    "alarm": "loud fire alarm blaring continuously, sharp electronic tone pulsing rapidly, high-frequency piercing beep pattern, overlapping with secondary alarm, strobe light clicking underneath, overwhelming and relentless",
    "snore": "extremely loud deep snoring from large adult, low rumbling nasal vibration on inhale, wet gurgling rattle, brief silence before explosive snort, rhythmic cycle repeating, close-mic bedroom quiet background amplifying every detail",
    "laugh": "deep genuine belly laughing building in intensity, breathy explosive ha-ha-ha rhythm, vocal cords vibrating with joy, gasping breath between waves, multiple people laughing together overlapping, contagious and continuous",
    "scream": "full-volume human scream of terror, raw vocal cords at maximum strain, high-pitched piercing tone with raspy edges, breath explosive before each scream, multiple overlapping screams layering, close-mic with zero reverb smear",
    "crowd": "massive stadium crowd at peak intensity, fifty thousand voices roaring simultaneously, individual shouts and chants emerging from mass sound, low-frequency pressure of collective human energy, continuous roar with waves of intensity",
    "city": "dense urban street at rush hour, multiple car engines layering, horns honking, bus diesel rumble, construction noise, pedestrians talking, distant sirens, all sounds simultaneously creating complex urban texture continuously",
    "lion": "adult male lion roaring at full power, enormous chest cavity creating deep sub-bass fundamental, raspy vocal texture on each roar, breath explosive before each blast, secondary growl underneath, roars repeating continuously with ferocious intensity",
    "tiger": "Bengal tiger snarling and roaring aggressively, deep chest resonance with raspy wet texture, teeth-baring growl building before each roar, explosive breath blast, close-mic with air movement from each vocalisation, continuous overlapping roars",
    "bear": "large grizzly bear growling and roaring, massive chest creating enormous low-frequency rumble, wet nasal texture on each growl, aggressive huffing between roars, claws on ground, continuous overlapping aggressive vocalisations",
    "elephant": "African elephant trumpeting at full volume, enormous trunk blast creating unique nasal honk with deep resonance, rumble vocalisation underneath through ground, multiple elephants responding, continuous trumpeting with zero pauses",
    "horse": "thoroughbred horse galloping at full speed, four hooves striking packed earth in rapid succession creating rhythmic thunder, heavy breathing and snorting, saddle leather creaking, continuous galloping rhythm with surrounding air movement",
    "cow": "large dairy cow mooing repeatedly, deep resonant nasal vocalisation with long sustain, secondary moo overlapping before first ends, barn acoustic underneath, close-mic with full low-frequency body resonance, continuous throughout",
    "pig": "large pig squealing and oinking in distress, high-pitched nasal squeal layering over deep oink, wet snout texture, rapid succession of calls overlapping, farm outdoor acoustic, continuous loud vocalisations throughout",
    "frog": "large pond full of bullfrogs croaking at night, deep resonant croak from multiple frogs layering, high-pitched tree frogs adding texture above, water surface reverb, continuous overlapping chorus with no pauses",
    "snake": "large snake hissing with sustained air expulsion through forked tongue, dry raspy texture, intermittent rattle from rattlesnake layering over hiss, close-mic with full breath texture, continuous aggressive hissing throughout",
    "bee": "enormous angry bee swarm buzzing at full intensity, thousands of wings beating simultaneously creating dense mid-frequency drone, pitch varying as swarm moves, individual bee close-mic over mass drone, continuous overwhelming buzz",
    "insect": "dense summer insect chorus, cicadas screaming at peak volume, crickets rhythmically chirping underneath, mosquito whine close-mic, beetle and moth wing beats, all layering simultaneously in continuous overwhelming natural texture",
    "monkey": "troop of howler monkeys at full cry, enormous low-frequency howl amplified by throat sac, multiple monkeys overlapping creating wall of sound, tree movement and branches cracking underneath, continuous overlapping howling throughout",
    "eagle": "bald eagle screaming with sharp piercing high-frequency cry, powerful wing beats creating air movement, repeated cries layering continuously, open sky outdoor acoustic, raw and penetrating call with no reverb smear",
    "crow": "large group of crows cawing aggressively, harsh raspy caw texture, rapid overlapping calls from multiple birds, wing beats between calls, close-mic outdoor dry air, continuous overlapping aggressive cawing throughout",
    "rooster": "rooster crowing at maximum volume, bright resonant cock-a-doodle-doo with sharp attack and long sustain, nasal texture on each crow, multiple crows overlapping before previous ends, close-mic outdoor morning air, continuous throughout",
    "duck": "large group of ducks quacking loudly and continuously, nasal honking quack texture, multiple ducks overlapping simultaneously, water splashing underneath, close-mic outdoor wet acoustic, relentless overlapping quacking throughout",
    "donkey": "donkey braying with full dramatic hee-haw, enormous nasal intake on hee followed by explosive honking haw, multiple brays overlapping, close-mic outdoor, ridiculous volume and texture, continuous overlapping braying throughout",
    "moan": "deep human moaning in pain or agony, low guttural vocal tone sustained, raspy breath texture, repeated moans layering continuously, close-mic dry acoustic, raw emotional intensity throughout",
    "groan": "deep human groaning in pain, low sustained guttural vocalisation, strained breath between groans, repeated overlapping, close-mic dry acoustic, continuous throughout",
    "wind howl": "howling wind through narrow gaps, high-pitched eerie whistle layering over deep pressure roar, gusts intensifying and fading rhythmically, outdoor exposed location, continuous haunting howl",
    "chainsaw": "petrol chainsaw running at full throttle, high-frequency blade screaming, two-stroke engine rattling, cutting through wood creating pitch variations, sawdust and mechanical noise, continuous aggressive mechanical roar",
    "drill": "electric power drill running continuously, high-pitched motor whine, bit spinning creating vibration, drilling into material adding crunch texture, close-mic, continuous mechanical noise throughout",
    "construction": "loud construction site, jackhammer pounding concrete, drills whirring, heavy machinery beeping reversing, workers shouting, metal clanging, all layering simultaneously in continuous industrial noise",
    "crowd cheer": "massive crowd erupting in cheer, tens of thousands of voices roaring simultaneously, clapping and stomping layering underneath, individual shouts emerging, continuous wave of human energy",
    "baby cry": "newborn baby crying desperately, high-pitched piercing wail, gasping breath before each cry, nasal quality, continuous overlapping cries with no pauses, close-mic indoor",
    "heartbeat fast": "racing heartbeat pounding rapidly, strong double-thud at 140bpm, adrenaline intensity, low-frequency cardiac muscle, close-mic chest, continuous fast pounding throughout",
    "ocean storm": "ocean in full storm, massive waves crashing violently, howling wind over water, spray and foam roaring, deep water movement underneath, overwhelming continuous force throughout",
    "female": "young woman speaking and laughing naturally, clear bright vocal tone, warm close-mic indoor recording, conversational speech with natural breath, continuous talking throughout",
    "woman": "young adult woman talking and laughing, clear bright female vocal tone, close-mic dry indoor, natural conversational speech, continuous voice throughout",
    "girl": "young girl talking and giggling, high bright vocal tone, natural speech rhythm, close-mic indoor, cheerful continuous vocalisation throughout",
    "female voice": "young woman speaking clearly, bright warm female vocal tone, close-mic studio quality, natural speech cadence, continuous clear voice throughout",
    "woman voice": "young adult woman speaking naturally, clear bright vocal tone, close-mic dry acoustic, warm conversational delivery, continuous throughout",
    "female laugh": "young woman laughing genuinely, bright high-pitched laughter, breathless giggles layering, close-mic indoor, warm and natural, continuous laughter throughout",
    "female scream": "young woman screaming in shock, high-pitched piercing scream, sharp attack, close-mic, raw and immediate, repeated screams overlapping throughout",
    "male": "adult man speaking with deep clear voice, warm baritone tone, close-mic dry indoor, natural conversational speech, continuous voice throughout",
    "man": "adult male speaking naturally, deep warm baritone, close-mic studio quality, natural speech rhythm, continuous throughout",
    "male voice": "adult man speaking clearly, deep resonant baritone, close-mic dry acoustic, authoritative tone, continuous clear voice throughout",
    "volcano": "massive volcano erupting violently, enormous low-frequency ground-shaking explosion, molten lava hissing and crackling, rocks and debris blasting through air, ash cloud roaring, earth rumbling continuously beneath, overlapping explosions with zero silence",
    "volcano eruption": "catastrophic volcanic eruption, earth-shattering explosion blasting from crater, deep seismic rumble shaking ground, superheated lava hissing on rock, pyroclastic debris raining down, continuous roaring explosion with ash cloud noise throughout",
    "earthquake": "violent earthquake shaking everything, deep low-frequency ground rumble, buildings creaking and cracking, glass shattering, objects falling and crashing, continuous seismic roar with zero pauses throughout",
    "avalanche": "massive snow avalanche thundering down mountain, millions of tons of snow roaring, trees snapping underneath, ground shaking, deep continuous roar building in intensity, debris and snow crashing continuously",
    "tsunami": "enormous tsunami wave roaring toward shore, massive water wall crashing, buildings and structures collapsing underneath, deep thunderous water roar, continuous devastating impact throughout",
    "tornado": "violent tornado roaring at full intensity, freight-train-like continuous roar, debris spinning and crashing, windows shattering, structures collapsing, howling wind vortex, overwhelming continuous destruction",
    "hurricane": "category five hurricane at full force, howling wind screaming at 200mph, rain hammering horizontally, structures groaning and collapsing, continuous overwhelming roar throughout",
    "meteor": "massive meteor impact explosion, earth-shattering shockwave blasting outward, ground shaking violently, fire and debris roaring, continuous catastrophic explosion throughout",
    "spaceship": "massive spaceship launching, enormous rocket engines blasting at full thrust, deep low-frequency ground shaking roar, steam and fire hissing, continuous overwhelming rocket noise throughout",
    "rocket": "rocket engine at full thrust, enormous deep low-frequency combustion roar, supersonic air being torn apart, ground shaking from acoustic pressure, continuous overwhelming blast throughout",
    "waterfall": "massive waterfall thundering into pool below, enormous continuous water volume creating deep roar, mist and spray hissing, surrounding rock echo amplifying, overwhelming continuous volume",
    "jungle": "dense jungle alive with continuous sound, howler monkeys screaming, tropical birds calling, insects droning, rain on leaves, all layers simultaneously in overwhelming natural cacophony",
    "underwater": "deep underwater ambience, low-frequency water pressure hum, distant whale song resonating, bubbles rising, muffled current movement, continuous deep oceanic atmosphere throughout",
    "space": "deep space ambience, low-frequency cosmic hum, distant pulsar rhythm, electromagnetic interference crackle, vast empty resonance, continuous eerie space atmosphere throughout",
    "battle": "intense military battle, continuous gunfire from multiple weapons, explosions blasting nearby, helicopter overhead, soldiers shouting, radio static, all layering simultaneously throughout",
    "crowd riot": "violent crowd riot, thousands screaming and shouting, glass shattering, explosions from teargas, continuous chaos noise overlapping throughout",
    "subway": "underground subway train arriving, screeching metal brakes on rails, wind rushing through tunnel, doors opening with pneumatic hiss, crowd noise, continuous urban underground noise",
    "factory": "industrial factory floor at full production, heavy machinery pounding rhythmically, metal pressing and stamping, conveyor belts humming, steam releasing, all layering in continuous industrial roar",
    "chainsaw forest": "chainsaw cutting through dense forest, engine screaming, wood splintering, massive tree cracking and crashing to ground, birds scattering, continuous aggressive mechanical and natural noise",
}

def elaborate_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower().strip()

    # Check cinematic library — longest key match first to avoid "male" matching inside "female"
    for key, description in sorted(CINEMATIC_SOUNDS.items(), key=lambda x: -len(x[0])):
        if key in prompt_lower:
            return description

    # LLM for unknown sounds
    result = _llm_elaborate(prompt)
    if result:
        return result

    return f"nonstop {prompt} sound blasting continuously, overlapping loud {prompt} sounds, extreme close-mic, extreme volume, zero silence, zero gaps throughout entire duration"

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

def freesound_search(prompt: str) -> str | None:
    """Search Freesound for a matching cinematic sound effect and download it."""
    if not FREESOUND_KEY:
        return None
    try:
        response = requests.get(
            "https://freesound.org/apiv2/search/text/",
            params={
                "query": prompt,
                "token": FREESOUND_KEY,
                "fields": "name,previews,duration",
                "filter": "duration:[12 TO 60]",
                "sort": "rating_desc",
                "page_size": 1,
            },
            timeout=10,
        )
        if response.status_code != 200:
            return None
        results = response.json().get("results", [])
        if not results:
            return None
        preview_url = results[0]["previews"]["preview-hq-mp3"]
        print(f"[Freesound] Match: {results[0]['name']}")
        audio = requests.get(preview_url, timeout=15)
        if audio.status_code == 200:
            path = "generated_audio.mp3"
            with open(path, "wb") as f:
                f.write(audio.content)
            return path
    except Exception as e:
        print(f"[Freesound] Error: {e}")
    return None


def generate_audio(elaborated_prompt: str) -> str:
    """
    Three-tier audio pipeline:
    1. Freesound — real professional cinematic sound effects
    2. ElevenLabs — AI sound generation
    3. Procedural fallback — pure Python synthesis
    """
    output_path = "generated_audio.wav"

    # Tier 1 — Freesound professional sound effects
    freesound_result = freesound_search(elaborated_prompt)
    if freesound_result:
        return freesound_result

    if ELEVENLABS_KEY:
        try:
            response = requests.post(
                "https://api.elevenlabs.io/v1/sound-generation",
                headers={
                    "xi-api-key": ELEVENLABS_KEY,
                    "Content-Type": "application/json",
                },
                json={"text": elaborated_prompt, "duration_seconds": 15.0, "prompt_influence": 0.9},
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
