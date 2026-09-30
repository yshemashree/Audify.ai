"""
Built-in library of real recordings (sounds/, see sounds/CREDITS.md).

Works fully offline. For prompts it recognises it builds the sound from real
recordings: glass breaks from real shatters, glass cracks from micro-fragments
cut out of real shatters, animals from real calls, and scenes as looped
recorded beds with real events sprinkled over them. Online sources, when
available, are folded in as extra takes or layers.
"""
import json
import os
import re
import logging
import threading
import numpy as np

import mixer
import synth

log = logging.getLogger("audify")

SR = mixer.SR
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
TARGET = 12.0

_manifest = None
_cache: dict[str, list[np.ndarray]] = {}
_lock = threading.Lock()


def _load_manifest():
    global _manifest
    if _manifest is None:
        try:
            with open(os.path.join(ROOT, "manifest.json")) as f:
                _manifest = json.load(f)
        except OSError:
            log.warning("[Library] sounds/manifest.json missing — built-in recordings disabled")
            _manifest = {}
    return _manifest


def takes(category: str) -> list[np.ndarray]:
    """Decoded stereo takes for a category (cached)."""
    with _lock:
        if category not in _cache:
            out = []
            for rel in _load_manifest().get(category, {}).get("files", []):
                try:
                    with open(os.path.join(ROOT, rel), "rb") as f:
                        x = mixer.decode(f.read())
                    if x is not None:
                        out.append(x)
                except OSError:
                    pass
            _cache[category] = out
        return [t.copy() for t in _cache[category]]


def available() -> list[str]:
    return list(_load_manifest())


# ── building blocks ──────────────────────────────────────────────────────────

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def bed(categories: list[str], seconds: float = TARGET, rng=None) -> np.ndarray | None:
    """Continuous background from recordings: every take of each category is
    chained with crossfades (so the loop point isn't obvious), then the
    categories are layered, the first one leading."""
    rng = rng or np.random.default_rng()
    n = int(seconds * SR)
    layers = []
    for i, cat in enumerate(categories):
        tk = takes(cat)
        if not tk:
            continue
        rng.shuffle(tk)
        tk = [t / _rms(t) for t in tk]  # level-match takes so the joins are seamless
        chain = tk[0]
        for t in tk[1:]:
            k = min(int(0.4 * SR), len(chain) // 3, len(t) // 3)
            ramp = np.linspace(0, 1, k)[:, None]
            chain = np.concatenate([chain[:-k], chain[-k:] * (1 - ramp) + t[:k] * ramp, t[k:]])
        layer = mixer.fit_length(chain, n)
        layers.append(layer / _rms(layer) * (1.0 if i == 0 else 0.55))
    return sum(layers) if layers else None


def hits(category_takes: list[np.ndarray], gap=(0.7, 1.1), seconds: float = TARGET, rng=None) -> np.ndarray:
    return mixer.sequence([mixer.trim(t) for t in category_takes], gap=gap, target=seconds, rng=rng)


def sprinkle(base: np.ndarray, category_takes: list[np.ndarray], per_second: float, gain: float, rng) -> np.ndarray:
    """Drop real one-off events (a horn, a siren, a drip) over a bed."""
    out = base.copy()
    if not category_takes:
        return out
    n = len(out)
    level = _rms(base) * gain
    for _ in range(max(1, int(n / SR * per_second))):
        t = mixer.trim(category_takes[rng.integers(len(category_takes))])
        t = t / _rms(t) * level
        start = int(rng.uniform(0, max(1, n - len(t))))
        end = min(n, start + len(t))
        out[start:end] += t[: end - start]
    return out


def _grains(shatters: list[np.ndarray], rng, count: int) -> list[np.ndarray]:
    """Real glass micro-fragments: short windows around transients in the tails
    of real shatter recordings — the individual shard ticks and pings."""
    grains = []
    sources = [s for s in shatters if len(s) > int(0.3 * SR)]
    if not sources:
        return grains
    for _ in range(count * 4):
        s = sources[rng.integers(len(sources))]
        env = np.abs(s).max(axis=1)
        start_zone = int(0.08 * SR)  # skip the main impact, keep the shard detail
        if len(env) - start_zone < int(0.05 * SR):
            continue
        idx = start_zone + int(np.argmax(env[start_zone:] * (rng.random(len(env) - start_zone) ** 6)))
        length = int(rng.uniform(0.015, 0.07) * SR)
        g = s[max(0, idx - int(0.001 * SR)): idx + length].copy()
        if len(g) < 64:
            continue
        g *= np.exp(-np.linspace(0, 5, len(g)))[:, None]
        g[: 32] *= np.linspace(0, 1, 32)[:, None]
        grains.append(_norm(g))
        if len(grains) >= count:
            break
    return grains


def crack(shatters: list[np.ndarray], seconds: float = TARGET, finale: bool = False, rng=None) -> np.ndarray | None:
    """Glass under stress built from real shard fragments: spidering clusters of
    ticks that escalate; optionally ends on a real full shatter."""
    rng = rng or np.random.default_rng()
    grains = _grains(shatters, rng, 160)
    if not grains:
        return None
    n = int(seconds * SR)
    out = np.zeros((n, 2))
    # faint low stress-creak underneath (synthesised groan, very quiet)
    creak = synth.band_noise(n, 120, 600, rng) * synth.smooth_env(n, 3, rng, 1.0) ** 3 * np.linspace(0.2, 1, n)
    out += 0.04 * creak[:, None]
    finale_len = 0
    last = None
    if finale:
        last = mixer.trim(max(shatters, key=lambda s: _rms(s) * len(s) ** 0.3))
        finale_len = len(last) + int(0.3 * SR)
    stop = n - finale_len
    t = 0.25
    while t * SR < stop - int(0.3 * SR):
        progress = t * SR / max(stop, 1)
        cluster = int(rng.uniform(4, 16) * (0.5 + 1.5 * progress))
        span = rng.uniform(0.04, 0.22)
        pan = rng.uniform(-0.7, 0.7)
        for i in range(cluster):
            g = grains[rng.integers(len(grains))]
            s0 = int((t + span * (i / cluster) ** 0.7) * SR)
            if s0 + len(g) >= stop:
                break
            amp = rng.uniform(0.25, 0.8) * (0.35 + 0.65 * progress) * (1 - 0.5 * i / cluster)
            p = np.clip(pan + rng.uniform(-0.2, 0.2), -1, 1)
            lr = np.array([np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)]) * np.sqrt(2)
            out[s0:s0 + len(g)] += g * lr * amp
        t += rng.uniform(0.3, 0.9) * (1.2 - 0.8 * progress)
    if finale and last is not None:
        s0 = n - len(last) - int(0.1 * SR)
        out[s0:s0 + len(last)] += _norm(last) * 1.4
    return out


# ── prompt routing ───────────────────────────────────────────────────────────

# (keywords, recipe). First match wins, so specific phrases come first.
ROUTES = [
    (("rain on glass", "rain on window", "rain on the window", "rain on a window", "window rain"),
     {"kind": "ambience", "bed": ["rain"], "sprinkle": ("water_drops", 0.6, 0.5)}),
    (("glass cracks and shatters", "glass cracking and shattering", "glass crack and shatter",
      "glass cracks then shatters", "cracks and shatters", "crack and shatter", "cracking then shattering",
      "glass breaking slowly"),
     {"kind": "impact", "crack": True, "finale": True}),
    (("glass crack", "glass cracking", "cracking glass", "glass crackling", "crackling glass", "ice crack",
      "ice cracking", "cracking ice", "window crack", "screen crack"),
     {"kind": "impact", "crack": True}),
    (("glass", "shatter", "shatters", "shattering", "window break", "window smash", "bottle", "smash", "smashing"),
     {"kind": "impact", "hits": "glass"}),
    (("thunderstorm", "thunder", "lightning", "storm"), {"kind": "ambience", "bed": ["thunderstorm", "rain"]}),
    (("waterfall",), {"kind": "ambience", "bed": ["water", "rain"], "sprinkle": ("birds", 0.15, 0.35)}),
    (("ocean", "waves", "wave", "sea", "beach", "surf"), {"kind": "ambience", "bed": ["ocean"]}),
    (("campfire", "fire", "fireplace", "bonfire", "crackling"), {"kind": "ambience", "bed": ["fire"]}),
    (("city", "traffic", "street", "road", "highway"),
     {"kind": "ambience", "synth": "traffic", "bed": ["engine"], "sprinkle": ("car_horn", 0.25, 0.8)}),
    (("keyboard", "typing"), {"kind": "ambience", "bed": ["keyboard"]}),
    (("clock", "ticking"), {"kind": "ambience", "bed": ["clock"]}),
    (("footsteps", "footstep", "walking"), {"kind": "ambience", "bed": ["footsteps"]}),
    (("rain", "rainfall", "drizzle", "downpour", "raining"), {"kind": "ambience", "bed": ["rain"]}),
    (("wind", "breeze", "gust", "gale"), {"kind": "ambience", "bed": ["wind"]}),
    (("water", "stream", "river", "pouring", "creek", "brook"), {"kind": "ambience", "bed": ["water"]}),
    (("dripping", "drips", "drip", "water drops"), {"kind": "ambience", "bed": ["water_drops"]}),
    (("crickets", "cricket", "cicada", "cicadas"), {"kind": "ambience", "bed": ["crickets"]}),
    (("insects", "insect", "bugs", "bees", "bee", "buzzing"), {"kind": "ambience", "bed": ["insects"]}),
    (("dog", "dogs", "puppy", "bark", "barking", "woof"), {"kind": "vocal", "hits": "dog"}),
    (("cat", "cats", "kitten", "meow", "meowing"), {"kind": "vocal", "hits": "cat"}),
    (("rooster", "cock a doodle", "crowing rooster"), {"kind": "vocal", "hits": "rooster"}),
    (("cow", "cows", "moo", "mooing", "cattle"), {"kind": "vocal", "hits": "cow"}),
    (("pig", "pigs", "oink", "hog"), {"kind": "vocal", "hits": "pig"}),
    (("sheep", "lamb", "baa", "goat", "goats"), {"kind": "vocal", "hits": "sheep"}),
    (("chicken", "chickens", "hen", "hens", "cluck", "clucking"), {"kind": "vocal", "hits": "chicken"}),
    (("frog", "frogs", "toad", "croak", "croaking", "ribbit"), {"kind": "vocal", "hits": "frog"}),
    (("crow", "crows", "raven", "caw", "cawing"), {"kind": "vocal", "hits": "crow"}),
    (("birds", "bird", "chirping", "songbird", "forest"), {"kind": "ambience", "bed": ["birds"]}),
    (("laugh", "laughing", "laughter", "giggle"), {"kind": "vocal", "hits": "laughing"}),
    (("baby", "infant", "crying baby", "baby crying"), {"kind": "vocal", "hits": "baby"}),
    (("snore", "snoring"), {"kind": "vocal", "hits": "snoring"}),
    (("train", "railway", "locomotive"), {"kind": "ambience", "bed": ["train"]}),
    (("helicopter", "chopper"), {"kind": "ambience", "bed": ["helicopter"]}),
    (("airplane", "aeroplane", "plane", "jet"), {"kind": "ambience", "bed": ["airplane"]}),
    (("chainsaw",), {"kind": "ambience", "bed": ["chainsaw"]}),
    (("siren", "ambulance", "police"), {"kind": "ambience", "bed": ["siren"]}),
    (("applause", "clapping", "clap"), {"kind": "ambience", "bed": ["applause"]}),
    (("car horn", "horn", "honk", "honking"), {"kind": "impact", "hits": "car_horn"}),
    (("fireworks", "firework"), {"kind": "impact", "hits": "fireworks"}),
    (("bell", "bells", "church bell"), {"kind": "impact", "hits": "bells"}),
    (("knock", "knocking", "door knock"), {"kind": "impact", "hits": "knock"}),
    (("engine", "car", "motor"), {"kind": "ambience", "bed": ["engine"]}),
]


def _earliest(text: str, groups):
    """(position, -length, keyword, payload) of the first keyword found in text."""
    best = None
    for keywords, payload in groups:
        for kw in keywords:
            m = re.search(r"\b" + re.escape(kw) + r"s?\b", text)
            if m:
                cand = (m.start(), -len(kw), kw, payload)
                if best is None or cand[:2] < best[:2]:
                    best = cand
    return best


def route(prompt: str):
    """Recipe for the prompt's main subject — the sound named first ('wolf howling
    in the forest' is about the wolf). Returns (keyword, recipe) or (None, None)
    when the main subject isn't something the library has recordings of."""
    text = prompt.lower()
    lib = _earliest(text, ROUTES)
    if lib is None:
        return None, None
    syn = _earliest(text, [(kws, None) for kws, _ in synth.RECIPES])
    if syn is not None and syn[:2] < lib[:2] and _earliest(syn[2], ROUTES) is None:
        return None, None  # e.g. 'wolf ...', 'spaceship ...': not in the library
    return lib[2], lib[3]


def build(prompt: str, extra_takes=None, sweetener=None, rng=None):
    """
    Build the sound for `prompt` from real recordings.
    extra_takes: more real takes (e.g. from Freesound) to mix in.
    sweetener:   an AI take (ElevenLabs) layered under impacts / beds.
    Returns (signal, kind) or None if the library has nothing for this prompt.
    """
    rng = rng or np.random.default_rng()
    kw, r = route(prompt)
    if r is None:
        return None
    extra = [t for t in (extra_takes or []) if t is not None]
    kind = r["kind"]
    out = None

    if r.get("crack"):
        out = crack(takes("glass") + extra, finale=r.get("finale", False), rng=rng)
    elif "hits" in r:
        tk = takes(r["hits"])
        if not tk and not extra:
            return None
        pool = extra + tk if kind == "impact" else [x for pair in zip(tk, extra) for x in pair] + tk[len(extra):] + extra[len(tk):]
        pool = pool or tk
        if kind == "impact" and sweetener is not None:
            pool = [mixer.align_layer(t, mixer.trim(sweetener), 0.45) for t in pool]
        elif kind == "vocal" and sweetener is not None:
            pool = pool + [sweetener]
        gap = (0.7, 1.1) if kind == "impact" else (0.4, 0.9)
        out = hits(pool, gap=gap, rng=rng)
    elif "bed" in r:
        out = bed(r["bed"], rng=rng)
        if out is None:
            return None
        if "synth" in r:
            s = mixer.to_stereo(synth.render(r["synth"]))[: len(out)]
            out = out + s / _rms(s) * _rms(out) * 0.7
        if "sprinkle" in r:
            cat, rate, gain = r["sprinkle"]
            out = sprinkle(out, takes(cat), rate, gain, rng)
        for layer in extra[:1] + ([sweetener] if sweetener is not None else []):
            out = mixer.layer(out, layer)

    if out is None:
        return None
    log.info("[Library] '%s' matched '%s' (%s)", prompt, kw, kind)
    return out, kind
