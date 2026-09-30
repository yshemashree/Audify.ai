"""
Build the bundled offline sound library (sounds/) from the ESC-50 dataset.

ESC-50 is 2,000 real 5-second field recordings in 50 classes, originally from
Freesound (https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0; ESC-10 subset
CC BY). For each category we want, this script downloads the candidate clips,
scores them (clean noise floor, no clipping, full bandwidth, strong event for
one-shots / steady level for ambiences), keeps the best takes from distinct
source recordings, trims silence, and saves them as MP3 with a manifest and
per-clip credits.

Usage:  python scripts/build_sound_library.py [--candidates 20]
"""
import argparse
import csv
import io
import json
import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mixer  # noqa: E402

BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
OUT = os.path.join(ROOT, "sounds")
SR = 44100

# library name: (ESC-50 class, kind, takes to keep)
CATEGORIES = {
    "glass": ("glass_breaking", "impact", 7),
    "dog": ("dog", "vocal", 5),
    "cat": ("cat", "vocal", 5),
    "rooster": ("rooster", "vocal", 3),
    "cow": ("cow", "vocal", 3),
    "pig": ("pig", "vocal", 3),
    "sheep": ("sheep", "vocal", 3),
    "chicken": ("hen", "vocal", 3),
    "frog": ("frog", "vocal", 3),
    "crow": ("crow", "vocal", 3),
    "laughing": ("laughing", "vocal", 3),
    "baby": ("crying_baby", "vocal", 2),
    "snoring": ("snoring", "vocal", 2),
    "rain": ("rain", "ambience", 3),
    "ocean": ("sea_waves", "ambience", 3),
    "fire": ("crackling_fire", "ambience", 3),
    "thunderstorm": ("thunderstorm", "ambience", 4),
    "wind": ("wind", "ambience", 3),
    "water": ("pouring_water", "ambience", 3),
    "water drops": ("water_drops", "ambience", 2),
    "birds": ("chirping_birds", "ambience", 3),
    "crickets": ("crickets", "ambience", 2),
    "insects": ("insects", "ambience", 2),
    "keyboard": ("keyboard_typing", "ambience", 3),
    "clock": ("clock_tick", "ambience", 2),
    "footsteps": ("footsteps", "ambience", 3),
    "engine": ("engine", "ambience", 2),
    "train": ("train", "ambience", 2),
    "helicopter": ("helicopter", "ambience", 2),
    "airplane": ("airplane", "ambience", 2),
    "chainsaw": ("chainsaw", "ambience", 2),
    "siren": ("siren", "ambience", 2),
    "applause": ("clapping", "ambience", 2),
    "car horn": ("car_horn", "impact", 3),
    "fireworks": ("fireworks", "impact", 3),
    "bells": ("church_bells", "impact", 2),
    "knock": ("door_wood_knock", "impact", 3),
}


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x


def score(x: np.ndarray, kind: str, category: str) -> float:
    """Higher is better. Rewards clean, full-band, well-exposed recordings."""
    if np.max(np.abs(x)) < 1e-3:
        return -1e9
    frames = x[: len(x) // 2205 * 2205].reshape(-1, 2205)  # 50 ms frames
    lvl = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    peak_db = lvl.max()
    floor_db = np.percentile(lvl, 10)
    active = np.mean(lvl > peak_db - 20)
    clipped = np.mean(np.abs(x) > 0.995)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    hf = spec[f > 6000].sum() / (spec.sum() + 1e-12)       # bandwidth / crispness
    dyn = peak_db - floor_db                                # event stands out of the noise

    s = -400 * clipped + 3 * min(hf, 0.3) * 100
    if kind == "impact":
        s += 1.2 * min(dyn, 60) - 20 * max(0, active - 0.6)
    elif kind == "vocal":
        s += 1.0 * min(dyn, 50) + 10 * min(active, 0.5)
    else:  # ambience: steady and continuous
        s += 40 * active - 0.5 * max(0, dyn - 25)
    if category == "glass":
        s += 150 * min(hf, 0.35)  # crystal-clear top end matters most for glass
        # ringing shards show up as sharp spectral peaks; crunching/pouring glass is flat noise
        band = spec[(f > 3000) & (f < 16000)]
        band = np.convolve(band, np.ones(8) / 8, mode="same")
        tonal = np.log10(np.percentile(band, 99.5) / (np.median(band) + 1e-12))
        s += 60 * tonal
        s -= 60 * max(0, active - 0.35)  # a shatter is an event, not a 3-second wash
    return float(s)


def encode_mp3(x: np.ndarray, bitrate: int = 160) -> bytes:
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(SR)
    enc.set_channels(1)
    enc.set_quality(2)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    return bytes(enc.encode(pcm.tobytes()) + enc.flush())


def fetch(fname: str) -> bytes | None:
    for _ in range(3):
        try:
            r = requests.get(f"{BASE}/audio/{fname}", timeout=30)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=20, help="clips to audition per category (max 40)")
    args = ap.parse_args()

    rows = list(csv.DictReader(io.StringIO(requests.get(f"{BASE}/meta/esc50.csv", timeout=30).text)))
    credits_src = requests.get(f"{BASE}/LICENSE", timeout=30).text
    os.makedirs(OUT, exist_ok=True)
    manifest, credits = {}, []

    for name, (esc_class, kind, keep) in CATEGORIES.items():
        clips = [r for r in rows if r["category"] == esc_class]
        n_cand = 40 if name == "glass" else args.candidates
        # spread candidates across different source recordings first
        clips.sort(key=lambda r: (r["take"], r["src_file"]))
        clips = clips[:n_cand]
        with ThreadPoolExecutor(16) as pool:
            audio = list(pool.map(lambda r: fetch(r["filename"]), clips))

        scored = []
        for r, data in zip(clips, audio):
            if data:
                x = read_wav(data)
                scored.append((score(x, kind, name), r, x))
        scored.sort(key=lambda t: -t[0])

        chosen, used_src = [], set()
        for s, r, x in scored:
            if r["src_file"] in used_src:
                continue
            used_src.add(r["src_file"])
            chosen.append((s, r, x))
            if len(chosen) == keep:
                break

        folder = os.path.join(OUT, name.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)
        files = []
        for i, (s, r, x) in enumerate(chosen, 1):
            x = mixer.trim(x[:, None], floor_db=-50)[:, 0] if kind != "ambience" else x
            x = x - x.mean()
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.95
            rel = f"{name.replace(' ', '_')}/{i}.mp3"
            with open(os.path.join(OUT, rel), "wb") as fh:
                fh.write(encode_mp3(x))
            files.append(rel)
            credits.append((rel, r["filename"], r["src_file"], r["esc10"] == "True"))
        manifest[name] = {"kind": kind, "files": files}
        print(f"{name:12} {esc_class:16} kept {len(files)}/{len(scored)}  best={chosen[0][0]:.0f}" if chosen else f"{name}: none")

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    with open(os.path.join(OUT, "CREDITS.md"), "w") as fh:
        fh.write(
            "# Sound credits\n\n"
            "Recordings in this folder are selected, trimmed and re-encoded clips from the\n"
            "[ESC-50 dataset](https://github.com/karolpiczak/ESC-50) by Karol J. Piczak,\n"
            "which are in turn excerpts of recordings on [Freesound](https://freesound.org).\n\n"
            "License: [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) "
            "(clips marked ESC-10 are CC BY). **Non-commercial use only** unless you replace them.\n\n"
            "K. J. Piczak. ESC: Dataset for Environmental Sound Classification. "
            "Proceedings of the 23rd Annual ACM Conference on Multimedia, 2015.\n\n"
            "| File | ESC-50 clip | Freesound source | License |\n|---|---|---|---|\n"
        )
        for rel, fname, src, esc10 in credits:
            fh.write(f"| `{rel}` | {fname} | [freesound.org/s/{src}](https://freesound.org/s/{src}/) | {'CC BY' if esc10 else 'CC BY-NC'} |\n")
        fh.write("\n## Original per-clip attributions (from ESC-50)\n\n```\n")
        wanted = {c[2] for c in credits}
        for line in credits_src.splitlines():
            if any(src in line for src in wanted):
                fh.write(line + "\n")
        fh.write("```\n")


if __name__ == "__main__":
    main()
