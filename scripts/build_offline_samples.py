"""
Pre-render the landing-page sample prompts into frontend/samples/ so the chips
play even when the backend can't be reached (no internet, server down, or the
page opened straight from disk). Uses only the built-in library and synth.

Usage:  python scripts/build_offline_samples.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["FREESOUND_API_KEY"] = os.environ["ELEVENLABS_API_KEY"] = os.environ["HUGGINGFACE_API_TOKEN"] = ""

import numpy as np  # noqa: E402
import tools  # noqa: E402

tools.FREESOUND_KEY = tools.ELEVENLABS_KEY = tools.HF_TOKEN = None
OUT = os.path.join(ROOT, "frontend", "samples")


def chip_prompts() -> list[str]:
    with open(os.path.join(ROOT, "frontend", "index.html")) as f:
        return re.findall(r"onclick=\"use\('([^']+)'\)\"", f.read())


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(7)
    mapping = {}
    for prompt in chip_prompts():
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
        description, key = tools.elaborate_prompt(prompt)
        path, engine = tools.generate_audio(prompt, description, key, os.path.join(OUT, slug))
        mapping[prompt] = {"file": os.path.basename(path), "engine": engine, "description": description}
        print(f"{prompt:32} -> {os.path.basename(path)} ({engine})")
    with open(os.path.join(OUT, "samples.js"), "w") as f:
        # a .js file, not .json: browsers block fetch() of local files, but a
        # <script src> loads fine even when index.html is opened from disk
        f.write("window.AUDIFY_SAMPLES = " + json.dumps(mapping, indent=1) + ";\n")


if __name__ == "__main__":
    main()
