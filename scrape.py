"""
Scrapes ESC-50 dataset metadata from GitHub and enriches descriptions
using the Freesound API. Outputs scraped_sounds.json.

Usage:
    python scrape.py                          # ESC-50 only (no API key needed)
    python scrape.py --freesound YOUR_KEY     # ESC-50 + Freesound enrichment
"""

import csv
import json
import time
import argparse
import requests
from io import StringIO

ESC50_CSV_URL = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"

ACOUSTIC_EXPANSIONS = {
    "rain": "steady rainfall, soft patter rhythm, outdoor reverb",
    "sea_waves": "ocean waves breaking on shore, rhythmic wash and retreat, open beach",
    "crackling_fire": "crackling campfire, dry wood snapping, warm mid-frequency crackle",
    "crickets": "cricket chirping, rhythmic high-pitched stridulation, quiet night outdoors",
    "chirping_birds": "morning bird song, melodic warble, multiple pitch variations, forest ambience",
    "water_drops": "water dripping, echo tap on still surface, hollow resonance",
    "wind": "strong wind through trees, rushing air, swaying branches",
    "pouring_water": "rushing water flow, continuous liquid turbulence, outdoor stream",
    "toilet_flush": "toilet flushing, rushing water swirl, bathroom reverb, gurgle tail",
    "thunderstorm": "deep rolling thunder crack, heavy rain on leaves, wide outdoor reverb",
    "crying_baby": "infant crying, high-pitched wail, short breath gaps, indoor room",
    "sneezing": "sharp sneeze burst, nasal explosive transient, close-mic, brief",
    "clapping": "crowd clapping, rhythmic hand impacts, hall reverb, applause texture",
    "breathing": "slow deep breathing, close-mic, soft nasal air, quiet interior",
    "coughing": "dry cough burst, throat clearing, close-mic, indoor room",
    "footsteps": "footsteps on wooden floor, hollow knock, moderate walking pace",
    "laughing": "warm human laughter, natural rhythm, indoor room reverb",
    "brushing_teeth": "toothbrush scrubbing, wet bristle on enamel, bathroom echo",
    "snoring": "deep rhythmic snoring, low nasal drone, bedroom quiet ambience",
    "drinking_sipping": "liquid sipping, mouth sounds, swallowing, close-mic",
    "door_knock": "firm triple knock on wood, hollow resonance, quiet interior",
    "mouse_click": "computer mouse click, sharp plastic snap, desk surface, close-mic",
    "keyboard_typing": "mechanical keyboard typing, crisp click-clack rhythm, office indoor",
    "door_wood_creaks": "creaking door hinge, slow metallic groan, horror atmosphere",
    "can_opening": "soda can pull tab, pressurised hiss, metallic crack, close-mic",
    "washing_machine": "washing machine cycle, drum rotation hum, water slosh, laundry room",
    "vacuum_cleaner": "vacuum cleaner motor roar, continuous high-frequency hum, indoor",
    "clock_alarm": "alarm clock buzzing, repetitive electronic pulse, sharp piercing tone",
    "clock_tick": "wall clock ticking, precise rhythmic tick-tock, quiet indoor room",
    "glass_breaking": "glass shattering, high-frequency crystal impact, cascading shards",
    "helicopter": "helicopter rotor overhead, rhythmic thwop, engine whine, air displacement",
    "chainsaw": "chainsaw running, aggressive engine buzz, high-frequency blade whine, outdoor",
    "siren": "emergency siren wail, rising and falling pitch, Doppler, outdoor street",
    "car_horn": "car horn honk, sharp mid-frequency blast, urban street ambience",
    "engine": "car engine idling, mechanical rumble, exhaust resonance, outdoor",
    "train": "train wheels on track, repetitive metallic clatter, track rhythm, tunnel reverb",
    "church_bells": "church bell toll, deep bronze resonance, long sustain decay, town square",
    "airplane": "airplane flyover, jet engine roar, Doppler sweep, open sky",
    "fireworks": "fireworks explosion, sharp crack and boom, crowd ambient, night outdoor",
    "hand_saw": "hand saw cutting wood, rhythmic scraping, sawdust texture, workshop",
    "dog": "medium dog bark, sharp burst, outdoor echo, alert defensive tone",
    "rooster": "rooster crowing, bright resonant call, outdoor farmyard, morning ambience",
    "pig": "pig oinking, nasal grunt, barnyard outdoor, close proximity",
    "cow": "cow mooing, deep resonant low, open field, calm farm ambience",
    "frog": "frog croaking, rhythmic call, pond outdoor, night ambience",
    "cat": "soft indoor meow, slight upward pitch inflection, close-mic, light reverb",
    "hen": "hen clucking, rhythmic short calls, barnyard outdoor, mid-frequency",
    "insects": "insect buzzing, continuous high-frequency drone, summer outdoor",
    "sheep": "sheep baaing, nasal resonant call, open meadow, farm ambience",
    "crow": "crow cawing, harsh dry call, open outdoor space, isolated single bird",
}

def fetch_esc50():
    print("Fetching ESC-50 metadata from GitHub...")
    r = requests.get(ESC50_CSV_URL, timeout=15)
    r.raise_for_status()
    reader = csv.DictReader(StringIO(r.text))
    rows = list(reader)
    print(f"  Got {len(rows)} entries across {len({r['category'] for r in rows})} categories")
    return rows

def build_dataset(rows):
    seen = set()
    sounds = []
    for row in rows:
        cat = row["category"]
        if cat in seen:
            continue
        seen.add(cat)
        label = cat.replace("_", " ")
        acoustic = ACOUSTIC_EXPANSIONS.get(cat, f"realistic {label} sound, natural acoustic environment")
        sounds.append({
            "id": f"esc50_{cat}",
            "category": cat,
            "label": label,
            "esc50_fold": row["fold"],
            "source": "ESC-50",
            "description": acoustic,
        })
    print(f"  Built {len(sounds)} unique category descriptions from ESC-50")
    return sounds

def enrich_with_freesound(sounds, api_key):
    print("Enriching with Freesound API...")
    base = "https://freesound.org/apiv2/search/text/"
    for s in sounds:
        params = {
            "query": s["label"],
            "fields": "name,description,tags,duration",
            "page_size": 1,
            "token": api_key,
        }
        try:
            r = requests.get(base, params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get("results", [])
            if results:
                hit = results[0]
                s["freesound_name"] = hit.get("name", "")
                s["freesound_tags"] = hit.get("tags", [])[:8]
                s["freesound_duration"] = hit.get("duration", 0)
                s["source"] = "ESC-50 + Freesound"
                print(f"  [{s['category']}] matched: {hit.get('name', '')}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [{s['category']}] Freesound error: {e}")
    return sounds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freesound", metavar="API_KEY", help="Freesound API key for enrichment")
    parser.add_argument("--output", default="scraped_sounds.json")
    args = parser.parse_args()

    rows = fetch_esc50()
    sounds = build_dataset(rows)

    if args.freesound:
        sounds = enrich_with_freesound(sounds, args.freesound)

    with open(args.output, "w") as f:
        json.dump(sounds, f, indent=2)

    print(f"\nSaved {len(sounds)} sounds to {args.output}")
    print("Categories scraped:")
    for s in sounds:
        print(f"  {s['category']:30s} — {s['description'][:60]}...")

if __name__ == "__main__":
    main()
