import chromadb
from chromadb.utils import embedding_functions

SOUND_DESCRIPTIONS = [
    {"id": "rain_01", "description": "steady rainfall on leaves, soft patter rhythm, light outdoor reverb, distant thunder rumble", "label": "rain"},
    {"id": "rain_02", "description": "heavy downpour on metal roof, high-density impact, sharp staccato drops, indoor echo", "label": "rain"},
    {"id": "rain_03", "description": "light drizzle on windowpane, gentle tapping, quiet urban background, calm atmosphere", "label": "rain"},
    {"id": "thunder_01", "description": "massive thunderclap overhead, deep low-frequency boom, long reverb tail, storm ambience", "label": "thunder"},
    {"id": "thunder_02", "description": "rolling thunder in distance, gradual build and fade, wide open sky reverb, low rumble", "label": "thunder"},
    {"id": "fire_01", "description": "crackling campfire, dry wood snapping, warm mid-frequency crackle, gentle air movement", "label": "fire"},
    {"id": "fire_02", "description": "large bonfire roar, intense heat hiss, continuous crackling, outdoor night ambience", "label": "fire"},
    {"id": "wind_01", "description": "strong wind through trees, rushing air, swaying branches, wide open outdoor space", "label": "wind"},
    {"id": "wind_02", "description": "howling wind at night, eerie resonant moan, gusts and lulls, mountain or open plain", "label": "wind"},
    {"id": "ocean_01", "description": "ocean waves breaking on shore, rhythmic wash and retreat, sea foam hiss, open beach", "label": "ocean"},
    {"id": "ocean_02", "description": "deep underwater ambience, muffled pressure, distant whale-like resonance, submerged reverb", "label": "ocean"},
    {"id": "heartbeat_01", "description": "slow steady heartbeat, close-mic chest thump, low frequency pulse, quiet medical room", "label": "heartbeat"},
    {"id": "heartbeat_02", "description": "racing heartbeat, rapid thumping, elevated stress rhythm, body-close resonance", "label": "heartbeat"},
    {"id": "cat_01", "description": "soft indoor meow, slight upward pitch inflection, close-mic, domestic shorthair, light room reverb", "label": "cat"},
    {"id": "cat_02", "description": "deep cat purring, low frequency vibration, continuous resonant rumble, close intimate recording", "label": "cat"},
    {"id": "cat_03", "description": "angry cat hiss, sharp air burst, defensive aggression, close-mic with short reverb", "label": "cat"},
    {"id": "dog_01", "description": "medium dog bark, sharp single burst, outdoor echo, alert defensive tone", "label": "dog"},
    {"id": "dog_02", "description": "small dog yapping, high-pitched rapid barks, indoor room reverb, excited energy", "label": "dog"},
    {"id": "dog_03", "description": "large dog growl, deep chest rumble, low menacing frequency, close proximity", "label": "dog"},
    {"id": "bird_01", "description": "morning bird song, melodic warble, multiple pitch variations, outdoor forest ambience", "label": "bird"},
    {"id": "bird_02", "description": "crow cawing, harsh dry call, open outdoor space, isolated single bird", "label": "bird"},
    {"id": "bird_03", "description": "owl hoot at night, haunting hollow tone, distant reverb, quiet night forest", "label": "bird"},
    {"id": "keyboard_01", "description": "mechanical keyboard typing, crisp click-clack rhythm, close-mic, office indoor acoustic", "label": "keyboard"},
    {"id": "keyboard_02", "description": "soft membrane keyboard tapping, muted keystrokes, quiet consistent rhythm, near-silence background", "label": "keyboard"},
    {"id": "clock_01", "description": "wall clock ticking, precise rhythmic tick-tock, quiet indoor room, slight reverb", "label": "clock"},
    {"id": "clock_02", "description": "grandfather clock chime, deep resonant bell tones, harmonic overtones, large room echo", "label": "clock"},
    {"id": "footsteps_01", "description": "footsteps on wooden floor, hollow knock, moderate pace walking, indoor room", "label": "footsteps"},
    {"id": "footsteps_02", "description": "footsteps on gravel, crunching texture, outdoor path, casual walking rhythm", "label": "footsteps"},
    {"id": "footsteps_03", "description": "running footsteps on pavement, rapid impact, urban outdoor, urgent fast pace", "label": "footsteps"},
    {"id": "car_01", "description": "car engine starting, ignition click, engine rumble building, mechanical resonance", "label": "car"},
    {"id": "car_02", "description": "car passing at speed, Doppler whoosh, engine roar rising and falling, outdoor road", "label": "car"},
    {"id": "car_03", "description": "car horn honk, sharp mid-frequency blast, urban street ambience, brief duration", "label": "car"},
    {"id": "water_01", "description": "water dripping in cave, echo tap on still water, hollow resonance, enclosed stone space", "label": "water"},
    {"id": "water_02", "description": "rushing stream, turbulent water flow, continuous white noise texture, outdoor nature", "label": "water"},
    {"id": "water_03", "description": "boiling water bubbles, rapid popping bursts, steam hiss, kitchen close-mic", "label": "water"},
    {"id": "crowd_01", "description": "large crowd cheering, stadium roar, collective human voices, massive reverb wash", "label": "crowd"},
    {"id": "crowd_02", "description": "busy cafe murmur, overlapping conversations, clattering cups, warm indoor ambience", "label": "crowd"},
    {"id": "noise_01", "description": "white noise hiss, broadband frequency wash, smooth featureless texture, neutral acoustic", "label": "noise"},
    {"id": "explosion_01", "description": "large explosion blast, deep low-frequency shockwave, debris rattle, outdoor wide reverb", "label": "explosion"},
    {"id": "explosion_02", "description": "small firecracker pop, sharp high transient, brief duration, outdoor open air", "label": "explosion"},
    {"id": "glass_01", "description": "glass shattering, high-frequency crystal impact, cascading shards, hard floor surface", "label": "glass"},
    {"id": "glass_02", "description": "wine glass ringing, sustained clear high tone, pure sine-like resonance, light touch", "label": "glass"},
    {"id": "door_01", "description": "wooden door slam, sharp impact, low frequency thud, hallway echo", "label": "door"},
    {"id": "door_02", "description": "creaking door hinge, slow metallic groan, horror atmosphere, close-mic detail", "label": "door"},
    {"id": "door_03", "description": "door knock, firm triple knock, hollow wooden resonance, quiet interior", "label": "door"},
    {"id": "wolf_01", "description": "lone wolf howl, long sustained note with vibrato, open mountain valley, night wind, distant echo tail, eerie silence", "label": "wolf"},
    {"id": "robot_01", "description": "robot booting up, rising electronic hum, servo whir, three ascending beep tones, industrial hiss, metallic resonance", "label": "robot"},
    {"id": "robot_02", "description": "robotic movement, servo motor whine, mechanical joints clicking, industrial environment", "label": "robot"},
    {"id": "bell_01", "description": "church bell toll, deep bronze resonance, long sustain decay, outdoor town square", "label": "bell"},
    {"id": "bell_02", "description": "small hand bell ring, clear bright tone, short sustain, close indoor space", "label": "bell"},
    {"id": "train_01", "description": "steam train approaching, rhythmic chug, whistle shriek, track clatter, Doppler effect", "label": "train"},
    {"id": "train_02", "description": "train wheels on track, repetitive metallic clatter, consistent rhythm, tunnel reverb", "label": "train"},
    {"id": "helicopter_01", "description": "helicopter rotor overhead, rhythmic thwop, engine whine, air pressure displacement", "label": "helicopter"},
    {"id": "gunshot_01", "description": "pistol gunshot, sharp crack, close-range, brief reverb tail, indoor room", "label": "gunshot"},
    {"id": "gunshot_02", "description": "distant rifle shot, muffled crack, echo off mountains, outdoor open space", "label": "gunshot"},
    {"id": "piano_01", "description": "single piano note, warm hammer strike, sustained harmonic decay, concert hall reverb", "label": "piano"},
    {"id": "piano_02", "description": "piano chord arpeggio, bright upper register, flowing finger movement, studio recording", "label": "piano"},
    {"id": "guitar_01", "description": "acoustic guitar strum, warm string resonance, wood body echo, close-mic recording", "label": "guitar"},
    {"id": "thunder_rain_01", "description": "thunderstorm full scene, heavy rain on leaves, lightning crack overhead, distant rolling thunder, outdoor", "label": "thunderstorm"},
    {"id": "forest_01", "description": "forest ambience, birds singing, wind in trees, insect drone, peaceful natural outdoor", "label": "forest"},
    {"id": "city_01", "description": "busy city street, traffic noise, car horns, pedestrian chatter, urban outdoor ambience", "label": "city"},
]

_chroma_client = None
_collection = None

def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    _collection = _chroma_client.get_or_create_collection(
        name="sounds",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if _collection.count() == 0:
        _collection.add(
            ids=[s["id"] for s in SOUND_DESCRIPTIONS],
            documents=[s["description"] for s in SOUND_DESCRIPTIONS],
            metadatas=[{"label": s["label"]} for s in SOUND_DESCRIPTIONS],
        )

    return _collection


def vector_search(query: str, threshold: float = 0.45):
    """
    Query the ChromaDB collection with cosine similarity.
    Returns (description, label, distance) if distance < threshold, else None.
    """
    col = _get_collection()
    results = col.query(query_texts=[query], n_results=1)
    if not results["distances"] or not results["distances"][0]:
        return None
    distance = results["distances"][0][0]
    if distance < threshold:
        description = results["documents"][0][0]
        label = results["metadatas"][0][0]["label"]
        return description, label, distance
    return None
