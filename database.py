import chromadb
from chromadb.utils import embedding_functions

SOUND_DESCRIPTIONS = [
    {"id": "rain_01", "description": "continuous heavy rain on hard surface, crisp high-density impact, loud sustained rainfall, close-mic outdoor, high-fidelity, no muffling", "label": "rain"},
    {"id": "rain_02", "description": "torrential downpour on metal roof, crisp sharp staccato drops, loud continuous impact, close-mic, high-fidelity recording", "label": "rain"},
    {"id": "rain_03", "description": "steady rainfall on leaves, continuous crisp patter, loud outdoor recording, close-mic, sustained throughout", "label": "rain"},
    {"id": "thunder_01", "description": "repeated sharp thunderclaps overhead, loud crisp crack each strike, continuous low-frequency rumble sustained, high-fidelity outdoor, no reverb wash", "label": "thunder"},
    {"id": "thunder_02", "description": "continuous thunderstorm, repeated crisp thunder cracks, torrential rain sustained, loud and defined throughout, high-fidelity", "label": "thunder"},
    {"id": "fire_01", "description": "continuous roaring campfire crackle, crisp dry wood snapping repeatedly, loud high-frequency pops, close-mic, full presence throughout", "label": "fire"},
    {"id": "fire_02", "description": "loud roaring bonfire, continuous crisp crackling and snapping, intense heat hiss, close-mic outdoor, high-fidelity, sustained", "label": "fire"},
    {"id": "wind_01", "description": "continuous loud wind rush, crisp gusts through trees, sustained howling, high-fidelity outdoor recording, no muffling", "label": "wind"},
    {"id": "wind_02", "description": "loud howling wind at night, crisp repeated gusts, sustained eerie moan, close-mic outdoor, high-fidelity throughout", "label": "wind"},
    {"id": "ocean_01", "description": "repeated loud ocean waves crashing on shore, crisp foam hiss, close-mic beach, high-fidelity, continuous wave impacts throughout", "label": "ocean"},
    {"id": "ocean_02", "description": "continuous loud ocean surf, crisp wave impacts repeating, sustained sea noise, high-fidelity outdoor beach recording", "label": "ocean"},
    {"id": "heartbeat_01", "description": "loud continuous human heartbeat, crisp close-mic chest thump, strong two-thud rhythm at 72bpm, dry acoustic, sustained throughout", "label": "heartbeat"},
    {"id": "heartbeat_02", "description": "loud rapid racing heartbeat, crisp repeated thumps, elevated fast rhythm, close-mic, high-fidelity, continuous throughout", "label": "heartbeat"},
    {"id": "cat_01", "description": "repeated loud crisp cat meows, close-mic, high-fidelity, sharp upward pitch inflection, dry indoor acoustic, continuous vocalisations throughout", "label": "cat"},
    {"id": "cat_02", "description": "loud continuous cat purring, crisp low-frequency vibration, close-mic, high-fidelity, sustained resonant rumble throughout", "label": "cat"},
    {"id": "cat_03", "description": "repeated loud cat hisses and meows, crisp sharp air bursts, close-mic, high-fidelity, continuous aggressive vocalisations", "label": "cat"},
    {"id": "dog_01", "description": "repeated loud sharp dog barks, large breed, close-mic, crisp attack each bark, dry outdoor air, continuous barking throughout", "label": "dog"},
    {"id": "dog_02", "description": "continuous loud dog barking, small breed, high-pitched crisp yaps, close-mic, high-fidelity, sustained throughout", "label": "dog"},
    {"id": "dog_03", "description": "loud continuous dog growling and barking, deep chest rumble, crisp close-mic, high-fidelity, menacing and sustained", "label": "dog"},
    {"id": "bird_01", "description": "continuous loud bird chorus, multiple species chirping repeatedly, crisp high-frequency calls, close-mic forest, high-fidelity throughout", "label": "bird"},
    {"id": "bird_02", "description": "repeated loud crow cawing, crisp harsh call each time, close-mic outdoor, high-fidelity, continuous throughout", "label": "bird"},
    {"id": "bird_03", "description": "repeated loud owl hoots, crisp haunting calls, close-mic night outdoor, high-fidelity, sustained throughout", "label": "bird"},
    {"id": "keyboard_01", "description": "continuous rapid mechanical keyboard typing, crisp loud click-clack, close-mic, dry office acoustic, sustained typing throughout", "label": "keyboard"},
    {"id": "keyboard_02", "description": "continuous keyboard typing, crisp repeated keystrokes, close-mic, high-fidelity, loud and defined throughout", "label": "keyboard"},
    {"id": "clock_01", "description": "crisp loud mechanical clock ticking, close-mic, sharp precise 1Hz rhythm, dry indoor acoustic, continuous throughout", "label": "clock"},
    {"id": "clock_02", "description": "loud repeated clock chimes, crisp resonant bell tones, close-mic, high-fidelity, continuous throughout", "label": "clock"},
    {"id": "footsteps_01", "description": "continuous footsteps on hardwood floor, crisp loud impact each step, close-mic, steady walking pace, dry indoor acoustic", "label": "footsteps"},
    {"id": "footsteps_02", "description": "continuous footsteps on gravel, loud crisp crunching texture, close-mic outdoor, high-fidelity, sustained walking throughout", "label": "footsteps"},
    {"id": "footsteps_03", "description": "continuous running footsteps on pavement, loud crisp rapid impacts, close-mic outdoor, high-fidelity, urgent pace throughout", "label": "footsteps"},
    {"id": "car_01", "description": "loud car engine running continuously, crisp mechanical rumble, close-mic outdoor, engine revving, high-fidelity, sustained throughout", "label": "car"},
    {"id": "car_02", "description": "loud continuous car engine roar, crisp mechanical noise, close-mic, high-fidelity, sustained throughout", "label": "car"},
    {"id": "car_03", "description": "repeated loud car horn honks, crisp sharp blasts, close-mic urban outdoor, high-fidelity, continuous", "label": "car"},
    {"id": "water_01", "description": "continuous loud water dripping, crisp sharp tap each drop, close-mic, high-fidelity, sustained throughout", "label": "water"},
    {"id": "water_02", "description": "continuous loud rushing water stream, crisp turbulent flow, close-mic outdoor, high-fidelity, sustained throughout", "label": "water"},
    {"id": "water_03", "description": "continuous loud boiling water, crisp rapid bubbling, steam hiss, close-mic kitchen, high-fidelity throughout", "label": "water"},
    {"id": "crowd_01", "description": "continuous loud crowd cheering, crisp overlapping voices, stadium roar, close-mic, high-fidelity, sustained throughout", "label": "crowd"},
    {"id": "crowd_02", "description": "continuous loud busy crowd noise, crisp overlapping voices, close-mic indoor, high-fidelity, sustained throughout", "label": "crowd"},
    {"id": "noise_01", "description": "loud continuous white noise, crisp broadband frequency wash, high-fidelity, full presence throughout", "label": "noise"},
    {"id": "explosion_01", "description": "loud massive explosion blast, crisp deep shockwave, debris rattle, high-fidelity outdoor, full impact", "label": "explosion"},
    {"id": "explosion_02", "description": "repeated loud firecracker explosions, crisp sharp cracks, close-mic outdoor, high-fidelity, continuous", "label": "explosion"},
    {"id": "glass_01", "description": "loud glass shattering, crisp high-frequency crystal impact, cascading shards, close-mic hard floor, high-fidelity", "label": "glass"},
    {"id": "glass_02", "description": "loud sustained glass ringing, crisp clear high tone, close-mic, high-fidelity, sustained resonance throughout", "label": "glass"},
    {"id": "door_01", "description": "repeated loud wooden door slams, crisp sharp impact, close-mic, high-fidelity, continuous throughout", "label": "door"},
    {"id": "door_02", "description": "continuous loud door creaking, crisp metallic groan, close-mic, high-fidelity, sustained throughout", "label": "door"},
    {"id": "door_03", "description": "repeated loud door knocks, crisp firm impacts, close-mic, high-fidelity, continuous knocking throughout", "label": "door"},
    {"id": "wolf_01", "description": "repeated loud wolf howls, crisp sustained notes with vibrato, close-mic outdoor night, high-fidelity, continuous howling throughout", "label": "wolf"},
    {"id": "robot_01", "description": "continuous loud robot sounds, crisp servo whirs and mechanical clicks, ascending beep tones repeating, high-fidelity, sustained throughout", "label": "robot"},
    {"id": "robot_02", "description": "loud continuous robotic mechanical noise, crisp servo motors and clicking joints, close-mic, high-fidelity, sustained", "label": "robot"},
    {"id": "bell_01", "description": "repeated loud church bell tolls, crisp deep bronze resonance, close-mic, high-fidelity, continuous throughout", "label": "bell"},
    {"id": "bell_02", "description": "repeated loud bell rings, crisp bright clear tone, close-mic, high-fidelity, continuous throughout", "label": "bell"},
    {"id": "train_01", "description": "continuous loud train sounds, crisp rhythmic chugging, whistle shriek repeating, track clatter, high-fidelity, sustained throughout", "label": "train"},
    {"id": "train_02", "description": "continuous loud train wheels on track, crisp metallic clatter, close-mic, high-fidelity, sustained throughout", "label": "train"},
    {"id": "helicopter_01", "description": "continuous loud helicopter rotor, crisp rhythmic thwop, engine whine sustained, close-mic, high-fidelity throughout", "label": "helicopter"},
    {"id": "gunshot_01", "description": "repeated loud gunshots, crisp sharp crack each shot, close-mic outdoor, high-fidelity, continuous firing", "label": "gunshot"},
    {"id": "gunshot_02", "description": "repeated loud rifle shots, crisp sharp cracks, close-mic outdoor, high-fidelity, continuous throughout", "label": "gunshot"},
    {"id": "piano_01", "description": "continuous loud piano playing, crisp warm hammer strikes, repeated notes and chords, close-mic, high-fidelity throughout", "label": "piano"},
    {"id": "piano_02", "description": "loud continuous piano melody, crisp bright notes repeating, close-mic studio, high-fidelity, sustained throughout", "label": "piano"},
    {"id": "guitar_01", "description": "continuous loud acoustic guitar strumming, crisp warm string resonance, close-mic, high-fidelity, sustained throughout", "label": "guitar"},
    {"id": "thunder_rain_01", "description": "continuous thunderstorm, repeated crisp thunderclaps, torrential rain sustained, loud and defined throughout, high-fidelity outdoor", "label": "thunderstorm"},
    {"id": "forest_01", "description": "continuous loud forest ambience, crisp bird calls repeating, wind in trees, close-mic outdoor, high-fidelity throughout", "label": "forest"},
    {"id": "city_01", "description": "continuous loud city street noise, crisp car horns and traffic, overlapping urban sounds, close-mic outdoor, high-fidelity", "label": "city"},
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
