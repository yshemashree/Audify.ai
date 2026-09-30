import os
import re
import math
import time
import logging
import requests
from dotenv import load_dotenv
from database import vector_search
from concurrent.futures import ThreadPoolExecutor
from synth import render
import mixer
import library

load_dotenv()
log = logging.getLogger("audify")

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")
FREESOUND_KEY = os.getenv("FREESOUND_API_KEY")

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

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
    "jungle": "dense jungle alive with continuous sound, howler monkeys screaming, tropical birds calling, insects droning, rain on leaves, all layers simultaneously in overwhelming natural cacophony",
    "underwater": "deep underwater ambience, low-frequency water pressure hum, distant whale song resonating, bubbles rising, muffled current movement, continuous deep oceanic atmosphere throughout",
    "space": "deep space ambience, low-frequency cosmic hum, distant pulsar rhythm, electromagnetic interference crackle, vast empty resonance, continuous eerie space atmosphere throughout",
    "battle": "intense military battle, continuous gunfire from multiple weapons, explosions blasting nearby, helicopter overhead, soldiers shouting, radio static, all layering simultaneously throughout",
    "crowd riot": "violent crowd riot, thousands screaming and shouting, glass shattering, explosions from teargas, continuous chaos noise overlapping throughout",
    "subway": "underground subway train arriving, screeching metal brakes on rails, wind rushing through tunnel, doors opening with pneumatic hiss, crowd noise, continuous urban underground noise",
    "factory": "industrial factory floor at full production, heavy machinery pounding rhythmically, metal pressing and stamping, conveyor belts humming, steam releasing, all layering in continuous industrial roar",
    "chainsaw forest": "chainsaw cutting through dense forest, engine screaming, wood splintering, massive tree cracking and crashing to ground, birds scattering, continuous aggressive mechanical and natural noise",
    "traffic": "heavy city traffic jam, multiple car engines idling and revving, horns honking repeatedly, truck diesel rumble, bus brakes hissing, motorbikes weaving, continuous dense urban road noise throughout",
    "traffic noise": "busy urban traffic, overlapping car engines, aggressive horn honking, truck rumble, screeching brakes, ambulance siren passing, pedestrian crossing beep, continuous dense road noise throughout",
    "road": "busy road traffic, constant stream of passing vehicles, car engines dopplering, truck airbrakes, motorbike acceleration, continuous urban road ambience throughout",
    "highway": "busy highway with fast moving vehicles, continuous whoosh of cars passing at speed, truck airblast, engine roar, tyre noise on asphalt, relentless high-speed traffic throughout",
}

# Exact descriptions for the sample prompts on the landing page. These keys are
# longer than the generic ones above, so they win the longest-match lookup.
CINEMATIC_SOUNDS.update({
    "thunderstorm": "violent thunderstorm directly overhead, sharp thunderclaps cracking every few seconds followed by deep rolling rumble, heavy rain pouring on every surface, gusting wind, continuous with no silence",
    "rain on glass": "heavy rain drumming on a window pane at night, crisp individual droplets tapping the glass, water streaming down in rivulets, soft muffled storm outside, intimate close-mic interior, continuous",
    "spaceship engine": "enormous spaceship engine humming in deep space, low pulsing reactor drone, layered detuned harmonics, subtle mechanical vibration and air vents hissing inside the hull, steady sci-fi ambience, continuous",
    "laser": "sci-fi laser blaster firing repeatedly, bright electronic pew zaps with fast descending pitch, energy charge whine between shots, crisp futuristic weapon sounds, continuous volley",
    "coffee shop": "busy coffee shop ambience, many people chatting softly at tables, cups and saucers clinking, espresso machine steaming and hissing, barista calling orders, warm indoor room tone, continuous",
    "cafe": "busy cafe ambience, overlapping conversation murmur, ceramic cups clinking, espresso machine hissing, chairs scraping, warm indoor room tone, continuous",
    "portal": "magical energy portal tearing open, swirling vortex whoosh rising in pitch, electric crackle and sparks, deep resonant hum pulsing, otherworldly shimmering tones, continuous",
    "robot powering up": "robot powering up, electrical hum starting, rising servo whine climbing in pitch, mechanical joints clicking into place, boot-up beeps and chirps, hydraulic hiss, continuous",
    "glass shattering": "single large glass window shattering, razor-sharp crack transient, heavy impact with low thump, explosive burst of crystal shards flying outward, fragments raining and bouncing on a hard floor, bright tinkling tail, VFX-grade hyper-real close-mic, crisp and clean",
    "glass": "thick glass pane smashing, razor-sharp crack transient, heavy impact thump, explosive burst of crystal shards, fragments raining and bouncing on hard floor, bright tinkling tail, VFX-grade hyper-real close-mic",
    "shatter": "glass shattering violently, razor-sharp crack, explosive burst of crystal shards, fragments raining and bouncing on hard floor, bright tinkling tail, VFX-grade hyper-real close-mic",
    "glass cracking": "glass under stress cracking, sharp spidering micro-fractures ticking outward, tense creaks, bright crystalline snaps growing louder, isolated, VFX-grade hyper-real close-mic, no music",
    "crackling glass": "glass under stress crackling, sharp spidering micro-fractures ticking outward, tense creaks, bright crystalline snaps, isolated, VFX-grade hyper-real close-mic, no music",
    "cracking glass": "glass under stress cracking, sharp spidering micro-fractures ticking outward, tense creaks, bright crystalline snaps growing louder, isolated, VFX-grade hyper-real close-mic, no music",
    "glass crackling": "glass under stress crackling, sharp spidering micro-fractures ticking outward, tense creaks, bright crystalline snaps, isolated, VFX-grade hyper-real close-mic, no music",
    "ice cracking": "thick ice sheet cracking, sharp splitting snaps with ringing pings, deep creaking groans, fractures spreading across the surface, VFX-grade hyper-real close-mic",
    "bottle": "glass bottle smashing on concrete, sharp crack, burst of shards, fragments bouncing and skittering, bright tinkling tail, VFX-grade close-mic",
    "window smash": "window pane smashing, razor-sharp crack, burst of crystal shards, fragments raining on the floor, bright tinkling tail, VFX-grade close-mic",
})

# Short, specific search terms for Freesound. Its text search matches every word,
# so a long description returns nothing — it needs a few precise keywords.
FREESOUND_QUERIES = {
    "thunderstorm": "thunderstorm rain thunder",
    "thunder": "thunder storm",
    "rain on glass": "rain window glass",
    "ocean": "ocean waves",
    "wave": "ocean waves",
    "campfire": "campfire crackling",
    "fire": "fire crackling",
    "cat": "cat meow",
    "kitten": "kitten meow",
    "dog": "dog barking",
    "spaceship engine": "spaceship engine hum",
    "spaceship": "spaceship engine",
    "wolf": "wolf howl",
    "laser": "laser gun sci-fi",
    "traffic": "city traffic",
    "city": "city traffic ambience",
    "coffee shop": "coffee shop ambience",
    "cafe": "cafe ambience",
    "portal": "portal magic energy",
    "keyboard": "keyboard typing",
    "waterfall": "waterfall",
    "robot powering up": "robot power up",
    "robot": "robot servo",
    "heartbeat": "heartbeat",
    "glass shattering": "glass shatter",
    "glass": "glass break",
    "shatter": "glass shatter",
    "glass cracking": "glass crack",
    "glass crackling": "glass crack",
    "crackling glass": "glass crack",
    "cracking glass": "glass crack",
    "ice cracking": "ice crack",
    "bottle": "bottle break",
    "window smash": "window break glass",
    "lion": "lion roar",
    "tiger": "tiger roar",
    "bear": "bear growl",
    "cow": "cow moo",
    "horse": "horse neigh",
    "sheep": "sheep bleat",
    "pig": "pig grunt",
    "duck": "duck quack",
    "rooster": "rooster crow",
    "crow": "crow caw",
    "eagle": "eagle scream",
    "elephant": "elephant trumpet",
    "monkey": "monkey chimpanzee",
    "frog": "frog croak",
    "donkey": "donkey bray",
    "snake": "snake hiss",
    "bee": "bee buzz",
    "insect": "crickets insects",
    "puppy": "puppy bark",
    "bird": "bird song",
}

ANIMAL_WORDS = {
    "cat", "cats", "kitten", "kitty", "meow", "meowing", "purr", "dog", "dogs", "puppy", "bark", "barking",
    "woof", "growl", "growling", "snarl", "wolf", "wolves", "coyote", "lion", "lions", "tiger", "tigers",
    "leopard", "jaguar", "panther", "bear", "bears", "grizzly", "cow", "cows", "moo", "mooing", "bull",
    "horse", "horses", "neigh", "neighing", "whinny", "pony", "goat", "goats", "sheep", "lamb", "baa",
    "pig", "pigs", "oink", "hog", "boar", "duck", "ducks", "quack", "quacking", "goose", "geese",
    "rooster", "chicken", "chickens", "hen", "cluck", "owl", "owls", "hoot", "crow", "crows", "raven",
    "caw", "eagle", "hawk", "falcon", "elephant", "elephants", "monkey", "monkeys", "chimp", "chimpanzee",
    "ape", "gorilla", "baboon", "frog", "frogs", "toad", "croak", "ribbit", "donkey", "mule", "bray",
    "bird", "birds", "parrot", "seagull", "gull", "dolphin", "whale", "snake", "rattlesnake", "cobra",
    "bee", "bees", "wasp", "cricket", "crickets", "cicada", "animal", "animals",
}
IMPACT_WORDS = {
    "glass", "shatter", "shatters", "shattering", "smash", "smashing", "crack", "cracks", "cracking",
    "bottle", "window", "break", "breaking", "explosion", "gunshot", "punch", "slam",
}


def classify(prompt: str, key: str | None) -> str:
    """ambience (continuous beds), impact (VFX one-shots) or vocal (animal calls)."""
    words = set(re.findall(r"[a-z]+", f"{prompt} {key or ''}".lower()))
    if re.search(r"\b(rain|raining|drizzle|downpour)\b", prompt.lower()) and words & {"glass", "window"}:
        return "ambience"  # rain on glass is a bed, not a break
    if words & IMPACT_WORDS and not words & {"fire", "campfire", "fireplace", "bonfire"}:
        return "impact"
    if words & ANIMAL_WORDS:
        return "vocal"
    return "ambience"


# Freesound search window and number of takes per kind of sound.
KIND_PROFILE = {
    "ambience": {"duration": "[6 TO 90]", "takes": 1, "el_seconds": 15.0},
    "impact": {"duration": "[0.4 TO 10]", "takes": 3, "el_seconds": 4.0},
    "vocal": {"duration": "[0.5 TO 20]", "takes": 3, "el_seconds": 6.0},
}

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "with", "for", "from", "by",
    "sound", "sounds", "noise", "audio", "effect", "effects", "some", "very", "really",
    "please", "make", "me", "like", "is", "are", "that", "this",
}


def match_cinematic(prompt: str):
    """Library key for the prompt, matched on whole words ('cat' does not fire on
    'location', 'male' not on 'female'). The sound named first wins ('a cat in the
    rain' is a cat), and at the same position the longer phrase wins."""
    text = prompt.lower()
    best = None
    for key in CINEMATIC_SOUNDS:
        m = re.search(r"\b" + re.escape(key) + r"s?\b", text)
        if m:
            rank = (m.start(), -len(key))
            if best is None or rank < best[0]:
                best = (rank, key)
    return best[1] if best else None


def freesound_query(prompt: str, key: str | None) -> list[str]:
    """Search queries to try in order: curated query, then the prompt's keywords."""
    queries = []
    if key:
        queries.append(FREESOUND_QUERIES.get(key, key))
    words = [w for w in re.findall(r"[a-z]+", prompt.lower()) if w not in STOPWORDS]
    if words:
        queries.append(" ".join(words[:4]))
        if len(words) > 2:
            queries.append(" ".join(words[:2]))
    return list(dict.fromkeys(q for q in queries if q))


def _llm_elaborate(prompt: str) -> str | None:
    """Ask Qwen (via the Hugging Face router) for an ElevenLabs sound prompt."""
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
                "max_tokens": 100,
                "temperature": 0.7,
            },
            timeout=(5, 12),
        )
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"].strip().strip('"')
            return text or None
        log.warning("[LLM] HTTP %s: %s", response.status_code, response.text[:200])
    except Exception as e:
        log.warning("[LLM] %s", e)
    return None


def elaborate_prompt(prompt: str) -> tuple[str, str | None]:
    """Return (description, library key or None)."""
    key = match_cinematic(prompt)
    if key:
        return CINEMATIC_SOUNDS[key], key
    result = _llm_elaborate(prompt)
    if result:
        return result, None
    return f"{prompt}, realistic high-fidelity sound effect, close-mic, clear and continuous throughout", None


def search_audio(elaborated_prompt: str) -> str:
    """
    Searches the ChromaDB vector database for a semantically similar expert sound description.
    Returns the matched description if cosine distance is below threshold, else 'NO_MATCH'.
    """
    try:
        match = vector_search(elaborated_prompt)
        if match:
            description, label, distance = match
            log.info("[ChromaDB] Match: '%s' (distance=%.3f)", label, distance)
            return description
    except Exception as e:
        log.warning("[ChromaDB] Search error: %s", e)
    return "NO_MATCH"


def freesound_fetch(queries: list[str], kind: str = "ambience") -> list[bytes]:
    """Find real recordings on Freesound; return up to N HQ preview mp3s.
    Among the most relevant results, well-rated and much-downloaded sounds win."""
    if not FREESOUND_KEY:
        return []
    profile = KIND_PROFILE[kind]
    deadline = time.monotonic() + 25  # never let Freesound eat the whole request
    found: list[bytes] = []
    for query in queries:
        if time.monotonic() > deadline or len(found) >= profile["takes"]:
            break
        try:
            response = requests.get(
                "https://freesound.org/apiv2/search/text/",
                params={
                    "query": query,
                    "token": FREESOUND_KEY,
                    "fields": "id,name,previews,duration,num_downloads,avg_rating",
                    "filter": f"duration:{profile['duration']}",
                    "sort": "score",
                    "page_size": 15,
                },
                timeout=(5, 10),
            )
            if response.status_code != 200:
                log.warning("[Freesound] HTTP %s: %s", response.status_code, response.text[:200])
                if response.status_code in (401, 403, 429):
                    break  # bad key or rate limited — other queries won't help
                continue
            results = response.json().get("results", [])
            ranked = sorted(
                enumerate(results),
                key=lambda ir: -ir[0] + 2.0 * math.log10(1 + (ir[1].get("num_downloads") or 0))
                + 0.6 * (ir[1].get("avg_rating") or 0),
                reverse=True,
            )
            for _, result in ranked:
                if time.monotonic() > deadline or len(found) >= profile["takes"]:
                    break
                url = (result.get("previews") or {}).get("preview-hq-mp3")
                if not url:
                    continue
                audio = requests.get(url, timeout=(5, 15))
                if audio.status_code == 200 and len(audio.content) > 1000:
                    log.info("[Freesound] '%s' -> %s", query, result.get("name"))
                    found.append(audio.content)
        except Exception as e:
            log.warning("[Freesound] %s", e)
    return found


EXAGGERATE_PREFIX = {
    "ambience": "Exaggerated, over-the-top Hollywood cinematic sound effect, huge and larger than life, extremely loud and punchy: ",
    "impact": "Single isolated VFX one-shot for film, hyper-real and exaggerated, razor-sharp attack, huge impact, clean tail, no music: ",
    "vocal": "Hyper-real exaggerated cinematic animal vocalisation, close-mic, powerful and clear, isolated, no music, no human voice: ",
}


def elevenlabs_fetch(description: str, kind: str = "ambience") -> bytes | None:
    """Generate a sound effect with ElevenLabs; return mp3 bytes."""
    if not ELEVENLABS_KEY:
        return None
    try:
        response = requests.post(
            "https://api.elevenlabs.io/v1/sound-generation",
            headers={"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"},
            json={
                "text": (EXAGGERATE_PREFIX[kind] + description)[:450],
                "duration_seconds": KIND_PROFILE[kind]["el_seconds"],
                "prompt_influence": 0.75,
            },
            timeout=(5, 60),
        )
        if response.status_code == 200 and response.content:
            return response.content
        log.warning("[ElevenLabs] HTTP %s: %s", response.status_code, response.text[:200])
    except Exception as e:
        log.warning("[ElevenLabs] %s", e)
    return None


def _save_raw(data: bytes, out_base: str) -> str:
    path = out_base + ".mp3"
    with open(path, "wb") as f:
        f.write(data)
    return path


def generate_audio(prompt: str, description: str, key: str | None, out_base: str) -> tuple[str, str]:
    """
    Returns (file path, engine name).
    Freesound (real recordings) and ElevenLabs (AI take) are fetched in parallel
    and combined according to the kind of sound:
      ambience — one recording and the AI take layered into a single bed
      impact   — each real take layered with the AI take, attacks aligned to the
                 sample, then sequenced as distinct hits (a VFX pack)
      vocal    — real and AI calls alternated with natural pauses
    If neither service answers, the offline synth takes over. Every result is
    mastered by the exaggeration chain for its kind.
    """
    kind = classify(prompt, key)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs_job = pool.submit(freesound_fetch, freesound_query(prompt, key), kind)
        el_job = pool.submit(elevenlabs_fetch, description, kind)
        fs_raw, el_raw = fs_job.result(), el_job.result()

    fs = [a for a in (mixer.decode(b) for b in fs_raw) if a is not None]
    el = mixer.decode(el_raw) if el_raw else None
    engines = "+".join(e for e, ok in (("freesound", fs), ("elevenlabs", el is not None)) if ok)

    # Built-in real recordings first: works offline, and online takes are mixed in.
    built = library.build(prompt, extra_takes=fs, sweetener=el)
    if built is not None:
        mix, lib_kind = built
        engines = "+".join(["library"] + ([engines] if engines else []))
        return mixer.write_audio(out_base, mixer.exaggerate(mix, lib_kind)), engines

    mix = None
    if kind == "ambience":
        if fs and el is not None:
            mix = mixer.layer(fs[0], el)
        elif fs or el is not None:
            mix = mixer.clamp_length(fs[0] if fs else el)
    elif fs or el is not None:
        takes = [mixer.trim(t) for t in fs]
        el_take = mixer.trim(el) if el is not None else None
        if kind == "impact":
            hits = [mixer.align_layer(t, el_take) for t in takes] if takes and el_take is not None else takes or [el_take]
            mix = mixer.sequence(hits, gap=(0.7, 1.1))
        else:
            calls = takes + ([el_take] if el_take is not None else [])
            mix = mixer.sequence(calls, gap=(0.5, 1.0))

    if mix is not None:
        log.info("[Mix] %s via %s", kind, engines)
        return mixer.write_audio(out_base, mixer.exaggerate(mix, kind)), engines

    # Audio arrived but couldn't be decoded — serve it untouched rather than fail.
    for data, engine in ((fs_raw[0] if fs_raw else None, "freesound"), (el_raw, "elevenlabs")):
        if data:
            return _save_raw(data, out_base), engine

    # Route the synth on what the user typed (plus the library key), not the long
    # description, which mentions many unrelated sounds ("rain", "wind", ...).
    x = render(f"{prompt} {key or ''}")
    return mixer.write_audio(out_base, mixer.exaggerate(x, kind)), "procedural"
