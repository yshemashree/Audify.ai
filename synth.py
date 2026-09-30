"""
Procedural sound synthesis — the offline fallback when Freesound and ElevenLabs
are unavailable. Every sample prompt on the landing page has a dedicated recipe.
Pure numpy, so a 12 s clip renders in well under a second.
"""
import re
import wave
import numpy as np
import animals

SR = 44100
DURATION = 12


# ── building blocks ──────────────────────────────────────────────────────────

def _t(n):
    return np.arange(n) / SR


def band_noise(n, lo, hi, rng, tilt=0.0):
    """White noise band-limited to [lo, hi] Hz via FFT; tilt < 0 darkens the spectrum."""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    mask = (freqs >= lo) & (freqs <= hi)
    if tilt:
        mask = mask * (np.maximum(freqs, 20) / 1000.0) ** tilt
    out = np.fft.irfft(spec * mask, n)
    return out / (np.abs(out).max() + 1e-9)


def smooth_env(n, rate_hz, rng, depth=0.5):
    """Slowly wandering envelope in [1-depth, 1]."""
    pts = max(4, int(n / SR * rate_hz) + 2)
    env = np.interp(np.linspace(0, pts - 1, n), np.arange(pts), rng.random(pts))
    return 1 - depth + depth * env


def decay(n, seconds):
    return np.exp(-_t(n) / max(seconds, 1e-4))


def place(buf, clip, start):
    start = int(start)
    if start >= len(buf) or start < 0:
        return
    end = min(len(buf), start + len(clip))
    buf[start:end] += clip[: end - start]


def glide_tone(freqs, harmonics=(1.0,), vibrato=0.0, vib_rate=5.0):
    """Oscillator following a per-sample frequency curve."""
    n = len(freqs)
    f = freqs * (1 + vibrato * np.sin(2 * np.pi * vib_rate * _t(n)))
    phase = 2 * np.pi * np.cumsum(f) / SR
    out = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        out += amp * np.sin(k * phase)
    return out


def _fade(x, seconds=0.05):
    k = min(len(x) // 2, int(seconds * SR))
    if k > 0:
        ramp = np.linspace(0, 1, k)
        if x.ndim == 2:
            ramp = ramp[:, None]
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


# ── recipes ──────────────────────────────────────────────────────────────────

def rain(n, rng, bright=False):
    out = 0.5 * band_noise(n, 300, 9000, rng, tilt=-0.3) * smooth_env(n, 0.5, rng, 0.2)
    drop_len = int(0.02 * SR)
    lo, hi = (2500, 6000) if bright else (1200, 4000)
    for _ in range(int(n / SR * (60 if bright else 120))):
        f = rng.uniform(lo, hi)
        d = np.sin(2 * np.pi * f * _t(drop_len)) * decay(drop_len, 0.004 if not bright else 0.008)
        place(out, d * rng.uniform(0.15, 0.6 if bright else 0.35), rng.integers(0, n))
    return out


def rain_on_glass(n, rng):
    out = 0.6 * rain(n, rng, bright=True)
    # occasional rivulets trickling down the pane
    out += 0.12 * band_noise(n, 1500, 5000, rng) * smooth_env(n, 1.5, rng, 0.9)
    return out


def thunder(n, rng):
    out = 0.45 * rain(n, rng)
    t = 0.3
    while t < n / SR - 1:
        L = int(rng.uniform(3, 5) * SR)
        rumble = band_noise(L, 20, 180, rng, tilt=-1) * decay(L, rng.uniform(1.0, 1.8))
        rumble *= smooth_env(L, 6, rng, 0.6)
        crack_len = int(0.35 * SR)
        crack = band_noise(crack_len, 400, 8000, rng) * decay(crack_len, 0.06)
        place(out, 1.4 * rumble, t * SR)
        place(out, 0.7 * crack, t * SR)
        t += rng.uniform(2.5, 4.0)
    return out


def fire(n, rng):
    out = 0.45 * band_noise(n, 40, 900, rng, tilt=-1) * smooth_env(n, 3, rng, 0.35)
    for _ in range(int(n / SR * 35)):
        L = int(rng.uniform(0.003, 0.02) * SR)
        pop = band_noise(L, 1500, 12000, rng) * decay(L, L / SR / 3)
        place(out, pop * rng.uniform(0.2, 0.9), rng.integers(0, n))
    return out


def wind(n, rng):
    t = _t(n)
    base = band_noise(n, 80, 2500, rng, tilt=-0.8)
    howl = band_noise(n, 300, 900, rng) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t + rng.random() * 6))
    return (0.7 * base + 0.4 * howl) * smooth_env(n, 0.4, rng, 0.6)


def ocean(n, rng):
    t = _t(n)
    swell = np.clip(np.sin(2 * np.pi * t / 6.0 - np.pi / 2), 0, None) ** 1.5
    wash = band_noise(n, 80, 6000, rng, tilt=-0.5) * (0.15 + 0.85 * swell)
    foam = band_noise(n, 2000, 10000, rng) * np.roll(swell, int(0.7 * SR)) ** 3
    return wash + 0.3 * foam


def waterfall(n, rng):
    roar = band_noise(n, 30, 8000, rng, tilt=-0.4) * smooth_env(n, 2, rng, 0.15)
    splash = band_noise(n, 1500, 9000, rng) * smooth_env(n, 8, rng, 0.8)
    out = roar + 0.35 * splash
    # jungle birds over the water
    return out + 0.25 * bird(n, rng, density=0.6)


def water(n, rng):
    out = 0.4 * band_noise(n, 200, 5000, rng, tilt=-0.4) * smooth_env(n, 4, rng, 0.5)
    L = int(0.06 * SR)
    for _ in range(int(n / SR * 12)):
        f0 = rng.uniform(400, 1200)
        bub = glide_tone(np.linspace(f0, f0 * 2.2, L)) * decay(L, 0.02)
        place(out, 0.3 * bub, rng.integers(0, n))
    return out


def heartbeat(n, rng, bpm=72):
    out = np.zeros(n)
    period = 60 / bpm
    L = int(0.18 * SR)
    t = _t(L)
    thump = np.sin(2 * np.pi * 55 * t) * decay(L, 0.05) + 0.4 * np.sin(2 * np.pi * 110 * t) * decay(L, 0.02)
    beat = 0.1
    while beat < n / SR:
        place(out, thump, beat * SR)
        place(out, 0.7 * thump, (beat + 0.28) * SR)
        beat += period
    return out + 0.02 * band_noise(n, 20, 200, rng)


def bird(n, rng, density=1.0):
    out = np.zeros(n)
    for _ in range(int(n / SR * 3 * density)):
        start = rng.integers(0, n)
        f0 = rng.uniform(2500, 5000)
        for k in range(rng.integers(2, 6)):
            L = int(rng.uniform(0.04, 0.12) * SR)
            curve = np.linspace(f0 * rng.uniform(0.8, 1.2), f0 * rng.uniform(1.1, 1.6), L)
            chirp = glide_tone(curve) * np.sin(np.pi * np.linspace(0, 1, L))
            place(out, 0.4 * chirp, start + k * int(0.1 * SR))
    return out


def keyboard(n, rng):
    out = 0.01 * band_noise(n, 50, 500, rng)
    t = 0.1
    while t < n / SR:
        L = int(0.03 * SR)
        click = band_noise(L, 1500, 9000, rng) * decay(L, 0.004)
        thock = np.sin(2 * np.pi * rng.uniform(180, 260) * _t(L)) * decay(L, 0.01)
        place(out, rng.uniform(0.5, 1) * (click + 0.5 * thock), t * SR)
        t += rng.uniform(0.05, 0.14) if rng.random() > 0.08 else rng.uniform(0.3, 0.6)
    return out


def clock(n, rng):
    out = np.zeros(n)
    L = int(0.02 * SR)
    for i in range(int(n / SR * 2)):
        f = 3200 if i % 2 == 0 else 2700
        tick = np.sin(2 * np.pi * f * _t(L)) * decay(L, 0.003) + 0.3 * band_noise(L, 2000, 8000, rng) * decay(L, 0.002)
        place(out, tick, (0.1 + i * 0.5) * SR)
    return out


def footsteps(n, rng):
    out = 0.01 * band_noise(n, 50, 1000, rng)
    t = 0.2
    while t < n / SR:
        L = int(0.12 * SR)
        step = band_noise(L, 60, 2500, rng, tilt=-0.8) * decay(L, 0.03)
        place(out, rng.uniform(0.6, 1.0) * step, t * SR)
        t += rng.uniform(0.5, 0.6)
    return out


def traffic(n, rng):
    out = 0.45 * band_noise(n, 30, 1500, rng, tilt=-1) * smooth_env(n, 0.3, rng, 0.4)
    # passing cars — noise swells with a doppler-ish engine tone
    for _ in range(int(n / SR * 0.8)):
        L = int(rng.uniform(2.5, 4) * SR)
        x = np.linspace(-1, 1, L)
        env = 1 / (1 + 8 * x ** 2)
        f = rng.uniform(80, 140) * (1 - 0.12 * np.tanh(3 * x))
        car = (0.4 * glide_tone(f, harmonics=(1, 0.5, 0.3)) + band_noise(L, 200, 4000, rng)) * env
        place(out, 0.6 * car, rng.integers(-L // 2, n))
    # the odd horn
    for _ in range(rng.integers(1, 3)):
        L = int(rng.uniform(0.3, 0.7) * SR)
        horn = glide_tone(np.full(L, rng.uniform(380, 460)), harmonics=(1, 0.8, 0.5, 0.3)) * np.minimum(1, np.linspace(0, 20, L)) * np.minimum(1, np.linspace(20, 0, L))
        place(out, 0.3 * horn, rng.integers(0, n - L))
    return out


def crowd(n, rng, cafe=False):
    out = np.zeros(n)
    for _ in range(14):
        # each voice: speech-band noise with syllable-rate amplitude and formant wobble
        v = band_noise(n, 150, 3500, rng, tilt=-0.6)
        syll = np.clip(smooth_env(n, rng.uniform(3, 6), rng, 1.0) - 0.35, 0, None)
        talk = smooth_env(n, 0.3, rng, 1.0) > 0.4
        out += v * syll * talk
    out /= np.abs(out).max() + 1e-9
    if cafe:
        L = int(0.25 * SR)
        for _ in range(int(n / SR * 1.2)):
            f = rng.uniform(2500, 4500)
            clink = sum(np.sin(2 * np.pi * f * r * _t(L)) for r in (1, 2.76, 5.4)) * decay(L, 0.05)
            place(out, 0.12 * clink, rng.integers(0, n))
        # espresso machine hiss now and then
        L = int(2.5 * SR)
        place(out, 0.18 * band_noise(L, 3000, 10000, rng) * np.sin(np.pi * np.linspace(0, 1, L)), rng.integers(0, n - L))
    return out


def spaceship(n, rng):
    t = _t(n)
    drone = np.zeros(n)
    for f, a in ((42, 1.0), (42.4, 0.8), (63, 0.5), (84.3, 0.4), (126, 0.25)):
        drone += a * np.sin(2 * np.pi * f * t + rng.random() * 6)
    drone *= 1 + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    rumble = band_noise(n, 20, 300, rng, tilt=-1)
    hum = 0.15 * np.sin(2 * np.pi * 240 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    air = 0.12 * band_noise(n, 1000, 6000, rng) * smooth_env(n, 0.5, rng, 0.7)
    return 0.5 * drone + 0.5 * rumble + hum + air


def laser(n, rng):
    out = np.zeros(n)
    t = 0.2
    while t < n / SR - 0.5:
        L = int(rng.uniform(0.18, 0.35) * SR)
        f0 = rng.uniform(1800, 3200)
        curve = f0 * np.exp(-np.linspace(0, 2.2, L))
        zap = glide_tone(curve, harmonics=(1, 0.5, 0.3, 0.2)) * decay(L, L / SR / 2.5)
        place(out, zap, t * SR)
        t += rng.choice([0.12, 0.25, 0.25, 0.6])
    return out


def portal(n, rng):
    t = _t(n)
    x = t / t[-1]
    swirl_f = 60 + 180 * x + 30 * np.sin(2 * np.pi * 0.7 * t)
    swirl = glide_tone(swirl_f, harmonics=(1, 0.6, 0.4, 0.3, 0.2), vibrato=0.05, vib_rate=9)
    whoosh = band_noise(n, 200, 5000, rng) * (0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.3 * t)))
    out = 0.4 * swirl + 0.4 * whoosh * (0.3 + x)
    # electric crackle
    for _ in range(int(n / SR * 25)):
        L = int(rng.uniform(0.005, 0.03) * SR)
        place(out, band_noise(L, 3000, 14000, rng) * decay(L, 0.005) * rng.uniform(0.2, 0.8), rng.integers(0, n))
    return out


def robot(n, rng):
    t = _t(n)
    x = np.clip(t / (n / SR * 0.6), 0, 1)
    whine_f = 80 + 1400 * x ** 1.6
    whine = glide_tone(whine_f, harmonics=(1, 0.5, 0.33, 0.25)) * (0.2 + 0.8 * x)
    hum = 0.3 * np.sin(2 * np.pi * 50 * t) * (x > 0.05) + 0.15 * np.sin(2 * np.pi * 100 * t)
    out = 0.35 * whine + hum
    # servo movements and boot beeps once powered
    for _ in range(8):
        L = int(rng.uniform(0.2, 0.5) * SR)
        f = rng.uniform(300, 700)
        servo = glide_tone(np.linspace(f, f * 1.5, L), harmonics=(1, 0.8, 0.6)) * np.sin(np.pi * np.linspace(0, 1, L))
        place(out, 0.25 * servo, rng.uniform(0.6, 0.95) * n)
    for i, f in enumerate((880, 1320, 1760)):
        L = int(0.12 * SR)
        place(out, 0.35 * np.sin(2 * np.pi * f * _t(L)) * np.minimum(1, np.linspace(10, 0, L)), (0.62 + 0.03 * i) * n)
    return out


# ── glass (VFX grade) ───────────────────────────────────────────────────────

def _pan(mono, pos):
    """Equal-power pan, pos in [-1, 1]."""
    a = (pos + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def _shard(rng, f0, length, dec):
    """One glass fragment ringing: inharmonic plate modes with a click on top."""
    t = _t(length)
    ring = np.zeros(length)
    for ratio, amp in ((1.0, 1.0), (2.32, 0.6), (4.25, 0.35), (6.63, 0.2)):
        if f0 * ratio < 18000:
            ring += amp * np.sin(2 * np.pi * f0 * ratio * t + rng.random() * 6) * decay(length, dec / ratio ** 0.5)
    click_len = min(length, int(0.002 * SR))
    ring[:click_len] += rng.standard_normal(click_len) * np.linspace(1, 0, click_len)
    return ring


def glass_hit(rng, seconds=3.0):
    """A single pane shattering, stereo: crack, impact, burst, flying shards,
    debris bouncing on the floor and a tinkling tail."""
    n = int(seconds * SR)
    out = np.zeros((n, 2))
    at = int(0.06 * SR)  # tiny pre-roll so the transient isn't clipped by fades

    # 1. crack transient — the sharp "snap" that sells the break
    L = int(0.012 * SR)
    crack = band_noise(L, 2500, 18000, rng) * decay(L, 0.002)
    out[at:at + L] += _pan(2.2 * crack, 0)

    # 2. body impact — low thump and mid clunk
    L = int(0.35 * SR)
    t = _t(L)
    thump = np.sin(2 * np.pi * (95 * np.exp(-t * 7) + 45) * t) * decay(L, 0.09)
    clunk = band_noise(L, 250, 1500, rng) * decay(L, 0.035)
    place(out[:, 0], 0.9 * thump + 0.5 * clunk, at)
    place(out[:, 1], 0.9 * thump + 0.5 * clunk, at)

    # 3. breakage burst — dense granular noise, very bright
    L = int(0.4 * SR)
    grains = (rng.random(L) < 0.08) * rng.standard_normal(L)
    burst = (0.6 * band_noise(L, 1800, 16000, rng) + 0.8 * grains) * decay(L, 0.09)
    burst2 = (0.6 * band_noise(L, 1800, 16000, rng) + 0.8 * (rng.random(L) < 0.08) * rng.standard_normal(L)) * decay(L, 0.09)
    out[at:at + L] += _pan(burst, -0.5) * 0.6 + _pan(burst2, 0.5) * 0.6

    # 4. flying shards — many ringing fragments scattered in time and space
    for _ in range(170):
        start = at + int(rng.exponential(0.12) * SR)
        length = int(rng.uniform(0.06, 0.5) * SR)
        f0 = rng.uniform(1800, 9000)
        sh = _shard(rng, f0, length, rng.uniform(0.02, 0.2)) * rng.uniform(0.04, 0.22)
        clip = _pan(sh, rng.uniform(-0.9, 0.9))
        end = min(n, start + length)
        if start < n:
            out[start:end] += clip[: end - start]

    # 5. debris landing and bouncing — sparser, later, each piece bounces 1–3 times
    for _ in range(70):
        land = at + int((0.25 + rng.gamma(1.6, 0.35)) * SR)
        f0 = rng.uniform(2500, 11000)
        amp = rng.uniform(0.05, 0.25) * np.exp(-(land - at) / SR / 1.2)
        pos = rng.uniform(-1, 1)
        gap = rng.uniform(0.04, 0.12)
        for b in range(rng.integers(1, 4)):
            length = int(rng.uniform(0.03, 0.12) * SR)
            s0 = land + int(gap * SR * b * (1 - 0.3 * b))
            if s0 + length < n:
                out[s0:s0 + length] += _pan(_shard(rng, f0, length, 0.02), pos) * amp * (0.55 ** b)

    return out


def glass(n, rng):
    """Several distinct shatters, spaced for easy cutting — a VFX pack."""
    out = np.zeros((n, 2))
    pos = int(0.1 * SR)
    while pos < n - int(1.5 * SR):
        hit = glass_hit(rng, rng.uniform(2.6, 3.2))
        end = min(n, pos + len(hit))
        out[pos:end] += hit[: end - pos] * rng.uniform(0.85, 1.0)
        pos += len(hit) + int(rng.uniform(0.2, 0.5) * SR)
    return out


def glass_crack(n, rng, finale=True):
    """Glass or ice under stress: creaks and spidering micro-fractures that
    escalate, ending in a shatter (VFX build-up)."""
    out = np.zeros((n, 2))
    end_crack = n - int(3.2 * SR) if finale else n
    # low creaking stress groan underneath
    creak = band_noise(n, 120, 700, rng) * smooth_env(n, 3, rng, 1.0) ** 3
    creak *= np.clip(np.linspace(0.2, 1.0, n), 0, 1)
    out += _pan(0.25 * creak, 0)
    t = 0.3
    while t * SR < end_crack:
        progress = t * SR / max(end_crack, 1)
        # one crack event: a cluster of ticks that spider outward over ~50–250 ms
        cluster = int(rng.uniform(6, 30) * (0.5 + progress))
        span = rng.uniform(0.05, 0.25)
        pos = rng.uniform(-0.8, 0.8)
        base = rng.uniform(2500, 7000)
        for i in range(cluster):
            s0 = int((t + span * (i / cluster) ** 0.7) * SR)
            length = int(rng.uniform(0.004, 0.03) * SR)
            if s0 + length >= n:
                break
            amp = rng.uniform(0.15, 0.6) * (0.4 + 0.8 * progress) * (1 - 0.5 * i / cluster)
            tick = _shard(rng, base * rng.uniform(0.8, 1.3), length, 0.004)
            out[s0:s0 + length] += _pan(tick * amp, pos + rng.uniform(-0.15, 0.15))
        # the main snap of the event: a very short bright click plus a glassy ping
        L = int(0.06 * SR)
        s0 = int(t * SR)
        if s0 + L < n:
            snap = band_noise(L, 3500, 18000, rng) * decay(L, 0.0012)
            ping = _shard(rng, base * 1.4, L, 0.02) * 0.5
            out[s0:s0 + L] += _pan((snap + ping) * (0.35 + 0.6 * progress), pos)
        t += rng.uniform(0.25, 0.9) * (1.2 - 0.8 * progress)
    if finale:
        hit = glass_hit(rng, 3.0)
        s0 = n - len(hit)
        out[s0:] += hit * 1.2
    return out


def ambient(n, rng):
    """Neutral fallback for sounds with no recipe — soft, evolving room tone."""
    return 0.6 * band_noise(n, 40, 3000, rng, tilt=-1) * smooth_env(n, 0.3, rng, 0.5)


# ── routing ──────────────────────────────────────────────────────────────────

# Ordered: multi-word and more specific phrases first.
RECIPES = [
    (("rain on glass", "rain on window", "rain on the window", "glass rain", "window rain"), rain_on_glass),
    (("thunderstorm", "thunder", "lightning", "storm"), thunder),
    (("glass crack", "glass cracking", "cracking glass", "glass crackling", "crackling glass", "ice crack", "ice cracking", "cracking ice", "window crack", "screen crack"),
     lambda n, r: glass_crack(n, r, finale=False)),
    (("cracks and shatters", "crack and shatter", "cracking then shattering", "glass breaking slowly"), glass_crack),
    (("glass", "shatter", "shattering", "shatters", "window break", "smash", "smashing", "bottle break"), glass),
    *[(kws, animals.RECIPES[name]) for kws, name in animals.KEYWORDS],
    (("coffee shop", "coffee", "cafe", "café", "restaurant"), lambda n, r: crowd(n, r, cafe=True)),
    (("waterfall", "jungle"), waterfall),
    (("ocean", "wave", "waves", "sea", "beach", "surf"), ocean),
    (("campfire", "fire", "fireplace", "crackling", "bonfire"), fire),
    (("portal", "teleport", "wormhole"), portal),
    (("laser", "blaster", "pew", "zap"), laser),
    (("spaceship", "spacecraft", "starship", "ufo", "space"), spaceship),
    (("robot", "android", "machine powering", "power up", "powering up"), robot),
    (("bird", "birds", "chirp", "chirping", "forest", "songbird"), bird),
    (("keyboard", "typing", "type"), keyboard),
    (("heartbeat", "heart", "pulse"), heartbeat),
    (("clock", "tick", "ticking"), clock),
    (("footstep", "footsteps", "walking", "steps"), footsteps),
    (("traffic", "city", "car", "cars", "road", "highway", "street", "engine"), traffic),
    (("crowd", "people", "chatter", "party", "stadium"), crowd),
    (("rain", "rainfall", "drizzle", "downpour"), rain),
    (("wind", "breeze", "gust"), wind),
    (("water", "stream", "river", "brook", "creek"), water),
]


def match_recipe(prompt: str):
    text = prompt.lower()
    for keywords, fn in RECIPES:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                return kw, fn
    return None, ambient


def render(prompt: str, duration: int = DURATION, seed: int | None = None) -> np.ndarray:
    """Render the best-matching recipe for `prompt` as a float signal in [-1, 1]
    (mono, or (n, 2) stereo for recipes that pan)."""
    rng = np.random.default_rng(seed)
    _, fn = match_recipe(prompt)
    n = duration * SR
    x = np.asarray(fn(n, rng), dtype=np.float64)[:n]
    if len(x) < n:
        x = np.pad(x, [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1))
    x -= x.mean(axis=0)
    x = np.tanh(1.2 * x / (np.abs(x).max() + 1e-9))  # gentle limiter
    return _fade(x / (np.abs(x).max() + 1e-9) * 0.89)


def synthesize(prompt: str, path: str, duration: int = DURATION, seed: int | None = None) -> str:
    """Render `prompt` to a 16-bit WAV at `path`."""
    x = render(prompt, duration, seed)
    pcm = (x * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1 if x.ndim == 1 else x.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
