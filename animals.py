"""
Animal vocalisations for the offline synth.

A small source-filter voice engine: a harmonic source whose pitch follows a
contour, shaped by moving formant resonances (the animal's vocal tract), plus
breath noise and "roughness" (the gravel in a roar or growl). Each animal is a
call recipe — pitch contour, formants, grit — sequenced with natural gaps.
"""
import numpy as np

SR = 44100


# ── engine ───────────────────────────────────────────────────────────────────

def curve(points, n):
    """Piecewise-linear contour through (fraction, value) points, n samples long."""
    xs, ys = zip(*points)
    return np.interp(np.linspace(0, 1, n), xs, ys)


def _band(n, lo, hi, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    x = np.fft.irfft(spec * ((f >= lo) & (f <= hi)), n)
    return x / (np.abs(x).max() + 1e-9)


def _formant_gain(freq, formants):
    g = np.full_like(freq, 0.003)
    for F, bw, amp in formants:
        g += amp / (1 + ((freq - F) / (bw / 2)) ** 2)
    return g


def voice(f0, formants, rng, rough=0.0, breath=0.1, tilt=1.0, vibrato=0.0, vib_rate=6.0,
          max_freq=10000.0, saturate=0.0):
    """
    f0        per-sample pitch contour (Hz)
    formants  list of (centre Hz or per-sample contour, bandwidth Hz, gain)
    rough     0..1 — gravel/growl amplitude modulation + subharmonics
    breath    0..1 — formant-shaped noise mixed with the tone
    tilt      spectral roll-off of the source (higher = darker, purer)
    """
    n = len(f0)
    t = np.arange(n) / SR
    wander = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 12), 1 + 0.012 * rng.standard_normal(12))
    f = f0 * wander * (1 + vibrato * np.sin(2 * np.pi * vib_rate * t + rng.random() * 6))
    phase = 2 * np.pi * np.cumsum(f) / SR
    formants = [(np.broadcast_to(F, (n,)), bw, a) for F, bw, a in formants]

    out = np.zeros(n)
    K = int(min(90, max_freq / max(f.min(), 20)))
    for k in range(1, K + 1):
        fk = k * f
        amp = k ** (-tilt) * _formant_gain(fk, formants)
        amp[fk > max_freq] = 0
        out += amp * np.sin(k * phase + rng.random() * 6)

    if rough:
        # period-doubling subharmonic plus fast random amplitude flutter
        out += rough * 0.5 * np.sin(0.5 * phase) * np.abs(out).mean() * 4
        flutter = _band(n, 18, 90, rng)
        out *= 1 + rough * 0.9 * flutter
    out /= np.abs(out).max() + 1e-9

    if breath:
        # approximate the vocal tract on the noise with the mean formant positions
        spec = np.fft.rfft(rng.standard_normal(n))
        fr = np.fft.rfftfreq(n, 1 / SR)
        mean_formants = [(float(np.mean(F)), bw * 1.5, a) for F, bw, a in formants]
        # the vocal tract and air absorb the top end: roll the noise off above ~max_freq/2
        rolloff = 1 / (1 + (fr / (max_freq * 0.5)) ** 4)
        noise = np.fft.irfft(spec * _formant_gain(fr, mean_formants) * rolloff, n)
        noise /= np.abs(noise).max() + 1e-9
        out = (1 - breath) * out + breath * noise

    if saturate:
        out = np.tanh((1 + 6 * saturate) * out)
    return out / (np.abs(out).max() + 1e-9)


def env(n, attack=0.05, release=0.25, shape=1.0):
    """Attack/release envelope; attack and release are fractions of the call."""
    x = np.linspace(0, 1, n)
    a = np.clip(x / max(attack, 1e-4), 0, 1)
    r = np.clip((1 - x) / max(release, 1e-4), 0, 1)
    return (a * r) ** shape


def secs(rng, lo, hi):
    return int(rng.uniform(lo, hi) * SR)


def sequence(n, rng, make_call, gap=(0.5, 1.2), start=(0.1, 0.3)):
    """Fill n samples with calls separated by natural pauses."""
    out = np.zeros(n)
    pos = int(rng.uniform(*start) * SR)
    while pos < n - SR // 4:
        call = make_call(rng)
        end = min(n, pos + len(call))
        out[pos:end] += call[: end - pos]
        pos += len(call) + int(rng.uniform(*gap) * SR)
    return out


def concat(*parts, gaps=None):
    gaps = gaps or [0.0] * (len(parts) - 1)
    pieces = []
    for i, p in enumerate(parts):
        pieces.append(p)
        if i < len(gaps):
            pieces.append(np.zeros(int(gaps[i] * SR)))
    return np.concatenate(pieces)


# ── calls ────────────────────────────────────────────────────────────────────

def meow(rng, kitten=False):
    L = secs(rng, 0.35, 0.55) if kitten else secs(rng, 0.7, 1.15)
    k = 1.8 if kitten else 1.0
    f0 = curve([(0, rng.uniform(420, 520) * k), (0.35, rng.uniform(680, 820) * k), (1, rng.uniform(380, 460) * k)], L)
    F1 = curve([(0, 350), (0.3, 950), (1, 450)], L)
    F2 = curve([(0, 1700), (0.3, 2000), (1, 1000)], L)
    x = voice(f0, [(F1, 250, 1.0), (F2, 400, 0.7), (3300, 600, 0.3)], rng, rough=0.05, breath=0.08, tilt=0.9, vibrato=0.01)
    return x * env(L, 0.12, 0.35)


def bark(rng, big=False, pitch=1.0):
    L = secs(rng, 0.16, 0.24) if big else secs(rng, 0.11, 0.17)
    k = (0.6 if big else 1.0) * pitch
    f0 = curve([(0, rng.uniform(420, 520) * k), (0.25, rng.uniform(540, 640) * k), (1, rng.uniform(240, 300) * k)], L)
    x = voice(f0, [(600 * (0.8 if big else 1), 300, 1.0), (1400, 500, 0.6), (2700, 800, 0.3)], rng,
              rough=0.45, breath=0.35, tilt=0.7, saturate=0.3)
    return x * env(L, 0.03, 0.6, 0.8)


def bark_burst(rng, big=False, pitch=1.0):
    barks = [bark(rng, big, pitch) for _ in range(rng.integers(2, 5))]
    return concat(*barks, gaps=[rng.uniform(0.12, 0.22) for _ in barks[1:]])


def growl(rng, low=90):
    L = secs(rng, 1.4, 2.4)
    f0 = curve([(0, low * 0.9), (0.4, low * 1.1), (1, low * 0.85)], L)
    x = voice(f0, [(400, 250, 1.0), (900, 400, 0.6), (2200, 800, 0.25)], rng, rough=1.0, breath=0.5, tilt=0.8, max_freq=5000)
    return x * env(L, 0.15, 0.3)


def howl(rng, pitch=1.0):
    L = secs(rng, 3.0, 4.5)
    base = rng.uniform(300, 360) * pitch
    f0 = curve([(0, base), (0.12, base * 1.7), (0.6, base * 1.65), (0.85, base * 1.3), (1, base * 1.05)], L)
    F1 = curve([(0, 380), (0.15, 600), (1, 420)], L)
    x = voice(f0, [(F1, 250, 1.0), (950, 350, 0.45), (2600, 700, 0.15)], rng, breath=0.05, tilt=1.4, vibrato=0.012, vib_rate=5,
              max_freq=7000)
    return x * env(L, 0.08, 0.3, 1.2)


def roar(rng, size=1.0):
    L = secs(rng, 1.8, 2.5)
    f0 = curve([(0, 140 * size), (0.15, 220 * size), (0.5, 170 * size), (1, 70 * size)], L)
    F1 = curve([(0, 300), (0.2, 650), (1, 380)], L)
    x = voice(f0, [(F1, 320, 1.0), (1150, 500, 0.6), (2500, 900, 0.3)], rng, rough=1.0, breath=0.55, tilt=0.8, saturate=0.4,
              max_freq=6000)
    return x * env(L, 0.08, 0.45)


def grunt(rng, low=110):
    L = secs(rng, 0.28, 0.4)
    f0 = curve([(0, low), (1, low * 0.65)], L)
    x = voice(f0, [(350, 250, 1.0), (900, 400, 0.5)], rng, rough=0.8, breath=0.65, tilt=0.9, max_freq=4000)
    return x * env(L, 0.08, 0.6)


def lion_call(rng):
    parts = [roar(rng)] + [grunt(rng) * (0.9 - 0.08 * i) for i in range(rng.integers(4, 7))]
    return concat(*parts, gaps=[0.35] + [rng.uniform(0.6, 0.9) + 0.1 * i for i in range(len(parts) - 2)])


def moo(rng):
    L = secs(rng, 1.4, 2.2)
    f0 = curve([(0, 105), (0.2, 150), (0.8, 140), (1, 100)], L) * rng.uniform(0.9, 1.1)
    F1 = curve([(0, 260), (0.25, 680), (1, 500)], L)
    F2 = curve([(0, 900), (0.25, 1150), (1, 850)], L)
    x = voice(f0, [(F1, 200, 1.0), (F2, 300, 0.55), (2400, 600, 0.2)], rng, rough=0.15, breath=0.1, tilt=1.1, max_freq=6000)
    return x * env(L, 0.12, 0.25)


def whinny(rng):
    L = secs(rng, 1.1, 1.5)
    f0 = curve([(0, 850), (0.1, 1300), (0.6, 900), (1, 420)], L)
    x = voice(f0, [(950, 350, 1.0), (1700, 500, 0.6), (3100, 800, 0.3)], rng, rough=0.25, breath=0.3, tilt=0.8,
              vibrato=0.07, vib_rate=11)
    x *= env(L, 0.05, 0.3)
    S = int(0.35 * SR)
    t = np.arange(S) / SR
    snort = _band(S, 200, 3500, rng) * (0.6 + 0.4 * np.sin(2 * np.pi * 32 * t)) * env(S, 0.05, 0.7)
    return concat(x, 0.7 * snort, gaps=[0.25])


def baa(rng, goat=False):
    L = secs(rng, 0.7, 1.0)
    base = rng.uniform(420, 500) if goat else rng.uniform(260, 320)
    f0 = curve([(0, base * 0.9), (0.2, base * 1.1), (1, base * 0.9)], L)
    x = voice(f0, [(750, 300, 1.0), (1300, 400, 0.6), (2600, 700, 0.3)], rng, rough=0.2, breath=0.2, tilt=0.8,
              vibrato=0.06, vib_rate=14 if goat else 11)
    t = np.arange(L) / SR
    return x * (0.6 + 0.4 * np.sin(2 * np.pi * (14 if goat else 11) * t)) * env(L, 0.1, 0.3)


def oink(rng):
    L = secs(rng, 0.14, 0.24)
    f0 = curve([(0, rng.uniform(190, 230)), (1, 150)], L)
    x = voice(f0, [(420, 250, 1.0), (1100, 400, 0.7), (2400, 700, 0.3)], rng, rough=0.6, breath=0.5, tilt=0.7)
    return x * env(L, 0.1, 0.5)


def pig_call(rng):
    if rng.random() < 0.25:
        L = secs(rng, 0.5, 0.7)
        f0 = curve([(0, 1400), (0.3, 2000), (1, 1300)], L)
        return voice(f0, [(2000, 600, 1.0), (3500, 900, 0.5)], rng, rough=0.4, breath=0.3, tilt=0.6, saturate=0.4) * env(L, 0.05, 0.3)
    grunts = [oink(rng) for _ in range(rng.integers(2, 4))]
    return concat(*grunts, gaps=[rng.uniform(0.08, 0.15) for _ in grunts[1:]])


def quack(rng):
    L = secs(rng, 0.14, 0.2)
    f0 = curve([(0, 290), (1, 240)], L)
    x = voice(f0, [(1100, 300, 1.2), (2600, 600, 0.6), (600, 300, 0.3)], rng, rough=0.4, breath=0.3, tilt=0.5, saturate=0.3)
    return x * env(L, 0.06, 0.4)


def quack_run(rng):
    qs = [quack(rng) for _ in range(rng.integers(3, 6))]
    return concat(*qs, gaps=[rng.uniform(0.05, 0.1) for _ in qs[1:]])


def crow_rooster(rng):
    syl = []
    for dur, a, b in ((0.16, 520, 640), (0.16, 680, 700), (0.2, 720, 760), (1.0, 820, 560)):
        L = int(dur * SR)
        f0 = curve([(0, a), (0.3, max(a, b) * 1.03), (1, b)], L)
        v = voice(f0, [(1000, 400, 1.0), (2200, 600, 0.7), (3600, 900, 0.3)], rng, rough=0.4, breath=0.2, tilt=0.6, saturate=0.3)
        syl.append(v * env(L, 0.1, 0.3))
    return concat(*syl, gaps=[0.05, 0.05, 0.06])


def cluck(rng):
    if rng.random() < 0.3:
        L = secs(rng, 0.3, 0.4)
        f0 = curve([(0, 500), (0.3, 720), (1, 430)], L)
    else:
        L = secs(rng, 0.06, 0.09)
        f0 = curve([(0, 480), (1, 400)], L)
    return voice(f0, [(1000, 400, 1.0), (2300, 600, 0.5)], rng, rough=0.3, breath=0.3, tilt=0.7) * env(L, 0.08, 0.5)


def hoot(rng):
    L = secs(rng, 0.3, 0.42)
    f0 = curve([(0, 320), (0.3, 370), (1, 300)], L)
    return voice(f0, [(360, 200, 1.0), (700, 300, 0.2)], rng, breath=0.12, tilt=2.2) * env(L, 0.2, 0.5, 1.5)


def owl_call(rng):
    return concat(hoot(rng), hoot(rng) * 0.8, hoot(rng) * 0.9, hoot(rng), gaps=[0.6, 0.15, 0.15])


def caw(rng):
    L = secs(rng, 0.28, 0.4)
    f0 = curve([(0, 550), (0.3, 660), (1, 440)], L)
    return voice(f0, [(1200, 400, 1.0), (2200, 700, 0.6)], rng, rough=0.7, breath=0.4, tilt=0.6, saturate=0.4) * env(L, 0.05, 0.4)


def caw_run(rng):
    cs = [caw(rng) for _ in range(rng.integers(2, 4))]
    return concat(*cs, gaps=[rng.uniform(0.15, 0.3) for _ in cs[1:]])


def screech(rng):
    L = secs(rng, 1.0, 1.4)
    f0 = curve([(0, 2300), (0.15, 2900), (1, 1700)], L)
    return voice(f0, [(2800, 700, 1.0), (4600, 1000, 0.5)], rng, rough=0.3, breath=0.35, tilt=0.6, vibrato=0.02,
                 vib_rate=18, max_freq=14000) * env(L, 0.05, 0.4)


def trumpet(rng):
    L = secs(rng, 1.3, 1.8)
    f0 = curve([(0, 360), (0.2, 520), (0.8, 500), (1, 420)], L)
    x = voice(f0, [(900, 400, 1.0), (2000, 600, 0.8), (3500, 900, 0.5)], rng, rough=0.5, breath=0.2, tilt=0.3,
              saturate=0.8, vibrato=0.015, vib_rate=7)
    return x * env(L, 0.06, 0.3)


def chimp(rng):
    hoots = []
    for i in range(rng.integers(5, 9)):
        L = secs(rng, 0.14, 0.22)
        p = 300 + 70 * i
        hoots.append(voice(curve([(0, p), (1, p * 1.25)], L), [(500 + 40 * i, 300, 1.0), (1300, 500, 0.5)], rng,
                           rough=0.2, breath=0.35, tilt=0.8) * env(L, 0.2, 0.4))
    L = secs(rng, 0.5, 0.8)
    scream = voice(curve([(0, 1100), (0.3, 1400), (1, 1000)], L), [(1800, 600, 1.0), (3200, 900, 0.6)], rng,
                   rough=0.6, breath=0.3, tilt=0.5, saturate=0.6) * env(L, 0.05, 0.3)
    return concat(*hoots, scream, gaps=[max(0.05, 0.25 - 0.025 * i) for i in range(len(hoots))])


def croak(rng):
    L = secs(rng, 0.25, 0.4)
    f0 = np.full(L, rng.uniform(28, 38))
    x = voice(f0, [(650, 300, 1.0), (1800, 500, 0.7)], rng, tilt=0.0, breath=0.05, max_freq=4000)
    return x * env(L, 0.1, 0.3)


def ribbit(rng):
    return concat(croak(rng), croak(rng) * 0.8, gaps=[0.08])


def hee_haw(rng):
    parts, gaps = [], []
    for _ in range(rng.integers(3, 5)):
        L = secs(rng, 0.3, 0.4)
        parts.append(voice(curve([(0, 700), (1, 850)], L), [(1500, 500, 1.0), (2800, 800, 0.5)], rng,
                           rough=0.4, breath=0.55, tilt=0.6) * env(L, 0.1, 0.3))
        L = secs(rng, 0.45, 0.6)
        parts.append(voice(curve([(0, 260), (1, 200)], L), [(600, 300, 1.0), (1300, 500, 0.6)], rng,
                           rough=0.7, breath=0.3, tilt=0.7, saturate=0.3) * env(L, 0.08, 0.3))
        gaps += [0.03, 0.05]
    return concat(*parts, gaps=gaps[:-1])


# ── continuous creatures ─────────────────────────────────────────────────────

def crickets(n, rng):
    out = np.zeros(n)
    t = np.arange(n) / SR
    for _ in range(rng.integers(4, 8)):
        f = rng.uniform(4200, 5200)
        rate = rng.uniform(1.5, 3.0)
        chirp_gate = (np.sin(2 * np.pi * rate * t + rng.random() * 6) > 0.6).astype(float)
        pulses = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 30 * t))
        out += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t) * chirp_gate * pulses
    return out


def buzz(n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for _ in range(rng.integers(2, 4)):
        dist = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 8), rng.uniform(0.2, 1.0, 8))
        f0 = rng.uniform(190, 250) * (1 + 0.04 * np.gradient(dist) * SR / 4) * (1 + 0.01 * np.sin(2 * np.pi * 7 * t))
        phase = 2 * np.pi * np.cumsum(f0) / SR
        saw = sum(np.sin(k * phase) / k for k in range(1, 30))
        out += saw * dist ** 2
    return out


def hiss(n, rng, rattle=False):
    out = np.zeros(n)
    pos = int(0.2 * SR)
    while pos < n - SR:
        L = secs(rng, 1.3, 2.2)
        h = _band(L, 2500, 11000, rng) * env(L, 0.25, 0.35)
        out[pos:pos + L] += h[: n - pos]
        pos += L + secs(rng, 0.5, 1.0)
    if rattle:
        t = np.arange(n) / SR
        out = 0.5 * out + _band(n, 3000, 12000, rng) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 55 * t))) * 0.6
    return out


# ── recipes (n, rng) -> signal ───────────────────────────────────────────────

def _chorus(n, rng, make, pitches=(1.0, 1.25, 0.85)):
    out = np.zeros(n)
    for i, p in enumerate(pitches):
        out += (1 - 0.25 * i) * sequence(n, rng, lambda r: make(r, p), gap=(0.8, 2.0), start=(0.1 + i * 1.3, 0.3 + i * 1.6))
    return out


RECIPES = {
    "cat": lambda n, r: sequence(n, r, meow, gap=(0.35, 0.9)),
    "kitten": lambda n, r: sequence(n, r, lambda rr: meow(rr, kitten=True), gap=(0.25, 0.6)),
    "dog": lambda n, r: sequence(n, r, bark_burst, gap=(0.5, 1.1)),
    "big dog": lambda n, r: sequence(n, r, lambda rr: bark_burst(rr, big=True), gap=(0.5, 1.1)),
    "puppy": lambda n, r: sequence(n, r, lambda rr: bark_burst(rr, pitch=1.6), gap=(0.3, 0.7)),
    "growl": lambda n, r: sequence(n, r, growl, gap=(0.2, 0.5)),
    "wolf": lambda n, r: _chorus(n, r, howl),
    "lion": lambda n, r: sequence(n, r, lion_call, gap=(0.8, 1.4)),
    "tiger": lambda n, r: sequence(n, r, lambda rr: concat(roar(rr, 1.15), growl(rr, 110), gaps=[0.2]), gap=(0.6, 1.2)),
    "bear": lambda n, r: sequence(n, r, lambda rr: concat(growl(rr, 70), roar(rr, 0.8), gaps=[0.3]), gap=(0.6, 1.2)),
    "cow": lambda n, r: sequence(n, r, moo, gap=(0.8, 1.8)),
    "horse": lambda n, r: sequence(n, r, whinny, gap=(0.8, 1.5)),
    "sheep": lambda n, r: sequence(n, r, baa, gap=(0.6, 1.4)),
    "goat": lambda n, r: sequence(n, r, lambda rr: baa(rr, goat=True), gap=(0.5, 1.2)),
    "pig": lambda n, r: sequence(n, r, pig_call, gap=(0.3, 0.9)),
    "duck": lambda n, r: sequence(n, r, quack_run, gap=(0.5, 1.2)),
    "rooster": lambda n, r: sequence(n, r, crow_rooster, gap=(1.2, 2.2)),
    "chicken": lambda n, r: sequence(n, r, cluck, gap=(0.1, 0.5)),
    "owl": lambda n, r: sequence(n, r, owl_call, gap=(1.0, 2.0)),
    "crow": lambda n, r: sequence(n, r, caw_run, gap=(0.6, 1.4)),
    "eagle": lambda n, r: sequence(n, r, screech, gap=(0.8, 1.6)),
    "elephant": lambda n, r: sequence(n, r, trumpet, gap=(1.0, 2.0)),
    "monkey": lambda n, r: sequence(n, r, chimp, gap=(0.8, 1.6)),
    "frog": lambda n, r: sequence(n, r, ribbit, gap=(0.3, 0.9)),
    "donkey": lambda n, r: sequence(n, r, hee_haw, gap=(1.0, 2.0)),
    "crickets": crickets,
    "bee": buzz,
    "snake": hiss,
    "rattlesnake": lambda n, r: hiss(n, r, rattle=True),
}

# prompt words -> recipe name (whole-word matched by synth.py)
KEYWORDS = [
    (("kitten", "kitty"), "kitten"),
    (("cat", "cats", "meow", "meowing", "purr"), "cat"),
    (("growl", "growling", "snarl", "snarling"), "growl"),
    (("big dog", "rottweiler", "german shepherd", "doberman", "pitbull"), "big dog"),
    (("puppy", "puppies"), "puppy"),
    (("dog", "dogs", "bark", "barking", "woof"), "dog"),
    (("wolf", "wolves", "coyote", "wolf howl", "howling wolf"), "wolf"),
    (("lion", "lions", "lion roar"), "lion"),
    (("tiger", "tigers", "leopard", "jaguar", "panther"), "tiger"),
    (("bear", "bears", "grizzly"), "bear"),
    (("cow", "cows", "moo", "mooing", "cattle", "bull"), "cow"),
    (("horse", "horses", "neigh", "neighing", "whinny", "pony"), "horse"),
    (("goat", "goats"), "goat"),
    (("sheep", "lamb", "baa"), "sheep"),
    (("pig", "pigs", "oink", "hog", "boar"), "pig"),
    (("duck", "ducks", "quack", "quacking", "goose", "geese"), "duck"),
    (("rooster", "cockerel", "cock a doodle"), "rooster"),
    (("chicken", "chickens", "hen", "cluck", "clucking"), "chicken"),
    (("owl", "owls", "hoot", "hooting"), "owl"),
    (("crow", "crows", "raven", "caw", "cawing"), "crow"),
    (("eagle", "hawk", "falcon"), "eagle"),
    (("elephant", "elephants", "trumpeting"), "elephant"),
    (("monkey", "monkeys", "chimp", "chimpanzee", "ape", "gorilla", "baboon"), "monkey"),
    (("frog", "frogs", "toad", "croak", "croaking", "ribbit"), "frog"),
    (("donkey", "donkeys", "mule", "hee haw"), "donkey"),
    (("cricket", "crickets", "cicada", "cicadas", "insects", "insect"), "crickets"),
    (("bee", "bees", "wasp", "wasps", "flies", "mosquito", "bumblebee"), "bee"),
    (("rattlesnake",), "rattlesnake"),
    (("snake", "snakes", "serpent", "cobra", "python snake"), "snake"),
]
