"""
Decoding, layering and "cinematic" mastering.

Freesound gives a real recording, ElevenLabs gives an AI-designed version of the
same sound. We decode both, layer them into one stereo track, and push the result
through an exaggerating chain: big low end, forward presence, heavy compression,
saturation and a loud limiter.
"""
import wave
import logging
import numpy as np

log = logging.getLogger("audify")

SR = 44100
MIN_SECONDS = 10
MAX_SECONDS = 20


# ── I/O ──────────────────────────────────────────────────────────────────────

def decode(data: bytes) -> np.ndarray | None:
    """Decode mp3/ogg/flac/wav bytes to float32 stereo (n, 2) at 44.1 kHz."""
    try:
        import miniaudio

        snd = miniaudio.decode(
            data,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=2,
            sample_rate=SR,
        )
        x = np.asarray(snd.samples, dtype=np.float32).reshape(-1, 2)
        return x if len(x) > SR // 2 else None
    except Exception as e:
        log.warning("[Mixer] decode failed: %s", e)
        return None


def write_wav(path: str, x: np.ndarray) -> str:
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def write_audio(base: str, x: np.ndarray) -> str:
    """Save as 192 kbps MP3 when an encoder is available (much smaller for the
    browser), otherwise as 16-bit WAV. Returns the path written."""
    try:
        import lameenc

        x = to_stereo(np.asarray(x))
        enc = lameenc.Encoder()
        enc.set_bit_rate(192)
        enc.set_in_sample_rate(SR)
        enc.set_channels(2)
        enc.set_quality(2)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        data = bytes(enc.encode(pcm.tobytes()) + enc.flush())
        path = base + ".mp3"
        with open(path, "wb") as f:
            f.write(data)
        return path
    except ImportError:
        return write_wav(base + ".wav", x)


def to_stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


# ── editing ──────────────────────────────────────────────────────────────────

def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def fit_length(x: np.ndarray, n: int, xfade: float = 0.5) -> np.ndarray:
    """Trim to n samples, or loop with a crossfade until it is n samples long."""
    if len(x) >= n:
        return x[:n].copy()
    k = min(int(xfade * SR), len(x) // 3)
    out = x.copy()
    while len(out) < n:
        ramp = np.linspace(0, 1, k)[:, None]
        seam = out[-k:] * (1 - ramp) + x[:k] * ramp
        out = np.concatenate([out[:-k], seam, x[k:]])
    return out[:n]


def fade(x: np.ndarray, fade_in: float = 0.03, fade_out: float = 0.6) -> np.ndarray:
    a, b = int(fade_in * SR), int(fade_out * SR)
    x = x.copy()
    if a:
        x[:a] *= np.linspace(0, 1, a)[:, None]
    if b:
        x[-b:] *= np.linspace(1, 0, b)[:, None]
    return x


def clamp_length(x: np.ndarray) -> np.ndarray:
    """Loop short clips up to MIN_SECONDS and trim long ones to MAX_SECONDS."""
    n = int(np.clip(len(x), MIN_SECONDS * SR, MAX_SECONDS * SR))
    return fit_length(x, n)


def layer(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Layer two takes of the same sound: the real recording as the body and the
    AI take on top, loudness-matched so neither buries the other."""
    n = int(np.clip(max(len(base), len(top)), MIN_SECONDS * SR, MAX_SECONDS * SR))
    base = fit_length(base, n)
    top = fit_length(top, n)
    target = max(_rms(base), _rms(top))
    base *= target / _rms(base)
    top *= target / _rms(top)
    # small offset so identical transients don't phase-cancel
    top = np.roll(top, int(0.012 * SR), axis=0)
    return 0.6 * base + 0.5 * top


def _mono_env(x: np.ndarray, window: float) -> np.ndarray:
    p = (np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)) ** 2
    w = max(1, int(window * SR))
    c = np.cumsum(np.concatenate([[0.0], p]))
    e = np.sqrt((c[w:] - c[:-w]) / w)
    return np.concatenate([np.full(w - 1, e[0]), e])


def trim(x: np.ndarray, floor_db: float = -45) -> np.ndarray:
    """Cut leading/trailing silence so one-shots start right on the hit."""
    env = _mono_env(x, 0.005)
    if env.max() <= 0:
        return x
    loud = np.nonzero(env > env.max() * 10 ** (floor_db / 20))[0]
    if not len(loud):
        return x
    start = max(0, loud[0] - int(0.005 * SR))
    end = min(len(x), loud[-1] + int(0.05 * SR))
    return x[start:end]


def onset(x: np.ndarray) -> int:
    """Sample index of the main attack (first point above 30 % of peak level)."""
    env = _mono_env(x, 0.003)
    return int(np.argmax(env > 0.3 * env.max()))


def align_layer(base: np.ndarray, top: np.ndarray, top_gain: float = 0.6) -> np.ndarray:
    """Stack two takes of an impact with their attacks lined up to the sample,
    so they hit as one bigger sound instead of flamming."""
    base, top = to_stereo(base), to_stereo(top)
    shift = onset(base) - onset(top)
    if shift > 0:
        top = np.concatenate([np.zeros((shift, 2)), top])
    elif shift < 0:
        base = np.concatenate([np.zeros((-shift, 2)), base])
    n = max(len(base), len(top))
    base = np.pad(base, ((0, n - len(base)), (0, 0)))
    top = np.pad(top, ((0, n - len(top)), (0, 0)))
    top = top * (_rms(base) / _rms(top)) * top_gain
    return base + top


def sequence(takes: list, gap=(0.6, 1.2), target=12.0, rng=None) -> np.ndarray:
    """Lay one-shots end to end with pauses, cycling through the takes for
    variation, until `target` seconds are filled (never cutting a take)."""
    rng = rng or np.random.default_rng()
    takes = [to_stereo(t) / (np.max(np.abs(t)) + 1e-9) for t in takes if t is not None and len(t)]
    parts, total, i = [], 0, 0
    while total < target * SR and total < MAX_SECONDS * SR:
        take = takes[i % len(takes)]
        if parts and total + len(take) > MAX_SECONDS * SR:
            break
        pad = np.zeros((int(rng.uniform(*gap) * SR) if parts else int(0.05 * SR), 2))
        parts += [pad, take]
        total += len(pad) + len(take)
        i += 1
    return np.concatenate(parts)


# ── exaggeration chain ───────────────────────────────────────────────────────

def _eq(x: np.ndarray) -> np.ndarray:
    """+7 dB below 120 Hz, +4 dB around 2–6 kHz presence, tame harsh top."""
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    gain = np.ones_like(f)
    gain *= 1 + (10 ** (7 / 20) - 1) / (1 + (f / 120) ** 4)                   # low shelf
    gain *= 1 + (10 ** (4 / 20) - 1) * np.exp(-((np.log2((f + 1) / 3500)) ** 2) / 0.8)  # presence
    gain *= 1 / (1 + (f / 16000) ** 6)                                         # soften fizz
    gain[f < 25] *= (f[f < 25] / 25) ** 2                                      # DC / sub-rumble
    return np.fft.irfft(spec * gain[:, None], len(x), axis=0)


def _envelope(x: np.ndarray, window: float) -> np.ndarray:
    p = np.max(np.abs(x), axis=1) ** 2
    w = max(1, int(window * SR))
    c = np.cumsum(np.concatenate([[0.0], p]))
    env = np.sqrt((c[w:] - c[:-w]) / w)
    return np.concatenate([np.full(w - 1, env[0]), env])


def _compress(x: np.ndarray, threshold_db=-20, ratio=4.0, makeup_db=8) -> np.ndarray:
    env = _envelope(x, 0.015)
    thr = 10 ** (threshold_db / 20) * np.max(env)
    gain = np.ones_like(env)
    over = env > thr
    gain[over] = (env[over] / thr) ** (1 / ratio - 1)
    # smooth the gain so it pumps musically instead of crackling
    k = int(0.03 * SR)
    gain = np.convolve(gain, np.ones(k) / k, mode="same")
    return x * gain[:, None] * 10 ** (makeup_db / 20)


def _air(x: np.ndarray, db: float = 5, corner: float = 7000) -> np.ndarray:
    """High shelf — the sparkle that makes glass and shards cut through."""
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    gain = 1 + (10 ** (db / 20) - 1) / (1 + (corner / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(spec * gain[:, None], len(x), axis=0)


def _transient_shaper(x: np.ndarray, amount: float = 2.5) -> np.ndarray:
    """Boost attacks relative to the body: fast envelope over slow envelope."""
    fast = _mono_env(x, 0.001)
    slow = _mono_env(x, 0.03) + 1e-6
    gain = np.clip((fast / slow) ** 1.5, 1, amount)
    k = int(0.002 * SR)
    gain = np.convolve(gain, np.ones(k) / k, mode="same")
    return x * gain[:, None]


def _hit_onsets(x: np.ndarray, min_gap: float = 0.4) -> list[int]:
    env = _mono_env(x, 0.005)
    thr = 0.35 * env.max()
    quiet = 0.08 * env.max()
    hits, armed, last = [], True, -int(min_gap * SR)
    for i in range(0, len(env), 64):
        if armed and env[i] > thr and i - last > min_gap * SR:
            hits.append(i)
            last, armed = i, False
        elif env[i] < quiet:
            armed = True
    return hits


def _sub_boom(n: int, onsets: list[int], strength: float = 0.5) -> np.ndarray:
    """Cinematic sweetener: a dropping sub-bass thump under every big hit."""
    out = np.zeros(n)
    L = int(0.6 * SR)
    t = np.arange(L) / SR
    boom = np.sin(2 * np.pi * (38 + 45 * np.exp(-t * 9)) * t) * np.exp(-t / 0.18)
    for o in onsets:
        end = min(n, o + L)
        out[o:end] += boom[: end - o]
    return strength * out


def reverb(x: np.ndarray, seconds: float = 1.2, mix: float = 0.18, seed: int = 7) -> np.ndarray:
    """Stereo algorithmic-style tail via FFT convolution with decaying noise."""
    rng = np.random.default_rng(seed)
    L = int(seconds * SR)
    t = np.arange(L) / SR
    ir = rng.standard_normal((L, 2)) * np.exp(-t / (seconds / 6.9))[:, None]
    f = np.fft.rfftfreq(L, 1 / SR)
    ir = np.fft.irfft(np.fft.rfft(ir, axis=0) / (1 + (f / 5000) ** 2)[:, None], L, axis=0)  # darker tail
    ir[: int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))[:, None]  # predelay-ish
    n = len(x) + L
    size = 1 << (n - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, size, axis=0) * np.fft.rfft(ir, size, axis=0), size, axis=0)[: len(x)]
    wet *= _rms(x) / (_rms(wet) + 1e-9)
    return (1 - mix) * x + mix * wet


def _limit(x: np.ndarray, drive: float = 1.5) -> np.ndarray:
    return np.tanh(drive * x / (np.max(np.abs(x)) + 1e-9)) / np.tanh(drive) * 0.966


def _widen(x: np.ndarray, amount: float) -> np.ndarray:
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return np.stack([mid + amount * side, mid - amount * side], axis=1)


def _normalize(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def exaggerate(x: np.ndarray, mode: str = "ambience") -> np.ndarray:
    """
    Make it big. Three flavours:
      ambience — heavy low end, upfront mids, squashed dynamics, saturated and loud
      impact   — VFX hits (glass, smashes): razor transients, sub boom under every
                 hit, extra air/sparkle, wide stereo and a big room tail
      vocal    — animal calls: chesty low end, forward presence, grit, room around it
    """
    x = to_stereo(np.asarray(x, dtype=np.float64))
    x = _normalize(x - x.mean(axis=0))

    if mode == "impact":
        x = _normalize(_eq(x))
        x = _normalize(_air(x, db=6, corner=6000))
        x = _normalize(_transient_shaper(x, 3.0))
        x = 0.8 * x + 0.35 * _compress(x, threshold_db=-24, ratio=3.0, makeup_db=6)
        x = _normalize(x)
        x += _sub_boom(len(x), _hit_onsets(x), 0.55)[:, None]
        x = _widen(_normalize(x), 1.6)
        x = reverb(x, seconds=1.4, mix=0.12)
        x = np.tanh(1.3 * _normalize(x)) / np.tanh(1.3)
        return fade(_limit(x, 1.2), fade_in=0.002, fade_out=0.3)

    if mode == "vocal":
        x = _normalize(_eq(x))
        x = 0.6 * x + 0.6 * _compress(x, threshold_db=-22, ratio=4.0, makeup_db=8)
        x = np.tanh(1.6 * _normalize(x)) / np.tanh(1.6)
        x = reverb(x, seconds=0.9, mix=0.14)
        x = _widen(x, 1.3)
        return fade(_limit(x, 1.4), fade_in=0.01, fade_out=0.3)

    x = _eq(x)
    x /= np.max(np.abs(x)) + 1e-9
    # parallel compression: keep the transients, slam the body underneath
    x = 0.7 * x + 0.6 * _compress(x)
    # saturation adds grit and perceived loudness
    x /= np.max(np.abs(x)) + 1e-9
    x = np.tanh(1.8 * x) / np.tanh(1.8)
    x = _widen(x, 1.4)
    return fade(_limit(x, 1.5))
