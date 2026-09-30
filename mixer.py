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


def exaggerate(x: np.ndarray) -> np.ndarray:
    """Make it big: heavy low end, upfront mids, squashed dynamics, saturated and loud."""
    x = to_stereo(np.asarray(x, dtype=np.float64))
    x -= x.mean(axis=0)
    x /= np.max(np.abs(x)) + 1e-9
    x = _eq(x)
    x /= np.max(np.abs(x)) + 1e-9
    # parallel compression: keep the transients, slam the body underneath
    x = 0.7 * x + 0.6 * _compress(x)
    # saturation adds grit and perceived loudness
    x /= np.max(np.abs(x)) + 1e-9
    x = np.tanh(1.8 * x) / np.tanh(1.8)
    # widen: mid/side with a boosted side channel
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    x = np.stack([mid + 1.4 * side, mid - 1.4 * side], axis=1)
    # brickwall-ish limiter to -0.3 dBFS
    x = np.tanh(1.5 * x / (np.max(np.abs(x)) + 1e-9)) / np.tanh(1.5) * 0.966
    return fade(x)
