# Feature extraction, landmark errors, and continuous claimed-user matching.
import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist


def five_landmarks(points):
    # Derive eyes, nose, and mouth corners from the HW2 68-point layout.
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (68, 2):
        raise ValueError("Expected an (n, 68, 2) landmark array.")
    return np.stack([points[:, 36:42].mean(1), points[:, 42:48].mean(1),
                     points[:, 30], points[:, 48], points[:, 54]], axis=1)


def face_features(points, mode="clean", amount=0.0):
    # Align to the eyes; perturb probe mouth corners in interocular units.
    delta = points[:, 1] - points[:, 0]
    scale = np.linalg.norm(delta, axis=1)
    if np.any(scale <= 0):
        raise ValueError("Coincident eyes cannot define a face coordinate system.")
    horizontal = delta / scale[:, None]
    downward = np.stack([-horizontal[:, 1], horizontal[:, 0]], axis=1)
    centered = points - points[:, :2].mean(1)[:, None]
    aligned = np.stack([np.einsum('npi,ni->np', centered, horizontal),
                        np.einsum('npi,ni->np', centered, downward)], axis=-1)
    aligned /= scale[:, None, None]
    if mode in ("outward", "combined"):
        aligned[:, 3, 0] -= amount
        aligned[:, 4, 0] += amount
    if mode == "one_corner":
        aligned[:, 3, 0] -= amount
        aligned[:, 3, 1] -= amount
    elif mode == "combined":
        aligned[:, 3, 1] -= amount
    elif mode not in ("clean", "outward"):
        raise ValueError(f"Unknown landmark error: {mode}")
    return aligned.reshape(len(points), -1)


def voice_features(path):
    # 26 features: mean and SD of 13 MFCCs, using 25 ms speech frames.
    rate, audio = wavfile.read(path)
    audio = audio.astype(float)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite mono audio: {path}")
    divisor = np.gcd(rate, 16000)
    audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    peak = np.max(np.abs(audio))
    if peak == 0:
        raise ValueError(f"Silent recording: {path}")
    audio /= peak
    active = np.flatnonzero(np.abs(audio) > 0.02)
    audio = audio[max(0, active[0] - 160):active[-1] + 161]
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    audio = np.pad(audio, (0, max(0, 400 - len(audio))))
    frames = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160]
    power = np.abs(np.fft.rfft(frames * np.hamming(400), n=512)) ** 2 / 512
    mel_max = 2595 * np.log10(1 + 8000 / 700)
    hz = 700 * (10 ** (np.linspace(0, mel_max, 28) / 2595) - 1)
    freq = np.fft.rfftfreq(512, 1 / 16000)
    filters = np.array([np.maximum(0, np.minimum((freq - a) / (b - a),
                        (c - freq) / (c - b))) for a, b, c in zip(hz, hz[1:], hz[2:])])
    mfcc = dct(np.log(np.maximum(power @ filters.T, 1e-12)),
               type=2, axis=1, norm="ortho")[:, :13]
    return np.concatenate([mfcc.mean(0), mfcc.std(0)])


def distance_scores(enrollment, labels, probes, users, k):
    # Each column is a claimed user; higher (less negative) means closer.
    columns = []
    for user in users:
        templates = enrollment[labels == user]
        if k < 1 or k > len(templates):
            raise ValueError("k must fit every user's enrollment set.")
        distances = cdist(probes, templates)
        columns.append(-np.partition(distances, k - 1, axis=1)[:, :k].mean(1))
    return np.column_stack(columns)


def score_groups(scores, probe_labels, users):
    genuine = probe_labels[:, None] == users[None, :]
    return scores[genuine], scores[~genuine]


def split_users(labels, rng):
    # Split samples within identities 60/20/remainder; never split identities.
    parts = [[], [], []]
    for user in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == user))
        if len(indices) < 5:
            raise ValueError(f"User {user} needs at least five samples.")
        n_train, n_val = int(0.6 * len(indices)), max(1, int(0.2 * len(indices)))
        for target, subset in zip(parts, np.split(indices, [n_train, n_train + n_val])):
            target.extend(subset)
    return [np.asarray(part, dtype=int) for part in parts]


def paired_probes(face_indices, voice_indices, face_labels, voice_labels, users, rng):
    # Pair without replacement inside each split and mapped identity.
    faces, voices = [], []
    for user in users:
        f = rng.permutation(face_indices[face_labels[face_indices] == user])
        v = rng.permutation(voice_indices[voice_labels[voice_indices] == user])
        n = min(len(f), len(v))
        faces.extend(f[:n]); voices.extend(v[:n])
    return np.asarray(faces), np.asarray(voices)
