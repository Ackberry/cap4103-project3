# Run the connected RQ1-RQ3 protocol; tune using validation data only.
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy
import matplotlib
from biometrics import (five_landmarks, face_features, distance_scores, score_groups,
                        split_users, paired_probes)
from evaluation import evaluate

ROOT = Path(__file__).resolve().parent
KNOWN = [('clean', 0.0)] + [(mode, amount) for mode in ('outward', 'one_corner')
                          for amount in (.02, .05, .10)]
UNSEEN = [('outward', .075), ('one_corner', .075), ('combined', .075)]


def name(condition):
    mode, amount = condition
    return mode if mode == 'clean' else f'{mode}_{amount*100:g}pct'


def main():
    parser = argparse.ArgumentParser(description='Run the connected RQ1-RQ3 protocol; tune using validation data only.')
    parser.add_argument('--face-data', type=Path, default=ROOT / 'data/caltech')
    parser.add_argument('--voice-data', type=Path, default=ROOT / 'data/voice.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    inputs = [args.face_data / 'X-68-Caltech.npy', args.face_data / 'y-68-Caltech.npy', args.voice_data]
    if not all(path.exists() for path in inputs):
        parser.error('Missing inputs. Run prepare_audio.py after downloading AudioMNIST.')
    face = five_landmarks(np.load(inputs[0]))
    original_face_labels = np.load(inputs[1])
    with np.load(inputs[2]) as cache:
        voice, original_voice_labels, voice_paths = [cache[key] for key in ('features', 'labels', 'paths')]
    if not np.isfinite(face).all() or not np.isfinite(voice).all():
        raise ValueError('Features contain nonfinite values.')
    rng = np.random.default_rng(42)
    n = min(len(np.unique(original_face_labels)), len(np.unique(original_voice_labels)))
    face_ids = rng.choice(np.unique(original_face_labels), n, replace=False)
    voice_ids = rng.choice(np.unique(original_voice_labels), n, replace=False)
    users = np.arange(n)
    fl = np.full(len(face), -1); vl = np.full(len(voice), -1)
    for user, f, v in zip(users, face_ids, voice_ids):
        fl[original_face_labels == f] = user
        vl[original_voice_labels == v] = user
    # Only mapped users participate. Preserve original input row numbers in manifests.
    fi, vi = np.flatnonzero(fl >= 0), np.flatnonzero(vl >= 0)
    fs = [fi[part] for part in split_users(fl[fi], rng)]
    vs = [vi[part] for part in split_users(vl[vi], rng)]
    pairs = [paired_probes(fs[i], vs[i], fl, vl, users, rng) for i in (1, 2)]
    face_enroll = face_features(face[fs[0]])
    voice_mean, voice_sd = voice[vs[0]].mean(0), voice[vs[0]].std(0)
    voice_sd[voice_sd == 0] = 1
    voice = (voice - voice_mean) / voice_sd
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'split_manifest.npz',
                        face_enrollment=fs[0], face_validation=fs[1], face_test=fs[2],
                        voice_enrollment=vs[0], voice_validation=vs[1], voice_test=vs[2],
                        validation_face_probes=pairs[0][0], validation_voice_probes=pairs[0][1],
                        test_face_probes=pairs[1][0], test_voice_probes=pairs[1][1],
                        voice_paths=voice_paths, voice_mean=voice_mean, voice_sd=voice_sd)

    def scores(split, k, condition):
        f, v = pairs[split]
        face_scores = distance_scores(face_enroll, fl[fs[0]], face_features(face[f], *condition), users, k)
        voice_scores = distance_scores(voice[vs[0]], vl[vs[0]], voice[v], users, k)
        return face_scores, voice_scores, fl[f]

    def metrics(matrix, labels, output=None, title=''):
        return evaluate(*score_groups(matrix, labels, users), output=output, title=title)

    max_k = min(np.count_nonzero(fl[fs[0]] == u) for u in users)
    max_k = min(max_k, min(np.count_nonzero(vl[vs[0]] == u) for u in users))
    candidates = [k for k in (1, 2, 3, 5, 7) if k <= max_k]
    k_trials = []
    for k in candidates:
        f, v, labels = scores(0, k, KNOWN[0])
        k_trials.append(dict(k=k, face_eer=metrics(f, labels)['eer'], voice_eer=metrics(v, labels)['eer']))
    chosen = min(k_trials, key=lambda r: ((r['face_eer'] + r['voice_eer']) / 2, r['k']))
    k = chosen['k']
    validation = {name(c): scores(0, k, c) for c in KNOWN}
    ranking = sorted(KNOWN[1:], key=lambda c: (-metrics(validation[name(c)][0], labels)['eer'], name(c)))
    important = ranking[:2]
    # RQ1 validation identifies the two most difficult errors for RQ2 tuning.
    tuning = [KNOWN[0], *important]
    bounds = []
    for modality in (0, 1):
        values = np.concatenate([validation[name(c)][modality].ravel() for c in tuning])
        low, high = float(values.min()), float(values.max())
        if low == high:
            raise ValueError('Cannot normalize constant validation scores.')
        bounds.append((low, high))

    def fuse(f, v, weight):
        normalized = [(x - lo) / (hi - lo) for x, (lo, hi) in zip((f, v), bounds)]
        return weight * normalized[0] + (1 - weight) * normalized[1]

    weight_trials = []
    for w in (.25, .5, .75):
        eers = [metrics(fuse(*validation[name(c)][:2], w), labels)['eer'] for c in tuning]
        weight_trials.append(dict(weight=w, mean_eer=float(np.mean(eers))))
    weight = min(weight_trials, key=lambda r: (r['mean_eer'], abs(r['weight']-.5), r['weight']))['weight']
    config = dict(seed=42, k=k, face_weight=weight, k_trials=k_trials, weight_trials=weight_trials,
                  normalization_bounds=bounds, rq2_conditions=[name(c) for c in tuning],
                  mapping=[dict(user=int(u), face=str(f), voice=str(v)) for u, f, v in zip(users, face_ids, voice_ids)],
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                  platform=platform.platform(), split_counts=dict(face=[len(x) for x in fs], voice=[len(x) for x in vs]),
                  paired_probe_counts=[len(p[0]) for p in pairs])
    # Write the frozen configuration before evaluating any test scores.
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    validation_rows = []
    for c in KNOWN:
        f, v, labels = validation[name(c)]
        validation_rows.append(dict(condition=name(c), **metrics(f, labels)))
    (args.output / 'rq1_validation.json').write_text(json.dumps(validation_rows, indent=2) + '\n')
    rows = []
    for condition in KNOWN + UNSEEN:
        f, v, labels = scores(1, k, condition)
        for system, matrix in [('face', f), ('voice', v), ('fusion', fuse(f, v, weight))]:
            title = f'{name(condition)} / {system}'
            folder = args.output / name(condition) / system
            result = metrics(matrix, labels, folder, title)
            rows.append(dict(phase='RQ3' if condition in UNSEEN else 'RQ1/RQ2',
                             condition=name(condition), system=system, **result))
    with (args.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'Locked k={k}, face weight={weight}; results: {args.output}')


if __name__ == '__main__':
    main()
