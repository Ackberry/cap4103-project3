# Turn completed experiment outputs into readable team handoff artifacts.
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'results'


def main():
    config = json.loads((ROOT / 'config.json').read_text())
    with (ROOT / 'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    conditions = list(dict.fromkeys(r['condition'] for r in rows))
    lookup = {(r['condition'], r['system']): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for system, offset in [('face', -.23), ('voice', 0), ('fusion', .23)]:
        for ax, metric in zip(axes, ['eer', 'd_prime']):
            values = [float(lookup[c, system][metric]) for c in conditions]
            if metric == 'eer':
                values = np.array(values) * 100
            ax.barh(np.arange(len(conditions)) + offset, values, height=.22, label=system.title())
    for ax in axes:
        ax.set_yticks(np.arange(len(conditions)), [c.replace('_', ' ').replace('pct', '%') for c in conditions])
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3)
    axes[0].set_xlabel('Equal error rate (%) — lower is better')
    axes[1].set_xlabel('d-prime — higher indicates greater separation')
    fig.suptitle(f"Held-out test results | k={config['k']} | face weight={config['face_weight']}")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(ROOT / 'comparison.png', dpi=180); plt.close(fig)
    lines = ['# Code-lead results handoff', '',
             f"Frozen settings: k = {config['k']}; face weight = {config['face_weight']}.",
             f"Paired validation/test probes: {config['paired_probe_counts']}.", '',
             'RQ1 validation selected these RQ2 tuning conditions: ' + ', '.join(config['rq2_conditions']) + '.',
             'RQ3 used the same frozen settings for 7.5% errors and the combined error.', '',
             '| Condition | Face EER | Voice EER | Fusion EER | Face d-prime | Voice d-prime | Fusion d-prime |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for c in conditions:
        scores = [lookup[c, s] for s in ('face', 'voice', 'fusion')]
        numbers = [f"{float(r['eer']):.2%}" for r in scores] + [f"{float(r['d_prime']):.3f}" for r in scores]
        lines.append('| ' + ' | '.join([c, *numbers]) + ' |')
    clean = float(lookup['clean', 'face']['eer'])
    worst = max(conditions[:7], key=lambda c: float(lookup[c, 'face']['eer']))
    improved = [c for c in conditions if float(lookup[c, 'fusion']['eer']) < min(float(lookup[c, s]['eer']) for s in ('face', 'voice'))]
    lines += ['', f"RQ1: clean face EER was {clean:.2%}; the highest known-condition face EER was "
              f"{float(lookup[worst, 'face']['eer']):.2%} ({worst}).",
              f"RQ2/RQ3: fusion had strictly lower EER than both individual modalities in {len(improved)} of {len(conditions)} conditions.",
              'Compare d-prime separately using the table; a lower EER does not imply every metric improved.', '',
              'Use comparison.png for the main comparison. Each condition/system folder contains an evaluation.png with distributions, ROC, and DET.', '',
              'Limitation: artificial identity pairing, only 20 users, few face probes, one split, and clean voice throughout. These results do not establish deployment performance or statistical significance.',
              'EER is an approximate empirical crossing, not an operational threshold.']
    (ROOT / 'HANDOFF.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
