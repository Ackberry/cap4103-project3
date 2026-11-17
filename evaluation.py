"""Adapted from PPA-3/hw3_submission/main.py (HW3 Evaluator).

Retains HW3's acceptance rule, population-variance d-prime, nearest-point
EER, and plots. Uses every observed score threshold instead of a fixed [0,1]
grid so negative distances and out-of-range normalized scores remain valid.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def evaluate(genuine, impostor, output=None, title=""):
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    for scores in (genuine, impostor):
        if not scores.size or not np.isfinite(scores).all():
            raise ValueError("Scores must be nonempty and finite.")
    values = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.r_[-np.inf, values, np.inf]
    fpr = 1 - np.searchsorted(np.sort(impostor), thresholds, side="left") / len(impostor)
    fnr = np.searchsorted(np.sort(genuine), thresholds, side="left") / len(genuine)
    index = np.argmin(np.abs(fpr - fnr))
    pooled = np.sqrt((genuine.var() + impostor.var()) / 2)
    metrics = dict(eer=float((fpr[index] + fnr[index]) / 2),
                   d_prime=float(abs(genuine.mean() - impostor.mean()) / (pooled + 1e-12)),
                   genuine_count=len(genuine), impostor_count=len(impostor))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / 'scores.npz', genuine=genuine, impostor=impostor)
        np.savetxt(output / 'rates.csv', np.column_stack([thresholds, fpr, fnr, 1-fnr]),
                   delimiter=',', header='threshold,fpr,fnr,tpr', comments='')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        lo, hi = values[0], values[-1]
        if lo == hi:
            lo -= 0.5; hi += 0.5
        bins = np.linspace(lo, hi, 26)
        for scores, label in [(genuine, 'Genuine'), (impostor, 'Impostor')]:
            axes[0].hist(scores, bins=bins, density=True, histtype='step', label=label)
        axes[0].set(xlabel='Matching score (higher = closer)', ylabel='Density', title='Score distributions')
        axes[0].legend()
        axes[1].plot(fpr, 1-fnr)
        axes[1].plot([0, 1], [0, 1], '--', color='gray')
        axes[1].set(xlabel='False acceptance rate', ylabel='True acceptance rate', title='ROC')
        # Clip only the display coordinates, never the metrics or raw scores.
        axes[2].plot(norm.ppf(np.clip(fpr, .0001, .9999)), norm.ppf(np.clip(fnr, .0001, .9999)))
        ticks = np.array([.001, .01, .1, .5, .9, .99])
        for axis in [axes[2].xaxis, axes[2].yaxis]:
            axis.set_ticks(norm.ppf(ticks), [f'{100*t:g}%' for t in ticks])
        axes[2].set(xlabel='False acceptance rate', ylabel='False rejection rate', title='DET (normal-deviate axes)')
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f"{title} | EER {metrics['eer']:.2%} | d-prime {metrics['d_prime']:.2f}")
        fig.tight_layout()
        fig.savefig(output / 'evaluation.png', dpi=160)
        plt.close(fig)
    return metrics
