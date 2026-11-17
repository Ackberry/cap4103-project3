# CAP4103 Project 3 — face and voice authentication

## Run

From this directory, create a Python environment and install `requirements.txt`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
.venv/bin/python run.py
.venv/bin/python export_results.py
```

The supplied `data/voice.npz` is the extracted real AudioMNIST feature cache.
The Caltech landmark arrays are included in `data/caltech/` (copied from HW2). To use another location:

```sh
.venv/bin/python run.py --face-data /path/to/homework2 --output results
```

To reproduce the voice cache from the official dataset:

```sh
curl -L --fail https://codeload.github.com/soerenab/AudioMNIST/zip/refs/heads/master -o AudioMNIST.zip
.venv/bin/python prepare_audio.py
```

AudioMNIST source: https://github.com/soerenab/AudioMNIST
The download is approximately 949 MB. Preparation selects 20 of the original
60 speakers using seed 42, retaining all 500 recordings per selected speaker.
Speaker identity, **not the spoken digit**, is the authentication label.

## Research questions and protocol

- **RQ1:** After selecting k on validation data, how do controlled mouth-corner
  directions and magnitudes affect face-authentication EER and d-prime?
- **RQ2:** For the difficult errors identified by RQ1 validation results, does
  score-level face/voice fusion improve performance compared with either alone?
- **RQ3:** Does the frozen RQ2 configuration compensate for unseen magnitudes
  and combinations?

Use 20 Caltech identities and 20 randomly selected AudioMNIST speakers with a
fixed artificial one-to-one identity mapping. Within each identity and modality,
shuffle samples into 60% enrollment, 20% validation, and the remainder test,
rounding down the first two counts. Identities intentionally appear in all three
sets; individual samples do not. This tests authentication of enrolled users.

Pair validation and test probes without replacement within each mapped identity
and split. Use the smaller available probe count; unused voice probes are not
scored. All enrollment recordings remain available for matching. Each paired
probe receives one genuine claim and 19 impostor claims. Face, voice, and fusion
use identical claims and probe pairs, so their comparisons are aligned.

### Features and matching

The original five-point Caltech arrays contain eye points and a nose point,
not mouth corners. Instead, derive five semantic landmarks from the 68-point
arrays: mean of eye indices 36–41 and 42–47, nose tip 30, and mouth corners 48
and 54 (zero-based). This is a derived representation, not the original HW2
five-point representation. Align the eye midpoint and eye axis and divide by
interocular distance; flatten the resulting five coordinates into 10 features.

Voice features are the temporal mean and population SD of 13 MFCCs (26 values):
16 kHz resampling, peak normalization, 2% peak silence trim with 10 ms margins,
0.97 pre-emphasis, 25 ms Hamming frames / 10 ms hop, 512-point FFT, 26 triangular
Mel bands from 0–8 kHz, log energies, orthonormal type-II DCT including c0.
Standardize voice features using enrollment statistics only.

For each claimed identity, use the negative mean Euclidean distance to its k
nearest enrollment samples. Higher scores indicate closer matches. Choose a
shared k from {1,2,3,5,7}, restricted to each user's enrollment count, minimizing
the mean clean validation EER of face and voice; break ties toward smaller k.
This dataset permits k in {1,2,3}.

### Landmark errors and fusion

Enrollment remains clean. Perturb only validation/test face probes; voice stays
clean. Eye-aligned image coordinates increase downward. At 2%, 5%, and 10% of
interocular distance, move both mouth corners outward, or move the image-left
corner both outward and upward by the stated amount on **each axis**.

Use the two largest face validation EERs among the six known errors to define
RQ2's important conditions (ties use condition name). Fit separate face and voice
min/max bounds on validation scores pooled over clean plus those two conditions.
Test face weights {0.25,0.50,0.75}; minimize mean validation EER over those three
conditions. Break ties toward 0.50, then the smaller weight. Save the configuration
before scoring the test set. Never clip normalized test scores or refit bounds.

Evaluate the frozen configuration on all known errors. RQ3 adds outward 7.5%,
one-corner upward/outward 7.5%, and a combined 7.5% error: both corners outward
plus the image-left corner upward. These conditions never enter tuning. The
combined error is an explicit implementation choice, since the team plan did
not specify its components.

### Evaluation

`evaluation.py` adapts the HW3 evaluator in `../PPA-3/hw3_submission/main.py`.
It retains acceptance at score >= threshold, population-variance d-prime, and
nearest-crossing EER. It replaces fixed 0–1 histogram bins and the fixed threshold
grid with full-range bins and all observed score thresholds plus endpoints.
DET uses normal-deviate axes, clipped only for plotting finite coordinates.

EER is the average FAR and FRR at their closest empirical operating point
(lower is better). It is a descriptive test metric, not a test-selected deployment
threshold. D-prime is the absolute mean separation divided by pooled SD
(higher usually means better separation); inspect distributions and ROC too,
since absolute d-prime alone does not diagnose reversed score ordering.

## Deliverables

- `results/summary.csv`: test metrics for every condition and system.
- `results/comparison.png`: labeled EER/d-prime comparisons.
- `results/HANDOFF.md`: selected settings, experiment links, and findings.
- `results/<condition>/<system>/evaluation.png`: distributions, ROC, and DET.
- Each plot folder also contains raw `scores.npz` and threshold/rate `rates.csv`.
- `results/config.json`: validation trials, frozen settings, mapping, input
  hashes, library versions, platform, seed, and sample counts.
- `results/rq1_validation.json`: evidence selecting RQ2 conditions.
- `results/split_manifest.npz`: original row indices, paired probes, voice
  filenames, and enrollment standardization statistics.
- `run.log`: execution completion; `preparation.log`: real-audio extraction log.

Run `python export_results.py` after `run.py` to regenerate the comparison and
handoff. The source files and saved feature cache can be shared without the
large archive and raw audio. The included Caltech arrays and voice feature cache
are sufficient to run all experiments.

## Limits

These are artificially paired people from separate datasets, not real multimodal
subjects. Only 20 identities and a small number of independent face probes are
available. Impostor claims reuse probes and are not independent observations.
Results describe one seeded split, not population-level certainty or statistical
significance. Voice evaluation uses a small subset of held-out recordings and
does not control spoken-digit balance. Controlled coordinate errors do not model
all detector failures, image conditions, spoofing, or voice degradation. No claim
of improvement is made unless the measured comparisons support it.
