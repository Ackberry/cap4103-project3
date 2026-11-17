# Code-lead results handoff

Frozen settings: k = 1; face weight = 0.25.
Paired validation/test probes: [68, 88].

RQ1 validation selected these RQ2 tuning conditions: clean, one_corner_10pct, one_corner_5pct.
RQ3 used the same frozen settings for 7.5% errors and the combined error.

| Condition | Face EER | Voice EER | Fusion EER | Face d-prime | Voice d-prime | Fusion d-prime |
|---|---:|---:|---:|---:|---:|---:|
| clean | 23.86% | 4.55% | 4.55% | 1.289 | 3.180 | 3.403 |
| outward_2pct | 24.13% | 4.55% | 4.55% | 1.282 | 3.180 | 3.413 |
| outward_5pct | 26.14% | 4.55% | 4.55% | 1.252 | 3.180 | 3.420 |
| outward_10pct | 25.00% | 4.55% | 4.55% | 1.143 | 3.180 | 3.413 |
| one_corner_2pct | 24.76% | 4.55% | 4.55% | 1.270 | 3.180 | 3.410 |
| one_corner_5pct | 25.21% | 4.55% | 4.55% | 1.218 | 3.180 | 3.406 |
| one_corner_10pct | 26.14% | 4.55% | 4.55% | 1.093 | 3.180 | 3.368 |
| outward_7.5pct | 27.12% | 4.55% | 4.55% | 1.208 | 3.180 | 3.420 |
| one_corner_7.5pct | 26.14% | 4.55% | 4.55% | 1.161 | 3.180 | 3.392 |
| combined_7.5pct | 25.21% | 4.55% | 4.55% | 1.103 | 3.180 | 3.384 |

RQ1: clean face EER was 23.86%; the highest known-condition face EER was 26.14% (outward_5pct).
RQ2/RQ3: fusion had strictly lower EER than both individual modalities in 0 of 10 conditions.
Compare d-prime separately using the table; a lower EER does not imply every metric improved.

Use comparison.png for the main comparison. Each condition/system folder contains an evaluation.png with distributions, ROC, and DET.

Limitation: artificial identity pairing, only 20 users, few face probes, one split, and clean voice throughout. These results do not establish deployment performance or statistical significance.
EER is an approximate empirical crossing, not an operational threshold.
