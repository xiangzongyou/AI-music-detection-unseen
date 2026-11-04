# FakeMusicCaps final results (3 random seeds, mean ± SD; 2,856 real clips)

## Overview

| Method | Seen: balanced acc. | Seen: real kept | Unseen: mean detection | Unseen: mean AUC |
|---|---|---|---|---|
| Baseline logistic regression | 0.910 ± 0.006 | 0.899 ± 0.008 | 0.350 ± 0.001 | 0.590 ± 0.005 |
| Baseline CNN | 0.975 ± 0.005 | 0.980 ± 0.010 | 0.499 ± 0.055 | 0.887 ± 0.022 |
| Envelope-removed logistic regression | 0.750 ± 0.010 | 0.739 ± 0.016 | 0.394 ± 0.008 | 0.585 ± 0.006 |
| Envelope-removed CNN | 0.947 ± 0.011 | 0.955 ± 0.002 | 0.562 ± 0.074 | 0.893 ± 0.022 |
| Envelope-augmented CNN | 0.966 ± 0.009 | 0.963 ± 0.017 | 0.495 ± 0.095 | 0.901 ± 0.020 |

## Unseen generators: detection rate

| Method | MusicGen_medium | audioldm2 | musicldm | mustango | stable_audio_open |
|---|---|---|---|---|---|
| Baseline logistic regression | 0.070 ± 0.003 | 0.644 ± 0.004 | 0.036 ± 0.003 | 0.951 ± 0.008 | 0.048 ± 0.010 |
| Baseline CNN | 0.330 ± 0.049 | 0.806 ± 0.154 | 0.127 ± 0.045 | 0.974 ± 0.012 | 0.258 ± 0.082 |
| Envelope-removed logistic regression | 0.248 ± 0.021 | 0.583 ± 0.016 | 0.325 ± 0.018 | 0.716 ± 0.011 | 0.100 ± 0.006 |
| Envelope-removed CNN | 0.184 ± 0.109 | 0.985 ± 0.008 | 0.460 ± 0.163 | 0.889 ± 0.081 | 0.292 ± 0.136 |
| Envelope-augmented CNN | 0.338 ± 0.200 | 0.912 ± 0.029 | 0.160 ± 0.037 | 0.776 ± 0.181 | 0.288 ± 0.145 |

## Unseen generators: AUC

| Method | MusicGen_medium | audioldm2 | musicldm | mustango | stable_audio_open |
|---|---|---|---|---|---|
| Baseline logistic regression | 0.461 ± 0.018 | 0.879 ± 0.004 | 0.214 ± 0.012 | 0.968 ± 0.005 | 0.430 ± 0.008 |
| Baseline CNN | 0.895 ± 0.016 | 0.984 ± 0.006 | 0.690 ± 0.080 | 0.994 ± 0.003 | 0.871 ± 0.018 |
| Envelope-removed logistic regression | 0.511 ± 0.008 | 0.744 ± 0.006 | 0.598 ± 0.014 | 0.778 ± 0.008 | 0.293 ± 0.017 |
| Envelope-removed CNN | 0.820 ± 0.017 | 0.994 ± 0.004 | 0.900 ± 0.008 | 0.976 ± 0.013 | 0.776 ± 0.130 |
| Envelope-augmented CNN | 0.884 ± 0.023 | 0.980 ± 0.008 | 0.830 ± 0.053 | 0.966 ± 0.023 | 0.845 ± 0.017 |