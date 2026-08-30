# LEAK'SİZ model — hava (wx_) KULLANMADAN

Üretim: `scripts/27_optuna_nowx.py` · 2026-08-27 14:25 · SEED=42
- 58 feature (hava çıkarıldı) · DÜZ model + anchor
- Referans: wx'li optuna LB 1.0648 (ama leak riski)

## 1. En iyi F1 RMSLE: **1.1105** (wx'li deep optuna: 1.1079)

- learning_rate: 0.038141
- num_leaves: 31
- min_data_in_leaf: 156
- feature_fraction: 0.55605
- bagging_fraction: 0.64118
- lambda_l1: 0.0096444
- lambda_l2: 11.065

## 2. Fold doğrulama (best params, blend)

| fold | blend | warm | cold |
|---|---|---|---|
| F1 | 1.1122 | 0.6114 | 2.0662 |
| F2 | 1.2493 | 0.8382 | 2.1390 |
| F3 | 1.2589 | 0.9606 | 1.9774 |

## 3. sub_nowx.csv (LEAK'SİZ) — kalibrasyon

| Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|
| -0.109 | +0.011 | +0.030 | +0.130 | 0.130 |

- **sub_nowx.csv yazıldı** — hava YOK, diskalifiye riski sıfır.
- F1 blend 1.1122 (wx'li ~1.113). Hava kaybının maliyeti = fark.
