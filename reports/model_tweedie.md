# Model Tweedie — sıfır-şişkin loss, ensemble çeşitliliği

Üretim: `scripts/23_tweedie.py` · 2026-08-25 22:56 · SEED=42
- 32 feature (29 + 3 anchor) · Tweedie var_power=1.2

## 1. Tweedie skorları (vs LightGBM hurdle+opt)

| fold | tweedie | hurdle+opt | Δ | warm | cold |
|---|---|---|---|---|---|
| F1 | **1.1464** | 1.1148 | +0.0316 | 0.6713 | 2.0851 |
| F2 | **1.2223** | 1.2401 | -0.0178 | 0.7928 | 2.1294 |
| F3 | **1.2831** | 1.2482 | +0.0349 | 1.0122 | 1.9570 |

## 2. Submission + ensemble (KRİTİK: korelasyon)

- submissions/sub_tw.csv yazıldı.
- Tweedie ↔ optuna(LGB) korelasyon: **0.9733** (yüksek → marjinal)
- Ensemble: sub_tw_opt_30/40/50 (tweedie ağırlığı) yazıldı.

## 3. Kalibrasyon (Tweedie)

| Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|
| -0.036 | +0.095 | +0.076 | +0.049 | 0.095 |
