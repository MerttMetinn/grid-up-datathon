# Model CatBoost hurdle — algoritma çeşitliliği (ensemble için)

Üretim: `scripts/21_catboost_hurdle.py` · 2026-08-25 21:02 · SEED=42
- 29 feature · cold 17 · kategorik ['static_guc_bucket', 'static_bolge', 'static_ilce_key']

## 1. CatBoost hurdle skorları (vs LightGBM hurdle+opt)

| fold | catboost | lgb hurdle+opt | Δ | warm | cold | AUC |
|---|---|---|---|---|---|---|
| F1 | **1.1254** | 1.1148 | +0.0106 | 0.6284 | 2.0804 | 0.943 |
| F2 | **1.2676** | 1.2401 | +0.0275 | 0.8680 | 2.1458 | 0.954 |
| F3 | **1.2491** | 1.2482 | +0.0009 | 0.9670 | 1.9381 | 0.925 |

## 2. Submission + ensemble

- submissions/sub_cat.csv yazıldı.
- CatBoost ↔ LightGBM(hurdle+opt) korelasyon: **0.9925** (yüksek→ensemble marjinal)
- Ensemble: sub_cat_lgb_50 (50/50), sub_cat_lgb_40 (40cat/60lgb) yazıldı.

## 3. Kohort-eş kalibrasyon (CatBoost)

| Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|
| -0.224 | -0.150 | -0.075 | -0.030 | 0.224 |

