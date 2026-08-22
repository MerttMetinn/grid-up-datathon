# Model Kartı — s2 (final)

Üretim: `scripts/14_save_model.py` · 2026-08-22 22:45 · SEED=42

## Ne bu model
Grid Up Datathon kazanan varyantı **s2** — trafo bazlı günlük tüketim tahmini.
Yerel CV: F1 blend **1.1244** (baseline b6: 1.2692), aylık kalibrasyon max sapma 0.099.

## Mimari
- **Hedef:** `log1p(tuketim)`, tahmin `expm1(model + init_score)`, `clip(0, None)`.
- **init_score (fiziksel çıpa):** mevsim-farkındalıklı anchor, `α=0.4` yumuşatma,
  cold sıfır düzeltmeli. warm: `lvl_median_log_full + α·mevsim_sapması`.
  cold: `log(guc·24) + α·log(LF_nz_mevsim) + log(1−zero_rate)`.
- **Ana model:** LightGBM, 58 feature (static_/cal_/lvl_/grp_/seas_),
  126 tur, 3 seed ([0, 1, 2]) log-uzayı ortalaması.
- **Cold model:** yalnızca cold örneklerle, 33 feature
  (static_+cal_+grp_), 73 tur. Cold satırlarda `w=0.45` ile b5 baseline'ı
  harmanlanır: `w·model_cold + (1−w)·b5`.

## Hiperparametreler (LightGBM)
```
objective = regression
learning_rate = 0.03
num_leaves = 127
min_data_in_leaf = 100
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 1
lambda_l2 = 1.0
seed = 42
feature_fraction_seed = 42
bagging_seed = 42
num_boost_round = 126 (ana) / 73 (cold)
```

## Artifact'lar
| dosya | içerik |
|---|---|
| `s2_main_seed{0,1,2}.txt` | ana booster (LightGBM native metin formatı) |
| `s2_cold_seed{0,1,2}.txt` | cold-only booster |

## Nasıl yüklenir / tahmin üretilir
```python
import lightgbm as lgb
from src.features import build_features, anchor_components
# ... build_features + anchor kur (bkz. 14_save_model.py), sonra:
bm = lgb.Booster(model_file="models/s2_main_seed0.txt")
pred_log = bm.predict(X_test[ALL_FEATURES]) + init_score
pred = np.clip(np.expm1(pred_log), 0, None)
```
Not: init_score (anchor) tahmin sırasında `anchor_components` ile YENİDEN kurulmalıdır —
booster onu içermez. Tam akış `scripts/14_save_model.py`'de.

## Yeniden üretilebilirlik
SEED=42 sabit, tur sayısı sabit. Bu script `sub_s.csv` ile birebir aynı tahmini
üretir (doğrulama: `sub_s_check.csv` ile diff).
