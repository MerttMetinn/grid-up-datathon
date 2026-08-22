# Model v3 — origin yayma + grp_ ayrıştırması

Üretim: `scripts/08_train_multi2.py` · 2026-08-22 17:07 · SEED=42

## 1. Skorlar (referans satırı: b6 blend)

### F1  (b6 blend = 1.2692)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| n1 | 1.1276 | 0.6324 | 2.0812 | **1.1274** | -0.1418 | 0.5271 | 1.3235 | 120 |
| n2 | 1.1237 | 0.6324 | 2.0716 | **1.1235** | -0.1457 | 0.5271 | 1.1058 | 120/63c |
| n3 | 1.1239 | 0.6324 | 2.0721 | **1.1237** | -0.1455 | 0.5271 | 1.0690 | 120/63c |

### F2  (b6 blend = 1.2654)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| n1 | 1.2398 | 0.7746 | 2.1974 | **1.2398** | -0.0256 | 0.6957 | 1.4475 | 88 |
| n2 | 1.2313 | 0.7746 | 2.1756 | **1.2312** | -0.0342 | 0.6957 | 1.3433 | 88/60c |
| n3 | 1.2160 | 0.7746 | 2.1366 | **1.2160** | -0.0494 | 0.6957 | 1.2400 | 88/60c |

### F3  (b6 blend = 1.3055)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| n1 | 1.3481 | 1.0897 | 1.9773 | **1.3382** | +0.0327 | 0.9924 | 1.1868 | 392 |
| n2 | 1.3573 | 1.0897 | 2.0044 | **1.3471** | +0.0416 | 0.9924 | 1.1766 | 392/127c |
| n3 | 1.3400 | 1.0897 | 1.9536 | **1.3305** | +0.0250 | 0.9924 | 1.0595 | 392/127c |

## 2. F2 feature importance (n1, gain) — ilk 25 + grup toplamları

| # | feature | gain payı |
|---|---|---|
| 1 | lvl_lf_median_90d | %75.27 |
| 2 | lvl_mean_log_28d | %6.71 |
| 3 | static_guc | %3.46 |
| 4 | lvl_mean_log_56d | %2.62 |
| 5 | static_ilce_key | %2.47 |
| 6 | lvl_zero_ratio_30d | %1.78 |
| 7 | grp_n_transformers | %1.71 |
| 8 | static_log_guc | %1.12 |
| 9 | lvl_mean_log_90d | %1.00 |
| 10 | grp_zero_rate_bucket | %0.79 |
| 11 | cal_doy_sin | %0.40 |
| 12 | static_bolge | %0.38 |
| 13 | lvl_zero_ratio_90d | %0.35 |
| 14 | lvl_std_log_90d | %0.30 |
| 15 | cal_horizon_days | %0.29 |
| 16 | lvl_history_days | %0.22 |
| 17 | lvl_zero_streak_days | %0.16 |
| 18 | cal_doy_cos | %0.15 |
| 19 | static_guc_bucket | %0.14 |
| 20 | grp_dow_ratio_ilce | %0.10 |
| 21 | lvl_trend_slope_90d | %0.09 |
| 22 | grp_lf_med_ilce_ay | %0.09 |
| 23 | lvl_days_since_last_nonzero | %0.08 |
| 24 | cal_ay | %0.07 |
| 25 | cal_dow | %0.04 |

| grup | toplam gain payı |
|---|---|
| static_ | %7.58 |
| cal_ | %0.97 |
| lvl_ | %88.57 |
| grp_ | %2.88 |
| seas_ | %0.00 |

## 3. (origin ayı × hedef ayı) kapsam matrisi

### F1 — valid ayları: [1, 2, 3]

| origin ayı | hedef ayları |
|---|---|
| 02 | 03, 04, 05, 06 |
| 03 | 04, 05, 06, 07 |
| 04 | 05, 06, 07, 08 |
| 05 | 06, 07, 08, 09 |
| 06 | 07, 08, 09, 10 |
| 07 | 08, 09, 10, 11 |
| 08 | 09, 10, 11, 12 |
| 09 | 10, 11, 12 |
| 10 | 11, 12 |
| 11 | 12 |
- Eğitimde hedef olarak görülen aylar: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
- **Valid'de olup eğitim hedefinde HİÇ görülmeyen aylar: [1, 2]**

### F2 — valid ayları: [4, 5, 6, 7]

| origin ayı | hedef ayları |
|---|---|
| 01 | 01, 02, 03 |
| 02 | 03 |
| 03 | 03 |
- Eğitimde hedef olarak görülen aylar: [1, 2, 3]
- **Valid'de olup eğitim hedefinde HİÇ görülmeyen aylar: [4, 5, 6, 7]**

### F3 — valid ayları: [9, 10, 11, 12]

| origin ayı | hedef ayları |
|---|---|
| 01 | 02, 03, 04, 05, 06 |
| 02 | 03, 04, 05, 06 |
| 03 | 04, 05, 06, 07 |
| 04 | 05, 06, 07, 08 |
| 05 | 06, 07, 08 |
| 06 | 07, 08 |
| 07 | 08 |
- Eğitimde hedef olarak görülen aylar: [2, 3, 4, 5, 6, 7, 8]
- **Valid'de olup eğitim hedefinde HİÇ görülmeyen aylar: [9, 10, 11, 12]**

## 4. Kontroller

| fold | eğitim satırı | cold payı | lvl_ NaN eğitim | lvl_ NaN valid | best_iter n1/cold |
|---|---|---|---|---|---|
| F1 | 2,001,625 | %21.2 | %34.7 | %22.5 | 120/63 |
| F2 | 255,713 | %21.0 | %23.3 | %22.2 | 88/60 |
| F3 | 1,202,176 | %21.3 | %33.6 | %23.4 | 392/127 |

- grp_ gain payı (F2): **%2.9** (önceki %2.3, hedef ≥%8)
- n3 harman ağırlığı: **w = 0.40** (F2'de optimize; w=1 saf cold modeli, w=0 saf b5)

## 5. Kabul kriterleri

- En iyi varyant (F2 blend'e göre): **n3**
- a) F2 blend ≤ 1.17: 1.2160 → SAĞLANMADI
- b) F3 blend < 1.3055: 1.3305 → SAĞLANMADI
- c) F1 blend ≤ 1.13: 1.1237 → SAĞLANDI
- d) grp_ gain ≥ %8: %2.9 → SAĞLANMADI
- **SONUÇ: KRİTER DÜŞTÜ — DUR**

- experiments/log.csv güncellendi (3 satır)
