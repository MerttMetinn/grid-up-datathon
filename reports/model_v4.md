# Model v4 — mevsim-nötr çıpalar (lvl_*_full) + p1/p2/p3

Üretim: `scripts/09_train_full.py` · 2026-08-22 17:20 · SEED=42

## 1. Skorlar

### F1  (b6 blend = 1.2692)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| p1 | 1.1235 | 0.6316 | 2.0721 | **1.1234** | -0.1458 | 0.5269 | 1.0690 | 116/63c |
| p2 | 1.1227 | 0.6316 | 2.0700 | **1.1225** | -0.1467 | 0.5269 | 1.0681 | 116/63c |
| p3 | 1.1223 | 0.6294 | 2.0715 | **1.1222** | -0.1470 | 0.5256 | 1.0837 | 116/63c,121/63c,107/74c |

### F2  (b6 blend = 1.2654)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| p1 | 1.2217 | 0.7861 | 2.1366 | **1.2217** | -0.0437 | 0.6978 | 1.2400 | 80/60c |
| p2 | 1.2219 | 0.7861 | 2.1369 | **1.2218** | -0.0436 | 0.6978 | 1.2440 | 80/60c |
| p3 | 1.2175 | 0.7787 | 2.1350 | **1.2174** | -0.0480 | 0.6956 | 1.2390 | 80/60c,84/60c,88/54c |

### F3  (b6 blend = 1.3055)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| p1 | 1.3332 | 1.0788 | 1.9536 | **1.3235** | +0.0180 | 0.9802 | 1.0595 | 642/127c |
| p2 | 1.3337 | 1.0788 | 1.9551 | **1.3240** | +0.0185 | 0.9802 | 1.0646 | 642/127c |
| p3 | 1.3302 | 1.0740 | 1.9535 | **1.3204** | +0.0149 | 0.9771 | 1.0645 | 642/127c,370/135c,420/125c |

## 2. ÖZEL KONTROL — F2 warm

- Önceki tur: 0.7746 (sabitti) · bu tur: **0.7787** (+0.0041)
- **DÜŞMEDİ** — lvl_*_full çıpaları F2 warm'a katkı vermedi; yaz rampası hâlâ fold içinden öğrenilemiyor.

## 3. Feature importance (ana model, gain)

### F2 — ilk 20

| # | feature | gain payı |
|---|---|---|
| 1 | lvl_lf_median_90d | %63.85 |
| 2 | lvl_lf_median_full | %16.50 |
| 3 | lvl_mean_log_28d | %4.71 |
| 4 | static_guc | %3.08 |
| 5 | static_ilce_key | %2.20 |
| 6 | lvl_zero_ratio_30d | %2.12 |
| 7 | grp_n_transformers | %1.75 |
| 8 | static_log_guc | %1.15 |
| 9 | lvl_mean_log_56d | %0.92 |
| 10 | grp_zero_rate_bucket | %0.87 |
| 11 | static_bolge | %0.42 |
| 12 | cal_doy_sin | %0.42 |
| 13 | cal_horizon_days | %0.27 |
| 14 | lvl_std_log_90d | %0.25 |
| 15 | lvl_zero_streak_days | %0.23 |
| 16 | lvl_mean_log_90d | %0.16 |
| 17 | static_guc_bucket | %0.14 |
| 18 | grp_dow_ratio_ilce | %0.12 |
| 19 | lvl_history_days | %0.12 |
| 20 | lvl_trend_slope_90d | %0.08 |
- **lvl_*_full ailesi gain payı: %16.6**

### F3 — ilk 20

| # | feature | gain payı |
|---|---|---|
| 1 | lvl_lf_median_90d | %65.11 |
| 2 | static_ilce_key | %6.25 |
| 3 | static_guc | %4.85 |
| 4 | lvl_lf_median_full | %3.84 |
| 5 | cal_doy_sin | %3.71 |
| 6 | grp_n_transformers | %1.70 |
| 7 | lvl_mean_log_28d | %1.62 |
| 8 | grp_zero_rate_bucket | %1.17 |
| 9 | static_log_guc | %1.13 |
| 10 | lvl_zero_streak_days | %1.02 |
| 11 | cal_hafta | %0.90 |
| 12 | lvl_mean_log_56d | %0.90 |
| 13 | lvl_trend_slope_90d | %0.88 |
| 14 | lvl_std_log_90d | %0.79 |
| 15 | cal_doy_cos | %0.67 |
| 16 | lvl_full_over_28d | %0.60 |
| 17 | lvl_history_days | %0.60 |
| 18 | cal_horizon_days | %0.57 |
| 19 | static_bolge | %0.51 |
| 20 | static_guc_bucket | %0.49 |
- **lvl_*_full ailesi gain payı: %5.4**

## 4. Cold model gain (yeni kriter d)

### F1 cold modeli — ilk 10 · grp_ toplamı %26.9

| # | feature | gain payı |
|---|---|---|
| 1 | static_ilce_key | %26.04 |
| 2 | static_guc | %23.35 |
| 3 | grp_n_transformers | %16.11 |
| 4 | grp_zero_rate_bucket | %10.11 |
| 5 | cal_doy_sin | %6.07 |
| 6 | static_log_guc | %3.58 |
| 7 | cal_doy_cos | %3.40 |
| 8 | cal_hafta | %2.93 |
| 9 | static_bolge | %2.79 |
| 10 | cal_ay | %1.61 |

### F2 cold modeli — ilk 10 · grp_ toplamı %39.2

| # | feature | gain payı |
|---|---|---|
| 1 | static_ilce_key | %24.04 |
| 2 | grp_n_transformers | %23.60 |
| 3 | static_guc | %21.46 |
| 4 | grp_zero_rate_bucket | %11.35 |
| 5 | static_bolge | %5.64 |
| 6 | static_log_guc | %2.99 |
| 7 | cal_horizon_days | %2.16 |
| 8 | grp_dow_ratio_ilce | %1.76 |
| 9 | cal_doy_sin | %1.67 |
| 10 | static_guc_bucket | %1.42 |

### F3 cold modeli — ilk 10 · grp_ toplamı %29.6

| # | feature | gain payı |
|---|---|---|
| 1 | static_ilce_key | %24.38 |
| 2 | static_guc | %17.27 |
| 3 | grp_n_transformers | %17.08 |
| 4 | cal_doy_sin | %13.42 |
| 5 | grp_zero_rate_bucket | %11.79 |
| 6 | static_bolge | %3.92 |
| 7 | static_log_guc | %2.63 |
| 8 | cal_hafta | %2.40 |
| 9 | static_guc_bucket | %1.84 |
| 10 | cal_horizon_days | %1.64 |

## 5. Harman ağırlığı w

- Fold başına optimum: F1=0.70 · F2=0.40 · F3=0.30
- p1 kullanılan (F2): 0.40 · p2/p3 kullanılan (3-fold ort.): **0.45**

## 6. Ablation — lvl_full_over_90d (F3)

- p2 (feature'la): 1.3240 · feature'sız: 1.3291 → katkı **+0.0051**

## 7. Kabul kriterleri

- En iyi varyant (F2 blend): **p3**
- a) F2 blend ≤ 1.205: 1.2174 → SAĞLANMADI
- b) F3 blend < 1.3055: 1.3204 → SAĞLANMADI
- c) F1 blend ≤ 1.13: 1.1222 → SAĞLANDI
- d) cold model grp_ gain ≥ %25 (F2): %39.2 → SAĞLANDI
- e) F2 warm < 0.7746: 0.7787 → SAĞLANMADI
- **SONUÇ: KRİTER DÜŞTÜ — DUR**

## 8. F3 kesim analizi — model−b6 nerede kaybediyor

| kesim | n | model | b6 | model−b6 |
|---|---|---|---|---|
| warm · sıfır | 6,404 | 2.9331 | 2.8044 | +0.1287 |
| warm · sıfırdışı | 240,048 | 0.9771 | 0.9564 | +0.0207 |
| cold · sıfır | 4,463 | 6.7619 | 6.8239 | -0.0620 |
| cold · sıfırdışı | 69,707 | 1.0645 | 1.0527 | +0.0118 |

| H_bucket | n | model | b6 | model−b6 |
|---|---|---|---|---|
| 0 (cold) | 74,170 | 1.9535 | 1.9605 | -0.0070 |
| 1-30 | 10,484 | 1.0505 | 1.1789 | -0.1284 |
| 31-90 | 16,294 | 1.1782 | 1.1982 | -0.0200 |
| 91-180 | 66,518 | 1.2322 | 1.2334 | -0.0012 |
| 181-300 | 36,277 | 1.0609 | 0.8245 | +0.2363 |
| 301-455 | 116,879 | 0.9624 | 0.9528 | +0.0096 |

- experiments/log.csv güncellendi (3 satır)
