# Model v1 — LightGBM (wx_ hariç tüm feature'lar)

Üretim: `scripts/06_train.py` · 2026-08-22 16:39 · SEED=42

## 1. Skorlar (RMSLE)

### F1

| varyant | all | warm | cold | blend | nz_all | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| v1 | 1.2489 | 0.8915 | 2.0605 | **1.2488** | 0.9094 | 0.7353 | 1.3675 | 87 |
| v2 | 1.2826 | 0.8674 | 2.1861 | **1.2824** | 0.9449 | 0.7156 | 1.5096 | 76 |
| v3 | 1.2666 | 0.8686 | 2.1419 | **1.2665** | 0.8613 | 0.6705 | 1.3439 | 76 |

### F2

| varyant | all | warm | cold | blend | nz_all | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| v1 | 2.0819 | 1.5467 | 3.3398 | **2.0818** | 1.8732 | 1.2264 | 3.2567 | 12 |
| v2 | 1.2594 | 0.7861 | 2.2331 | **1.2594** | 1.0149 | 0.7153 | 1.6928 | 75 |
| v3 | 1.2247 | 0.7773 | 2.1553 | **1.2246** | 0.9243 | 0.6951 | 1.4725 | 89 |

### F3

| varyant | all | warm | cold | blend | nz_all | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|---|
| v1 | 1.5353 | 0.9461 | 2.6862 | **1.5152** | 1.3833 | 0.8023 | 2.5072 | 57 |
| v2 | 1.2388 | 0.9093 | 1.9714 | **1.2267** | 0.9398 | 0.7864 | 1.3398 | 70 |
| v3 | 1.2322 | 0.9216 | 1.9343 | **1.2207** | 0.9249 | 0.8117 | 1.2379 | 458 |

- En iyi varyant (F1 blend): **v1 = 1.2488**
- b6 referans: 1.2692 → fark **+0.0204**
- Hedef çıta: 1.07 → kalan mesafe +0.1788

## 2. F1 feature importance (gain) — v1, ilk 25

| # | feature | gain payı |
|---|---|---|
| 1 | lvl_mean_log_90d | %53.55 |
| 2 | lvl_zero_streak_days | %23.32 |
| 3 | lvl_mean_log_56d | %6.58 |
| 4 | static_guc | %3.13 |
| 5 | lvl_days_since_last_nonzero | %1.91 |
| 6 | static_ilce_key | %1.63 |
| 7 | static_log_guc | %0.99 |
| 8 | lvl_is_dead_flag | %0.93 |
| 9 | grp_lf_med_il_bucket_ay | %0.85 |
| 10 | lvl_std_log_90d | %0.80 |
| 11 | grp_seasonal_ilce_ay | %0.72 |
| 12 | lvl_trend_slope_90d | %0.68 |
| 13 | cal_ay | %0.58 |
| 14 | grp_zero_rate_ilce_ay | %0.54 |
| 15 | lvl_lf_median_90d | %0.54 |
| 16 | cal_hafta | %0.53 |
| 17 | lvl_mean_log_28d | %0.52 |
| 18 | cal_doy_cos | %0.50 |
| 19 | grp_lf_med_ilce_ay | %0.48 |
| 20 | lvl_history_days | %0.33 |
| 21 | cal_doy_sin | %0.23 |
| 22 | static_guc_bucket | %0.20 |
| 23 | grp_n_transformers | %0.12 |
| 24 | static_bolge | %0.08 |
| 25 | static_has_bad_rows | %0.07 |

## 3. leak_check (F1)

- Kontrol edilen feature: 43 · uyarılı: 8

| feature | shift σ | valid NaN | tek-feature RMSLE | uyarı |
|---|---|---|---|---|
| seas_lag364_available | +8.35 | %0.0 | 2.1134 | dağılım kayması +8.35σ |
| lvl_history_days | -0.61 | %22.2 | 2.1157 | dağılım kayması -0.61σ |
| cal_is_ramadan | +1.32 | %0.0 | 2.1291 | dağılım kayması +1.32σ |
| cal_doy_cos | +0.81 | %0.0 | 2.1327 | dağılım kayması +0.81σ |
| grp_zero_rate_ilce_ay | +0.66 | %0.0 | 2.1333 | dağılım kayması +0.66σ |
| cal_hafta | -1.44 | %0.0 | 2.1412 | dağılım kayması -1.44σ |
| cal_ay | -1.51 | %0.0 | 2.1428 | dağılım kayması -1.51σ |
| cal_doy_sin | +1.18 | %0.0 | 2.1507 | dağılım kayması +1.18σ |

En güçlü 10 tek-feature RMSLE (bilgi amaçlı):

| feature | tek-feature RMSLE | valid NaN |
|---|---|---|
| lvl_mean_log_28d | 1.2225 | %22.9 |
| lvl_mean_log_56d | 1.2366 | %22.6 |
| lvl_mean_log_90d | 1.2630 | %22.5 |
| lvl_lf_median_90d | 1.4846 | %22.5 |
| lvl_trend_slope_90d | 1.7540 | %22.6 |
| static_log_guc | 1.8061 | %0.0 |
| static_guc | 1.8062 | %0.0 |
| grp_zero_rate_bucket | 1.8379 | %0.0 |
| static_guc_bucket | 1.8379 | %0.0 |
| lvl_zero_streak_days | 1.8450 | %22.2 |

## 4. Kabul kriteri

- Şart: F1 blend ≤ b6 − 0.15 = 1.1192
- Gerçekleşen: 1.2488 → **GEÇEMEDİ**

- experiments/log.csv güncellendi (3 satır)
