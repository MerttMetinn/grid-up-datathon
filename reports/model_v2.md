# Model v2 — çok-origin LightGBM

Üretim: `scripts/07_train_multi.py` · 2026-08-22 16:52 · SEED=42

## 1. Skorlar

### F1

| varyant | all | warm | cold | blend | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|
| m1 | 1.1430 | 0.6898 | 2.0549 | **1.1428** | 0.6013 | 1.2852 | 109 |
| m2 | 1.1221 | 0.6843 | 2.0090 | **1.1220** | 0.5830 | 1.1532 | 102 |
| m3 | 1.1368 | 0.6843 | 2.0458 | **1.1366** | 0.5830 | 1.1662 | 102/67c |

### F2

| varyant | all | warm | cold | blend | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|
| m1 | 1.2676 | 0.7941 | 2.2439 | **1.2675** | 0.7278 | 1.6669 | 73 |
| m2 | 1.2273 | 0.7737 | 2.1666 | **1.2272** | 0.6912 | 1.4233 | 87 |
| m3 | 1.2324 | 0.7737 | 2.1796 | **1.2323** | 0.6912 | 1.4241 | 87/80c |

### F3

| varyant | all | warm | cold | blend | nz_warm | nz_cold | best_iter |
|---|---|---|---|---|---|---|---|
| m1 | 1.4265 | 1.1682 | 2.0645 | **1.4166** | 1.0158 | 1.4145 | 54 |
| m2 | 1.4402 | 1.2163 | 2.0127 | **1.4315** | 1.0635 | 1.3342 | 196 |
| m3 | 1.4595 | 1.2163 | 2.0717 | **1.4501** | 1.0635 | 1.3773 | 196/235c |

## 2. F1 feature importance (m2, gain) — ilk 25 + grup toplamları

| # | feature | gain payı |
|---|---|---|
| 1 | lvl_mean_log_28d | %38.33 |
| 2 | lvl_lf_median_90d | %23.21 |
| 3 | static_ilce_key | %8.41 |
| 4 | static_guc | %7.53 |
| 5 | lvl_zero_ratio_30d | %4.44 |
| 6 | cal_doy_cos | %3.48 |
| 7 | lvl_std_log_90d | %1.90 |
| 8 | static_log_guc | %1.57 |
| 9 | grp_n_transformers | %1.49 |
| 10 | lvl_zero_streak_days | %1.29 |
| 11 | lvl_trend_slope_90d | %1.23 |
| 12 | cal_doy_sin | %1.19 |
| 13 | cal_hafta | %0.99 |
| 14 | grp_zero_rate_bucket | %0.82 |
| 15 | lvl_history_days | %0.67 |
| 16 | static_guc_bucket | %0.65 |
| 17 | cal_horizon_days | %0.52 |
| 18 | lvl_mean_log_90d | %0.49 |
| 19 | lvl_mean_log_56d | %0.48 |
| 20 | static_bolge | %0.34 |
| 21 | lvl_days_since_last_nonzero | %0.27 |
| 22 | cal_ay | %0.27 |
| 23 | lvl_zero_ratio_90d | %0.24 |
| 24 | static_il | %0.14 |
| 25 | lvl_is_dead_flag | %0.02 |

| grup | toplam gain payı |
|---|---|
| static_ | %18.64 |
| cal_ | %6.46 |
| lvl_ | %72.57 |
| grp_ | %2.32 |
| seas_ | %0.00 |

## 3. Zorunlu kontroller

| fold | eğitim satırı | cold satır payı (hedef ~%22) | lvl_ NaN eğitim | lvl_ NaN valid |
|---|---|---|---|---|
| F1 | 1,194,778 | %21.0 | %37.4 | %22.5 |
| F2 | 207,596 | %20.9 | %22.3 | %22.2 |
| F3 | 751,031 | %21.6 | %33.0 | %23.4 |

- **best_iter < 150 uyarısı:** [('F1', 'm1', 109), ('F1', 'm2', 102), ('F1', 'm3_cold', 67), ('F2', 'm1', 73), ('F2', 'm2', 87), ('F2', 'm3_cold', 80), ('F3', 'm1', 54)] — leakage tamamen gitmemiş olabilir veya model erken doyuyor.
- lvl_ ailesi toplam gain: **%72.6** (önceki model: %77+) — **yeterince düşmedi, rapor ediliyor.**

## 4. Referanslar ve kabul

- Referanslar: b6=1.2692 · v3=1.2665 · çıta=1.07 · cold tabanı=1.78 · b5 nz_cold=1.102
- En iyi varyant (F1 blend): **m2 = 1.1220**

- a) F1 blend ≤ 1.15: 1.1220 → SAĞLANDI
- b) F2 < 1.2246 ve F3 < 1.2207: 1.2272 / 1.4315 → SAĞLANMADI
- c) nz_cold ≤ 1.10: 1.1532 → SAĞLANMADI
- **SONUÇ: KRİTER DÜŞTÜ — DUR**

- experiments/log.csv güncellendi (3 satır)
