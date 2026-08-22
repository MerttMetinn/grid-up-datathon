# Model v7 — A×B çarpımı (s1/s2/s3) + p3r/r3 referans

Üretim: `scripts/13_train_final.py` · 2026-08-22 20:15 · SEED=42

## 1. Alpha grid (B) — α seçimi kalibrasyona göre

| alpha | Nis | May | Haz | Tem | max|sapma| | F1 blend |
|---|---|---|---|---|---|---|
| 0.4 | -0.099 | +0.026 | +0.025 | +0.087 | 0.099 | 1.1244 |
| 0.5 | -0.100 | +0.026 | +0.037 | +0.113 | 0.113 | 1.1297 |
| 0.6 | -0.101 | +0.025 | +0.048 | +0.139 | 0.139 | 1.1347 |
| 0.7 | -0.103 | +0.028 | +0.050 | +0.173 | 0.173 | 1.1357 |
| 0.85 | -0.114 | +0.020 | +0.072 | +0.230 | 0.230 | 1.1661 |
| 1.0 | -0.129 | +0.005 | +0.087 | +0.274 | 0.274 | 1.1734 |

- **Seçilen α* = 0.4** (min max|aylık sapma| = 0.099)

## 2. Skorlar (5 varyant × 3 fold, 1 seed CV)

### F1  (b6 = 1.2692)

| varyant | all | warm | cold | blend | model−b6 |
|---|---|---|---|---|---|
| p3r | 1.1256 | 0.6382 | 2.0700 | **1.1254** | -0.1438 |
| r3 | 1.1763 | 0.7173 | 2.1059 | **1.1762** | -0.0930 |
| s1 | 1.1497 | 0.6629 | 2.1023 | **1.1496** | -0.1196 |
| s2 | 1.1245 | 0.6218 | 2.0849 | **1.1244** | -0.1448 |
| s3 | 1.1298 | 0.6439 | 2.0743 | **1.1297** | -0.1395 |

### F2  (b6 = 1.2654)

| varyant | all | warm | cold | blend | model−b6 |
|---|---|---|---|---|---|
| p3r | 1.2220 | 0.7864 | 2.1369 | **1.2220** | -0.0434 |
| r3 | 1.2399 | 0.8236 | 2.1343 | **1.2399** | -0.0255 |
| s1 | 1.2406 | 0.8253 | 2.1337 | **1.2406** | -0.0248 |
| s2 | 1.2432 | 0.8279 | 2.1368 | **1.2431** | -0.0223 |
| s3 | 1.2396 | 0.8220 | 2.1356 | **1.2396** | -0.0258 |

### F3  (b6 = 1.3055)

| varyant | all | warm | cold | blend | model−b6 |
|---|---|---|---|---|---|
| p3r | 1.3209 | 1.0581 | 1.9551 | **1.3109** | +0.0054 |
| r3 | 1.2637 | 0.9541 | 1.9694 | **1.2522** | -0.0533 |
| s1 | 1.2707 | 0.9756 | 1.9540 | **1.2597** | -0.0458 |
| s2 | 1.2593 | 0.9526 | 1.9596 | **1.2479** | -0.0576 |
| s3 | 1.2697 | 0.9706 | 1.9592 | **1.2585** | -0.0470 |

## 3. KOHORT-EŞ aylık kalibrasyon (birincil karar) — max|sapma| eşik 0.15

| varyant | Nis | May | Haz | Tem | max|sapma| | ✓ |
|---|---|---|---|---|---|---|
| p3r | -0.016 | +0.145 | +0.083 | -0.018 | 0.145 | ✓ |
| r3 | -0.122 | +0.006 | +0.077 | +0.270 | 0.270 | ✗ |
| s1 | -0.134 | -0.015 | -0.060 | -0.097 | 0.134 | ✓ |
| s2 | -0.099 | +0.026 | +0.025 | +0.087 | 0.099 | ✓ |
| s3 | -0.069 | +0.023 | -0.084 | -0.233 | 0.233 | ✗ |

## 4. Kalibrasyon — warm / cold ayrı (kohort-eş taban)

### warm

| varyant | Nis | May | Haz | Tem |
|---|---|---|---|---|
| p3r | -0.018 | +0.057 | -0.049 | -0.181 |
| r3 | -0.123 | -0.086 | -0.017 | +0.196 |
| s1 | -0.136 | -0.130 | -0.203 | -0.230 |
| s2 | -0.100 | -0.068 | -0.090 | -0.036 |
| s3 | -0.070 | -0.062 | -0.231 | -0.416 |

### cold

| varyant | Nis | May | Haz | Tem |
|---|---|---|---|---|
| p3r | +0.185 | +0.453 | +0.427 | +0.387 |
| r3 | +0.076 | +0.331 | +0.321 | +0.453 |
| s1 | +0.141 | +0.388 | +0.311 | +0.234 |
| s2 | +0.091 | +0.354 | +0.323 | +0.393 |
| s3 | +0.076 | +0.318 | +0.300 | +0.221 |

## 5. Cold seviye bias — sıfır düzeltmesi öncesi/sonrası

| varyant | cold_adj | cold bias (cold−warm eş grup) |
|---|---|---|
| r3 | False | +0.0642 |
| s2 | True | +0.1509 |
| s3 | True | +0.2375 |

## 6. F1 ana model gain — mevsim çapası çift-sayımı kırıldı mı

- s3 F1: lvl_lf_median_90d %1.4
- kalan mevsim feature (FEATS_A'da doy/ay yok): cal_horizon_days %4.7 · seas_ toplam %0.0
- ilk 8: static_ilce_key %23.7 · grp_zero_rate_bucket %9.4 · lvl_full_over_28d %8.5 · static_guc %7.1 · grp_n_transformers %6.7 · lvl_lf_median_full %5.8 · cal_horizon_days %4.7 · lvl_zero_streak_days %4.4

## 7. Kabul kriterleri

- Kazanan (min max|sapma|): **s2**
- a) max|aylık sapma| ≤ 0.15: 0.099 → SAĞLANDI
- b) F1 blend ≤ 1.14: 1.1244 → SAĞLANDI
- c) F3 blend ≤ 1.3155: 1.2479 → SAĞLANDI
- d) |cold bias| ≤ 0.15: 0.1509 → SAĞLANMADI
- **SONUÇ: KRİTER DÜŞTÜ**

- submissions/sub_s.csv yazıldı (kazanan=s2, 3-seed).

- experiments/log.csv güncellendi (5 satır)
