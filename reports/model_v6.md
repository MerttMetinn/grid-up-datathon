# Model v6 — mevsim-farkındalıklı çapa (r1/r2/r3 + p3r referans)

Üretim: `scripts/12_train_anchor.py` · 2026-08-22 18:33 · SEED=42

## 1. Skorlar

### F1  (b6 = 1.2692)

| varyant | all | warm | cold | blend | model−b6 |
|---|---|---|---|---|---|
| r1 | 1.1781 | 0.7138 | 2.1146 | **1.1779** | -0.0913 |
| r2 | 1.1852 | 0.7138 | 2.1324 | **1.1850** | -0.0842 |
| r3 | 1.1748 | 0.7138 | 2.1064 | **1.1747** | -0.0945 |
| p3r | 1.1265 | 0.6389 | 2.0715 | **1.1263** | -0.1429 |

### F2  (b6 = 1.2654)

| varyant | all | warm | cold | blend | model−b6 |
|---|---|---|---|---|---|
| r1 | 1.2357 | 0.8205 | 2.1273 | **1.2357** | -0.0297 |
| r2 | 1.2404 | 0.8205 | 2.1395 | **1.2403** | -0.0251 |
| r3 | 1.2385 | 0.8205 | 2.1345 | **1.2384** | -0.0270 |
| p3r | 1.2196 | 0.7829 | 2.1350 | **1.2195** | -0.0459 |

### F3  (b6 = 1.3055)

| varyant | all | warm | cold | blend | model−b6 |
|---|---|---|---|---|---|
| r1 | 1.2688 | 0.9567 | 1.9794 | **1.2572** | -0.0483 |
| r2 | 1.2646 | 0.9567 | 1.9678 | **1.2532** | -0.0523 |
| r3 | 1.2647 | 0.9567 | 1.9681 | **1.2533** | -0.0522 |
| p3r | 1.3196 | 1.0568 | 1.9535 | **1.3096** | +0.0041 |

## 2. Çapa etkisi (F1, aynı feature seti, r3 kurgusu)

- Eski çapa log(guc·24): blend 1.1263 · yeni mevsim-farkındalıklı çapa: 1.1747 → fark +0.0483

## 3. F1 gain — lvl_lf_median_90d payı

- **yeni çapa:** lvl_lf_median_90d %0.7 · ilk 5: static_ilce_key %23.9 · lvl_full_over_28d %8.5 · static_guc %7.6 · cal_doy_cos %7.2 · lvl_lf_median_full %7.0
- **eski çapa:** lvl_lf_median_90d %28.1 · ilk 5: lvl_lf_median_90d %28.1 · lvl_mean_log_28d %20.8 · lvl_lf_median_full %14.6 · static_ilce_key %7.2 · lvl_lf_median_364d %6.9

## 4. Tam-eğitim aylık kalibrasyon (d') — 1 seed

- 2025 tabanı: tüm test trafoları (kompozisyon tam eş DEĞİL — bazılarının 2025 verisi yok) ve tam-kapsamlı kohort (1,825 trafo, ≥110/122 gün).

| varyant | Nis | May | Haz | Tem | max |sapma| | d' |
|---|---|---|---|---|---|---|
| r1 | +0.096 | +0.234 | +0.199 | +0.301 | 0.301 | ✗ |
| r2 | +0.097 | +0.252 | +0.218 | +0.331 | 0.331 | ✗ |
| r3 | +0.096 | +0.242 | +0.224 | +0.298 | 0.298 | ✗ |
| p3r | +0.204 | +0.380 | +0.230 | +0.015 | 0.380 | ✗ |

Tam-kapsamlı kohort tabanıyla (kompozisyon-eş):

| varyant | Nis | May | Haz | Tem |
|---|---|---|---|---|
| r1 | -0.124 | -0.000 | +0.052 | +0.268 |
| r2 | -0.124 | +0.018 | +0.071 | +0.299 |
| r3 | -0.124 | +0.007 | +0.077 | +0.266 |
| p3r | -0.016 | +0.145 | +0.083 | -0.018 |

## 5. Aylık kalibrasyon — warm / cold ayrı (tüm-trafo tabanı)

### warm

| varyant | Nis | May | Haz | Tem |
|---|---|---|---|---|
| r1 | +0.095 | +0.149 | +0.131 | +0.226 |
| r2 | +0.095 | +0.149 | +0.131 | +0.226 |
| r3 | +0.095 | +0.149 | +0.131 | +0.226 |
| p3r | +0.203 | +0.292 | +0.098 | -0.148 |

### cold

| varyant | Nis | May | Haz | Tem |
|---|---|---|---|---|
| r1 | +0.250 | +0.533 | +0.377 | +0.488 |
| r2 | +0.356 | +0.615 | +0.445 | +0.594 |
| r3 | +0.289 | +0.567 | +0.468 | +0.479 |
| p3r | +0.405 | +0.688 | +0.573 | +0.420 |

## 6. Anchor kalibrasyonu (init_score aylık ortalaması vs 2025+drift)

| ay | anchor ort. | 2025+drift | fark |
|---|---|---|---|
| 04 | 6.1718 | 6.3167 | -0.1449 |
| 05 | 6.2705 | 6.2746 | -0.0041 |
| 06 | 6.6972 | 6.6959 | +0.0013 |
| 07 | 7.0448 | 7.1153 | -0.0704 |

## 7. Kabul kriterleri

- En iyi varyant (F1 blend): **r3**
- a) F1 blend ≤ 1.13: 1.1747 → SAĞLANMADI
- b) F3 blend ≤ 1.3155: 1.2533 → SAĞLANDI
- c) d' dört ayda sapma ≤ 0.12: ✗ → SAĞLANMADI
- **SONUÇ: KRİTER DÜŞTÜ — DUR**

- experiments/log.csv güncellendi (4 satır)
