# Baseline sonuçları

Üretim: `scripts/03_run_baselines.py` · 2026-08-22 16:25 · SEED=42

## 1. Fold doğrulaması (verify_fold)

| fold | cold_row_share | hedef | lag364_cov_pm7 | hedef | h_median | hedef | durum |
|---|---|---|---|---|---|---|---|
| F1 | 0.2217 | 0.2216 | 0.3267 | 0.35 | 107 | 105 | OK |
| F2 | 0.2216 | 0.2216 | 0.0000 | 0.35 | 104 | 105 | OK · lag364: yapısal N/A |
| F3 | 0.2313 | 0.2216 | 0.0272 | 0.35 | 104 | 105 | OK · lag364: yapısal N/A |

## 2. Baseline skorları (RMSLE)

### F1  (train_end=2025-12-31, valid=2026-01-01..2026-03-31)

| baseline | all | warm | cold | blend |
|---|---|---|---|---|
| b1_global | 2.1460 | 2.1300 | 2.2012 | 2.1460 |
| b2_trafo | 1.3059 | 0.9006 | 2.2012 | 1.3058 |
| b3_trafo_ay | 1.3570 | 0.9929 | 2.2012 | 1.3569 |
| b4_trafo_ay_hi | 1.3553 | 0.9899 | 2.2012 | 1.3552 |
| b5_guc_lf | 1.8357 | 1.7525 | 2.1020 | 1.8357 |
| b6_hibrit | 1.2692 | 0.9006 | 2.1020 | 1.2691 |

### F2  (train_end=2025-03-31, valid=2025-04-01..2025-07-31)

| baseline | all | warm | cold | blend |
|---|---|---|---|---|
| b1_global | 2.2110 | 2.1620 | 2.3752 | 2.2110 |
| b2_trafo | 1.3507 | 0.8589 | 2.3752 | 1.3507 |
| b3_trafo_ay | 1.3507 | 0.8589 | 2.3752 | 1.3507 |
| b4_trafo_ay_hi | 1.3507 | 0.8589 | 2.3752 | 1.3507 |
| b5_guc_lf | 1.9577 | 1.8984 | 2.1529 | 1.9577 |
| b6_hibrit | 1.2655 | 0.8589 | 2.1529 | 1.2654 |

### F3  (train_end=2025-08-31, valid=2025-09-01..2025-12-31)

| baseline | all | warm | cold | blend |
|---|---|---|---|---|
| b1_global | 2.0293 | 1.9928 | 2.1463 | 2.0278 |
| b2_trafo | 1.3811 | 1.0465 | 2.1463 | 1.3687 |
| b3_trafo_ay | 1.3811 | 1.0465 | 2.1463 | 1.3687 |
| b4_trafo_ay_hi | 1.3811 | 1.0465 | 2.1463 | 1.3687 |
| b5_guc_lf | 1.7469 | 1.6773 | 1.9605 | 1.7441 |
| b6_hibrit | 1.3157 | 1.0465 | 1.9605 | 1.3055 |

## 3. b5 fallback seviyesi kullanımı (F1)

| seviye | satır | pay |
|---|---|---|
| ilce_key+ay_no+haftaici | 299,929 | %100.00 |

## 4. Kabul kriterleri

- verify_fold üç metrikte hedefe yakın: SAĞLANDI
- b5 cold satırlarda b1/b2/b3'ten iyi (F1): SAĞLANDI (b5=2.1020 vs b1=2.2012, b2=2.2012, b3=2.2012)
- b6 global olarak hepsinden iyi (F1): SAĞLANDI (b6=1.2692)

- experiments/log.csv güncellendi (6 satır)
