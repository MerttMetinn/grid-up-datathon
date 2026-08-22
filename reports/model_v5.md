# Model v5 — mevsim düzeltmesi (q1/q2)

Üretim: `scripts/11_train_season.py` · 2026-08-22 18:05 · SEED=42

## 1. Skorlar

### F1  (b6 blend = 1.2692)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold |
|---|---|---|---|---|---|---|---|
| q1 | 1.1265 | 0.6389 | 2.0715 | **1.1263** | -0.1429 | 0.5315 | 1.0837 |
| q2 | 1.1381 | 0.6649 | 2.0715 | **1.1380** | -0.1312 | 0.5583 | 1.0837 |

### F2  (b6 blend = 1.2654)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold |
|---|---|---|---|---|---|---|---|
| q1 | 1.2179 | 0.7796 | 2.1350 | **1.2179** | -0.0475 | 0.6974 | 1.2390 |
| q2 | 1.2210 | 0.7858 | 2.1350 | **1.2210** | -0.0444 | 0.7154 | 1.2390 |

### F3  (b6 blend = 1.3055)

| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold |
|---|---|---|---|---|---|---|---|
| q1 | 1.3196 | 1.0568 | 1.9535 | **1.3096** | +0.0041 | 0.9620 | 1.0645 |
| q2 | 1.2734 | 0.9804 | 1.9535 | **1.2625** | -0.0430 | 0.8907 | 1.0645 |

## 2. F1 mevsim feature gain payları (ana model)

| feature | gain payı |
|---|---|
| lvl_season_adjusted_90d | %0.00 |
| lvl_season_adjusted_28d | %0.00 |
| lvl_season_gap | %0.00 |
| lvl_median_log_full | %0.16 |
| lvl_lf_median_364d | %6.85 |

## 3. F3 H 181-300 kesimi (önceki: +0.2363)

- n=36,277 · model 0.9253 · b6 0.8245 → fark **+0.1008**

## 4. w_warm

- w_warm = **0.75** (0.50–1.00 grid, 3-fold warm ortalaması)
- F1 maliyeti: q1 1.1263 → q2 1.1380 (+0.0117)

## 5. Tam-eğitim Temmuz/Mayıs sağlık oranı (1 seed)

- q1: Temmuz/Mayıs = **1.61×** (beklenen ~1.86×, eşik ≥1.6)
- q2: Temmuz/Mayıs = **1.51×** (beklenen ~1.86×, eşik ≥1.6)

## 6. Kabul kriterleri

- En iyi varyant (F1 blend): **q1**
- a) F1 blend ≤ 1.11: 1.1263 → SAĞLANMADI
- b) F3 blend ≤ 1.3155: 1.3096 → SAĞLANDI
- c) F3 H181-300 farkı ≤ +0.10: +0.1008 → SAĞLANMADI
- d) Temmuz/Mayıs ≥ 1.6: 1.61× → SAĞLANDI
- **SONUÇ: KRİTER DÜŞTÜ — DUR**

- experiments/log.csv güncellendi (2 satır)
