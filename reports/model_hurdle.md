# Model hurdle — ölü-trafo iki-aşamalı model

Üretim: `scripts/19_train_hurdle.py` · 2026-08-23 21:02 · SEED=42

## 1. Skorlar — hurdle vs s2+wx referans

| fold | hurdle blend | s2+wx | Δ | warm | cold | AUC(zero) |
|---|---|---|---|---|---|---|
| F1 | **1.1285** | 1.1447 | -0.0162 | 0.6428 | 2.0726 | 0.949 |
| F2 | **1.2327** | 1.2486 | -0.0159 | 0.8146 | 2.1276 | 0.955 |
| F3 | **1.2481** | 1.2759 | -0.0278 | 0.9671 | 1.9352 | 0.919 |

- Önceki sıfır AUC (statik hücre, reports/diagnosis): 0.56 → hurdle classifier: 0.941 ort

## 2. Kohort-eş aylık kalibrasyon (hurdle, 3-seed)

| Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|
| -0.120 | -0.000 | -0.051 | -0.031 | 0.120 |

## 3. Karar

- F1 Δ: -0.0162 · 3-fold ort Δ: -0.0199
- **SONUÇ: hurdle KABUL — submission üretildi**

- submissions/sub_hurdle.csv yazıldı.
