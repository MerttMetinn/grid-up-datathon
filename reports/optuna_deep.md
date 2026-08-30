# optuna derinleştirme — 75 feature (hava dahil) + 60 trial

Üretim: `scripts/26_optuna_deep.py` · 2026-08-27 13:55 · SEED=42
- 75 feature · DÜZ model (hurdle/b5 YOK) · arkadaş ref 1.06483

## 1. En iyi F1 RMSLE (tek fold, tek seed): **1.1079**

En iyi parametreler:
- learning_rate: 0.048247
- num_leaves: 31
- min_data_in_leaf: 344
- feature_fraction: 0.96896
- bagging_fraction: 0.53516
- lambda_l1: 0.0014582
- lambda_l2: 0.60174

## 2. Fold doğrulama (best params, blend)

| fold | blend | warm | cold | best_iter |
|---|---|---|---|---|
| F1 | 1.1130 | 0.6186 | 2.0605 | 262 |
| F2 | 1.2548 | 0.8463 | 2.1423 | 12 |
| F3 | 1.2630 | 0.9692 | 1.9745 | 18 |

## 3. sub_opt_deep.csv — kalibrasyon

| Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|
| -0.096 | +0.008 | -0.068 | -0.024 | 0.096 |

- **sub_opt_deep.csv yazıldı.** F1 blend 1.1130 (arkadaş 29-feat optuna LB 1.06483).
- Karşılaştırma: LB'de arkadaşın optuna'sını (1.0648) geçerse 75-feature değerli.
