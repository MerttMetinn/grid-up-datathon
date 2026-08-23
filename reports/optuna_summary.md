# Optuna Ozet Raporu (F1 / F2 / F3 -> full_train)

## Fold Bazli En Iyi Sonuclar (her fold kendi optimum parametreleriyle)

| Fold | RMSLE | Best Iteration |
|---|---|---|
| F1 | 1.11098 | 239 |
| F2 | 1.24452 | 49 |
| F3 | 1.24531 | 41 |
| **Ortalama** | **1.20027** | - |

## Fold Bazli En Iyi Hiperparametreler

| Parametre | F1 | F2 | F3 |
|---|---|---|---|
| learning_rate | 0.0342771 | 0.0120203 | 0.0227532 |
| num_leaves | 41 | 75 | 223 |
| min_data_in_leaf | 324 | 70 | 107 |
| feature_fraction | 0.66821 | 0.730132 | 0.602194 |
| bagging_fraction | 0.626021 | 0.755471 | 0.833291 |
| lambda_l1 | 6.24514 | 0.012173 | 9.18031 |
| lambda_l2 | 15.4021 | 5.44111 | 0.107324 |

## Birlestirilmis (Compromise) Parametreler

*Log-olcekli parametreler (learning_rate, lambda_l1, lambda_l2) icin geometrik ortalama, digerleri icin aritmetik ortalama kullanildi.*

| Parametre | Deger |
|---|---|
| learning_rate | 0.0210857 |
| num_leaves | 113 |
| min_data_in_leaf | 167 |
| feature_fraction | 0.666845 |
| bagging_fraction | 0.738261 |
| lambda_l1 | 0.887017 |
| lambda_l2 | 2.07964 |

## Birlestirilmis Parametrelerin Fold Dogrulamasi

| Fold | RMSLE | Best Iteration |
|---|---|---|
| F1 | 1.12491 | 306 |
| F2 | 1.24540 | 29 |
| F3 | 1.25619 | 54 |
| **Ortalama** | **1.20884** | **130** |
