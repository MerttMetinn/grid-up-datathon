# Model hurdle+optuna — mimari + feature selection + optuna params

Üretim: `scripts/20_hurdle_optuna.py` · 2026-08-25 20:05 · SEED=42

- Feature: 29 (arkadaşın seçimi) · cold model: 17

## 1. Skorlar — hurdle+optuna vs bizim hurdle

| fold | hurdle+opt | eski hurdle | Δ | warm | cold | AUC |
|---|---|---|---|---|---|---|
| F1 | **1.1148** | 1.1285 | -0.0137 | 0.6171 | 2.0666 | 0.951 |
| F2 | **1.2401** | 1.2327 | +0.0074 | 0.8293 | 2.1270 | 0.954 |
| F3 | **1.2482** | 1.2481 | +0.0001 | 0.9667 | 1.9361 | 0.925 |

## 2. Kohort-eş aylık kalibrasyon

| Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|
| -0.258 | -0.140 | -0.061 | -0.002 | 0.258 |

- submissions/sub_hurdle_opt.csv yazıldı.
- F1 blend 1.1148 (eski hurdle 1.1285, optuna-düz LB 1.065)
