# LEAK'SİZ recency weighting — son aylara ağırlık

Üretim: `scripts/28_recency.py` · 2026-08-27 15:20 · SEED=42
- 58 feature (hava YOK) · halflife grid [90, 180, 365, None]

## 1. Halflife grid — fold blend skorları

| halflife | F1 | F2 | F3 |
|---|---|---|---|
| 90 | 1.1074 | 1.2969 | 1.3525 |
| 180 | 1.1108 | 1.2923 | 1.3435 |
| 365 | 1.1123 | 1.2975 | 1.3354 |
| yok | 1.1135 | 1.2943 | 1.3352 |

- En iyi halflife (F1'e göre): **90** → F1 1.1074 (ağırlıksız 1.1135)

## 2. sub_recency.csv — kalibrasyon

| Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|
| -0.098 | +0.015 | -0.028 | +0.026 | 0.098 |

- **sub_recency.csv yazıldı** (halflife=90). Leak'siz. sub_nowx_lo LB 1.06525 referans.
