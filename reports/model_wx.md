# Model wx — hava with/without (s2 üzerinde)

Üretim: `scripts/16_train_wx.py` · 2026-08-23 12:24 · SEED=42

## 1. Skorlar — t0 (wx yok) vs t1 (wx var)

### F1  (b6=1.2692 · s2=1.1244)

| var | all | warm | cold | blend | wx Δ(blend) |
|---|---|---|---|---|---|
| t0 | 1.1410 | 0.6450 | 2.1005 | **1.1409** |  |
| t1 | 1.1448 | 0.6581 | 2.0956 | **1.1447** | +0.0038 |

### F2  (b6=1.2654 · s2=1.2432)

| var | all | warm | cold | blend | wx Δ(blend) |
|---|---|---|---|---|---|
| t0 | 1.2394 | 0.8213 | 2.1360 | **1.2394** |  |
| t1 | 1.2486 | 0.8411 | 2.1330 | **1.2486** | +0.0092 |

### F3  (b6=1.3055 · s2=1.2479)

| var | all | warm | cold | blend | wx Δ(blend) |
|---|---|---|---|---|---|
| t0 | 1.3141 | 1.0385 | 1.9699 | **1.3036** |  |
| t1 | 1.2867 | 0.9996 | 1.9589 | **1.2759** | -0.0277 |

## 2. F2 wx_ önem payı (karar fold'u)

- **wx_ ailesi toplam gain (F2): %5.1**
- ilk 12: static_ilce_key %28.9 · static_guc %13.2 · grp_n_transformers %8.1 · grp_zero_rate_bucket %8.0 · lvl_median_log_full %6.2 · lvl_full_over_28d %4.8 · static_log_guc %4.3 · cal_horizon_days %3.1 · static_guc_bucket %2.0 · static_bolge %2.0 · lvl_std_log_90d %2.0 · lvl_lf_median_90d %1.5

## 3. Kohort-eş aylık kalibrasyon (tam eğitim, 3-seed)

| var | Nis | May | Haz | Tem | max|sapma| |
|---|---|---|---|---|---|
| t0 | -0.103 | +0.024 | +0.020 | +0.081 | 0.103 |
| t1 | -0.118 | -0.012 | -0.049 | -0.022 | 0.118 |

## 4. Karar

- F2 wx Δ (karar fold'u): **+0.0092** (wx yardım etmiyor)
- F1 wx Δ: +0.0038 · F3 wx Δ: -0.0277
- Kalibrasyon: t0 max 0.103 → t1 max 0.118
- wx_ gain payı (F2): %5.1

- **SONUÇ: wx marjinal/olumsuz — karar kullanıcıya**

