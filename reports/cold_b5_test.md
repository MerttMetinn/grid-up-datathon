# Cold b5 harmanı testi — W grid

Üretim: `scripts/22_cold_b5_test.py` · 2026-08-25 22:48
- W=0 cold saf model (b5 yok) · W=0.45 mevcut

## 1. Fold CV — W etkisi (cold RMSLE / blend)

### F1
| W | cold RMSLE | blend |
|---|---|---|
| 0.00 | 2.1020 | 1.1295 |
| 0.25 | 2.0807 | 1.1207 |
| 0.45 | 2.0668 | 1.1150 |
| 0.70 | 2.0536 | 1.1096 |

### F2
| W | cold RMSLE | blend |
|---|---|---|
| 0.00 | 2.1529 | 1.2488 |
| 0.25 | 2.1292 | 1.2398 |
| 0.45 | 2.1246 | 1.2380 |
| 0.70 | 2.1368 | 1.2427 |

### F3
| W | cold RMSLE | blend |
|---|---|---|
| 0.00 | 1.9605 | 1.2533 |
| 0.25 | 1.9422 | 1.2470 |
| 0.45 | 1.9367 | 1.2451 |
| 0.70 | 1.9414 | 1.2467 |

## 2. Tam-eğitim submission + kalibrasyon (her W)

| W | Nis | May | Haz | Tem | cold ort. tahmin log |
|---|---|---|---|---|---|
| 0.00 | -0.258 | -0.119 | -0.045 | +0.002 | 7.273 |
| 0.25 | -0.258 | -0.133 | -0.058 | -0.005 | 7.231 |
| 0.45 | -0.258 | -0.144 | -0.068 | -0.010 | 7.197 |
| 0.70 | -0.259 | -0.158 | -0.081 | -0.017 | 7.155 |

- Submissionlar: sub_w00 (b5 yok), sub_w25, sub_w45 (mevcut), sub_w70.
- Hipotez: W küçüldükçe cold tahmini düşer. LB 'düşük iyi' ise sub_w00 < sub_w45.
