# Tahmin-seviyesi sağlık kontrolü (pred_sanity)

Üretim: `scripts/10_submit.py` · 2026-08-22 17:40 · SEED=42

## a–c. Aylık ortalama log1p — tahmin vs geçen yıl + drift

| ay | 2026 tahmin | 2025 gerçek (test trafoları) | beklenen (2025+0.102) | fark |
|---|---|---|---|---|
| 04 | 6.5247 | 6.2147 | 6.3167 | +0.2080 |
| 05 | 6.6516 | 6.1726 | 6.2746 | +0.3769 |
| 06 | 6.9172 | 6.5939 | 6.6959 | +0.2213 |
| 07 | 7.1259 | 7.0133 | 7.1153 | +0.0106 |

## d. Temmuz/Mayıs oranı (geometrik)

- Tahmin edilen (2026): **1.61×** · 2025 gerçek (test trafoları): 2.32× · beklenen ~1.86×
- Ramp makul yakalanıyor (1.61× ≥ 1.6).

## e. İlçe bazında Temmuz/Mayıs (test satırı en çok 10 ilçe)

| ilçe | tahmin 2026 | gerçek 2025 |
|---|---|---|
| İZMİR>ÖDEMİŞ | 1.84× | 2.78× |
| İZMİR>BORNOVA | 1.52× | 2.97× |
| İZMİR>MENDERES | 1.61× | 2.03× |
| İZMİR>URLA | 1.56× | 1.47× |
| İZMİR>TİRE | 1.53× | 2.49× |
| İZMİR>BUCA | 1.46× | 2.04× |
| MANİSA>SALİHLİ | 1.72× | 2.46× |
| İZMİR>SEFERİHİSAR | 1.39× | 1.51× |
| İZMİR>BAYINDIR | 1.45× | 2.54× |
| İZMİR>BAYRAKLI | 1.67× | 2.51× |

## f. Cold trafolar (train'de hiç yok)

| ay | tahmin ort. log1p |
|---|---|
| 04 | 6.7251 |
| 05 | 6.9530 |
| 06 | 7.2592 |
| 07 | 7.5400 |

- Cold Temmuz/Mayıs: **1.80×**

- sub_b6.csv ve sub_p3.csv `submissions/` altında, doğrulamadan geçti. LB'ye kullanıcı yükleyecek.
