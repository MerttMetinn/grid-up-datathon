# Veri Keşif Raporu (recon)

Üretim: `scripts/00_recon.py` · Tarih: 2026-08-22 12:38

## YAPI

### 1. Satır ve tekil trafo sayıları

| | satır | tekil tanim |
|---|---|---|
| train | 1,226,237 | 5,344 |
| test  | 714,688 | 7,036 |

### 2. KRİTİK — tanim kümeleri kesişimi

- Kesişim: **5,012** trafo
- Test'te olup train'de OLMAYAN: **2,024** (test trafolarının %28.77'i)
- Train'de olup test'te olmayan: **332** (train trafolarının %6.21'i)

> **UYARI:** Test'te train'de hiç görülmemiş trafo oranı %1'in üzerinde → **cold-start stratejisi gerekecek** (lokasyon × güç grubu medyanına düşüş vb.).

### 3. Tarih aralıkları

- **train**: 2025-01-01 → 2026-03-31 · takvim 455 gün · veride 455 tekil gün
- **test**: 2026-04-01 → 2026-07-31 · takvim 122 gün · veride 122 tekil gün

### 4. Trafo başına gün sayısı (train)

Tekil gün sayısı dağılımı:

| istatistik | değer |
|---|---|
| min | 1 |
| 1% | 1 |
| 5% | 2 |
| 25% | 82 |
| 50% | 170 |
| 75% | 453 |
| 95% | 455 |
| 99% | 455 |
| max | 455 |

- Tam panel (455 gün) olan trafo: 1,253 / 5,344 (%23.4)
- Kendi aralığında boşluğu (gap) olan trafo: 1,240 (toplam eksik gün: 69,816)
- Aynı trafo+gün mükerrer satır: 0
- Sonradan başlayan trafo (ilk gün > 2025-01-01): 3,285
- Erken susan trafo (son gün < 2026-03-31): 1,415

- Test panel: trafo başına gün min=1, max=122 (**dengesiz**)

### 5. Bellek (train)

- Optimizasyonsuz (int64/float64/object string): ~295 MB
- Optimize (category tanim/lokasyon, float32, datetime64): **24 MB**

## HEDEF

### 6. tuketim istatistikleri

| istatistik | değer |
|---|---|
| min | 0.00 |
| %25 | 260.49 |
| medyan | 1,075.20 |
| %75 | 2,748.48 |
| max | 50,403,052.00 |
| ortalama | 3,251.90 |
| NaN | 0 |
| negatif | 0 |
| sıfır | 57,536 (%4.69) |

### 7. log1p(tuketim) histogramı

```
[ 0.00,  1.18)     63,820  #######
[ 1.18,  2.36)      8,739  #
[ 2.36,  3.55)     32,245  ###
[ 3.55,  4.73)     87,426  #########
[ 4.73,  5.91)    174,504  ##################
[ 5.91,  7.09)    279,435  #############################
[ 7.09,  8.28)    387,711  ########################################
[ 8.28,  9.46)    177,658  ##################
[ 9.46, 10.64)      8,999  #
[10.64, 11.82)      3,838  #
[11.82, 13.01)      1,816  #
[13.01, 14.19)          7  #
[14.19, 15.37)          8  #
[15.37, 16.55)         10  #
[16.55, 17.74)         21  #
```

### 8. Yük faktörü = tuketim / (guc*24)

- Hesaplanabilen satır: 1,226,237 (guc=0 veya NaN nedeniyle düşen: 0)
- Medyan: 0.1059 · %95: 0.3654 · max: 58.66
- **1'i aşan satır: 1,821 (%0.149)** — veri hatası sinyali
- 1'i aşan satırların dokunduğu trafo: 37

### 9. Ardışık sıfır blokları (30+ gün)

- 30+ gün ardışık sıfır bloğu sayısı: 356
- Bu bloklara sahip tekil trafo: **320**
- En uzun blok: 455 gün
- Train'in son gününde hâlâ sıfır bloğunda olan trafo (kapanmış aday): **158**

## KOLONLAR

### 10. guc

- Tekil guc değeri (train, satır bazında): 41
- Trafo bazında guc: min=40 · medyan=400 · max=35,900
- guc=0 satır: 0 · guc NaN satır: 0

En yaygın 10 guc değeri (trafo sayısı):

| guc (kVA) | trafo |
|---|---|
| 400 | 1,172 |
| 1,000 | 894 |
| 250 | 891 |
| 160 | 623 |
| 630 | 578 |
| 1,250 | 474 |
| 100 | 387 |
| 50 | 149 |
| 1,600 | 40 |
| 800 | 19 |

- guc'u zaman içinde değişen trafo: 0

### 11. lokasyon

- Tekil lokasyon (train): 47
- `İL>BÖLGE>İLÇE` (2 adet `>`) formatına uyan satır: %73.28
- Jenerik `GEDİZ EDAŞ` içeren satır: 0 (%0.00) · trafo: 0
- Ayrıştırma (formata uyanlar): il=1 · bölge=3 · ilçe=30
- İller: ['İZMİR']
- NaN lokasyon satırı: 0
- Formata uymayan tekil değer örnekleri: ['MANİSA>GÖRDES', 'MANİSA>SARIGÖL', 'MANİSA>KULA', 'MANİSA>AKHİSAR', 'MANİSA>SOMA', 'MANİSA>DEMİRCİ', 'MANİSA>TURGUTLU', 'MANİSA>KÖPRÜBAŞI']
- Lokasyonu zaman içinde değişen trafo: 0

### 12. test id ↔ sample_submission

- test.csv id formatı `tanim_YYYY-MM-DD` mi: **EVET**
- sample_submission satır sayısı: 714,688 · test satır: 714,688
- Küme olarak birebir eşleşme: **EVET**
- Sıra da aynı mı: **EVET**
- test id mükerrer: 0

## ZAMAN

### 13. Aylık toplam tüketim (train)

| ay | toplam | ortalama/satır | satır |
|---|---|---|---|
| 2025-01 | 192,924,144 | 3,013.2 | 64,027 |
| 2025-02 | 189,926,672 | 3,259.6 | 58,267 |
| 2025-03 | 168,595,952 | 2,585.6 | 65,206 |
| 2025-04 | 158,083,040 | 2,468.2 | 64,048 |
| 2025-05 | 136,968,528 | 2,036.3 | 67,262 |
| 2025-06 | 242,894,864 | 3,573.2 | 67,976 |
| 2025-07 | 412,953,824 | 5,459.2 | 75,643 |
| 2025-08 | 350,295,392 | 4,396.3 | 79,680 |
| 2025-09 | 285,244,992 | 3,880.9 | 73,499 |
| 2025-10 | 178,406,576 | 2,247.0 | 79,399 |
| 2025-11 | 259,403,920 | 2,976.4 | 87,154 |
| 2025-12 | 340,114,976 | 3,212.0 | 105,889 |
| 2026-01 | 370,314,688 | 3,290.3 | 112,546 |
| 2026-02 | 324,100,320 | 3,090.3 | 104,875 |
| 2026-03 | 377,368,576 | 3,124.8 | 120,766 |

### 14. Haftanın gününe göre ortalama tüketim

| gün | ortalama |
|---|---|
| Pzt | 2,751.3 |
| Sal | 3,896.8 |
| Çar | 2,843.7 |
| Per | 3,128.8 |
| Cum | 3,357.8 |
| Cmt | 3,922.7 |
| Paz | 2,857.9 |

### 15. 2025 Nisan–Temmuz aylık toplamları (test dönemi kıyası)

| ay | toplam | ortalama/satır |
|---|---|---|
| 2025-04 | 158,083,040 | 2,468.2 |
| 2025-05 | 136,968,528 | 2,036.3 |
| 2025-06 | 242,894,864 | 3,573.2 |
| 2025-07 | 412,953,824 | 5,459.2 |

## DİKKAT ÇEKEN ANOMALİLER

- Test'te train'de görülmemiş 2,024 trafo (%28.77) → cold-start gereksinimi.
- Yük faktörü > 1 olan 1,821 satır (%0.149, 37 trafo); max yük faktörü 58.7.
- 30+ gün ardışık sıfır bloğu olan 320 trafo; 158 tanesi train sonunda hâlâ sıfırda (kapanmış aday).
- Kendi tarih aralığında boşluğu olan 1,240 trafo (toplam 69,816 eksik gün) → panel dengesiz.
- Test paneli dengesiz: trafo başına gün 1–122 arası.
