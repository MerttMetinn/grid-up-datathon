# Teşhis Raporu — RMSLE ayrıştırması

Üretim: `scripts/04_diagnose.py` · 2026-08-22 16:27 · SEED=42

## 1. RMSLE ayrıştırması — sıfırların MSE payı

| fold | baseline | kesim | n | y=0 satır | y=0 pay | y=0 SSE | **y=0 MSE payı** | y>0 SSE | rmsle |
|---|---|---|---|---|---|---|---|---|---|
| F1 | b1_global | warm | 233,439 | 8,413 | %3.60 | 391,948 | **%37.01** | 667,119 | 2.1300 |
| F1 | b1_global | cold | 66,490 | 4,690 | %7.05 | 218,500 | **%67.82** | 103,673 | 2.2012 |
| F1 | b2_trafo | warm | 233,439 | 8,413 | %3.60 | 56,507 | **%29.84** | 132,832 | 0.9006 |
| F1 | b2_trafo | cold | 66,490 | 4,690 | %7.05 | 218,500 | **%67.82** | 103,673 | 2.2012 |
| F1 | b3_trafo_ay | warm | 233,439 | 8,413 | %3.60 | 66,637 | **%28.96** | 163,495 | 0.9929 |
| F1 | b3_trafo_ay | cold | 66,490 | 4,690 | %7.05 | 218,500 | **%67.82** | 103,673 | 2.2012 |
| F1 | b4_trafo_ay_hi | warm | 233,439 | 8,413 | %3.60 | 66,133 | **%28.91** | 162,594 | 0.9899 |
| F1 | b4_trafo_ay_hi | cold | 66,490 | 4,690 | %7.05 | 218,500 | **%67.82** | 103,673 | 2.2012 |
| F1 | b5_guc_lf | warm | 233,439 | 8,413 | %3.60 | 388,207 | **%54.15** | 328,723 | 1.7525 |
| F1 | b5_guc_lf | cold | 66,490 | 4,690 | %7.05 | 218,718 | **%74.45** | 75,072 | 2.1020 |
| F1 | b6_hibrit | warm | 233,439 | 8,413 | %3.60 | 56,507 | **%29.84** | 132,832 | 0.9006 |
| F1 | b6_hibrit | cold | 66,490 | 4,690 | %7.05 | 218,718 | **%74.45** | 75,072 | 2.1020 |
| F2 | b1_global | warm | 195,961 | 9,684 | %4.94 | 444,789 | **%48.56** | 471,191 | 2.1620 |
| F2 | b1_global | cold | 55,796 | 3,182 | %5.70 | 146,150 | **%46.43** | 168,629 | 2.3752 |
| F2 | b2_trafo | warm | 195,961 | 9,684 | %4.94 | 22,208 | **%15.36** | 122,346 | 0.8589 |
| F2 | b2_trafo | cold | 55,796 | 3,182 | %5.70 | 146,150 | **%46.43** | 168,629 | 2.3752 |
| F2 | b3_trafo_ay | warm | 195,961 | 9,684 | %4.94 | 22,208 | **%15.36** | 122,346 | 0.8589 |
| F2 | b3_trafo_ay | cold | 55,796 | 3,182 | %5.70 | 146,150 | **%46.43** | 168,629 | 2.3752 |
| F2 | b4_trafo_ay_hi | warm | 195,961 | 9,684 | %4.94 | 22,208 | **%15.36** | 122,346 | 0.8589 |
| F2 | b4_trafo_ay_hi | cold | 55,796 | 3,182 | %5.70 | 146,150 | **%46.43** | 168,629 | 2.3752 |
| F2 | b5_guc_lf | warm | 195,961 | 9,684 | %4.94 | 477,824 | **%67.66** | 228,412 | 1.8984 |
| F2 | b5_guc_lf | cold | 55,796 | 3,182 | %5.70 | 177,798 | **%68.75** | 80,816 | 2.1529 |
| F2 | b6_hibrit | warm | 195,961 | 9,684 | %4.94 | 22,208 | **%15.36** | 122,346 | 0.8589 |
| F2 | b6_hibrit | cold | 55,796 | 3,182 | %5.70 | 177,798 | **%68.75** | 80,816 | 2.1529 |
| F3 | b1_global | warm | 246,452 | 6,404 | %2.60 | 299,428 | **%30.59** | 679,293 | 1.9928 |
| F3 | b1_global | cold | 74,170 | 4,463 | %6.02 | 208,674 | **%61.07** | 133,007 | 2.1463 |
| F3 | b2_trafo | warm | 246,452 | 6,404 | %2.60 | 50,365 | **%18.66** | 219,557 | 1.0465 |
| F3 | b2_trafo | cold | 74,170 | 4,463 | %6.02 | 208,674 | **%61.07** | 133,007 | 2.1463 |
| F3 | b3_trafo_ay | warm | 246,452 | 6,404 | %2.60 | 50,365 | **%18.66** | 219,557 | 1.0465 |
| F3 | b3_trafo_ay | cold | 74,170 | 4,463 | %6.02 | 208,674 | **%61.07** | 133,007 | 2.1463 |
| F3 | b4_trafo_ay_hi | warm | 246,452 | 6,404 | %2.60 | 50,365 | **%18.66** | 219,557 | 1.0465 |
| F3 | b4_trafo_ay_hi | cold | 74,170 | 4,463 | %6.02 | 208,674 | **%61.07** | 133,007 | 2.1463 |
| F3 | b5_guc_lf | warm | 246,452 | 6,404 | %2.60 | 356,293 | **%51.38** | 337,089 | 1.6773 |
| F3 | b5_guc_lf | cold | 74,170 | 4,463 | %6.02 | 207,825 | **%72.90** | 77,249 | 1.9605 |
| F3 | b6_hibrit | warm | 246,452 | 6,404 | %2.60 | 50,365 | **%18.66** | 219,557 | 1.0465 |
| F3 | b6_hibrit | cold | 74,170 | 4,463 | %6.02 | 207,825 | **%72.90** | 77,249 | 1.9605 |

> **Sonuç (1):** F1'de b6 hibritin toplam kareli hatasının %56.97'i gerçek-sıfır satırlardan geliyor — bu satırların payı yalnızca %4.37 iken; metriğin ana kaldıracı seviye tahmini değil, **sıfırları bilmek**.

## 2. F1 valid sıfır profili

### 2a. Sıfır satır payı

- global: %4.37 · warm: %3.60 · cold: %7.05

### 2b. Cold trafolar

| trafo sıfır oranı | trafo | satır | sıfır satır |
|---|---|---|---|
| %0 | 1,516 | 61,320 | 0 |
| %0-5 | 4 | 205 | 5 |
| %5-25 | 5 | 187 | 18 |
| %25-75 | 4 | 190 | 79 |
| %75-99 | 0 | 0 | 0 |
| %100 | 91 | 4,588 | 4,588 |

### 2c. Warm trafolar

| trafo sıfır oranı | trafo | satır | sıfır satır |
|---|---|---|---|
| %0 | 2,809 | 222,237 | 0 |
| %0-5 | 14 | 1,149 | 23 |
| %5-25 | 16 | 1,143 | 162 |
| %25-75 | 11 | 911 | 382 |
| %75-99 | 14 | 1,201 | 1,048 |
| %100 | 91 | 6,798 | 6,798 |

> **Sonuç (2):** Sıfırlar dağınık değil — cold'da sıfır satırların %97.83'i, warm'da %93.26'i sıfır oranı %75+ olan 'ölü/yarı ölü' trafolarda toplanmış; problem 'hangi gün sıfır' değil, büyük ölçüde '**hangi trafo ölü**' problemi.

## 3. Sıfır tahmin edilebilir mi (train geneli)

### 3a. ilce_key bazında sıfır oranı (uç 10'ar)

| ilce_key | sıfır oranı | satır |
|---|---|---|
| İZMİR>BAYINDIR | %13.76 | 32,990 |
| İZMİR>URLA | %12.35 | 47,094 |
| İZMİR>KARŞIYAKA | %11.03 | 21,379 |
| MANİSA>AKHİSAR | %8.33 | 36,800 |
| MANİSA>KÖPRÜBAŞI | %8.14 | 4,818 |
| İZMİR>KONAK | %7.46 | 52,260 |
| İZMİR>ALİAĞA | %7.38 | 21,944 |
| MANİSA>SARUHANLI | %6.99 | 12,244 |
| MANİSA>AHMETLİ | %6.90 | 5,809 |
| İZMİR>ÇİĞLİ | %6.46 | 17,855 |
| İZMİR>GAZİEMİR | %1.96 | 13,858 |
| İZMİR>KARABURUN | %1.72 | 12,991 |
| MANİSA>KULA | %1.52 | 22,421 |
| İZMİR>SEFERİHİSAR | %1.52 | 36,818 |
| İZMİR>FOÇA | %0.95 | 11,493 |
| MANİSA>SELENDİ | %0.36 | 12,449 |
| MANİSA>KIRKAĞAÇ | %0.10 | 8,039 |
| İZMİR>SELÇUK | %0.01 | 13,433 |
| MANİSA>GÖLMARMARA | %0.00 | 3,291 |
| İZMİR>BALÇOVA | %0.00 | 5,703 |

### 3b. guc_bucket bazında

| guc_bucket | sıfır oranı |
|---|---|
| <=160 | %4.10 |
| 250-400 | %4.20 |
| 630-1000 | %4.93 |
| 1250-1600 | %5.32 |
| >1600 | %15.01 |

### 3c. ay bazında

| ay | sıfır oranı |
|---|---|
| 2025-01 | %7.30 |
| 2025-02 | %7.41 |
| 2025-03 | %7.30 |
| 2025-04 | %7.42 |
| 2025-05 | %7.42 |
| 2025-06 | %5.03 |
| 2025-07 | %2.31 |
| 2025-08 | %2.49 |
| 2025-09 | %2.67 |
| 2025-10 | %2.86 |
| 2025-11 | %3.51 |
| 2025-12 | %4.29 |
| 2026-01 | %4.35 |
| 2026-02 | %4.47 |
| 2026-03 | %4.53 |

### 3d. days_since_entry bazında

| days_since_entry | sıfır oranı | satır |
|---|---|---|
| 0 | %8.20 | 5,344 |
| 1-7 | %6.74 | 33,006 |
| 8-30 | %6.64 | 103,270 |
| 31-90 | %6.75 | 245,895 |
| 90+ | %3.75 | 838,722 |

### 3e. Hücre-oranı AUC (F1, hücre = ilce_key × guc_bucket × ay_no)

- AUC (tüm valid): **0.5572** · sadece cold satırlar: **0.5336**

> **Sonuç (3):** Sıfır oranı ilçeye göre %0.00–%13.76 bandında, yeni giriş gününde %8.20 ve statik hücre bilgisiyle bile AUC 0.56 (cold'da 0.53) — sıfır olasılığı kısmen tahmin edilebilir, model bu sinyali kullanabilmeli.

## 4. Sıfırsız skorlar (yalnız gerçek > 0)

### F1

| baseline | all | warm | cold |
|---|---|---|---|
| b1_global | 1.6393 | 1.7218 | 1.2952 |
| b2_trafo | 0.9081 | 0.7683 | 1.2952 |
| b3_trafo_ay | 0.9651 | 0.8524 | 1.2952 |
| b4_trafo_ay_hi | 0.9635 | 0.8500 | 1.2952 |
| b5_guc_lf | 1.1865 | 1.2086 | 1.1022 |
| b6_hibrit | 0.8514 | 0.7683 | 1.1022 |

### F2

| baseline | all | warm | cold |
|---|---|---|---|
| b1_global | 1.6365 | 1.5904 | 1.7903 |
| b2_trafo | 1.1036 | 0.8104 | 1.7903 |
| b3_trafo_ay | 1.1036 | 0.8104 | 1.7903 |
| b4_trafo_ay_hi | 1.1036 | 0.8104 | 1.7903 |
| b5_guc_lf | 1.1377 | 1.1073 | 1.2394 |
| b6_hibrit | 0.9222 | 0.8104 | 1.2394 |

### F3

| baseline | all | warm | cold |
|---|---|---|---|
| b1_global | 1.6194 | 1.6822 | 1.3813 |
| b2_trafo | 1.0669 | 0.9564 | 1.3813 |
| b3_trafo_ay | 1.0669 | 0.9564 | 1.3813 |
| b4_trafo_ay_hi | 1.0669 | 0.9564 | 1.3813 |
| b5_guc_lf | 1.1566 | 1.1850 | 1.0527 |
| b6_hibrit | 0.9789 | 0.9564 | 1.0527 |

> **Sonuç (4):** Sıfırlar atılınca b6 F1 skoru 1.269 → 0.851 — seviye tahmini kalitesi göründüğünden çok daha iyi; skorun büyük kısmı sıfır problemine gömülü.

## 5. Cold'da sabit tahmin vs fiziksel çıpa (F1)

### 5a-b. Tüm cold satırlar (66,490)

- Oracle sabit (valid'den log-ortalama = 6.476 → 648 kWh): RMSLE **2.1733**
- b5 (guc×24×LF): RMSLE **2.1020** (fark +0.0713)

### 5c. Sadece gerçek>0 cold satırlar (61,800)

- Oracle sabit (log-ortalama 6.968 → 1,061 kWh): RMSLE **1.2874**
- b5: RMSLE **1.1022** (fark +0.1852)

> **Sonuç (5):** guc ölçeklemesi sıfır-dışı satırlarda sabitten +0.185 RMSLE kazandırıyor (tüm cold'da +0.071) — yani `guc` bilgisi gerçek seviye sinyali taşıyor; oracle sabitin bile 2.17'de kalması cold probleminin seviyeden çok sıfır/heterojenlik problemi olduğunu doğruluyor.

## 6. Kaçınılmaz MSE tavanı (cold, F1)

- p (sıfır oranı) = 0.0705 · L (sıfır-dışı ort. log1p) = 6.968
- Kaçınılmaz MSE = p(1-p)L² = 3.1829 → **RMSLE tabanı = 1.7841**
- b5 mevcut cold RMSLE = 2.1020 → taban ile ara: **0.3180**

> **Sonuç (6):** Sıfırlar hiç ayırt edilemese bile taban 1.78; b5'in 2.10'lik cold skoru ile taban arasındaki 0.32'lik aralık, cold tarafında modellemeyle kazanılabilir alandır (sıfır olasılığı + seviye heterojenliği).

## 7. YoY drift (sabit kohort) → lag_364 düzeltmesi

| ay | 2025 | 2026 | fark (log1p) |
|---|---|---|---|
| 01 | 6.6437 | 6.7816 | +0.1379 |
| 02 | 6.7011 | 6.7258 | +0.0248 |
| 03 | 6.5522 | 6.6966 | +0.1444 |
| **ort** | · | · | **+0.1023** |

> **Sonuç (7):** Sabit kohortta YoY drift Oca–Mar ortalaması **+0.102 log1p** (çarpan olarak ×1.108) — lag_364 feature'larına önerilen düzeltme: `seas_lag364_log1p + 0.102` (ya da modele ham ver, `cal_year` benzeri sinyalle öğrenmesine izin ver; baseline kullanımında katsayı buradan).

