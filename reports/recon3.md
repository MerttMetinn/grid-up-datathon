# Recon-3 Raporu

Üretim: `scripts/02_recon3.py` · Tarih: 2026-08-22 15:48

## 1. Toplu giriş teyidi

### 1a. Cold trafoların test'e giriş günü — en yoğun 10 tarih

| tarih | gün | ayın günü | trafo | cold'ların payı |
|---|---|---|---|---|
| 2026-05-11 | Pzt | 11 | 1,326 | %65.51 |
| 2026-04-30 | Per | 30 | 114 | %5.63 |
| 2026-05-03 | Paz | 3 | 105 | %5.19 |
| 2026-05-07 | Per | 7 | 88 | %4.35 |
| 2026-05-13 | Çar | 13 | 36 | %1.78 |
| 2026-05-30 | Cmt | 30 | 28 | %1.38 |
| 2026-07-01 | Çar | 1 | 19 | %0.94 |
| 2026-06-08 | Pzt | 8 | 13 | %0.64 |
| 2026-06-29 | Pzt | 29 | 12 | %0.59 |
| 2026-05-14 | Per | 14 | 11 | %0.54 |

- Tek tepe: 2026-05-11 günü 1,326 trafo (%65.51) — **belirgin toplu giriş**
- Giriş görülen tekil gün sayısı: 80 / 122

### 1b. Train'de sonradan başlayan trafolar — en yoğun 10 tarih

| tarih | gün | ayın günü | trafo | payı |
|---|---|---|---|---|
| 2026-03-26 | Per | 26 | 329 | %10.02 |
| 2025-07-28 | Pzt | 28 | 177 | %5.39 |
| 2025-11-25 | Sal | 25 | 167 | %5.08 |
| 2025-09-10 | Çar | 10 | 99 | %3.01 |
| 2025-11-18 | Sal | 18 | 96 | %2.92 |
| 2025-06-17 | Sal | 17 | 46 | %1.40 |
| 2025-12-10 | Çar | 10 | 45 | %1.37 |
| 2025-12-18 | Per | 18 | 45 | %1.37 |
| 2025-09-12 | Cum | 12 | 35 | %1.07 |
| 2025-11-21 | Cum | 21 | 34 | %1.04 |

- Toplu giriş günü tanımı: ≥30 trafo/gün → 15 gün, 1,233 trafo (%37.53)

### 1c. Tarih örüntüsü — haftanın günü / ayın günü

- Train toplu giriş günlerinin haftanın günü dağılımı: Pzt=2 · Sal=4 · Çar=4 · Per=3 · Cum=2
- Ayın 1'ine denk gelen toplu gün: 0 / 15
- Ayın günü dağılımı (toplu günler): 4 · 7 · 10 · 12 · 17 · 18 · 19 · 21 · 23 · 25 · 26 · 28
- Test cold tepesi: 2026-05-11 (Pzt, ayın 11'i)

> **Sonuç (1):** Cold girişlerinin %65.51'i tek günde (2026-05-11) — train'de de girişlerin %37.53'i 15 toplu güne yığılmış; bu tek tek saha kurulumu değil, dönemsel toplu sisteme alım imzasıdır.

## 2. Ramp-up testi

- Uygun trafo (≥90 gün geçmiş, taban>0): toplu-giriş kohortu 550 · tek-tük kohortu 1,177

Medyan norm = log1p(tuketim) / trafonun 60–90. gün ortalama log1p'i

| days_since_entry | toplu giriş kohortu | n | tek tük kohortu | n |
|---|---|---|---|---|
| 0 | 0.952 | 550 | 0.883 | 1,177 |
| 1 | 0.997 | 540 | 0.992 | 1,100 |
| 2 | 0.995 | 537 | 0.991 | 1,082 |
| 3 | 0.991 | 538 | 0.992 | 1,080 |
| 4 | 0.994 | 537 | 0.992 | 1,079 |
| 5 | 0.995 | 537 | 0.996 | 1,075 |
| 6 | 0.998 | 534 | 0.997 | 1,076 |
| 7-13 | 1.001 | 3,747 | 0.996 | 7,495 |
| 14-20 | 1.004 | 3,742 | 0.998 | 7,490 |
| 21-27 | 1.008 | 3,750 | 0.999 | 7,604 |
| 28-34 | 1.009 | 3,739 | 0.999 | 7,741 |
| 35-41 | 1.015 | 3,738 | 1.001 | 7,751 |
| 42-48 | 1.006 | 3,732 | 0.999 | 7,791 |
| 49-55 | 1.006 | 3,709 | 1.000 | 7,826 |
| 56-62 | 1.007 | 3,701 | 1.001 | 7,888 |
| 63-69 | 1.002 | 3,714 | 1.001 | 7,917 |
| 70-76 | 1.000 | 3,754 | 1.002 | 7,947 |
| 77-83 | 1.000 | 3,764 | 1.001 | 7,952 |
| 84-90 | 1.002 | 3,794 | 1.000 | 7,930 |

- İlk hafta (0–6 gün) medyan norm: toplu=0.991 · tek-tük=0.985 (1.0 = olgun seviye)

> **Sonuç (2): RAMP YOK** — yeni giren trafo ilk günden olgun seviyede (ilk hafta medyan norm toplu 0.99, tek-tük 0.98); `guc × LF` doğrudan çalışır, ramp feature'ı gereksiz.

## 3. Test trafolarının geçmiş uzunluğu (H) dağılımı

### 3a. H histogramı

| H aralığı | trafo | pay |
|---|---|---|
| 0 (cold) | 2,024 | %28.77 |
| 1-30 | 748 | %10.63 |
| 31-90 | 593 | %8.43 |
| 91-180 | 1,199 | %17.04 |
| 181-300 | 578 | %8.21 |
| 301-455 | 1,894 | %26.92 |
| **toplam** | 7,036 | · |

- H medyanı: 105 gün · warm'larda medyan: 174 gün

### 3b. H dağılımı × guc_bucket (satırlar guc_bucket, pay %)

| guc_bucket | trafo | 0 (cold) | 1-30 | 31-90 | 91-180 | 181-300 | 301-455 |
|---|---|---|---|---|---|---|---|
| <=160 | 1,453 | %22.8 | %7.7 | %10.1 | %20.7 | %10.3 | %28.4 |
| 250-400 | 2,602 | %24.6 | %10.5 | %8.8 | %22.0 | %8.8 | %25.4 |
| 630-1000 | 2,100 | %33.9 | %12.4 | %8.8 | %12.7 | %6.9 | %25.3 |
| 1250-1600 | 813 | %40.8 | %12.4 | %3.8 | %6.4 | %6.0 | %30.5 |
| >1600 | 68 | %16.2 | %2.9 | %1.5 | %10.3 | %7.4 | %61.8 |

### 3c. Test'e giriş tarihi (tüm test trafoları) — en yoğun 10 gün

| tarih | gün | trafo | pay |
|---|---|---|---|
| 2026-04-01 | Çar | 3,928 | %55.83 |
| 2026-05-11 | Pzt | 2,222 | %31.58 |
| 2026-05-03 | Paz | 141 | %2.00 |
| 2026-04-30 | Per | 119 | %1.69 |
| 2026-05-07 | Per | 102 | %1.45 |
| 2026-05-13 | Çar | 54 | %0.77 |
| 2026-05-30 | Cmt | 30 | %0.43 |
| 2026-05-05 | Sal | 20 | %0.28 |
| 2026-07-01 | Çar | 20 | %0.28 |
| 2026-06-29 | Pzt | 16 | %0.23 |

- Giriş görülen tekil gün: 91 / 122 · ilk gün (2026-04-01) girenler: 3,928 (%55.83)

- Profil CSV yazıldı: `data/processed/test_history_profile.csv` (7,036 satır)

> **Sonuç (3):** Test trafolarının %28.77'i cold, warm'ların H medyanı 174 gün ve dağılım guc_bucket'a göre kayda değer değişiyor — `make_folds` H örneklemesini bu CSV'deki bucket-bazlı dağılımdan yapmalı.

## 4. Sıfır bloğu devam oranı

### 4a-b. Blok sayıları ve devam oranı

- 30+ gün sıfır bloğu (toplam): 356
- Biten (tüketim yeniden başladı): 87
- Veri sonuna kadar süren (sansürlü): 269
- **q = biten / toplam = 0.244**

### 4c. Dönüş seviyesi L

- Biten blok sonrası ilk 30 günün trafo-bazlı ortalama log1p'i: **L ortalama = 3.195** · medyan = 1.339 · %25–%75 = 0.10–6.20
- x* = q·L = 0.244 × 3.195 = **0.781** (log1p ölçeği) → tahmin ≈ **1.2 kWh**

### 4d. Blok uzunluğuna göre q

| blok uzunluğu | toplam | biten | q |
|---|---|---|---|
| 30-60 | 60 | 38 | 0.633 |
| 61-120 | 67 | 18 | 0.269 |
| 121-240 | 196 | 27 | 0.138 |
| 240+ | 33 | 4 | 0.121 |

> **Sonuç (4):** 30+ günlük sıfır bloklarının %24.44'i yeniden tüketime dönüyor (q=0.24) ve dönüş seviyesi L≈3.2 log1p; blok uzadıkça q düşüyor — kapanmış-aday trafo tahmini x*=q·L≈0.78 (≈1 kWh) civarında olmalı, sert 0 override yanlış.

## 5. İlçe Temmuz/Mayıs oranı — üç yöntem

| ilçe | trafo | aritmetik | medyan | geometrik |
|---|---|---|---|---|
| İZMİR>KONAK | 68 | 5.01× | 1.56× | 1.47× |
| İZMİR>KINIK ⚠️az-örnek | 5 | 5.00× | 2.88× | 2.79× |
| İZMİR>KARABAĞLAR | 54 | 3.53× | 1.66× | 1.68× |
| İZMİR>BAYRAKLI | 42 | 3.23× | 2.04× | 1.89× |
| İZMİR>BEYDAĞ ⚠️az-örnek | 7 | 3.01× | 2.38× | 2.24× |
| MANİSA>SARIGÖL ⚠️az-örnek | 7 | 2.81× | 4.53× | 4.27× |
| İZMİR>KİRAZ | 39 | 2.40× | 1.96× | 1.96× |
| MANİSA>ALAŞEHİR | 31 | 2.31× | 2.36× | 2.22× |
| İZMİR>BAYINDIR | 12 | 2.28× | 4.60× | 2.55× |
| İZMİR>TİRE | 16 | 2.27× | 2.61× | 2.29× |
| İZMİR>ÖDEMİŞ | 59 | 2.20× | 2.79× | 2.72× |
| İZMİR>TORBALI | 53 | 2.11× | 2.07× | 2.39× |
| İZMİR>BUCA | 54 | 2.10× | 1.76× | 1.71× |
| İZMİR>ÇEŞME | 34 | 2.09× | 1.85× | 2.16× |
| MANİSA>SARUHANLI | 12 | 2.09× | 2.20× | 2.23× |
| MANİSA>YUNUSEMRE | 47 | 2.03× | 1.87× | 2.11× |
| İZMİR>KARŞIYAKA | 29 | 1.99× | 1.88× | 1.83× |
| MANİSA>AHMETLİ ⚠️az-örnek | 3 | 1.98× | 2.37× | 2.00× |
| İZMİR>SELÇUK | 19 | 1.97× | 1.98× | 2.11× |
| MANİSA>ŞEHZADELER | 16 | 1.96× | 1.93× | 2.65× |
| MANİSA>KÖPRÜBAŞI ⚠️az-örnek | 5 | 1.96× | 1.39× | 1.60× |
| İZMİR>GAZİEMİR | 18 | 1.94× | 1.88× | 2.07× |
| İZMİR>MENEMEN | 40 | 1.92× | 1.92× | 1.81× |
| İZMİR>DİKİLİ | 13 | 1.92× | 2.11× | 1.73× |
| MANİSA>SALİHLİ | 29 | 1.90× | 1.86× | 2.02× |
| MANİSA>GÖRDES | 25 | 1.89× | 3.06× | 2.66× |
| İZMİR>MENDERES | 47 | 1.88× | 1.87× | 1.82× |
| İZMİR>KARABURUN ⚠️az-örnek | 8 | 1.87× | 2.04× | 1.94× |
| MANİSA>SELENDİ | 17 | 1.87× | 1.64× | 1.74× |
| MANİSA>TURGUTLU | 35 | 1.84× | 1.43× | 1.73× |
| İZMİR>SEFERİHİSAR | 21 | 1.75× | 1.67× | 1.64× |
| İZMİR>BORNOVA | 75 | 1.75× | 1.66× | 1.64× |
| İZMİR>ÇİĞLİ | 22 | 1.74× | 1.93× | 1.71× |
| İZMİR>URLA | 37 | 1.73× | 1.76× | 1.72× |
| MANİSA>GÖLMARMARA ⚠️az-örnek | 4 | 1.73× | 1.67× | 1.89× |
| MANİSA>AKHİSAR | 36 | 1.73× | 1.63× | 1.63× |
| İZMİR>NARLIDERE ⚠️az-örnek | 3 | 1.69× | 1.55× | 1.71× |
| İZMİR>KEMALPAŞA | 63 | 1.68× | 1.75× | 1.56× |
| İZMİR>ALİAĞA | 13 | 1.68× | 1.74× | 1.62× |
| İZMİR>BERGAMA | 29 | 1.67× | 2.08× | 1.78× |
| MANİSA>KIRKAĞAÇ ⚠️az-örnek | 7 | 1.54× | 1.40× | 1.71× |
| İZMİR>FOÇA | 11 | 1.52× | 1.31× | 1.52× |
| İZMİR>GÜZELBAHÇE | 10 | 1.50× | 1.49× | 1.51× |
| İZMİR>BALÇOVA ⚠️az-örnek | 6 | 1.50× | 1.64× | 1.50× |
| MANİSA>SOMA | 23 | 1.42× | 2.00× | 1.34× |
| MANİSA>KULA | 31 | 1.35× | 1.50× | 1.44× |
| MANİSA>DEMİRCİ | 18 | 1.15× | 1.57× | 1.30× |

- **Konak kontrolü:** aritmetik 5.01× · medyan 1.56× · geometrik 1.47×

- Yöntemler arası Spearman sıra korelasyonu: aritmetik–medyan 0.56 · aritmetik–geometrik 0.67 · medyan–geometrik 0.79

> **Sonuç (5):** Konak'ın 5.0× aritmetik oranı medyanda 1.56×, geometrikte 1.47× — robust yöntemlerde eriyor, aritmetik oran birkaç büyük trafonun eseri; grp_ mevsimsel indeks robust (medyan/geometrik) istatistikle kurulmalı; az-örnekli ilçeler (⚠️, trafo<10) her yöntemde güvenilmez.
