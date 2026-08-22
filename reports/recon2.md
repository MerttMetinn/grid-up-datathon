# Recon-2 Raporu

Üretim: `scripts/01_recon2.py` · Tarih: 2026-08-22 13:23

## 1. Cold-start profili

### 1a. Test satırlarının cold payı

- Cold trafo: 2,024 / 7,036 (%28.77)
- **Cold SATIR: 158,369 / 714,688 (%22.16)**

### 1b. guc dağılımı — cold vs warm (trafo bazında)

| istatistik | cold | warm |
|---|---|---|
| min | 40 | 40 |
| %25 | 250 | 250 |
| medyan | 630 | 400 |
| %75 | 1,000 | 1,000 |
| max | 30,930 | 35,900 |

guc frekans tablosu (trafo sayısı ve kolon içi pay):

| guc | cold | cold % | warm | warm % |
|---|---|---|---|---|
| 1,000 | 446 | 22.0 | 817 | 16.3 |
| 400 | 351 | 17.3 | 1,114 | 22.2 |
| 1,250 | 312 | 15.4 | 434 | 8.7 |
| 250 | 286 | 14.1 | 848 | 16.9 |
| 630 | 246 | 12.2 | 542 | 10.8 |
| 160 | 187 | 9.2 | 590 | 11.8 |
| 100 | 93 | 4.6 | 363 | 7.2 |
| 50 | 40 | 2.0 | 136 | 2.7 |
| 1,600 | 19 | 0.9 | 38 | 0.8 |
| 800 | 16 | 0.8 | 18 | 0.4 |
| 40 | 6 | 0.3 | 12 | 0.2 |
| 1,630 | 4 | 0.2 | 5 | 0.1 |
| 500 | 3 | 0.1 | 4 | 0.1 |
| 63 | 3 | 0.1 | 9 | 0.2 |
| 200 | 2 | 0.1 | 1 | 0.0 |
| (diğer 27 değer) | 10 | 0.5 | 81 | 1.6 |

### 1c. İlçe dağılımı — cold yoğunlaşması

Cold trafo sayısına göre ilk 15 ilçe:

| ilçe | cold | ilçedeki trafo | ilçe içi cold oranı | tüm cold'lar içindeki pay |
|---|---|---|---|---|
| İZMİR>BORNOVA | 219 | 485 | %45.2 | %10.8 |
| İZMİR>MENDERES | 132 | 320 | %41.2 | %6.5 |
| İZMİR>BAYRAKLI | 111 | 232 | %47.8 | %5.5 |
| İZMİR>MENEMEN | 105 | 243 | %43.2 | %5.2 |
| İZMİR>KARŞIYAKA | 75 | 141 | %53.2 | %3.7 |
| İZMİR>KARABAĞLAR | 74 | 226 | %32.7 | %3.7 |
| İZMİR>DİKİLİ | 71 | 122 | %58.2 | %3.5 |
| İZMİR>KONAK | 66 | 230 | %28.7 | %3.3 |
| İZMİR>TORBALI | 65 | 189 | %34.4 | %3.2 |
| İZMİR>BUCA | 61 | 257 | %23.7 | %3.0 |
| İZMİR>ÖDEMİŞ | 57 | 445 | %12.8 | %2.8 |
| MANİSA>YUNUSEMRE | 53 | 159 | %33.3 | %2.6 |
| MANİSA>SOMA | 53 | 153 | %34.6 | %2.6 |
| İZMİR>URLA | 52 | 278 | %18.7 | %2.6 |
| MANİSA>SALİHLİ | 51 | 252 | %20.2 | %2.5 |

İl bazında:

| il | cold | toplam | cold oranı |
|---|---|---|---|
| MANİSA | 459 | 1,679 | %27.3 |
| İZMİR | 1,565 | 5,357 | %29.2 |

- İlk 5 ilçe tüm cold trafoların %31.7'ini içeriyor; ilçe içi cold oranı %11–%58 arası değişiyor.

### 1d. Cold trafoların test'teki gün sayısı ve devreye giriş tarihi

- Gün sayısı: min=1 · %25=82 · medyan=82 · %75=82 · max=122
- 122 günün tamamında görünen cold trafo: 1 (%0.05)
- İlk gün 2026-04-01 olan cold trafo: 1 (%0.05)

İlk görüldükleri tarih (aylık histogram):

| ay | trafo | pay |
|---|---|---|
| 2026-04 | 183 | %9.04 |
| 2026-05 | 1,666 | %82.31 |
| 2026-06 | 105 | %5.19 |
| 2026-07 | 70 | %3.46 |

### 1e. Train'de 2025-01-01 sonrası başlayan warm trafolar vs cold devreye giriş

- Train'de sonradan başlayan trafo: 3,285 / 5,344 (%61.47)

Train sonradan-başlama aylık histogram (455 günlük pencere):

| ay | trafo | aylık ort. yeni trafo/gün |
|---|---|---|
| 2025-01 | 138 | 4.5 |
| 2025-02 | 33 | 1.2 |
| 2025-03 | 40 | 1.3 |
| 2025-04 | 48 | 1.6 |
| 2025-05 | 135 | 4.4 |
| 2025-06 | 231 | 7.7 |
| 2025-07 | 264 | 8.5 |
| 2025-08 | 51 | 1.6 |
| 2025-09 | 272 | 9.1 |
| 2025-10 | 255 | 8.2 |
| 2025-11 | 595 | 19.8 |
| 2025-12 | 350 | 11.3 |
| 2026-01 | 255 | 8.2 |
| 2026-02 | 142 | 5.1 |
| 2026-03 | 476 | 15.4 |

- Train'de günlük ort. yeni trafo girişi: 7.2/gün · Test'te ilk gün sonrası cold girişi: 16.7/gün

> **Özet (1):** Test satırlarının %22.16'i cold; cold trafolar warm'dan daha büyük güçlü (medyan 630 vs 400 kVA) ve belirli ilçelerde yoğun (ilçe içi cold oranı %11–%58); neredeyse hiçbiri test başında yok — %82'i 2026-05'te devreye giriyor, yani cold'lar train'deki filo büyümesinin (günde 7.2 yeni trafo) devamı ama daha hızlı (16.7/gün).

## 2. lag_364 kapsamı

| kapsam | exact | ±3 gün | ±7 gün | satır |
|---|---|---|---|---|
| test — tüm satırlar | %33.80 | %34.39 | %35.01 | 714,688 |
| test — sadece warm satırlar | %43.42 | %44.18 | %44.98 | 556,319 |
| F1 fold — valid tüm satırlar | %45.66 | %46.04 | %46.33 | 338,187 |
| F1 fold — history'de görülen (warm) satırlar | %49.57 | %49.98 | %50.30 | 311,480 |

- Test lag hedef aralığı: 2025-04-02 → 2025-08-01 (train içinde)
- F1 lag hedef aralığı: 2025-01-02 → 2025-04-01

> **Özet (2):** Test genelinde ±7 gün kapsam %35.0 (warm'da %45.0) — STRATEGY_v2 eşiklerine göre: %20–50 bandı → lag_364 eklenir ama grp_ mevsimsel indeks öncelikli; pencere genişletmenin katkısı marjinal (exact %33.8 → ±7 %35.0, +1.2 puan), yani boşluklar birkaç günlük kaymalardan değil geçmişin hiç olmamasından kaynaklanıyor.

## 3. Haftanın günü anomalisi — normalize kontrol

| gün | ham ortalama | normalize oran (trafo-ay içi) | sıfır satır payı |
|---|---|---|---|
| Pzt | 2,751.3 | 0.9910 | %14.37 |
| Sal | 3,896.8 | 0.9961 | %14.42 |
| Çar | 2,843.7 | 1.0065 | %14.20 |
| Per | 3,128.8 | 1.0147 | %14.22 |
| Cum | 3,357.8 | 1.0084 | %14.32 |
| Cmt | 3,922.7 | 1.0060 | %14.22 |
| Paz | 2,857.9 | 0.9773 | %14.25 |

- Ham tabloda göreli açıklık: %36.0 · normalize tabloda açıklık: 0.0375 (3.7 puan)
- Normalize hesapta dışlanan satır (trafo-ay ortalaması 0): 50,519

> **Özet (3):** Salı/Cumartesi tepesi kompozisyon artefaktı — normalize açıklık %3.7'e düşüyor (ham %36.0); gerçek trafo-içi dow etkisi küçük (Paz en düşük 0.977, Per en yüksek 1.015) ve sıfır satırlar güne eşit dağılmış.

## 4. LF>1 ve sıfır-blok trafoları test'te

### 4a. LF>1 trafoları

- LF>1 satırı olan trafo: 37
- Bunlardan test'te olan: **36** (%97.30)

### 4b. Train sonunda sıfır bloğunda olan trafolar

- Kapanmış aday (30+ gün sıfır, train sonunda hâlâ sıfır): 158
- Bunlardan test'te olan: **158** (%100.00)
- Test'teki gün sayıları: min=2 · medyan=122 · max=122 · toplam satır=18,629 (test satırlarının %2.61'i)
- 122 günün tamamında istenen: 137 trafo

> **Özet (4):** Bozuk-LF trafolarının 36/37 tanesi ve kapanmış-aday trafoların 158/158 tanesi test'te tahmin bekliyor; kapanmış adaylar test satırlarının %2.61'ini kaplıyor ve 137 tanesi 122 günün tamamında istendiği için sıfır-override kuralının etki alanı küçük ama cezası büyük.

## 5. Mevsimsellik tabanı — tam panel kohortu

- Kohort: 1,253 tam panelli trafo · 570,115 satır

Aylık ortalama log1p(tuketim) — sabit kohort:

| ay | ort. log1p | bar |
|---|---|---|
| 2025-01 | 6.6437 | `########` |
| 2025-02 | 6.7011 | `###########` |
| 2025-03 | 6.5522 | `####` |
| 2025-04 | 6.4950 | `#` |
| 2025-05 | 6.4702 | `#` |
| 2025-06 | 6.7922 | `################` |
| 2025-07 | 7.0880 | `##############################` |
| 2025-08 | 7.0027 | `##########################` |
| 2025-09 | 6.7370 | `#############` |
| 2025-10 | 6.5106 | `##` |
| 2025-11 | 6.5842 | `######` |
| 2025-12 | 6.7603 | `##############` |
| 2026-01 | 6.7816 | `###############` |
| 2026-02 | 6.7258 | `############` |
| 2026-03 | 6.6966 | `###########` |

- Geometrik-ortalama ölçekte Temmuz/Mayıs oranı (kohort geneli): 1.86×

İlçe bazında Temmuz/Mayıs oranı (kohort, ortalama tuketim):

| ilçe | trafo | Mayıs ort. | Temmuz ort. | Temmuz/Mayıs |
|---|---|---|---|---|
| İZMİR>KONAK | 68 | 4,641 | 23,270 | **5.01×** |
| İZMİR>KINIK | 5 | 325 | 1,624 | **5.00×** |
| İZMİR>KARABAĞLAR | 54 | 7,427 | 26,218 | **3.53×** |
| İZMİR>BAYRAKLI | 42 | 6,101 | 19,736 | **3.23×** |
| İZMİR>BEYDAĞ | 7 | 314 | 944 | **3.01×** |
| MANİSA>SARIGÖL | 7 | 794 | 2,228 | **2.81×** |
| İZMİR>KİRAZ | 39 | 403 | 966 | **2.40×** |
| MANİSA>ALAŞEHİR | 31 | 416 | 962 | **2.31×** |
| İZMİR>BAYINDIR | 12 | 1,255 | 2,858 | **2.28×** |
| İZMİR>TİRE | 16 | 576 | 1,310 | **2.27×** |
| İZMİR>ÖDEMİŞ | 59 | 869 | 1,909 | **2.20×** |
| İZMİR>TORBALI | 53 | 1,594 | 3,359 | **2.11×** |
| İZMİR>BUCA | 54 | 2,776 | 5,817 | **2.10×** |
| İZMİR>ÇEŞME | 34 | 2,224 | 4,657 | **2.09×** |
| MANİSA>SARUHANLI | 12 | 611 | 1,278 | **2.09×** |
| MANİSA>YUNUSEMRE | 47 | 1,876 | 3,814 | **2.03×** |
| İZMİR>KARŞIYAKA | 29 | 3,027 | 6,024 | **1.99×** |
| MANİSA>AHMETLİ | 3 | 362 | 717 | **1.98×** |
| İZMİR>SELÇUK | 19 | 634 | 1,247 | **1.97×** |
| MANİSA>ŞEHZADELER | 16 | 1,094 | 2,142 | **1.96×** |
| MANİSA>KÖPRÜBAŞI | 5 | 8,952 | 17,506 | **1.96×** |
| İZMİR>GAZİEMİR | 18 | 6,278 | 12,149 | **1.94×** |
| İZMİR>MENEMEN | 40 | 2,064 | 3,961 | **1.92×** |
| İZMİR>DİKİLİ | 13 | 904 | 1,731 | **1.92×** |
| MANİSA>SALİHLİ | 29 | 964 | 1,828 | **1.90×** |
| MANİSA>GÖRDES | 25 | 181 | 343 | **1.89×** |
| İZMİR>MENDERES | 47 | 959 | 1,803 | **1.88×** |
| İZMİR>KARABURUN | 8 | 374 | 701 | **1.87×** |
| MANİSA>SELENDİ | 17 | 171 | 319 | **1.87×** |
| MANİSA>TURGUTLU | 35 | 1,845 | 3,394 | **1.84×** |
| İZMİR>SEFERİHİSAR | 21 | 1,295 | 2,273 | **1.75×** |
| İZMİR>BORNOVA | 75 | 2,446 | 4,270 | **1.75×** |
| İZMİR>ÇİĞLİ | 22 | 1,683 | 2,932 | **1.74×** |
| İZMİR>URLA | 37 | 1,152 | 1,998 | **1.73×** |
| MANİSA>GÖLMARMARA | 4 | 705 | 1,220 | **1.73×** |
| MANİSA>AKHİSAR | 36 | 1,804 | 3,116 | **1.73×** |
| İZMİR>NARLIDERE | 3 | 1,593 | 2,691 | **1.69×** |
| İZMİR>KEMALPAŞA | 63 | 659 | 1,106 | **1.68×** |
| İZMİR>ALİAĞA | 13 | 1,886 | 3,162 | **1.68×** |
| İZMİR>BERGAMA | 29 | 1,131 | 1,886 | **1.67×** |
| MANİSA>KIRKAĞAÇ | 7 | 1,232 | 1,892 | **1.54×** |
| İZMİR>FOÇA | 11 | 767 | 1,164 | **1.52×** |
| İZMİR>GÜZELBAHÇE | 10 | 1,073 | 1,613 | **1.50×** |
| İZMİR>BALÇOVA | 6 | 10,532 | 15,807 | **1.50×** |
| MANİSA>SOMA | 23 | 1,500 | 2,132 | **1.42×** |
| MANİSA>KULA | 31 | 306 | 411 | **1.35×** |
| MANİSA>DEMİRCİ | 18 | 255 | 292 | **1.15×** |

> **Özet (5):** Yaz rampası sabit kohortta da gerçek (Temmuz/Mayıs geometrik ortalamada 1.86×, dip ay Mayıs) ama ilçeler arası fark çok büyük (1.15×–5.01×): en sert patlama İZMİR>KONAK / İZMİR>KINIK / İZMİR>KARABAĞLAR'ta, en zayıf MANİSA>DEMİRCİ'de — mevsimsel düzeltme ilçe bazında yapılmadan tek küresel eğriyle açıklanamaz.
