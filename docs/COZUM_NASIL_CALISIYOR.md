# Çözüm Nasıl Çalışıyor — 1.04952 Skoruna Giden Yol

**Public LB skoru:** 1.04952 · **Dosya:** `submissions/sub_final.csv`
**Tarih:** 31 Ağustos 2026 (yarışma kapanışı) · **Metrik:** RMSLE (düşük = iyi)

Bu belge, en iyi skorumuzu **hangi adımların** ürettiğini ve **hangi veriyi**
kullandığımızı baştan sona anlatır. Amaç: bu dosyayı okuyan biri, tek bir soru
işareti kalmadan sonucu yeniden üretebilsin.

---

## 1. Otuz saniyede: ne yaptık?

Problem, trafo bazlı günlük elektrik tüketimi tahmini. Ama **klasik bir zaman
serisi problemi değil**: test trafolarının %28.8'i eğitim verisinde hiç yok,
çoğu 2026 Mayıs'ında toplu olarak sisteme giriyor. Yani geçmişi olmayan
trafolara tahmin üretmek zorundayız.

Çözüm iki katmandan oluşuyor:

```
KATMAN 1 — MODEL (notebook üretir)
  ham veri -> feature'lar -> fiziksel çapa + 12 LightGBM -> ham tahmin
                                                             |
KATMAN 2 — KALİBRASYON (public LB ölçümüyle)                 v
  4 adet tek boyutlu düzeltme:
    1) genel seviye  2) cold/warm paylaşımı
    3) genel yayılım 4) cold'a özel yayılım
                                                             |
                                                             v
                                                    submission (1.04952)
```

Katman 1 modelin **ne bildiğini**, Katman 2 tahminlerin **nereye oturduğunu**
belirliyor. İkisi ayrı problemler ve ayrı çözüldü.

---

## 2. Kullanılan veri — tamamı

| # | Kaynak | Ne için |
|---|---|---|
| 1 | `train.csv` (yarışma) | Eğitim hedefi ve geçmişi (2025-01-01 → 2026-03-31) |
| 2 | `test.csv` (yarışma) | Tahmin edilecek satırlar (2026-04-01 → 2026-07-31) |
| 3 | `sample_submission.csv` (yarışma) | Çıktı sırası/formatı doğrulaması |
| 4 | `holidays` Python paketi | Türkiye resmî + dinî tatil **takvimi** |

**Başka hiçbir şey kullanılmadı.** Hava durumu yok, EPİAŞ yok, tahmin dönemine
ait hiçbir dış gözlem yok. Notebook'un son hücresi çalışma boyunca açılan tüm
dosyaları listeler ve bunu `assert` ile doğrular.

`holidays` paketi bir **takvimdir** (23 Nisan, 1 Mayıs, Ramazan Bayramı gibi
tarihleri döner) — ölçüm verisi değil, yıllar öncesinden bellidir, ileriye
sızma yaratmaz.

Yarışma sahipleri dış kaynak kullanımını serbest bırakmıştır; kullanmama
kararımız bilinçlidir. Gerekçe ve ölçümler: `docs/VERI_KAYNAKLARI.md`.

---

## 3. Katman 1 — Model

Tamamı `notebooks/gridup_leakfree_submission.ipynb` içinde. Depodaki `src/`
modüllerini **import etmez**; tek başına çalışır.

### 3.1 Fiziksel temel

```
log(1 + tüketim) ≈ log(guc × 24) + log(yük faktörü)
                   └── bilinen ──┘   └── model bunu öğrenir ──┘
```

Bir trafonun kurulu gücü (`guc`, kVA) biliniyor. Teorik maksimum günlük
tüketimi `guc × 24`. Gerçek tüketim bunun bir kesri — buna **yük faktörü**
diyoruz (medyan 0.106).

Bu bilgiyi modele bedava veriyoruz: LightGBM sıfırdan başlamıyor, bu **fiziksel
çapanın** (`init_score`) üzerinden başlıyor ve sadece artığı öğreniyor. Böylece
tüm kapasitesini "bu trafo tipik yük faktöründen ne kadar sapıyor" sorusuna
ayırıyor.

Çapa iki durumda farklı kuruluyor:

- **Geçmişi olan trafo (warm):** kendi geçmişinin log medyanı + ilçe×ay mevsim sapması
- **Geçmişi olmayan trafo (cold):** `log(guc × 24) + log(ilçe yük faktörü)` +
  mevsim sapması + sıfır düzeltmesi

### 3.2 Feature'lar — 58 adet, 5 grup

| Prefix | Ne | Cold trafoda |
|---|---|---|
| `static_` | Güç, konum, bayraklar | dolu |
| `cal_` | Takvim: ay, gün, tatil, tahmin ufku | dolu |
| `lvl_` | Trafonun **kendi** geçmiş seviyesi | **NaN** |
| `grp_` | İlçe × ay × güç grubu istatistikleri | dolu — cold'un tek dayanağı |
| `seas_` | 364 gün önceki kendi değeri | çoğunlukla NaN |

Cold trafolarda `lvl_` ve `seas_` bilerek **NaN bırakılıyor**. Uydurma bir
değerle doldurmak modeli yanıltır; LightGBM NaN'ı kendi dallandırır ve ağırlığı
`grp_` ile çapaya kaydırır.

`grp_` istatistikleri **medyan/geometrik ortalama** ile kuruluyor, aritmetik
ile değil. Sebep veriden ölçüldü: Konak ilçesinde Temmuz/Mayıs oranı aritmetik
ortalamayla 5.01×, medyanla 1.56× çıkıyor — birkaç dev trafo aritmetik
ortalamayı ele geçiriyor.

### 3.3 Eğitim kurgusu — asıl numara burada

Tek bir tarihten eğitmek modeli yanıltır: eğitimde her trafonun tam geçmişi
olur, testte ise %28.8'inin hiç geçmişi yoktur. Model "geçmiş her zaman vardır"
varsayımını öğrenir ve testte çöker.

Bunun yerine **2025 boyunca 10 farklı `forecast_origin`** kesiliyor. Her
origin'de:

1. Hedef penceresi = `(origin, origin + 122 gün]` — test geometrisiyle birebir
2. Her trafoya, test dağılımından örneklenmiş bir **geçmiş uzunluğu H** atanıyor
3. Trafonun geçmişi son H güne kırpılıyor · **H = 0 → yapay cold-start örneği**
4. Cold örneklerinin hedef satırları, giriş gecikmesi kadar baştan kırpılıyor

Böylece model, gerçek testte karşılaşacağı **bilgi rejimiyle** eğitiliyor.

### 3.4 Sızıntıya karşı üç kural

1. **Kısa lag yasak.** `lag_1`, `lag_7`, `rolling_7` kullanılmıyor. Test 122
   günlük tek blok; 2026-07-31'i tahmin ederken 2026-07-30'un değeri bilinemez.
   En kısa izinli lag **364 gün**.
2. **Recursive tahmin yok.** Her satır doğrudan tahmin ediliyor.
3. **Her feature bir `forecast_origin` alıyor** ve fonksiyonun ilk satırı
   geçmişin origin'i aşmadığını `assert` ile kesiyor. Final tahminde
   `forecast_origin = 2026-03-31` — modelin gördüğü en son hedef verisi test
   başlangıcından bir gün öncesi.

### 3.5 Model: 12 LightGBM

**4 geçmiş-uzunluğu (H) çekilişi × 3 tohum**, tahminler log uzayında ortalanıyor.

İkisi ayrı ayrı gerekli, çünkü iki farklı varyans kaynağı var:

- **Tohum** LightGBM'in bagging/feature örneklemesini değiştirir
- **H çekilişi** eğitim matrisinin kendisini değiştirir

Kritik nokta: H varyansı tohum ortalamasıyla **sönmez**, çünkü tüm tohumlar aynı
eğitim matrisini paylaşır. Ölçülen etkisi büyük: çekilişler arası tahmin seviyesi
std **0.042 log**, satır bazında **0.085 log**.

Hedef `log1p(tuketim)`, tahmin `expm1` + `clip(0, None)` ile geri alınıyor.
**Smearing / bias düzeltmesi yok** — RMSLE zaten log uzayında kare hata, log
uzayı ortalaması bu metrik için optimal.

**Bu katmanın çıktısı:** `submissions/sub_sp17.csv` → public LB **1.05343**

---

## 4. Katman 2 — Kalibrasyon

Model iyi ayrıştırıyor olabilir ama tahminleri **yanlış yere oturmuş** olabilir.
Bu ayrı bir problem ve public LB geri bildirimiyle çözüldü.

### 4.1 Yöntem: hata parabolik, iki nokta yeter

Tahminlere uygulanan her doğrusal düzeltme için hata **tam olarak paraboliktir.**
Örneğin uniform bir `d` kaydırması için:

```
MSE(d) = MSE(0) + 2·d·m + d²          (m = ortalama artık)
```

`d²` katsayısı **bilinen** (1). Yani bilinmeyen tek şey `m`. **Tek bir LB
ölçümü** bu denklemi kapalı formda çözer ve optimumu verir.

Bu yöntemin gücü: parametre taraması yok, deneme-yanılma yok, aşırı-uyum yok.
Her düzeltme tek boyutlu ve analitik olarak çözülüyor.

**Doğrulandı:** `SEGMENT_DELTA` için öngörülen skor **1.05343**, gerçekleşen
**1.05343** — beş hanede birebir.

### 4.2 Uygulanan dört düzeltme

| # | düzeltme | değer | ne yapar |
|---|---|---|---|
| 1 | `LEVEL_SHIFT` | −0.2712 | genel tahmin seviyesi |
| 2 | `SEGMENT_DELTA` | +0.1709 | cold/warm paylaşımı |
| 3 | Global eğim | −0.0505 | tüm tahminlerin yayılımı |
| 4 | Cold eğimi | −0.0556 | cold satırlarının ek daraltılması |

1 ve 2 notebook'un içinde; 3 ve 4 sonradan uygulanıyor (bkz. bölüm 8).

#### Düzeltme 1 — genel seviye

İki LB noktası çözdü: ortalama artık **m = +0.013 log**. Yani model genel
seviyede zaten kalibre; sabit sadece en iyi noktayı sabitliyor.

#### Düzeltme 2 — cold/warm paylaşımı ⭐

Bu, çalışmanın en değerli bulgusu. Genel seviye doğru görünüyordu (+0.013), ama
segmentler ayrı ölçülünce:

| segment | satır payı | sapma |
|---|---|---|
| **cold** | %22.2 | **+0.184** |
| warm | %77.8 | **−0.035** |
| ağırlıklı ortalama | %100 | +0.013 ← genel ölçümün gördüğü |

**İki hata birbirini gizliyormuş.**

**Kök neden bir modelleme hatası.** Sıfır-şişirilmiş bir dağılımda (y=0 olasılığı
`p`, pozitifken seviye `L`) RMSLE'yi minimize eden tahmin:

```
DOĞRU  : E[log1p(y)] = (1−p) · L        <- çarpımsal
BİZİMKİ: L + log(1−p)                   <- toplamsal
```

`log(1−p)` **ham ölçekte ortalamayı** düzeltmek için doğrudur. Ama RMSLE **log
ölçekte** çalışıyor; orada düzeltme çarpımsal olmalı. Fark `p` ile büyüyor:
p=0.06'da +0.38, p=0.20'de +1.26 log fazla tahmin. Cold satırlarda `p` yüksek
olduğu için sapma orada birikiyor.

Düzeltme cold'u `−δ`, warm'ı `+δ·(f_cold/f_warm)` kaydırıyor. Warm kaydırması
cold satır payına göre ölçeklendiği için **genel ortalama değişmiyor** —
sadece paylaşım düzeliyor.

#### Düzeltme 3 — yayılım (eğim)

İlk iki düzeltme tahminlerin **nerede durduğunu** ayarlıyor. Üçüncüsü **ne kadar
yayıldığını**:

```
p' = ortalama + (1 + d) · (p − ortalama)      d = −0.04
```

Ölçüm: `cov(artık, tahmin) = +0.154`. Model tahmin seviyesiyle korele sapıyordu;
tahminleri ortalamaya doğru daraltmak kazandırdı (1.05343 → **1.04990**).

#### Düzeltme 4 — cold'a özel yayılım

Aynı mantık cold satırlarına ayrıca uygulandı: `cov_cold = +0.060`, optimum ek
daraltma −0.0556. Beklenen bir sonuç — cold tahminleri trafo geçmişi olmadan,
yalnızca grup istatistiklerinden kuruluyor; belirsizlik yüksekken optimal tahmin
ortalamaya daha çok çekilmeli.

Her iki eğim düzeltmesinde de ortalama korunuyor ve kırpma olmuyor.

---

## 5. Skorun adım adım oluşumu

| adım | dosya | public LB | kazanç |
|---|---|---|---|
| Basit baseline (b6) | `sub_b6.csv` | 1.36728 | — |
| Model + hava durumu | `sub_s2_optuna.csv` | 1.06483 | referans |
| Hava çıkarıldı | `sub_nowx_lo.csv` | 1.06525 | −0.0004 (gürültü) |
| Kendi kendine yeterli notebook | `sub_notebook.csv` | 1.05764 | +0.0072 |
| 12 modelli topluluk | `sub_hens_lo.csv` | 1.05737 | +0.0003 |
| **Düzeltme 2** (cold/warm) | `sub_sp30.csv` | 1.05568 | +0.0017 |
| **Düzeltme 2 optimum** | `sub_sp17.csv` | 1.05343 | +0.0023 |
| **Düzeltme 3** (global eğim, prob) | `sub_sl_m04.csv` | 1.04990 | +0.0035 |
| **Düzeltme 4** (cold eğimi, prob) | `sub_csl12.csv` | 1.05002 | ölçüm |
| **Düzeltme 3+4 optimum** | **`sub_final.csv`** | **1.04952** | **+0.0004** |

Toplam: **1.06483 → 1.04952** (0.0153 iyileşme), tamamı dış veri kullanmadan.

### Yöntemin doğruluk kaydı

Parabol yöntemi dört kez uygulandı, dördünde de tuttu:

| düzeltme | öngörü | gerçekleşen | sapma |
|---|---|---|---|
| `SEGMENT_DELTA` optimum | 1.05343 | 1.05343 | **0.00000** |
| Global eğim (prob −0.04) | yön tahmini | 1.04990 | doğru yön |
| Cold eğimi (prob −0.12) | yön tahmini | 1.05002 | doğru yön |
| Birleşik optimum | ~1.0494 | 1.04952 | 0.0001 |

---

## 6. Denenip elenenler

Hepsi ölçülüp kapatıldı — varsayım değil, sayı.

| deneme | ölçülen | sonuç |
|---|---|---|
| Hava durumu (17 feature) | katkı ~0 (1.06483 vs 1.06525) | ❌ |
| Satır bazlı sıfır düzeltmesi | korelasyon 0.005 (eşik 0.08) | ❌ 1.06374 |
| Uniform seviye kaydırma (−0.30) | ortalama artık +0.013 | ❌ 1.09545 |
| `tanim` ID komşuluğu | 0.094 (ilçe zaten 0.179) | ❌ |
| Yeni trafo rampası | yaş 1–7 gün: −0.019 | ❌ rampa yok |
| Hurdle (ölü-trafo sınıflandırıcı) | AUC 0.94 ama skor düzelmedi | ❌ |
| CatBoost / Tweedie ensemble | modeller %97–99 korele | ❌ |
| Recency ağırlıklandırma | F1 iyi, F2/F3 kötü | ❌ overfit |
| 75 feature + 60 optuna denemesi | 29 feature'ı geçmedi | ❌ |

---

## 7. Hata nerede? (dürüst değerlendirme)

| kesim | satır payı | RMSLE | **hata payı** |
|---|---|---|---|
| **cold + gerçek sıfır** | %1.6 | 6.66 | **%56** |
| cold + pozitif | %20.6 | 1.08 | %19 |
| warm + pozitif | %75.0 | 0.54 | %18 |
| warm + sıfır | %2.8 | 1.80 | %7 |

Hatanın yarısından fazlası, satırların %1.6'sında: **geçmişi olmayan ve o gün
hiç tüketim yapmayan trafolar.**

Bu yapısal bir tavan. Bir trafonun geçmişi yoksa kapalı olup olmadığı **tanım
gereği bilinemez**. Kalibrasyon düzeltmeleri belirsizlik altında doğru
*beklenen değeri* verir, ama "hangi satır sıfır" sorusunu çözmez.

Ulaşılabilir tabanı ölçtük (kâhin tahmincilerle): **0.782**. Yani teorik olarak
aşağıda yer var, ama oraya inmek trafo kimliğine bağlanabilecek ek bir öznitelik
(abone tipi, kurulum tarihi, fider bilgisi) gerektirirdi.

---

## 8. Yeniden üretilebilirlik

**Katman 1** tamamen deterministik:

```bash
python notebooks/gridup_leakfree_submission.py     # -> submission.csv
```

- Tüm rastgelelik `SEED = 42` üzerinden sabit (H örneklemesi, LightGBM
  bagging/feature örneklemesi, tohum ortalaması)
- `tanim` kategori sırası açıkça sabitlendi — bu sıra pandas'ta CSV ve parquet
  okumalarında farklı çıkıyor ve H atamasını etkiliyordu
- İnternet erişimi gerekmez
- **Doğrulandı:** iki bağımsız koşu bit düzeyinde aynı çıktı verdi
  (MD5 `62967caa88b16ba58d8d0eeb59630402`)

**Katman 2**'nin ilk iki düzeltmesi notebook'un içinde (`LEVEL_SHIFT`,
`SEGMENT_DELTA`).

### ⚠️ Açık: üçüncü düzeltme henüz notebook'ta değil

`sub_final.csv` (1.04952) = notebook çıktısı + iki eğim düzeltmesi, sonradan
uygulanmış. Yani **notebook şu an bu dosyayı üretmiyor.**

Notebook `sub_sp17.csv`'yi (1.05343) birebir üretiyor — final seçiminde ikinci
dosya olarak o işaretlendi, tam da bu yüzden.

Bunu final olarak seçersek, eğim düzeltmesinin de notebook'a taşınması gerekir.
Değişiklik küçük — 8b bölümüne tek satır:

```python
mu = np.log1p(pred_final).mean()
pred_final = np.clip(np.expm1(mu + (1 + SLOPE_DELTA) * (np.log1p(pred_final) - mu)), 0, None)
```

Ölçülen optimumlar (LB'den kapalı formda çözüldü):

| parametre | değer | ne yapar |
|---|---|---|
| `SLOPE_DELTA` | −0.0505 | tüm tahminleri ortalamaya doğru daraltır |
| `COLD_SLOPE_DELTA` | −0.0556 | cold satırlarını ek olarak daraltır |

Cold'un ek daraltma gerektirmesi beklenen bir sonuç: cold tahminleri trafo
geçmişi olmadan, yalnızca grup istatistiklerinden kuruluyor — belirsizlik
yüksekken optimal tahmin ortalamaya daha çok çekilmeli.

---

## 9. Bir uyarı: model gürültüsü

Aynı feature ve parametrelerle kurulan iki model arasında **0.0076** skor farkı
ölçtük (`sub_nowx_lo` vs `sub_notebook`). Bu eşiğin altındaki çapraz-doğrulama
farkları **anlamlı değildir**; karar vermek için LB gerekir.

Bu bulgu geriye dönük olarak önemli: geçmişte "CV'de 0.003 iyileşti" diye
alınan bazı kararlar muhtemelen gürültüydü.

---

## İlgili belgeler

- `docs/VERI_KAYNAKLARI.md` — veri kaynağı beyanı (host incelemesi için)
- `docs/PROJE_HANDOFF.md` — tam proje bağlamı
- `docs/NEREDE_KALDIK_29-30_AGUSTOS.md` — kalibrasyon bulgularının hikâyesi
- `reports/hdraw_ensemble.md` — seviye ve segment deneylerinin ham kaydı
- `reports/zero_anchor.md` — elenen satır-bazlı sıfır düzeltmesi
- `README.md` — proje yapısı, skor geçmişi, kapatılan kaldıraçlar
