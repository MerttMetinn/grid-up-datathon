# Grid Up Datathon — Yol Haritası

**Görev:** Trafo bazlı günlük elektrik tüketimi tahmini
**Metrik:** RMSLE (düşük = iyi)
**Train:** Ocak 2025 – Mart 2026 · **Test:** Nisan – Temmuz 2026

---

## 0. Temel Gerçek (her kararın dayanağı)

Test seti, train'in bittiği yerden başlayan **kesintisiz 4 aylık gelecek bloğu**dur.
Bunun üç doğrudan sonucu var:

1. **Kısa lag'ler (lag_1, lag_7, rolling_7) test'te HESAPLANAMAZ.** Bu feature'ları kullanan
   her mimari, validasyonda harika görünüp leaderboard'da çöker.
2. **lag_364 / lag_365 KULLANILABİLİR.** Nisan–Temmuz 2025 train içinde. Bu, projenin en
   güçlü tek sinyali olacak.
3. **Recursive (autoregressive) tahmin yapılmamalı.** 120 günlük ufukta hata birikimi
   yıkıcıdır. Direct multi-horizon yaklaşım kullanılacak.

> Altın kural: Bir feature'ı yazmadan önce sor — *"Bu değer 2026-07-31 satırı için,
> sadece 2026-03-31'e kadarki hedef verisiyle hesaplanabilir mi?"* Cevap hayırsa, feature yok.

---

## 1. Proje İskeleti

```
gridup/
├── data/
│   ├── raw/           # train.csv, test.csv, sample_submission.csv
│   ├── external/      # open-meteo cache, tatil takvimi, ilçe koordinatları
│   └── processed/     # feature'lı parquet dosyaları
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 99_final_submission.ipynb   # jüriye gidecek, tek tuşla çalışan
├── src/
│   ├── config.py       # yollar, sabitler, SEED, tarih sınırları
│   ├── data.py         # yükleme + temizlik
│   ├── weather.py      # Open-Meteo çekme + cache
│   ├── calendar_tr.py  # tatil / bayram / okul / turizm takvimi
│   ├── features.py     # feature engineering (forecast_origin parametreli)
│   ├── validation.py   # zaman bazlı split + RMSLE
│   ├── train.py
│   └── predict.py
├── experiments/
│   └── log.csv         # HER denemenin kaydı
└── submissions/
```

**İlk 20'ye girersen notebook'un inceleniyor.** O yüzden en baştan seed'i sabit,
yorumlu, tek komutla çalışan bir pipeline kur. Sonradan toparlamak çok pahalı.

---

## 2. Faz 1 — EDA

### Yapısal kontroller
- [ ] Kaç tekil `tanim`? Train ve test'teki trafo kümeleri aynı mı? Test'te olup train'de
      olmayan trafo var mı? → cold-start stratejisi gerekir mi?
- [ ] Her trafo için gün sayısı tam mı? Boşluklar var mı? Panel dengeli mi?
- [ ] Trafo bazında serinin başlangıç/bitiş tarihi — sonradan devreye giren veya susan trafolar

### Hedef değişken
- [ ] `tuketim` dağılımı; `log1p` sonrası normale yaklaşıyor mu?
- [ ] Sıfır tüketimli gün oranı. Gerçek mi (kesinti/arıza) yoksa eksik kayıt mı?
      **Uzun sıfır blokları var mı?** (→ kapanmış trafo kuralı gerekir)
- [ ] Negatif değerler, absürt yüksek değerler (sayaç sarması, ters okuma)
- [ ] Yük faktörü = `tuketim / (guc * 24)`. **1'i aşan satırlar veri hatası sinyalidir.**
- [ ] `guc = 0` veya null olan trafolar

### Zamansal
- [ ] Toplam günlük tüketim zaman serisi. Yaz piki ne kadar keskin? Kış piki?
      (Ege → klima kaynaklı yaz piki baskın beklenir, test dönemi tam buna denk geliyor)
- [ ] Haftanın günü etkisi (sanayide pazar düşüşü belirgin, meskende değil)
- [ ] Ramazan / Kurban Bayramı / resmî tatillerde ne oluyor?

### Lokasyon
- [ ] `lokasyon` kaç tekil değer? "İL>BÖLGE>İLÇE" parse et
- [ ] Kaç satır jenerik "GEDİZ EDAŞ"? → **doldurmaya çalışma**, ayrı "bilinmiyor" kategorisi
- [ ] Sahil ilçeleri (Muğla, Aydın, Çeşme, Kuşadası...) yazın patlıyor mu?
      → turizm sezonu feature'ı, test dönemi tam sezon açılışı

### Segmentasyon (kritik)
Trafoları davranışa göre kümele. Girdiler: `guc`, yıllık ortalama log-tüketim,
yaz/kış oranı, hafta içi/sonu oranı, varyasyon katsayısı, sıfır gün oranı.

Beklenen 3–5 arketip: **mesken · sanayi · tarımsal sulama (yazın patlar) ·
turizm (sezonluk) · kamu/aydınlatma**

Bu kümelenme hem feature olarak, hem hata analizinde, hem de jüri sunumunda kullanılacak.

---

## 3. Faz 2 — Validasyon Kurgusu

> **Burada hata yaparsan geri kalan her şey boşa gider.**
> Random KFold KESİNLİKLE kullanma.

### Ana fold — birincil karar mekanizman
```
Train: 2025-01-01 → 2025-12-31
Valid: 2026-01-01 → 2026-03-31
```
3 aylık ileri ufuk + lag_364 mevcut (Oca–Mar 2025'ten). Test'e yapı olarak en yakın fold.
**Model seçimi ve hiperparametre kararları buradan verilir.**

### İkincil fold — mevsim testi
```
Train: 2025-01-01 → 2025-03-31
Valid: 2025-04-01 → 2025-07-31
```
Test ile **birebir aynı takvim penceresi ve aynı ufuk uzunluğu**.
Dezavantaj: sadece 3 ay eğitim, lag_364 yok → mutlak skoru yanıltıcı.
Karar için değil, **"modelim yaz geçişini yakalayabiliyor mu?"** sanity check'i için.

### Üçüncül fold — robustluk
```
Train: → 2025-08-31
Valid: 2025-09-01 → 2025-12-31
```

### Karar kuralı
Bir değişiklik ana fold'da iyileştirip ikincil fold'da bozuyorsa → **kabul etme.**
Overfit ediyorsundur.

### Metrik
```python
import numpy as np

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
```

**Skoru sadece global değil, şu kırılımlarda da raporla:**
- ay bazında
- trafo segmenti bazında
- `guc` çeyrekleri bazında
- **ufuk haftası bazında** (1. hafta vs 16. hafta — bozulma ne kadar?)

Jüri sunumundaki hikâye bu tablolardan çıkacak.

---

## 4. Faz 3 — Baseline'lar

Sıfır feature ile hızla gönder; **local validasyon ile leaderboard tutarlı mı** gör.

| # | Baseline | Not |
|---|---|---|
| 1 | Global sabit (train medyanı) | Taban çizgi |
| 2 | Trafo medyanı: `expm1(median(log1p(tuketim)))` | |
| 3 | **Trafo × ay medyanı** (geçen yılın aynı ayı) | Gerçek rakibin bu. Şaşırtıcı derecede güçlü |
| 4 | Trafo × ay × haftaiçi/haftasonu medyanı | |

**Bu baseline'ları geçemeyen hiçbir ML modelini kabul etme.**
3 numaralı baseline, ensemble bileşeni olarak sonuna kadar elde kalsın.

---

## 5. Faz 4 — Dış Veri

Organizatör açıkça izin verdi (hava durumu API + tatil takvimi + kaynak belirtme yeterli).
**Kullanmamak net kayıptır.**

### 5a. Hava durumu — Open-Meteo Archive API

Bugün Ağustos 2026 olduğu için **Nisan–Temmuz 2026 verisi arşivde mevcut** →
test dönemi için *gerçek gözlemlenmiş* hava verisi çekilebilir (tahmin değil). Büyük avantaj.

**Adımlar**
1. `lokasyon`'dan ilçe listesi → her ilçe için lat/lon tablosu.
   Fallback zinciri: ilçe → il merkezi → bölge ortalaması → jenerik Ege merkezi
2. `archive-api.open-meteo.com/v1/archive`, aralık: `2024-12-01 → 2026-08-01`
3. Değişkenler:
   - `temperature_2m_max/min/mean`, `apparent_temperature_max/mean`
   - `relative_humidity_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`
   - `shortwave_radiation_sum`, `sunshine_duration`
4. **CACHE'LE.** Her çalıştırmada API'ye gitme; parquet'e yaz.
   Notebook'ta kaynağı açıkça belirt (kural gereği yeterli).

**Türetilecek feature'lar**
- `CDD = max(0, T_mean - 22)` · `HDD = max(0, 18 - T_mean)`
  → eşikleri validasyonla optimize et (Ege için CDD eşiği 21–24 arası dene)
- `CDD²`, `CDD³` — **klima tüketimi sıcaklıkla doğrusal değil, üstel artar**
- Sıcaklığın 3/7 günlük hareketli ortalaması → termal kütle + alışkanlık gecikmesi
- `T_max - T_min` (günlük amplitüd)
- Nem × sıcaklık etkileşimi (apparent_temperature zaten kısmen içeriyor)
- **"İlk sıcak gün" anomalisi:** son 7 gün ortalamasına göre sapma
  → klimaların ilk açıldığı gün sıçraması

### 5b. Takvim ve özel günler

`holidays.TR` paketi resmî + dinî tatilleri verir; **doğruluğunu manuel kontrol et.**

Test dönemine düşen kritik günler:
- 23 Nisan, 1 Mayıs, 19 Mayıs, 15 Temmuz
- **Kurban Bayramı (Mayıs 2026 sonu)** — test döneminin tam ortasında.
  Hem tatil hem büyük göç: sahil ilçelerinde artış, sanayide çöküş.
- **Okul kapanışı (Haziran ortası)** → tatil bölgelerinde nüfus sıçraması

**Feature'lar:** `is_holiday`, `holiday_name` (kategorik), `days_to_holiday` /
`days_since_holiday` (±3), `is_bridge_day`, `is_weekend`, `day_of_week`, `month`,
`day_of_year` (sin/cos), `week_of_year`, `is_ramadan` (train'de Şub–Mar 2026'da var,
test'te yok ama modelin öğrenmesi için kalsın)

---

## 6. Faz 5 — Feature Engineering

Feature'lar üç kategoriye ayrılır. **Her biri için test'te hesaplanabilirlik doğrulanır.**

### A) Statik trafo özellikleri
- `guc`, `log(guc)`
- Lokasyon hiyerarşisi: `il`, `bolge`, `ilce` (ayrı kategorikler), `is_generic_lokasyon`
- Trafonun train dönemindeki: ortalama/medyan log-tüketim, std, yük faktörü,
  haftasonu/haftaiçi oranı, yaz/kış oranı, sıfır gün oranı, veri başlangıcı (yaş)
- Segment/küme etiketi (Faz 1'den)
- **Sıcaklık duyarlılığı:** train'de `log1p(tuketim) ~ CDD` regresyonunun eğimi.
  "Bu trafo klima ağırlıklı mı?" sorusunu tek sayıya indirir; yaz tahmininde çok değerli.

### B) Seviye (level) özellikleri
Tahmin başlangıcından ÖNCEKİ dönemden hesaplanır, **ufuk boyunca sabit kalır**.
- Tahmin başlangıcından önceki son 28 / 56 / 90 günün ortalama log-tüketimi
- Son 90 günün lineer trend eğimi → trafo büyüyor mu, küçülüyor mu

> **Sızıntı uyarısı:** Bu feature'lar validasyon fold'larında da aynı mantıkla
> (fold başlangıcından geriye bakarak) hesaplanmalı.
> `features.py` mutlaka `forecast_origin` parametresi almalı.

### C) Mevsimsel / geçmiş-yıl özellikleri
- `lag_364` (52 hafta önce, aynı haftanın günü) — `lag_365`'ten genelde daha iyi
- `lag_364`'ün ±3 ve ±7 günlük pencere ortalaması/medyanı (gürültü azaltır)
- Geçen yıl aynı ay × aynı haftanın günü trafo medyanı
- **Mevsimsel indeks (güçlü yapı):**
  `oran = lag_364_penceresi / trafonun_geçen_yıl_ortalaması`
  → bu indeks güncel seviye ile çarpılır

### D) Zaman + hava (Faz 4'ten)

### Kategorik işleme
LightGBM / CatBoost native kategorik desteğini kullan.
`tanim`'ı doğrudan kategorik vermek 10K+ trafoda riskli → yerine B/A'daki trafo-seviyesi
istatistikleri. Yine de bir varyantta dene, LightGBM bazen iyi başa çıkıyor.

---

## 7. Faz 6 — Modelleme

### Ana model: LightGBM
- Hedef: `y = log1p(tuketim)`, `objective='regression'` (L2)
  → log uzayında RMSE'yi minimize etmek **doğrudan RMSLE'yi minimize etmektir**
- Tahmin: `expm1(pred)` + `clip(0, None)`
- Alternatif dene: `objective='huber'` / `'regression_l1'` — aykırılara dayanıklı,
  log uzayında bazen daha iyi genelliyor
- Başlangıç: `learning_rate=0.03`, `num_leaves=127`, `min_data_in_leaf=100`,
  `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=1`,
  `n_estimators` → early stopping

### Ağırlıklandırma
Metrik log ölçekte → **her satır eşit ağırlıklı, küçük trafolar büyükler kadar önemli.**
Büyüklüğe göre ağırlık VERME. Tarihe göre hafif recency ağırlığı (son 6 ay ×1.5) denenebilir.

### İkinci model: CatBoost
Lokasyon hiyerarşisi gibi yüksek kardinaliteli kategorikleri daha iyi işler → ensemble çeşitliliği.

### Üçüncü yaklaşım: Segment bazlı modeller
Sanayi ve mesken çok farklı davranıyorsa 3–4 segment modeli tek global modeli geçebilir.
**Önce global modeli sağlamlaştır**, sonra dene.

### Denemeye değer
- Ay bazında ayrı modeller (Nisan modeli / Temmuz modeli)
- Feature olarak `horizon_days` (tahmin başlangıcına uzaklık) — ufuk bozulması varsa yararlı

### YAPMA
- ❌ Recursive / autoregressive tahmin
- ❌ Test döneminden herhangi bir hedef bilgisi türetme
- ❌ Random KFold

---

## 8. Faz 7 — Son İşlemler ve Hata Analizi

- `np.clip(pred, 0, None)` (kural gereği zaten)
- Log uzayında L2 → tahmin orijinal ölçekte medyana yakınsar.
  Bu RMSLE için **doğru** davranıştır. **"Smearing correction" UYGULAMA**, skoru bozar.
- **Kapanmış trafo kuralı:** uzun süredir sıfır tüketen trafolara yüksek tahmin vermek
  büyük log cezası yer. Ayrı kural düşün.
- **Yeni/eksik trafo:** lokasyon × güç grubu × segment medyanına düş.
- **Hata analizi:** en yüksek kareli log hatalı 100 trafo-günü incele.
  Sistematik mi (belirli ilçe? bayram? sıcak dalgası?) yoksa gürültü mü?
  → Genelde en büyük skor sıçraması buradan gelir.

---

## 9. Faz 8 — Ensemble ve Final

- LightGBM + CatBoost + baseline#3'ü **log uzayında** ağırlıklı ortalama ile harmanla.
  Ağırlıkları ana fold'da optimize et, ikincil fold'da doğrula.
- 3–5 farklı seed ile eğit, ortala (ucuz varyans azaltma)
- Final: **tüm train verisiyle** yeniden eğit
  (`n_estimators` = validasyondaki en iyi iterasyon × 1.1)

### Submission checklist
- [ ] Satır sayısı test ile aynı
- [ ] `id` sırası/formatı `sample_submission` ile birebir
- [ ] NaN yok, negatif yok
- [ ] Tahmin dağılımı train'inkine makul benziyor (log-histogram üst üste çiz)
- [ ] Ay bazında tahmin ortalaması geçen yılın aynı ayına yakın mı? (mantık kontrolü)

---

## 10. Jüri / Notebook Hazırlığı

İlk günden itibaren biriktir:
- **`experiments/log.csv`** — her denemenin feature seti, model, 3 fold skoru, LB skoru
- **İş anlamı olan bulgular** (jüri bunları dinlemek ister, model mimarisini değil):
  - "Sahil ilçelerinde Haziran'da %X sıçrama"
  - "CDD 24°C üstünde tüketim üstel artıyor"
  - "Kurban Bayramı'nda sanayi -%Y, turizm +%Z"
- Feature importance + SHAP grafikleri
- **Operasyonel değer çerçevesi:** trafo yükü tahmini → yatırım planlama,
  aşırı yüklenme riski erken uyarı, bakım planlaması

---

## 11. Öncelik Sırası (zaman kısıtlıysa)

1. **Validasyon kurgusu doğru olsun** ← en yüksek getiri
2. Baseline #3 + uçtan uca submission pipeline
3. LightGBM + takvim + statik + level feature'ları
4. `lag_364` ailesi + mevsimsel indeks
5. Open-Meteo + CDD/HDD nonlineer terimler
6. Hata analizi → hedefli feature ekleme
7. Ensemble + seed averaging

---

## 12. Risk Kaydı

| Risk | Belirti | Önlem |
|---|---|---|
| Sızıntı (level feature) | Validasyon çok iyi, LB kötü | `forecast_origin` disiplini, fold başına yeniden hesap |
| Kısa lag kullanımı | Test'te NaN patlaması | Feature whitelist kontrolü |
| Yaz rejimini kaçırma | İkincil fold'da kötü skor | CDD nonlineer terimler, sıcaklık duyarlılığı feature'ı |
| Ana fold'a overfit | Fold'lar arası çelişki | İki fold'da birden iyileşme şartı |
| Kapanmış trafolar | Birkaç satır devasa hata | Sıfır bloğu kuralı |
| Notebook reprodüksiyonu | Jüri aşamasında çalışmıyor | Sabit seed, cache'li dış veri, tek komut çalıştırma |
