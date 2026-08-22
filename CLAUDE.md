# CLAUDE.md — Grid Up Datathon

Claude Code her oturumda bunu okur. Buradaki her sayı `reports/recon*.md` ile
doğrulanmıştır — varsayım yoktur. Genel çerçeve: `docs/ROADMAP.md` ·
Gerekçeler: `docs/STRATEGY_v3.md`

---

## Problem

Trafo bazlı günlük elektrik tüketimi tahmini. Metrik **RMSLE** (düşük = iyi).

| | değer |
|---|---|
| Train | 2025-01-01 → 2026-03-31 · 1,226,237 satır · 5,344 trafo |
| Test | 2026-04-01 → 2026-07-31 · 714,688 satır · 7,036 trafo |
| Kolonlar | `id`, `tanim`, `guc` (kVA), `tarih`, `tuketim`, `lokasyon` |
| Bellek | dtype optimizasyonu ile 24 MB — pandas rahat yeter |

> Bu bir "zaman serisi devam ettirme" problemi **değil**. Test satırlarının %22'si
> hiç görülmemiş trafolara ait, warm'ların geçmiş medyanı 174 gün, lag_364 kapsamı %35.
> Bu, **kısıtlı geçmişle kesitsel tahmin** problemidir.
>
> Modelin öğrendiği asıl şey: `log1p(tuketim) ≈ log(guc × 24) + log(yük_faktörü)`

---

## Kilitlenmiş veri gerçekleri

### Cold-start
- **2,024 trafo (%28.8) train'de yok** → **158,369 satır (%22.2)**
- Girişlerin **%65.5'i tek gün: 2026-05-11**. İdari toplu sisteme alım (fiziksel kurulum değil)
- **RAMP YOK** — yeni trafo ilk günden olgun seviyede (ilk hafta medyan norm 0.98–0.99).
  `days_since_entry` ramp feature'ları GEREKSİZ. Sadece gün 0 düşük (kısmi okuma)
  → `static_is_first_day` flag'i yeterli
- Cold'lar daha büyük güçlü: medyan **630 vs 400 kVA**. Cold oranı guc_bucket'a göre
  %16.2 (>1600) ile %40.8 (1250-1600) arasında değişiyor → **stratifikasyon zorunlu**

### Geçmiş uzunluğu (H)
| H | trafo | pay |
|---|---|---|
| 0 (cold) | 2,024 | %28.8 |
| 1-30 | 748 | %10.6 |
| 31-90 | 593 | %8.4 |
| 91-180 | 1,199 | %17.0 |
| 181-300 | 578 | %8.2 |
| 301-455 | 1,894 | %26.9 |

H medyanı 105 (warm'da 174). **Test trafolarının %19'unun geçmişi 90 günden kısa.**
Hazır profil: `data/processed/test_history_profile.csv`

### Hedef ve veri kalitesi
- `tuketim`: NaN/negatif yok. Medyan 1,075. **Sıfır oranı %4.69 — sıfır satırları ATMA**
- Yük faktörü = `tuketim/(guc*24)`, medyan 0.106. **>1 olan 1,821 satır / 37 trafo bozuk**
  (36'sı test'te) → eğitimden çıkar, trafoyu atma, `static_has_bad_rows` flag'i ver
- **158 kapanmış aday** (30+ gün sıfır, train sonunda hâlâ sıfır), hepsi test'te,
  18,629 satır (%2.61)
- Sıfır bloğu dönüş oranı **q = 0.244**, dönüş seviyesi **L ≈ 3.20 log1p**
  → optimal tahmin `x* = q·L ≈ 0.78` (≈1.2 kWh). **SERT 0 OVERRIDE YAPMA**
- q blok uzunluğuna göre: 30-60 gün → 0.63 · 61-120 → 0.27 · 121-240 → 0.14 · 240+ → 0.12

### lag_364
Kapsam (±7 gün): test tümü **%35.0** · test warm %45.0 · F1 valid %46.3
Pencere genişletmenin katkısı marjinal (+1.2 puan) — boşluk kayma değil, geçmiş yokluğu.
→ Ekle ama yatırım yapma. `seas_lag364_available` flag'ini de ver.

### Mevsimsellik
Sabit kohortta yaz rampası gerçek: dip **Mayıs**, tepe **Temmuz**, geometrik **1.86×**.
**Test dönemi tam bu rampanın üstünde.**

**Aritmetik ortalama YANILTICI** (Konak 5.01× → medyan 1.56×, geometrik 1.47×).
Spearman aritmetik–medyan sadece 0.56. → `grp_` mevsimsel istatistikleri
**medyan veya geometrik ortalama** ile kur.

Robust sıralamada tepe: **SARIGÖL 4.3× · ÖDEMİŞ 2.7× · GÖRDES 2.7×** — yani yaz
rampasının sürücüsü kentsel klima değil, **tarımsal sulama**.
→ Hava feature'ları sadece CDD/HDD olmamalı; **yağış açığı, toprak nemi, ET0** eklenmeli.
Trafo sayısı <10 olan 10 ilçe güvenilmez, shrinkage uygula.

### Kolonlar
- `tanim` **string** ('202917T') — sayıya çevirme, category tut
- `lokasyon` 47 tekil, **iki format**: `İZMİR>BÖLGE>İLÇE` (%73.3), `MANİSA>İLÇE` (%26.7).
  "GEDİZ EDAŞ" jenerik değeri **yok**
- `guc` 41 tekil değer, 40–35,900 kVA, trafo içinde sabit, eksik yok
- Haftanın günü etkisi **artefakt**: ham açıklık %36 → normalize %3.7.
  `cal_dow` + `cal_is_weekend` ver, geç
- Submission: `id` = `tanim_YYYY-MM-DD`, sample_submission ile küme VE sıra birebir

---

## MUTLAK KURALLAR

1. **Kısa lag yasak** (`lag_1`, `lag_7`, `rolling_7`). Kullanılabilir: `lag_364`+
2. **Recursive tahmin yasak.** Direct multi-horizon
3. **Random KFold yasak.** Sadece `src/validation.py` fold'ları
4. **Validasyon test'in bilgi rejimini eşlemek ZORUNDA.** Cold-start ayrı vaka değil,
   `H = 0` halidir. `make_folds` her valid trafosuna `test_history_profile.csv`'den
   guc_bucket-stratified bir `H` atar ve sadece son `H` gününü bırakır
5. **Fold doğrulama kontrolü zorunlu:** cold satır payı ≈ **%22.2**,
   lag_364 ±7 kapsamı ≈ **%35.0**, H medyanı ≈ **105**. Sapma varsa modele geçme
6. **Skor her zaman warm/cold ayrı raporlanır.** Birleşik tahmin:
   `sqrt(0.778*warm_mse + 0.222*cold_mse)`
7. **Feature testi:** *"2026-07-31 satırı için sadece 2026-03-31'e kadarki hedef
   verisiyle hesaplanabilir mi? Trafo hiç görülmemişse ne olur?"*
8. **`features.py` her zaman `forecast_origin` alır.** Sızıntının bir numaralı kaynağı
9. **Hedef:** `log1p(tuketim)` üzerinde eğit, `expm1` + `clip(0, None)`.
   **Smearing / bias correction YOK** — log uzayı ortalaması RMSLE için zaten optimal
10. **Örnek ağırlığı eşit.** Trafo büyüklüğüne göre ağırlık verilmez
11. **Seed:** `src/config.py` içindeki `SEED` her yerde

---

## Sözleşmeler — değiştirmeden önce sor

```python
# src/config.py
SEED = 42
TRAIN_START, TRAIN_END = "2025-01-01", "2026-03-31"
TEST_START,  TEST_END  = "2026-04-01", "2026-07-31"
COLD_ROW_SHARE = 0.2216
GUC_BUCKETS = [(0,160), (161,400), (401,1000), (1001,1600), (1601,None)]

# src/validation.py
def rmsle(y_true, y_pred) -> float: ...
def make_folds(df, profile, seed=SEED) -> list[dict]: ...
def verify_fold(fold) -> dict: ...      # cold_row_share, lag364_cov, h_median
    # F2/F3'te lag364_cov kontrolü YAPISAL N/A (lag penceresi veri başlangıcından
    # önce) — sadece cold_row_share ve h_median kontrol edilir

# src/baselines.py
# b6 = warm'da b2 (trafo medyanı), cold'da b5 (guc×LF). b3/b4 referans olarak kalır
# (kırpılmış geçmiş rejiminde trafo×ay medyanı bayat: YoY seviye kayması ~+0.14 log)
def evaluate(df, y_true_col, y_pred_col) -> pd.DataFrame: ...
    # kırılımlar: global | warm/cold | H-bucket | ay | guc_bucket | il | ilce
    #             | ufuk-haftası | zero_streak durumu

# src/features.py
def build_features(df, forecast_origin: str, history: pd.DataFrame) -> pd.DataFrame: ...
FEATURE_GROUPS: dict[str, list[str]]

# src/weather.py
def get_weather(coords, start, end) -> pd.DataFrame: ...   # ~38 ilçe, parquet cache
```

**Kolon prefix:** `static_` `lvl_` `seas_` `cal_` `wx_` `grp_`

---

## Çalışma disiplini

- Her deney `experiments/log.csv`: `timestamp, exp_id, feature_set, model,
  f1_all, f1_warm, f1_cold, f1_blend, f2_all, f3_all, lb, note`
- **warm ve cold skorunu ayrı görmeden karar verme.** Global iyileşip cold kötüleşiyorsa
  testte zarar verir
- **Fold rolleri (düzeltilmiş — bilgi rejimi esas):**

  | fold | bilgi rejimi | rol |
  |---|---|---|
  | **F1** | train 12 ayı kapsıyor, valid ayının geçen yılı train'de MEVCUT | **BİRİNCİL** — test'in bilgi rejiminin tek eşi. Mevsimsel feature'lar SADECE burada doğrulanabilir |
  | F2 | train sadece Oca–Mar; valid ayının geçen yılı YOK | yön kontrolü. Mutlak eşik uygulanmaz, sadece b6'ya göre delta okunur |
  | F3 | aynı kusur (valid ayının geçen yılı yok) | kırılganlık alarmı. **b6 + 0.01 içinde kalmak yeterli**, geçmek zorunlu değil |

  **NOT:** Tam mevsimsel bilgiyle yaz doğrulaması bu veri setinde yapısal olarak
  İMKÂNSIZ. Yaz riski LB + tahmin-seviyesi sağlık kontrolüyle yönetilir.
  "F2 warm kendini geçsin" tipi kriterler geri çekildi.

- **TEŞHİS (pred_sanity v1): model yazı kaçırmıyor, BAHARI şişiriyor.**
  Aylık sapmalar: Nisan +0.21 / Mayıs +0.38 / Haziran +0.22 / Temmuz +0.01.
  Sebep: `lvl_lf_median_90d` çapası (gain %65) origin'de kış penceresini görüyor,
  Mayıs ise yılın dip ayı. Kanıt: cold Temmuz/Mayıs 1.80× (grp_ tabanlı) vs
  warm 1.61× (lvl_ tabanlı). **F1 bu kusuru YAPISAL OLARAK ölçemez**
  (F1'de çapa da hedef de kış).

- **Sağlık kriteri (d'): AYLIK KALİBRASYON.** Temmuz/Mayıs oranı kriteri GERİ
  ÇEKİLDİ (yanlış şeyi ölçüyordu). Yerine: her ay için
  `|tahmin_ay − (2025_gerçek_ay + 0.102)| ≤ 0.12` — Nis/May/Haz/Tem ayrı ayrı.
  2025 tabanı kompozisyon açısından tam eş değil (bazı test trafolarının
  2025 verisi yok) — raporda belirtilir, tam-kapsamlı trafolarla da hesaplanır.
- **Hedef çıta: blend 1.07** — sıfır problemi mükemmel çözülse ulaşılacak seviye.
  Referanslar: b6 blend F1=1.2692 · F2=1.2654 · F3=1.3055
- `guc × 24 × LF[ilce, ay]` cold baseline'ı sonuna kadar referans kalır
- Dış veri cache'lenir, `data/external/` silinmez

---

## Yapılmayacaklar

- ❌ `data/raw/` değiştirmek
- ❌ Sıfır tüketim satırlarını eğitimden atmak
- ❌ Kapanmış trafolara sert 0 override
- ❌ Aritmetik ortalamayla mevsimsel indeks kurmak
- ❌ `days_since_entry` ramp feature'ları (RAMP YOK)
- ❌ Cold trafolara warm feature uydurmak (NaN kalır veya `grp_`'ye düşer)
- ❌ `tanim`'ı sayıya çevirmek
- ❌ Her çalıştırmada Open-Meteo'yu yeniden çağırmak
- ❌ Sözleşmeleri habersiz değiştirmek

---

## Ortam

```
Python 3.11 · pandas · numpy · pyarrow
lightgbm · catboost · scikit-learn
requests · holidays
matplotlib · seaborn · shap
```