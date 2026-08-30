# Proje Devir Dosyası (Handoff) — Grid Up Datathon

**Amaç:** Yeni bir sohbet/kişi bu dosyayı okuyup **sıfırdan tam bağlam** kurabilsin.
Buraya kadar yapılan her şey, öğrenilen her ders ve sonraki adımlar burada.

**Son güncelleme:** 30 Ağustos 2026 · **Yarışma bitişi:** 1 Eylül 2026
**İlgili dosyalar:** `CLAUDE.md` (kurallar), `docs/DECISIONS.md` (karar günlüğü),
`docs/NEREDE_KALDIK.md` + `docs/NEREDE_KALDIK_26-27_AGUSTOS.md` (günlük notlar)

---

## 1. PROBLEM (30 saniyede)

- **Görev:** Trafo bazlı **günlük elektrik tüketimi** tahmini (İzmir + Manisa).
- **Metrik:** RMSLE (düşük = iyi).
- **Train:** Ocak 2025 – Mart 2026 · **Test:** Nisan – Temmuz 2026.
- **Ana zorluk:** Test trafolarının %28.8'i eğitimde HİÇ YOK (cold-start). Çoğu
  Mayıs 2026'da toplu devreye giriyor. Bu bir "zaman serisi devam" değil,
  **geçmişsiz kesitsel tahmin** problemi.
- **Fiziksel temel:** `tüketim ≈ guc × 24 × yük_faktörü`. Model esasen yük faktörünü öğrenir.

---

## 2. ŞU ANKİ DURUM (en önemli tablo)

| Model | Kaggle (public) skoru | Not |
|---|---|---|
| Basit baseline (b6) | 1.36728 | başlangıç |
| optuna (arkadaş, sena branch) | 1.06483 | HAVA içerir (leak riski) |
| sub_nowx_lo (leak'siz) | 1.06525 | eski güvenli aday |
| sub_notebook (notebook çıktısı) | 1.05764 | teslim notebook'u kuruldu |
| sub_hens_lo (12 model topluluğu) | 1.05737 | H çekilişi ortalaması |
| sub_lv030 (seviye −0.30) | 1.09545 | ❌ hesap hatası, bkz. 29-30 notu |
| **sub_sp30 (cold/warm ayrı kalibrasyon)** | **1.05568** | **mevcut en iyi** |
| sub_zc (satır bazlı sıfır düzeltmesi) | 1.06374 | ❌ korelasyon 0.005, sinyal yok |
| `sub_sp17` (optimum δ=0.171) | öngörü **1.05343** | gönderilmeyi bekliyor |
| Lider | 0.99046 | ulaşılabilir taban 0.78 — imkânsız değil |

**Sıralama:** ~133. (public). İlk 20 (1.01529) skorla ulaşılamıyor.
**Plato:** ~1.053. Kaldıraçların tamamı ölçülüp kapatıldı — ayrıntı ve sayılar
`docs/NEREDE_KALDIK_29-30_AGUSTOS.md` madde 8'de.

**En değerli bulgu (29-30 Ağustos):** Genel seviye kalibrasyonu doğru görünüyordu
(+0.013) ama segmentler ayrı ölçülünce cold **+0.184**, warm **−0.035** çıktı —
iki hata birbirini gizliyormuş. Kök neden: anchor'ın sıfır düzeltmesi RMSLE için
yanlış formda (`L + log(1−p)` yerine `(1−p)·L` olmalı). Düzeltildi, +0.0022 kazandı.

---

## 3. DOSYA / MODEL KONUMLARI

- **En iyi güvenli submission:** `submissions/sub_nowx_lo.csv` (leak'siz, 1.06525)
- **En iyi wx'li submission:** `submissions/sub_s2_optuna.csv` (1.06483, arkadaşın, leak riski)
- Kayıtlı modeller: `models/` (s2, wx, optuna_final)
- Arkadaşın optuna kodu + sonuçları: **`sena` git branch'i**
  (`scripts/20_optuna_train.py`, `reports/optuna_summary.md`, `model_features.json`)
- Kod: `src/` (config, data, features, train, validation, baselines, predict, weather, dataset)
- Deney scriptleri: `scripts/00_recon.py` → `scripts/28_recency.py`
- Tüm rapor/kanıt: `reports/*.md`

---

## 4. YAPILAN TÜM DENEMELER + SONUÇLARI

### Model evrimi (yerel CV, F1 blend)
| Aşama | Ne | F1 | Sonuç |
|---|---|---|---|
| v1 | LightGBM tek origin | 1.2488 | self-leakage, elendi |
| v2 | çok-origin eğitim | 1.1220 | sızıntı çözüldü |
| p3 | mevsim-nötr çıpalar + seed avg | 1.1222 | iyi |
| s2 | mevsim-farkındalıklı anchor (α=0.4) | 1.1244 | en iyi mimari |
| hurdle | ölü-trafo classifier + nonzero regresör | 1.1285 | AUC 0.94 |
| hurdle+opt | + arkadaşın feature+param | 1.1148 | en iyi CV |

### Kaggle'da denenen ve BAŞARISIZ olanlar
| Deneme | Sonuç |
|---|---|
| Seviye kaydırma (tahmini düşür) | ❌ optuna zaten optimal seviyede |
| CatBoost (farklı algoritma) | ❌ korelasyon 0.99, çeşitlilik yok |
| Tweedie (sıfır-şişkin loss) | ❌ korelasyon 0.97, genel kötü |
| Cold b5 harman W=1.0 (sub_final) | ❌ CV'de iyi, Kaggle'da KÖTÜ (1.079) |
| tanim (trafo ID) pattern | ❌ anlamsız numara |
| optuna 75-feature + 60 trial | ❌ 29-feature'ı geçmedi |
| Ensemble (optuna+tweedie+cat) | ❌ seviyeyi bozuyor |
| Recency weighting (halflife=90) | ❌ F1-overfit (F2/F3 bozuluyor) |

### BAŞARILI / önemli bulgu
| Deneme | Sonuç |
|---|---|
| **Hava'yı çıkarmak (sub_nowx)** | ✅ skor AYNI (1.0652 vs 1.0648) — hava GEREKSİZMİŞ |

---

## 5. KRİTİK DERSLER (en değerli kısım)

1. **Basit model kazanıyor.** Arkadaşın düz LightGBM (29 feat + optuna param) tüm
   karmaşık mimarilerimizi (hurdle, ensemble, CatBoost) geçti. Karmaşıklık işe yaramıyor.

2. **CV cold'da YANILTIYOR.** Bizim simüle cold (holdout warm trafo) ≠ gerçek cold
   (Mayıs'ta kurulan fiziksel trafo). CV'de iyi görünen cold ayarları Kaggle'da kötü
   çıktı (sub_final örneği). → **Cold kararları SADECE Kaggle ile verilmeli.**

3. **Seviye kaydırma öldü.** optuna zaten optimal seviyede tahmin ediyor (−0.19 sapma).
   Daha fazla düşürmek/yükseltmek skoru değiştirmiyor.

4. **Model çeşitliliği yok.** Tüm modeller aynı anchor + feature kullandığı için
   %97-99 korele. Ensemble faydası yok.

5. **Hatanın %56'sı "cold + ölü trafo"** satırlarında (verilerin %1.6'sı). Cold trafonun
   geçmişi yok → ölü olduğu bilinemiyor → YAPISAL TAVAN. Lider bile tam çözemez.

6. **HAVA DURUMU GEREKSİZ.** Open-Meteo gerçekleşmiş hava (forward leak) skora ~0 katkı
   yaptı. Leak'siz model (sub_nowx_lo) aynı skoru veriyor.

7. **`tanim` anlamsız** rastgele ID (pattern yok).

8. **Recency/F1-özel ayarlar overfit riski** — F1 iyileşip F2/F3 bozan değişiklik reddedilir.

---

## 6. ⚠️ FORWARD LEAK DURUMU (en kritik açık konu)

### Sorun
Biz Nisan–Temmuz 2026 için **gerçekleşmiş hava** verisi çektik ve kullandık = forward leak
(gerçekte o günü tahmin ederken hava bilinemez). Host: *"gerçekleşmiş hava durumu verisi
kullanmak uygun olmayacaktır"* dedi ama **net karar (diskalifiye?) vermedi.**

### Bizim durumumuz — İYİ HABER
Hava'yı test ettik, **skora katkısı ~0.** Leak'siz modelimiz (`sub_nowx_lo`) aynı skoru
veriyor (1.06525 vs 1.06483). Yani **leak riskini almadan aynı yerdeyiz.**

### Güçlü leak (trafo tüketimi) erişilemez
EPİAŞ tüketim verisi ulusal/bölgesel — trafo/ilçe bazlı DEĞİL. Trafo hedefi açık kaynak
değil. Yani "leak serbestmiş gibi" davransak bile **bizi yükseltecek veri YOK** (hava
tavanı 1.065).

### Lider 0.994 nasıl?
İki olasılık: (a) bizde olmayan özel veri, (b) public %30'a overfit → private'da düşer.

---

## 7. FİNAL STRATEJİSİ

**Kaggle 2 submission final seçtirir.** Öneri:
- **1. final:** `sub_nowx_lo` (leak'siz, 1.06525) → GÜVENLİ, diskalifiye riski sıfır
- **2. final:** optuna wx'li (1.06483) → SADECE host hava'yı serbest bırakırsa

**Neden leak'siz güvenli en iyi bahis:**
- Final private %70 ile belirlenir. Public'te 136. ama private çok farklı olabilir.
- Eğer leak/overfit takımlar private'da çökerse, temiz modelimiz yükselir.
- **136 → ilk 20 için asıl yolumuz: private'da leak'lilerin elenmesi.**

---

## 7b. TESLİM NOTEBOOK'U (29 Ağustos'ta eklendi)

Host ilk 20 takımın notebook'unu inceleyecek. Buna karşı **kendi kendine yeterli,
leak'siz** bir teslim notebook'u kuruldu:

```
notebooks/gridup_leakfree_submission.ipynb    # teslim edilecek (22 hücre)
notebooks/gridup_leakfree_submission.py       # kaynak (percent format)
scripts/29_build_notebook.py                  # .py -> .ipynb cevirici
docs/VERI_KAYNAKLARI.md                       # veri kaynagi beyani
submissions/sub_notebook.csv                  # notebook'un kendi ciktisi
```

**Özellikleri:**
- `src/` modüllerini **import etmez** — tüm kod içinde, Kaggle'da olduğu gibi çalışır
- Sadece `train.csv` + `test.csv` + `sample_submission.csv` + `holidays` paketi okur
- `test_history_profile.csv`'ye bağımlı değil — profili train+test'ten yeniden türetir
  (tüm sayılar birebir tuttu: cold %28.8/%22.2, H medyanı 105, toplu giriş 2026-05-11)
- Son hücrede **sızıntı denetimi**: açılan tüm dosyaları listeler, yarışma dışı
  dosya okunmadığını `assert` ile doğrular
- **Determinizm doğrulandı:** iki bağımsız tam koşu bit düzeyinde aynı çıktı verdi

**ÖNEMLİ BULGU — `sub_nowx_lo` yeniden üretilemez:**
O dosya `data/processed/*.parquet` cache'inin `tanim` kategori sırasına bağlı bir
rastgele çekilişle üretilmiş; ham CSV'den bu sıra türetilemez. Notebook aynı
yöntemi uygular ama farklı bir H çekilişi kullanır (korelasyon 0.992, seviye farkı
−0.027 log). Test-zamanı feature matrisi ve anchor iki yolda **bit düzeyinde aynı**
(doğrulandı) — fark sadece eğitimdeki H örneklemesinde.

→ **Bu yüzden `sub_notebook.csv` LB'de doğrulanmalı.** Skoru onaylanırsa final
submission o olmalı, çünkü teslim notebook'unun gerçekten ürettiği dosya odur.
`sub_nowx_lo`'yu final tutarsak, incelemede "notebook bu dosyayı üretmiyor" denir.

**Bekleyen LB testleri (öncelik sırası):**
1. `submissions/sub_notebook.csv` — notebook çıktısı, ~1.065 bekleniyor
2. `submissions/sub_recency.csv` — leak'siz, hiç denenmedi (F1 1.1074 ama F3 kötü)

---

## 8. SONRAKİ ADIMLAR (yeni chat'te ilk işler)

### En kritik (senin, kod değil):
0. **`sub_notebook.csv` ve `sub_recency.csv`'yi submit et** (bkz. 7b) — final
   seçimi bu iki skora bağlı.
1. **Host'tan NET cevap al:** "Forward-leak (gerçekleşmiş hava/EPİAŞ) kullanan takımlar
   private'da elenecek/dezavantaj görecek mi? Leak kullanmayanlar korunacak mı?"
   Bu cevap her şeyi belirler.
2. ~~**Notebook'ta veri kaynaklarını şeffaf yaz**~~ ✅ YAPILDI — bkz. 7b ve
   `docs/VERI_KAYNAKLARI.md`.

### Kod tarafında kalan (hepsi marjinal, dürüst beklenti düşük):
- Leak'siz cold-start: LB-güdümlü (CV yanıltıyor) — birkaç sıra kazanabilir
- Private sağlamlık için leak'siz modellerden dikkatli ensemble
- **Büyük sıçrama beklenmiyor** — leak'siz tavan ~1.065

### Yapılmayacaklar (denendi, işe yaramadı):
- ❌ Seviye kaydırma · ❌ CatBoost/Tweedie ensemble · ❌ recency (F1-overfit)
- ❌ cold b5 harman artırma · ❌ 75-feature optuna · ❌ hava ekleme (gereksiz)

---

## 9. TEKNİK ÖZET (yeni chat için hızlı referans)

- **Feature (75, `src/features.py`):** static_ (güç/konum), cal_ (takvim), lvl_ (geçmiş
  seviye), grp_ (grup istatistikleri), seas_ (lag_364), wx_ (hava — LEAK, çıkarılabilir).
- **Fiziksel çapa (anchor):** `init_score = guc×24×LF` mevsim-düzeltmeli, α=0.4 yumuşatma.
  `src/features.py: anchor_components / assemble_anchor`.
- **Validasyon (`src/validation.py`):** 3 fold, geçmiş-uzunluğu eşlemeli (cold simülasyonu).
  F1 birincil (kış), F2 yaz-yön, F3 kırılganlık. **UYARI: cold ölçümü güvenilmez.**
- **En iyi model kurgusu:** düz LightGBM + anchor init_score + 29 seçili feature (hava'sız) +
  optuna params. Hurdle/b5/ensemble EKLEME (hepsi geçmedi).
- **Leak'siz en iyi:** `scripts/27_optuna_nowx.py` → `sub_nowx.csv`, sonra optuna seviyesine
  −0.205 kaydırma → `sub_nowx_lo.csv` (1.06525).

---

**TEK CÜMLE ÖZET:** Leak'siz olarak ~1.065'te (136. sıra) sağlam bir yerdeyiz; hava/karmaşıklık
işe yaramadı; ilk 20 için tek gerçek şans host'un forward-leak'i private'da elemesi — o yüzden
`sub_nowx_lo`'yu güvenli final tut ve host'tan net cevap iste.
