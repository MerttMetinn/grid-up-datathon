# Karar Günlüğü — Grid Up Datathon

Bu dosya, projede alınan **her önemli kararı gerekçesiyle** kronolojik olarak kaydeder.
Amaç: sonuca değil, sonuca *nasıl* varıldığına dair kalıcı bir iz bırakmak.
Her satır bir gözlemin veya kararın "neden"idir.

İlgili dosyalar: `CLAUDE.md` (kilitli kurallar), `reports/*.md` (kanıtlar),
`experiments/log.csv` (tüm deney skorları).

---

## 0. Problem tanımı (değişmeyen)

- **Görev:** Trafo bazlı günlük elektrik tüketimi tahmini. Metrik **RMSLE**.
- **Train:** 2025-01-01 → 2026-03-31 (1,226,237 satır, 5,344 trafo)
- **Test:** 2026-04-01 → 2026-07-31 (714,688 satır, 7,036 trafo)
- **Erişilebilir bilgi:** `tanim`, `guc` (kVA), `tarih`, `lokasyon`; hedef `tuketim`.

> **En kritik erken kavrayış (recon sonrası):** Bu bir "zaman serisi devam ettirme"
> problemi DEĞİL. Test satırlarının %22'si hiç görülmemiş trafolara ait, warm
> trafoların geçmiş medyanı 174 gün, lag_364 kapsamı %35. Bu, **kısıtlı geçmişle
> kesitsel tahmin** problemidir. Modelin öğrendiği asıl şey:
> `log1p(tuketim) ≈ log(guc × 24) + log(yük_faktörü)` — yani yük faktörü.

---

## 1. Keşif turları (recon 1–3)

### recon-1 (`reports/recon.md`) — temel yapı
- Train 5,344 trafo, test 7,036. **Test'in %28.8'i (2,024 trafo) train'de YOK** → cold-start
  projenin ana problemi olarak tanımlandı.
- `tanim` saf sayı değil (`'202917T'` gibi) → **string/category tutulur, sayıya çevrilmez.**
- `lokasyon` iki format: `İZMİR>BÖLGE>İLÇE` (%73) ve `MANİSA>İLÇE` (%27). Beklenen jenerik
  "GEDİZ EDAŞ" değeri **bu veride yok** → parse mantığı parça sayısına göre kuruldu.
- Bellek: dtype optimizasyonuyla 24 MB → **pandas yeterli, polars gerekmedi.**
- Yük faktörü >1 olan 1,821 satır / 37 trafo **bozuk ölçüm** (fiziksel imkânsız).
- 320 trafoda 30+ gün ardışık sıfır; 158'i train sonunda hâlâ sıfırda (kapanmış aday).

### recon-2 (`reports/recon2.md`) — cold-start + lag profili
- Cold **satır** oranı %22.2 (trafo oranı %28.8'den düşük — cold'lar daha az gün içeriyor).
- Cold trafolar warm'dan **daha büyük güçlü** (medyan 630 vs 400 kVA); cold oranı
  guc_bucket'a göre %16–%41 arası → **stratifikasyon zorunlu.**
- **lag_364 kapsamı sadece %35** (±7 gün penceresiyle bile). Pencere genişletme katkısı
  marjinal → boşluk kayma değil, geçmiş yokluğu. **Karar: lag_364 eklenir ama yatırım yapılmaz.**
- Haftanın günü etkisi **artefakt**: ham açıklık %36 → trafo-içi normalize %3.7.
  **Karar: `cal_dow` + `cal_is_weekend` ver, geç. DOW feature mühendisliği yapma.**
- Mevsimsellik gerçek ama **aritmetik ortalama yanıltıcı** (Konak 5.01× → medyan 1.56×).
  **Karar: `grp_` mevsim istatistikleri medyan/geometrik ile kurulur.**

### recon-3 (`reports/recon3.md`) — toplu giriş + ramp + q
- **Cold trafoların %65.5'i tek gün: 2026-05-11.** İdari toplu sisteme alım, fiziksel
  kurulum değil (takvim örüntüsü yok, aynı desen train'de de var).
- **RAMP YOK** — yeni giren trafo ilk günden olgun seviyede (ilk hafta medyan norm 0.98–0.99).
  **Karar: `days_since_entry` ramp feature'ları GEREKSİZ; sadece `static_is_first_day` flag.**
- Geçmiş uzunluğu (H) dağılımı guc_bucket'a göre çıkarıldı → `test_history_profile.csv`.
- **Sıfır bloğu dönüş oranı q = 0.244**, dönüş seviyesi L ≈ 3.20 log1p. Optimal tahmin
  `x* = q·L ≈ 0.78` (≈1.2 kWh). **Karar: kapanmış trafolara SERT 0 override YAPMA** —
  log-L2 kaybının optimumu zaten q·L; modele doğru feature'ları vermek yeterli.

---

## 2. Ölçüm altyapısı — en yüksek getirili yatırım

> "Ölçemiyorsan hiçbir şey yapmıyorsun." Model kurmadan önce validasyon doğru kuruldu.

### Validasyon (`src/validation.py`)
- **Cold-start ayrı vaka değil, `H = 0` halidir.** `make_folds` her valid trafosuna
  `test_history_profile.csv`'den **guc_bucket-stratified** bir `H` atar ve sadece son
  `H` gününü bırakır. Böylece cold + kısa-geçmiş aynı anda simüle edilir.
- **H ataması bucket içinde kantil eşlemeli:** derin geçmişli trafoya uzun H. Bağımsız
  rastgele atama lag_364 kapsamını çökertiyordu (0.14'e) — eşleme test seviyesine (0.33) çıkardı.
- **Cold satır payı kalibrasyonu:** doğal cold'lar valid'de az satırlı olduğundan hedef
  %22.2'ye ulaşmak için en sığ geçmişli warm trafolar cold'a terfi eder.
- **`verify_fold` zorunlu kontrolü:** cold satır payı ≈%22.2, lag_364 ≈%35, H medyanı ≈105.
  F1 üçünü de tutturuyor. **F2/F3'te lag_364 YAPISAL N/A** (lag hedef penceresi
  2025-01-01'den önce; hiçbir fold kuramaz).

### Fold rolleri (üç kez revize edildikten sonra son hali)
| fold | bilgi rejimi | rol |
|---|---|---|
| **F1** | train 12 ay, valid ayının geçen yılı MEVCUT | **BİRİNCİL** — test'in bilgi rejiminin tek eşi |
| F2 | train sadece Oca–Mar, geçen yıl YOK | yön kontrolü, b6'ya göre delta |
| F3 | aynı kusur (yaz origin → kış hedefi) | kırılganlık alarmı, b6+0.01 içinde kal |

> **Kilit ders:** Tam mevsimsel bilgiyle yaz doğrulaması bu veri setinde **yapısal olarak
> imkânsız** (hiçbir fold hem yaz hedefi hem geçen-yıl-yaz geçmişi içeremez). Yaz riski
> LB + tahmin-seviyesi sağlık kontrolüyle yönetilir, tek bir CV skoruyla değil.

### RMSLE ayrıştırması (`reports/diagnosis.md`) — en önemli teşhis
- **Metriğin ana kaldıracı seviye tahmini değil, SIFIRLARI BİLMEK.** F1'de b6'nın kareli
  hatasının **%57'si**, satırların yalnızca %4.4'ünü oluşturan gerçek-sıfır satırlardan geliyor.
- Sıfırlar dağınık değil: cold'da sıfır satırların %98'i, warm'da %93'ü "ölü/yarı-ölü"
  trafolarda (sıfır oranı %75+). **Problem "hangi gün sıfır" değil, "hangi trafo ölü".**
- Ama statik bilgiyle sıfır tahmini zayıf: hücre-oranı AUC 0.56 (cold'da 0.53).
- **YoY drift +0.102 log1p** (sabit kohort, Oca–Mar). lag_364 ve anchor'da düzeltme katsayısı.

---

## 3. Baseline'lar (`src/baselines.py`, `reports/baseline_results.md`)

| # | tanım | rol |
|---|---|---|
| b1 | global medyan | taban çizgi |
| b2 | trafo medyanı | **warm'da şaşırtıcı güçlü (F1 warm 0.90)** |
| b3 | trafo × ay medyanı | kırpılmış geçmişte b2'den kötü (YoY bayat) |
| b5 | `guc×24×LF_med[ilçe,ay,haftaici]` | **cold baseline'ı — fiziksel çıpa** |
| **b6** | **warm→b2, cold→b5 hibrit** | **yenilmesi gereken referans: F1 1.2692** |

- **b6'nın warm bacağı b3 değil b2** (sözleşme güncellemesi): kırpılmış geçmiş rejiminde
  trafo×ay medyanı bayat kalıyor, b2 daha iyi. b3/b4 referans olarak tutuldu.
- **b5 (guc çıpası) sıfır-dışı cold satırlarda oracle-sabitten +0.185 iyi** → `guc` gerçek
  seviye sinyali taşıyor, kazanç sıfırlardan gelmiyor.

---

## 4. Model evrimi (v1 → v7) — her adım bir dersle

Tüm skorlar `experiments/log.csv`'de. Aşağıda **neden bir sonraki adıma geçildiği** var.

| ver | ne denendi | F1 blend | sonuç / ders |
|---|---|---|---|
| v1 | LightGBM, tek origin, tüm feature | 1.2488 | **Self-leakage:** `lvl_mean_90d` origin'e komşu eğitim satırlarının hedefini içeriyor; model %54 gain'le ona yaslanıp early-stop 87 turda underfit. Fold'lar arası tutmadı. |
| v2 | çok-origin (test geometrisi) + H-örnekleme | 1.1220 | Eğitim satırları HEP origin-sonrası → self-leakage bitti, −0.13 sıçrama. Ama F3 çöktü (origin'ler kış hedefi üretemiyor). |
| n1-3 | origin'leri fold'un tamamına yay + `grp_` ayrıştırması | 1.1235 | F3 mevsim boşluğu kapandı. Cold model + b5 harmanı nz_cold'u b5'in altına indirdi. |
| p1-3 | mevsim-nötr `lvl_*_full` çıpaları + seed averaging | **1.1222** | En iyi F1. Ama F2 warm çakılı, F3 hâlâ b6+0.02. |
| q1-2 | `lvl_season_*` feature + warm b2-sigortası | 1.1263 | Mevsim feature'ları F1'de **%0 gain** (cal_ay+lvl aynı bilgiyi taşıyor). F3 kapandı ama q2 rampı sönümledi. |
| r1-3 | mevsim bilgisini feature'dan **init_score'a** taşı | 1.1747 | Çapa `lvl_lf_median_90d` gain'ini %28→%0.7 düşürdü, F3'ü kurtardı (−0.05). AMA **Temmuz çift-sayımı** (anchor + cal_ay ikisi de mevsim) + F1 geriledi. |
| **s2** | **anchor'ı α=0.4 yumuşat + cold sıfır düzeltmesi** | **1.1244** | **En iyi bileşim** — aşağıda. |

### Neden çok-origin eğitim (v2, kalıcı karar)
Test her satır için origin=2026-03-31'den 1–122 gün ilerisini tahmin ediyor. Eğitimi de
**aynı geometriyle** kurduk: her fold'da 3–10 forecast_origin kesilir, eğitim satırları
HER ZAMAN origin-sonrası, feature'lar yalnızca origin-öncesi geçmişten. Bu hem self-leakage'ı
yok eder hem `cal_horizon_days` sinyalini kazandırır.

### Neden init_score çapası (r/s serisi, kalıcı karar)
Fiziksel model `log1p(tuketim) ≈ log(guc×24) + log(LF)`. Bunu **init_score** olarak verince
model artık sadece **artığı** (residual) öğrenir. Mevsim-farkındalıklı çapa:
- warm: `lvl_median_log_full + mevsim_indeksi[ilçe, hedef_ay] − yıllık_ort[ilçe]`
- cold: `log(guc×24) + log(LF_nz_med[ilçe, hedef_ay, haftaici])`

### Neden α=0.4 yumuşatma (`reports/model_v7.md`)
Tam çapa (α=1.0) mevsimi **çift sayıyordu**: anchor Temmuz'u yükseltiyor + `cal_ay`/`cal_doy`
feature'ları da yükseltiyor → model residual'da Temmuz'u bir daha şişiriyor (+0.27 sapma).
`anchor_soft = base + α·(mevsim_sapması)` ile α grid'lendi. Seçim **F1 skoruna göre değil,
aylık kalibrasyona göre** (min max|sapma|). α=0.4 hem kalibrasyonda (0.099) hem F1'de (1.1244)
en iyi — tek yönlü, iç köşe yok.

---

## 5. Tahmin-seviyesi sağlık kontrolü — yaz doğrulamasının tek yerel aracı

CV yaz rejimini ölçemediği için (bkz. fold rolleri), **tahminin kendisi** 2025 gerçeğiyle
karşılaştırıldı (`reports/pred_sanity.md`, `reports/model_v7.md`).

- **Kriter d' (aylık kalibrasyon):** her ay için `|tahmin − (2025_gerçek + 0.102)| ≤ 0.12`.
  Kohort-eş tabanla (2025 Nis–Tem'de ≥110/122 gün veriye sahip trafolar) hesaplanır —
  kompozisyon artefaktından arınmış.
- **s2 sonucu:** Nis −0.099, May +0.026, Haz +0.025, Tem +0.087 → **max 0.099**. Projede ilk
  kez dört ay da eşik içinde. (İlk teşhis: model yazı kaçırmıyor, **baharı şişiriyordu**;
  anchor + α=0.4 bunu çözdü.)

---

## 6. Kalıcı MUTLAK KURALLAR (CLAUDE.md'den özet)

1. Kısa lag yasak (`lag_1/7`, `rolling_7`) — test'te hedef yok. `lag_364`+ serbest.
2. Recursive tahmin yasak — direct multi-horizon.
3. Random KFold yasak — sadece `validation.py` zaman+H-eşlemeli fold'ları.
4. Validasyon test'in bilgi rejimini eşlemek zorunda (cold = H=0).
5. Skor her zaman **warm/cold ayrı** raporlanır; birleşik `sqrt(0.778·warm_mse + 0.222·cold_mse)`.
6. Hedef `log1p` üzerinde eğitilir, `expm1 + clip(0)` ile geri çevrilir. **Smearing YOK.**
7. Örnek ağırlığı eşit — trafo büyüklüğüne göre ağırlık verilmez.
8. `tanim` sayıya çevrilmez; sıfır satırları eğitimden atılmaz; kapanmış trafoya sert 0 yazılmaz.
9. `grp_` istatistikleri medyan/geometrik + shrinkage, yalnızca fold train penceresinden
   (valid trafoları asla girmez — sızıntının bir numaralı kaynağı).

---

## 7. Açık uçlar / sonraki adımlar

- **Cold sıfır düzeltmesi (`log(1−zero_rate)` anchor'a) işe yaramadı** — cold model init'i
  residual'da telafi ediyor, cold bias'ı düşürmek yerine artırdı. **Öneri: kaldır.**
- **Hava (`wx_`) ailesi henüz yok.** Open-Meteo arşivi (Nis–Tem 2026 gerçek gözlem mevcut) +
  ilçe bazlı CDD/HDD nonlineer + yağış açığı/toprak nemi (tarımsal sulama sürücüsü). F2 fold'u
  içinden doğrulanamaz; katkısı ancak tam-eğitim + LB'de görülür.
- **LB kalibrasyonu bekleniyor:** `sub_b6`, `sub_p3`, `sub_s` yüklenince hangi fold'un LB'yi
  öngördüğü ölçülüp sonraki kararlar ona bağlanacak.
- Ensemble (CatBoost çeşitliliği) ve segment-bazlı modeller denenmedi.

---

## 8. Deney skoru özeti (referans)

| model | F1 blend | F2 blend | F3 blend | not |
|---|---|---|---|---|
| b6 (baseline) | 1.2692 | 1.2654 | 1.3055 | yenilmesi gereken |
| lgbm v2 (m2) | 1.1220 | 1.2312 | 1.4315 | F3 çöktü |
| lgbm p3 | 1.1222 | 1.2174 | 1.3204 | en iyi F1, F3 zayıf |
| **lgbm s2** | **1.1244** | **1.2432** | **1.2479** | **dengeli en iyi; kalibrasyon 0.099** |

s2 = çok-origin + mevsim-farkındalıklı anchor (α=0.4) + cold-only model + b5 harmanı (w=0.45),
3-seed. Üretilen submission: `submissions/sub_s.csv`.
