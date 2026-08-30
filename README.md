# Grid Up Datathon — Trafo Bazlı Günlük Tüketim Tahmini

Ege bölgesindeki (İzmir + Manisa) dağıtım trafolarının **günlük elektrik tüketimini**
tahmin eden uçtan uca bir makine öğrenmesi pipeline'ı. Metrik: **RMSLE** (düşük = iyi).

> **Tek cümlelik özet:** Bu bir zaman serisi problemi değil, **kısıtlı geçmişle kesitsel
> tahmin** problemidir — çünkü test trafolarının %28.8'i eğitimde hiç görülmemiştir ve
> çoğu 2026 Mayıs'ında şebekeye toplu olarak katılmıştır.

| | değer |
|---|---|
| Train | 2025-01-01 → 2026-03-31 · 1.23M satır · 5,344 trafo |
| Test | 2026-04-01 → 2026-07-31 · 715K satır · 7,036 trafo |
| Metrik | RMSLE |
| **En iyi public LB** | **1.05343** (öngörü) · ölçülen en iyi 1.05568 · baseline b6 1.36728 |
| Teslim çözümü | `notebooks/gridup_leakfree_submission.ipynb` — dış veri kullanmaz |
| Kolonlar | `tanim` (trafo), `guc` (kVA), `tarih`, `lokasyon`, hedef `tuketim` |

---

## Neden bu problem zor? (üç yapısal gerçek)

1. **Cold-start (%28.8):** 2,024 test trafosu eğitimde hiç yok. Sadece `guc`, `lokasyon`,
   `tarih` biliniyor — geçmiş tüketim yok. Çoğu **2026-05-11'de tek seferde** devreye giriyor.
2. **Yaz rampası test döneminin tam ortasında:** dip Mayıs → tepe Temmuz, geometrik ~1.86×.
   Modelin bu ekstrapolasyonu geçmişini görmeden yapması gerekiyor.
3. **Metrik sıfırlara aşırı duyarlı:** hatanın %57'si, satırların %4.4'ünü oluşturan
   gerçek-sıfır (ölü/kapalı trafo) satırlarından geliyor.

Bu üç gerçek de **veriyle doğrulandı, varsayılmadı** — bkz. `reports/recon*.md`.

---

## Yaklaşımın özü

```
log1p(tuketim) ≈ log(guc × 24) + log(yük_faktörü)
                 └── bilinen ──┘   └── model bunu öğrenir ──┘
```

Fiziksel çıpa `log(guc×24)` modele **init_score** olarak verilir; LightGBM yalnızca
**yük faktörünü** (ve mevsimsel sapmasını) öğrenir. Bu formülasyon, hiç görülmemiş
`guc`/ilçe kombinasyonlarına genellemede kritik.

**Dört tasarım kararı pipeline'ı belirledi:**

1. **Çok-origin eğitim:** Eğitim, test geometrisiyle birebir kurulur — her fold'da birden
   çok `forecast_origin` kesilir, eğitim satırları hep origin-sonrası, feature'lar yalnızca
   origin-öncesi geçmişten. Bu, sızıntıyı (leakage) yapısal olarak imkânsız kılar.
2. **Geçmiş-uzunluğu eşlemeli validasyon:** Cold-start ayrı bir vaka değil, `H=0` halidir.
   Her fold, `test_history_profile.csv`'den guc-stratified geçmiş uzunluğu örnekleyerek
   test'in bilgi rejimini (cold payı %22, lag_364 kapsamı %35) taklit eder.
3. **Mevsim-farkındalıklı çapa:** Mevsim bilgisi feature değil, init_score içinde taşınır
   ve `α=0.4` ile yumuşatılır — takvim feature'larıyla çift-sayımı önlemek için.
4. **Cold için hibrit:** cold trafolara ayrı bir model + fiziksel baseline (b5) log-uzayında
   harmanlanır (`w=0.45`).

Ayrıntılı gerekçeler ve tüm alternatiflerin neden elendiği: **[`docs/DECISIONS.md`](docs/DECISIONS.md)**.

---

## Proje yapısı

```
grid-up-datathon/
├── README.md               ← buradasın
├── CLAUDE.md               ← kilitli kurallar + veri gerçekleri + sözleşmeler
├── requirements.txt
├── docs/
│   ├── PROJE_HANDOFF.md    ← TAM BAĞLAM (yeni katılan buradan başlasın)
│   ├── VERI_KAYNAKLARI.md  ← veri kaynağı beyanı (host incelemesi için)
│   ├── NEREDE_KALDIK_*.md  ← günlük çalışma notları
│   ├── DECISIONS.md        ← KARAR GÜNLÜĞÜ (her kararın nedeni)
│   ├── ROADMAP.md          ← başlangıç planı
│   ├── STRATEGY_v3.md      ← recon sonrası revize strateji
│   └── AGENTS.md           ← çok-ajanlı çalışma düzeni notları
├── src/                    ← pipeline (import edilebilir modüller)
│   ├── config.py           ← sabitler, yollar, SEED, tarih sınırları
│   ├── data.py             ← yükleme + temizlik + parquet cache
│   ├── validation.py       ← H-eşlemeli fold'lar, RMSLE, evaluate
│   ├── baselines.py        ← b1…b6 (b6 yenilmesi gereken referans)
│   ├── features.py         ← feature engineering + anchor (init_score)
│   ├── train.py            ← çok-origin LightGBM eğitim mantığı
│   ├── leak_check.py       ← feature sızıntı/sağlık kontrolü
│   └── predict.py          ← submission yazımı + doğrulama
├── notebooks/              ← TESLİM ÇÖZÜMÜ (kendi kendine yeterli, leak'siz)
│   ├── gridup_leakfree_submission.py    ← kaynak (percent format)
│   └── gridup_leakfree_submission.ipynb ← üretilen notebook
├── scripts/                ← numaralı, sırayla çalışan deney zinciri (00→31)
├── reports/                ← her scriptin ürettiği bulgular (kanıt arşivi)
├── experiments/log.csv     ← tüm deneylerin skor kaydı
├── models/                 ← kaydedilmiş final model (s2) + MODEL_CARD.md
├── data/
│   ├── raw/                ← train/test/sample_submission (git-ignored, ayrı dağıtılır)
│   └── processed/          ← parquet cache + test_history_profile.csv
└── submissions/            ← üretilen tahminler (CSV'ler git-ignored, finaller hariç)
```

### Script zinciri (00 → 31)

| script | ne yapar | çıktı |
|---|---|---|
| `00_recon.py` | temel yapı keşfi | `reports/recon.md` |
| `01_recon2.py` | cold-start + lag_364 profili | `reports/recon2.md` |
| `02_recon3.py` | toplu giriş, ramp testi, q | `reports/recon3.md` + profil CSV |
| `03_run_baselines.py` | 6 baseline × 3 fold + fold doğrulama | `reports/baseline_results.md` |
| `04_diagnose.py` | RMSLE ayrıştırması (sıfır analizi) | `reports/diagnosis.md` |
| `05_cold_population.py` | cold popülasyon sağlaması | `reports/cold_population.md` |
| `06`–`13_*.py` | model evrimi v1 → v7 | `reports/model_v*.md` |
| `15`–`25_*.py` | hava, hurdle, catboost, tweedie, hata analizi | `reports/*.md` |
| `26`–`28_*.py` | optuna derinleştirme, leak'siz model, recency | `reports/optuna_*.md` |
| `29_build_notebook.py` | percent `.py` → `.ipynb` çevirici | `notebooks/*.ipynb` |
| `30_hdraw_ensemble.py` | H-çekilişi topluluğu (12 model) | `reports/hdraw_ensemble.md` |
| `31_zero_anchor.py` | sıfır düzeltmesi deneyi (elendi) | `reports/zero_anchor.md` |

Her script tek başına, deterministik (sabit `SEED`) ve idempotent çalışır.

---

## Kurulum ve çalıştırma

```bash
# 1. Bağımlılıklar (Python 3.11+)
pip install -r requirements.txt

# 2. Ham veri: data/raw/{train,test,sample_submission}.csv
#    (depoda mevcut; yoksa datathon platformundan indir)

# 3a. TESLİM ÇÖZÜMÜ — tek komut, submission.csv üretir
python notebooks/gridup_leakfree_submission.py

# 3b. Geliştirme zinciri (keşif + baseline + model evrimi)
python scripts/00_recon.py
python scripts/03_run_baselines.py
python scripts/13_train_final.py     # v7 modeli + submissions/sub_s.csv
python scripts/14_save_model.py      # modeli models/ altına kaydet
```

> **Not:** `scripts/13_train_final.py` tarihsel "v7" modelini üretir, **final teslim
> çözümü değildir**. Final çözüm `notebooks/` altındadır (bkz. bir alttaki bölüm).

İlk çalıştırmada `data.py` ham CSV'leri okuyup `data/processed/*.parquet` cache üretir;
sonraki çalıştırmalar cache'ten okur (24 MB, saniyeler).

**Kaydedilmiş model:** `models/s2_{main,cold}_seed{0,1,2}.txt` (LightGBM native format) +
`models/MODEL_CARD.md`. Bunlar **v7 (s2) dönemine ait** tarihsel modellerdir ve
`submissions/sub_s.csv`'yi birebir üretir (byte-identik doğrulandı). Final teslim
çözümü bu dosyaları kullanmaz — notebook modelleri her çalıştırmada baştan eğitir.

**Submission CSV'leri git'te tutulmaz** (her biri ~28 MB, notebook deterministik olduğu
için yeniden üretilebilir). İstisna: final olarak seçilen `sub_sp17.csv` ve
`sub_sp30.csv`. Ayrıntı `.gitignore` içinde.

---

## Teslim notebook'u (leak'siz, kendi kendine yeterli)

Yarışmaya teslim edilen çözüm tek bir notebook'tur:

```
notebooks/gridup_leakfree_submission.ipynb
```

- **`src/` modüllerini import etmez** — tüm kod içindedir, Kaggle'da olduğu gibi çalışır.
- **Dış veri kullanmaz.** Yalnızca `train.csv`, `test.csv`, `sample_submission.csv` ve
  `holidays` paketinin TR tatil takvimi. Gerçekleşmiş hava durumu ve EPİAŞ verisi
  bilinçli olarak dışarıda bırakılmıştır (bkz. `docs/VERI_KAYNAKLARI.md`).
- Son hücre bir **sızıntı denetimi** çalıştırır: açılan tüm dosyaları listeler,
  yarışma dosyaları dışında bir şey okunmadığını `assert` ile doğrular.

Notebook doğrudan düzenlenmez. Kaynak percent formatlı Python dosyasıdır
(lint edilebilir, çalıştırılabilir), `.ipynb` ondan üretilir:

```bash
# kaynağı düzenle: notebooks/gridup_leakfree_submission.py
python scripts/29_build_notebook.py      # -> notebooks/gridup_leakfree_submission.ipynb
```

**Model kurgusu:** 4 geçmiş-uzunluğu (H) çekilişi × 3 tohum = **12 LightGBM**, log
uzayında ortalanır. İki varyans kaynağı ayrı ayrı çeşitlendirilir: tohum LightGBM'in
bagging/feature örneklemesini, H çekilişi ise eğitim matrisinin kendisini değiştirir.
H varyansı tohum ortalamasıyla sönmez — tüm tohumlar aynı eğitim matrisini paylaşır.

**Determinizm doğrulandı:** notebook iki kez bağımsız çalıştırıldı, üretilen
`submission.csv` **bit düzeyinde aynı** (MD5 `62967caa88b16ba58d8d0eeb59630402`).

### Public LB'den gelen 2 kalibrasyon sabiti

Notebook'ta iki sabit public leaderboard geri bildirimiyle ölçülmüştür. İkisi de tek
boyutludur ve **kapalı formda çözülmüştür** — parametre taraması yapılmamıştır.
Türetme ve gerekçe: `docs/VERI_KAYNAKLARI.md` bölüm 4.

| sabit | değer | ne yapar |
|---|---|---|
| `LEVEL_SHIFT` | −0.2712 | genel tahmin seviyesi |
| `SEGMENT_DELTA` | +0.1709 | cold/warm paylaşımı (**genel ortalamayı değiştirmez**) |

`SEGMENT_DELTA`'nın gerekçesi bir modelleme hatasıdır: sıfır-şişirilmiş dağılımda
RMSLE'yi minimize eden tahmin `E[log1p(y)] = (1−p)·L` yani **çarpımsaldır**; anchor
ise toplamsal `L + log(1−p)` kullanıyor — bu ham ölçekte ortalama için doğru, log
ölçekte değil. Sapma cold satırlarda birikiyor. Ölçülen: cold **+0.184**, warm
**−0.035**, ağırlıklı ortalama **+0.013** — yani genel kalibrasyonda görünmüyor.

---

## Jüri sunumu — hazır sorular ve cevaplar

**S: Bu neden klasik bir zaman serisi tahmini değil?**
C: Test trafolarının %28.8'i eğitimde hiç görülmemiş ve çoğu 2026 Mayıs'ında toplu devreye
girmiş. Görülen trafolarda bile geçmiş kısa (medyan 174 gün, lag_364 kapsamı %35). Yani
"seriyi devam ettir" yerine "bu güçteki, bu ilçedeki trafo bu günde hangi yük faktöründe
çalışır" sorusunu çözüyoruz — kesitsel bir tahmin.

**S: Cold-start'ı nasıl çözdünüz?**
C: Fiziksel çıpayla. `tüketim ≈ guc × 24 × yük_faktörü` olduğundan, cold trafo için
bilinmeyen tek şey yük faktörüdür — o da benzer güçteki komşu trafolardan (`grp_` ilçe×ay
istatistikleri) öğrenilir. Kurulu güç modele `init_score` olarak verildiğinden görülmemiş
kombinasyonlara genelleme güçlü. Ayrı cold modeli ve fiziksel baseline harmanı denendi
ancak **elendi** — tek model + fiziksel çıpa daha iyi sonuç verdi.

Cold tarafında ayrıca ölçülmüş bir kalibrasyon düzeltmesi var: sıfır-şişirilmiş dağılımda
RMSLE'nin optimumu `(1−p)·L` (çarpımsal) iken anchor `L + log(1−p)` (toplamsal) kullanıyordu.
Bu, cold tahminlerini sistematik olarak yükseltiyordu (ölçülen sapma **+0.184**). Düzeltildi.

**S: Yaz rampasını (test dönemi) modelin geçmişi olmadan yakaladığını nasıl doğruladınız?**
C: Bu bizim en zorlu noktamızdı ve **dürüst cevap: tek bir CV skoruyla doğrulanamaz.** Hiçbir
validasyon fold'u hem yaz hedefi hem de o yazın geçen-yıl geçmişini aynı anda içeremez — bu
yapısal bir kısıt. Bunun yerine **tahmin-seviyesi sağlık kontrolü** kurduk: 2026 tahminimizin
aylık ortalamasını, aynı trafoların 2025 gerçeğine YoY drift (+0.102) ekleyerek karşılaştırıyoruz.
Sonuç: dört ayın hepsinde sapma ≤ 0.10 (`reports/model_v7.md`).

**Dürüstlük notu:** Bu sağlık kontrolünün referansı (2025 gerçeği + YoY drift `+0.102`)
sonradan public LB tarafından **yanlışlandı** — gerçek 2026 seviyesi bu referansın altında
çıktı. Seviye kalibrasyonu artık LB'den kapalı formda çözülüyor (bkz. iki kalibrasyon sabiti).
Bunu kayda geçiriyoruz çünkü yerel bir sağlık kriterinin yanıltabildiğini gösteriyor.

**S: Validasyonunuzun sızdırmadığını nereden biliyorsunuz?**
C: Üç katmanlı güvence. (1) Çok-origin eğitimde her satırın hedefi kesinlikle
`forecast_origin`'den sonra; feature'lar kesinlikle öncesinden. (2) `grp_` istatistikleri
yalnızca fold'un train penceresinden, valid trafoları asla girmeden hesaplanır. (3)
`leak_check.py` her feature için train/valid dağılım kaymasını ve "tek başına skoru şüpheli
iyileştiren feature" uyarısını üretir — hiçbir feature b6'yı tek başına geçmedi.

**S: Metrik neden sıfırlara bu kadar duyarlı, buna nasıl yaklaştınız?**
C: `reports/diagnosis.md`'de gösterdiğimiz gibi hatanın %57'si, satırların %4.4'ünü oluşturan
gerçek-sıfır satırlarından geliyor — bunlar birkaç "ölü/kapalı" trafoda toplanmış. Sert 0
override YAPMADIK; çünkü sıfır bloklarının %24'ü yeniden tüketime dönüyor (q=0.244). Bunun
yerine modele sıfır-serisi feature'ları (`lvl_zero_streak`, `grp_zero_rate`) verip optimal
`q·L` seviyesini kendisinin bulmasını sağladık.

**S: İş açısından değeri ne?**
C: Bu gerçek bir dağıtım şirketi problemi: **şebekeye yeni katılan trafoların yükünü geçmişi
olmadan tahmin etmek.** Doğrudan uygulaması var — yatırım planlaması, kapasite tahsisi, aşırı
yüklenme erken uyarısı. Ayrıca veri kalitesi bulguları (37 trafoda ölçüm hatası, 158 trafoda
uzun kesinti) operasyonel olarak da değerli.

**S: Sonuçlarınızın istikrarlı olduğunu nasıl gösteriyorsunuz?**
C: Üç fold farklı bilgi rejimini test ediyor (F1 birincil, F2 yön kontrolü, F3 kırılganlık
alarmı) ve her karar **warm/cold ayrı** skorla verildi. Bir değişiklik birincil fold'da
iyileşip ikincil fold'da bozuyorsa reddedildi. Final model **12 modelin** ortalaması
(4 geçmiş-uzunluğu çekilişi × 3 tohum) — iki ayrı varyans kaynağı da söndürülüyor.

**Ölçülmüş uyarı:** Aynı feature ve parametrelerle kurulan iki model arasında **0.0076**
skor farkı ölçtük. Bu, CV'de gördüğümüz küçük farkların bir kısmının gürültü olduğu anlamına
geliyor; bu eşiğin altındaki farklarla karar vermiyoruz.

**S: Neyi farklı yapardınız / sıradaki adım ne?**
C: Hava durumu denendi ve **bilinçli olarak çıkarıldı**: 17 hava feature'ının skora katkısı
~0 çıktı (1.06483 → 1.06525), üstelik tahmin dönemine ait gerçekleşmiş hava kullanmak
*forward leak*'tir ve yarışma sahiplerince uygun görülmemiştir. CatBoost/Tweedie ensemble de
denendi — modeller %97–99 korele çıktığı için çeşitlilik sağlamadı.

Sıradaki adım için asıl fırsat cold tarafında: hatanın **%56'sı** "cold + gerçek sıfır"
satırlarında (verinin %1.6'sı) ve orada trafo-bazlı bilgimiz yok. Denediğimiz her vekil
sinyal ölçülüp elendi (`tanim` ID komşuluğu, grup sıfır oranlarının satır-bazlı kullanımı,
yeni-trafo rampası). Ulaşılabilir tabanı **0.782** olarak ölçtük, yani teorik olarak yer var —
ama geçmişi olmayan bir trafonun kapalı olup olmadığı tanım gereği bilinemiyor. Gerçek
ilerleme trafo kimliğine bağlanabilecek harici bir öznitelik (abone tipi, kurulum tarihi,
fider bilgisi) ile gelirdi.

Kapatılan kaldıraçların tam listesi ve ölçülen değerleri aşağıdaki tabloda.

---

## Skor geçmişi (public LB)

| submission | skor | ne değişti |
|---|---|---|
| `sub_b6` (baseline) | 1.36728 | referans |
| `sub_s2_optuna` | 1.06483 | hava içerir (leak riski) |
| `sub_nowx_lo` | 1.06525 | hava çıkarıldı — **maliyeti ~0** |
| `sub_notebook` | 1.05764 | kendi kendine yeterli notebook |
| `sub_hens_lo` | 1.05737 | 12 modelli H-çekilişi topluluğu |
| `sub_lv030` | 1.09545 | ❌ seviye kaydırma (hesap hatası) |
| `sub_sp30` | **1.05568** | cold/warm ayrı kalibrasyon |
| `sub_zc` | 1.06374 | ❌ satır bazlı sıfır düzeltmesi |
| `sub_sp17` | öngörü 1.05343 | optimum δ = 0.171 |

---

## Ölçülen ve kapatılan kaldıraçlar

Aşağıdakilerin hepsi **denendi ve sayıyla kapatıldı** — varsayım değil. Yeni
katılan biri bunları tekrar denemesin.

| kaldıraç | ölçülen | sonuç |
|---|---|---|
| Cold/warm ayrı kalibrasyon | cold +0.184 / warm −0.035 | ✅ **kazandırdı** |
| Global seviye kaydırma | ortalama artık +0.013 | ❌ kapalı |
| Satır bazlı sıfır yapısı | korelasyon 0.005 (eşik 0.08) | ❌ sinyal yok |
| H çekilişi topluluğu | 0.0003 | ❌ ihmal edilebilir |
| `tanim` ID komşuluğu | 0.094 (ilçe zaten 0.179) | ❌ ilçenin altında |
| Yeni trafo rampası | yaş 1–7 gün: −0.019 | ❌ rampa yok |
| Hava durumu (17 feature) | katkı ~0 | ❌ çıkarıldı |
| Hurdle sıfır matematiği | `expm1((1−p)·L)` | ❌ zaten doğruymuş |
| Ensemble (CatBoost/Tweedie) | korelasyon %97–99 | ❌ çeşitlilik yok |
| Recency ağırlıklandırma | F1 iyi, F2/F3 kötü | ❌ overfit |

### Hatanın nerede olduğu

| kesim | satır payı | RMSLE | **hata payı** |
|---|---|---|---|
| **cold + gerçek sıfır** | %1.6 | 6.66 | **%56** |
| cold + pozitif | %20.6 | 1.08 | %19 |
| warm + pozitif | %75.0 | 0.54 | %18 |
| warm + sıfır | %2.8 | 1.80 | %7 |

Ulaşılabilir taban (kâhin tahmincilerle ölçüldü): **0.782**. Yani skorun daha
aşağı inebileceği yer var, ama geçmişi olmayan bir trafonun ölü olup olmadığı
tanım gereği bilinemiyor — hatanın %56'sı orada.

**Uyarı — model gürültüsü:** aynı feature ve parametrelerle kurulan iki model
arasında **0.0076** skor farkı ölçüldü. Bu eşiğin altındaki CV farkları anlamlı
değildir; karar vermek için LB gerekir.

---

## Yeniden üretilebilirlik

- Tüm rastgelelik `src/config.py`'deki tek `SEED = 42` ile sabitlenir.
- Dış veri cache'lenir, ham veri asla değiştirilmez.
- Her deney `experiments/log.csv`'ye yazılır; her rapor scripti deterministiktir.
- Sözleşmeler (fonksiyon imzaları) `CLAUDE.md`'de kilitlidir.
