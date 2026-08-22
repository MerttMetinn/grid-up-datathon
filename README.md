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
| En iyi model (yerel CV) | **F1 blend 1.1244** (baseline b6: 1.2692) |
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
├── scripts/                ← numaralı, sırayla çalışan deney zinciri (00→13)
├── reports/                ← her scriptin ürettiği bulgular (kanıt arşivi)
├── experiments/log.csv     ← tüm deneylerin skor kaydı
├── models/                 ← kaydedilmiş final model (s2) + MODEL_CARD.md
├── data/
│   ├── raw/                ← train/test/sample_submission (git-ignored, ayrı dağıtılır)
│   └── processed/          ← parquet cache + test_history_profile.csv
└── submissions/            ← sub_b6, sub_p3, sub_s (üretilen tahminler)
```

### Script zinciri (00 → 13)

| script | ne yapar | çıktı |
|---|---|---|
| `00_recon.py` | temel yapı keşfi | `reports/recon.md` |
| `01_recon2.py` | cold-start + lag_364 profili | `reports/recon2.md` |
| `02_recon3.py` | toplu giriş, ramp testi, q | `reports/recon3.md` + profil CSV |
| `03_run_baselines.py` | 6 baseline × 3 fold + fold doğrulama | `reports/baseline_results.md` |
| `04_diagnose.py` | RMSLE ayrıştırması (sıfır analizi) | `reports/diagnosis.md` |
| `05_cold_population.py` | cold popülasyon sağlaması | `reports/cold_population.md` |
| `06`–`13_*.py` | model evrimi v1 → v7 | `reports/model_v*.md` |

Her script tek başına, deterministik (sabit `SEED`) ve idempotent çalışır.

---

## Kurulum ve çalıştırma

```bash
# 1. Bağımlılıklar (Python 3.11+)
pip install -r requirements.txt

# 2. Ham veriyi yerleştir (repoda YOK — datathon platformundan indir)
#    data/raw/train.csv, test.csv, sample_submission.csv

# 3. Keşiften final modele tüm zinciri çalıştır (veya tek tek)
python scripts/00_recon.py
python scripts/03_run_baselines.py
python scripts/13_train_final.py     # en iyi model + submissions/sub_s.csv
python scripts/14_save_model.py      # final modeli models/ altına kaydet
```

İlk çalıştırmada `data.py` ham CSV'leri okuyup `data/processed/*.parquet` cache üretir;
sonraki çalıştırmalar cache'ten okur (24 MB, saniyeler).

**Kaydedilmiş model:** `models/s2_{main,cold}_seed{0,1,2}.txt` (LightGBM native format) +
`models/MODEL_CARD.md`. Pipeline deterministik (SEED=42, sabit tur) olduğundan bu modeller
`submissions/sub_s.csv`'yi **birebir** üretir (byte-identik doğrulandı). Yükleme talimatı
model kartında.

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
istatistikleri) öğrenilir. Cold satırlar için ayrı bir model + fiziksel baseline harmanı
kullanıyoruz. Kurulu güç modele init_score olarak verildiğinden görülmemiş kombinasyonlara
genelleme güçlü.

**S: Yaz rampasını (test dönemi) modelin geçmişi olmadan yakaladığını nasıl doğruladınız?**
C: Bu bizim en zorlu noktamızdı ve **dürüst cevap: tek bir CV skoruyla doğrulanamaz.** Hiçbir
validasyon fold'u hem yaz hedefi hem de o yazın geçen-yıl geçmişini aynı anda içeremez — bu
yapısal bir kısıt. Bunun yerine **tahmin-seviyesi sağlık kontrolü** kurduk: 2026 tahminimizin
aylık ortalamasını, aynı trafoların 2025 gerçeğine YoY drift (+0.102) ekleyerek karşılaştırıyoruz.
Sonuç: dört ayın hepsinde sapma ≤ 0.10 (`reports/model_v7.md`).

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
iyileşip ikincil fold'da bozuyorsa reddedildi. Final model 3 farklı seed ortalaması (varyans
azaltma). Tüm 30+ deneyin skoru `experiments/log.csv`'de izlenebilir.

**S: Neyi farklı yapardınız / sıradaki adım ne?**
C: Hava durumu (`wx_`) ailesi henüz eklenmedi — Open-Meteo arşivinden Nis–Tem 2026 için
*gerçek gözlemlenmiş* sıcaklık çekilebilir (tahmin değil), ilçe bazlı CDD/HDD nonlineer
terimlerle yaz rampası daha keskin yakalanabilir. Ayrıca CatBoost ile ensemble çeşitliliği ve
segment-bazlı modeller denenebilir. Cold sıfır düzeltmesinin işe yaramadığını da tespit ettik
(model init'i telafi ediyor) — bu kaldırılacak.

---

## Yeniden üretilebilirlik

- Tüm rastgelelik `src/config.py`'deki tek `SEED = 42` ile sabitlenir.
- Dış veri cache'lenir, ham veri asla değiştirilmez.
- Her deney `experiments/log.csv`'ye yazılır; her rapor scripti deterministiktir.
- Sözleşmeler (fonksiyon imzaları) `CLAUDE.md`'de kilitlidir.
