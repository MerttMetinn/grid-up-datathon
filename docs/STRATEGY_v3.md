# STRATEGY v3 — Recon-2 Sonrası (v2'nin yerine geçer)

`docs/ROADMAP.md` genel çerçeve olarak durur. Çelişki olursa **bu dosya kazanır.**
Veri gerçekleri: `reports/recon.md`, `reports/recon2.md`

---

## 0. Tek cümlelik problem tanımı

> Test satırlarının %22'si, train'de hiç görülmemiş trafolara ait ve bunların çoğu
> **2026 Mayıs'ında toplu olarak** devreye giriyor. Kalan %78 için bile geçmiş kısa
> (lag_364 kapsamı %45). Yani bu bir "zaman serisi devam ettirme" problemi değil,
> **kısıtlı geçmişle kesitsel tahmin** problemi.

Modelin öğreneceği asıl şey: **belirli bir güçteki, belirli bir ilçedeki bir trafonun,
belirli bir günde hangi yük faktöründe çalıştığı.**

---

## 1. Cold-start yeniden yorumu

| Bulgu | Değer |
|---|---|
| Cold trafo | 2,024 / 7,036 (%28.8) |
| Cold **satır** | 158,369 / 714,688 (%22.2) |
| Test başında var olan cold trafo | **1 tane** |
| 2026-05'te giren | 1,666 (%82.3) |
| Gün sayısı %25/%50/%75 | **82 / 82 / 82** |

Üç çeyreğin de 82 olması, büyük bir kümenin **aynı gün** girdiğini gösterir
(122 − 82 = 40 gün sonra ≈ **2026-05-11**). Bu, tek tek saha kurulumu değil,
**toplu sisteme alım** imzasıdır.

### Ayırt edilmesi gereken iki senaryo

| | Davranış | Modelleme sonucu |
|---|---|---|
| **A. Fiziksel yeni kurulum** | Tüketim sıfırdan rampalanır | `days_since_entry` feature'ı ZORUNLU; ilk 30-60 gün düşük tahmin |
| **B. Toplu sisteme alım** | İlk günden olgun seviye | `guc × LF` doğrudan çalışır, ramp feature'ı gereksiz |

**Bu train verisinde ölçülebilir** (2025-11'de 595, 2026-03'te 476 toplu giriş var).
Recon-3 madde 2 bunu cevaplayacak. **Ölçmeden cold modelini kurma.**

### Cold profil farkları (holdout'ta taklit edilmeli)
- Güç medyanı **630 vs 400 kVA** — cold'lar daha büyük. 1,000 ve 1,250 kVA aşırı temsilli
- İlçe içi cold oranı %11 (Ödemiş) – %58 (Dikili); kentsel İzmir baskın
- Bornova tek başına cold'ların %10.8'i

---

## 2. Validasyon — geçmiş-uzunluğu eşlemeli fold

> **Kavramsal sadeleştirme:** Cold-start ayrı bir vaka değil, `history_length = 0` halidir.
> Fold'u geçmiş uzunluğuna göre kurarsan hem cold'u hem kısa-geçmişi aynı anda simüle edersin,
> ve lag_364 kapsamı kendiliğinden test seviyesine iner.

### `make_folds` algoritması

1. **Hedef dağılımı çıkar.** Her test trafosu için `H = train'de bulunduğu gün sayısı`
   (cold'lar için 0). Bu dağılımı `guc_bucket` bazında ayrı ayrı sakla.
   Ayrıca test'e giriş tarihi dağılımını (günlük) sakla.
2. **Zaman bölmesi** yap (train_end / valid_start / valid_end).
3. Valid penceresinde görünen her trafoya, kendi `guc_bucket`'ının dağılımından
   örneklenmiş bir `H` ata.
4. O trafonun train kısmında **yalnızca `train_end`'den geriye doğru son `H` gününü** bırak,
   gerisini sil. `H=0` ise trafo train'den tamamen çıkar (= cold).
5. Silinen satırlar `grp_` istatistiklerine de **girmez.**
6. Valid tarafında da giriş tarihi dağılımını eşle: `H=0` olan trafoların valid satırlarının
   bir kısmını başlangıçtan kırp, ki **cold satır payı ≈ %22** olsun.

### Fold'lar

| Fold | Train | Valid | Rol |
|---|---|---|---|
| **F1 — ana** | → 2025-12-31 | 2026-01-01 → 2026-03-31 | Birincil karar |
| **F2 — mevsim** | → 2025-03-31 | 2025-04-01 → 2025-07-31 | Yaz rampası sanity check. Karar için KULLANMA |
| **F3 — robustluk** | → 2025-08-31 | 2025-09-01 → 2025-12-31 | Farklı rejim |

### Doğrulama kontrolü (fold kurulduktan sonra ZORUNLU)
Fold'da şu üçünü ölç ve test değerleriyle karşılaştır:
- cold satır payı → hedef **%22.2**
- lag_364 ±7 kapsamı → hedef **%35.0**
- geçmiş uzunluğu medyanı → test'ten hesaplanan değer

Sapma büyükse fold kurgusu bozuktur, modele geçme.

### Skor birleştirme
RMSLE karesel olduğu için fold'lardan LB tahmini şöyle kurulur:
```
tahmini_lb = sqrt(0.778 * warm_mse + 0.222 * cold_mse)
```
`experiments/log.csv`'ye `warm`, `cold` ve bu birleşik değeri ayrı ayrı yaz.

---

## 3. lag_364 — düşük öncelik, silinmez

| Kapsam (±7 gün) | Değer |
|---|---|
| Test — tümü | **%35.0** |
| Test — warm | %45.0 |
| F1 valid — tümü | %46.3 |

Pencere genişletmenin katkısı marjinal (exact %33.8 → ±7 %35.0). Boşluk **kayma değil,
geçmişin hiç olmaması.** Daha geniş pencere denemeye gerek yok.

**Karar:** `seas_lag364_*` feature'ları eklenir ama yatırım oraya yapılmaz.
Mevsimsellik esas olarak `grp_` ilçe indeksleri + hava üzerinden kurulur.
`seas_lag364_available` (0/1) flag'i de feature olarak verilir.

---

## 4. Mevsimsellik — ilçe kırılımı zorunlu

Sabit kohortta (1,253 tam panelli trafo) yaz rampası gerçek: dip **Mayıs**, tepe **Temmuz**,
geometrik ölçekte **1.86×**. **Test dönemi (Nisan–Temmuz) tam bu rampanın üstünde.**

İlçeler arası fark devasa:

| Grup | Temmuz/Mayıs |
|---|---|
| İZMİR>KONAK, İZMİR>KINIK | ~5.0× |
| İZMİR>KARABAĞLAR, BAYRAKLI | 3.2–3.5× |
| Kıyı/turizm (ÇEŞME, URLA, SEFERİHİSAR) | 1.7–2.1× |
| İç Manisa (DEMİRCİ, KULA, SOMA) | 1.15–1.42× |

**Tek küresel mevsim eğrisi bu veriyi açıklayamaz.** Zorunlu feature'lar:
- `grp_seasonal_index[ilce, ay]` — ilçenin o aydaki seviyesi / yıllık seviyesi
- `grp_cdd_slope[ilce]` — ilçenin sıcaklık duyarlılığı katsayısı
- `grp_seasonal_index[ilce, guc_bucket, ay]` (yeterli örnek varsa)
- `wx_cdd` × `grp_cdd_slope[ilce]` etkileşimi

> **Uyarı:** Yukarıdaki oranlar aritmetik ortalama. Konak'ın 5.0×'i 68 trafodan birkaçından
> geliyor olabilir. Recon-3 madde 5'te medyan ve geometrik ortalama ile doğrula.

---

## 5. Sıfır / kapanmış trafolar — DÜZELTME

**v2'deki "0 tahmin et" önerisi YANLIŞTI. Sert override yapma.**

Log uzayında L2 kaybının optimumu, devam etme olasılığı `q` ve tipik seviye `L` iken:
```
x* = q · L          (x* = log1p ölçeğinde optimal tahmin)
```
Örnek: `q = 0.2`, `L = 7` → `x* = 1.4` → tahmin ≈ **3 kWh**. Sıfır değil, bin de değil.

Bu tam olarak `log1p` hedefi üzerinde L2 ile eğitilen bir modelin **kendiliğinden**
bulduğu değerdir. Yapılması gereken: modele doğru feature'ları vermek.

- `lvl_zero_streak_days` — forecast_origin'de kaç gündür sıfır
- `lvl_zero_ratio_90d` — son 90 günde sıfır gün oranı
- `lvl_days_since_last_nonzero`

**Kapsam:** 158 trafo, 18,629 satır (%2.61), 137'si tam 122 gün isteniyor.
Alan küçük ama satır başına ceza büyük (gerçek 0 iken 1,000 tahmin → kareli log hatası ≈ 47).

Recon-3 madde 4, `q`'nun gerçek değerini ölçecek.

---

## 6. Haftanın günü — düşük yatırım

Artefakt doğrulandı: ham açıklık %36 → trafo-içi normalize edilince **%3.7**.
Gerçek etki küçük ama tutarlı: Pazar en düşük (0.977), Perşembe en yüksek (1.015).

**Karar:** `cal_dow` ve `cal_is_weekend` ver, geç. DOW üzerine feature mühendisliği yapma.

---

## 7. Bozuk yük faktörü

37 trafonun **36'sı test'te**. Yani bu trafolar için tahmin üretmek zorundayız.

- Eğitimde `LF > 1` olan satırlar **çıkarılır** (clip edilmez)
- Ama trafo tamamen atılmaz — diğer satırları eğitimde kalır
- `static_has_bad_rows` flag'i feature olarak verilir (ölçüm sorunlu trafo sinyali)

---

## 8. Feature grupları — revize öncelik

| Grup | Prefix | Öncelik | Not |
|---|---|---|---|
| Grup istatistikleri | `grp_` | **1** | Cold'un can damarı + mevsimsellik |
| Statik | `static_` | **1** | `guc`, `log_guc`, ilçe, il, bölge, `guc_bucket` |
| Takvim | `cal_` | 2 | Tatil, bayram, `days_to_holiday`, ay, doy sin/cos |
| Hava | `wx_` | 2 | CDD/HDD + nonlineer + ilçe etkileşimi |
| Seviye | `lvl_` | 3 | Sadece warm'da dolu; sıfır-streak dahil |
| Mevsimsel lag | `seas_` | 4 | Kapsam %35, düşük yatırım |

### Fiziksel çıpa (mimarinin temeli)
```
log1p(tuketim) ≈ log(guc × 24) + log(yük_faktörü)
```
`log(guc × 24)`'ü **offset / init_score** olarak vermeyi dene — model artık sadece
yük faktörünü öğrenir. Cold-start'ta görülmemiş `guc` kombinasyonlarına genelleme
bu formülasyonda belirgin şekilde daha iyi olur. Feature olarak vermekle karşılaştır.

---

## 9. Öncelik sırası (revize)

1. **Recon-3** — toplu giriş mi ramp mı, geçmiş dağılımı, `q`, ilçe oranı robustluğu
2. `config.py` + `data.py` + `validation.py` (geçmiş-uzunluğu eşlemeli) + `evaluate`
3. **Fold doğrulama kontrolü** (%22.2 / %35.0 / geçmiş medyanı)
4. Baseline'lar — özellikle `guc × 24 × LF[ilce, ay]` + uçtan uca submission
5. LightGBM + `grp_` + `static_` + `cal_`, geçmiş-dropout ile
6. Open-Meteo (~38 ilçe) + ilçe bazlı CDD etkileşimi ← **yaz rampası burada kazanılır**
7. `lvl_` + `seas_`
8. Hata analizi → ensemble

---

## 10. Jüri hikâyesi

Bu veri gerçek bir operasyonel problemi içeriyor: **şebekeye yeni katılan trafoların
tüketimini geçmişi olmadan tahmin etmek.** Yatırım planlaması ve kapasite tahsisi için
doğrudan değeri var.

Anlatılacaklar:
- Cold-start'ı ayrı ölçtük (warm/cold ayrı RMSLE) ve validasyonu test'in bilgi rejimine
  eşleyerek kurduk — geçmiş uzunluğu dağılımı eşlemesi
- Fiziksel çıpa: kurulu güç × yük faktörü; model aslında yük faktörünü öğreniyor
- Yaz rampası ilçeye göre 1.15×–5.0× arasında değişiyor — kentsel İzmir'de klima yükü baskın,
  iç Manisa'da neredeyse yok. Tek merkezi model bunu kaçırır
- Veri kalitesi bulguları: 37 trafoda ölçüm hatası, 158 trafoda uzun kesinti
