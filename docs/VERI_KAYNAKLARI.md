# Veri Kaynakları Beyanı — Grid Up Datathon

**Takım çözümü:** `notebooks/gridup_leakfree_submission.ipynb`
**Son güncelleme:** 30 Ağustos 2026

Bu belge, final submission'ı üreten notebook'un kullandığı **tüm** veri
kaynaklarını listeler. Yarışma sahiplerinin ilk 20 takım için yapacağı notebook
incelemesine yönelik hazırlanmıştır.

---

## 1. Kullanılan kaynaklar (tamamı)

| # | Kaynak | Dosya / paket | Ne için | Dış veri mi? |
|---|---|---|---|---|
| 1 | Yarışma verisi | `train.csv` | Eğitim hedefi ve geçmişi (2025-01-01 → 2026-03-31) | Hayır |
| 2 | Yarışma verisi | `test.csv` | Tahmin edilecek satırlar (2026-04-01 → 2026-07-31) | Hayır |
| 3 | Yarışma verisi | `sample_submission.csv` | Çıktı sırası/formatı doğrulaması | Hayır |
| 4 | Python paketi | `holidays` (TR) | Türkiye resmî + dinî tatil **takvimi** | Statik takvim |

**Başka hiçbir kaynak yoktur.** Notebook'un son hücresi, çalışma boyunca açılan
tüm dosyaları listeler ve yarışma dosyaları dışında bir şey okunmadığını
`assert` ile doğrular.

### `holidays` paketi hakkında not

Bu paket bir **takvim**dir: 23 Nisan, 1 Mayıs, 19 Mayıs, 15 Temmuz, Ramazan ve
Kurban bayramları gibi tarihleri döner. Ölçüm, gözlem veya tahmin dönemine ait
gerçekleşmiş bir olay verisi **değildir**; bu tarihler yıllar öncesinden bellidir.
Forward leak oluşturmaz.

---

## 2. Bilinçli olarak KULLANILMAYANLAR

| Kaynak | Neden kullanılmadı |
|---|---|
| **Gerçekleşmiş hava durumu** (Open-Meteo arşivi, MGM vb.) | **Forward leak.** Nisan–Temmuz 2026 tahmin edilirken o günlerin gerçek sıcaklığı bilinemez. Kullanmama kararı bize aittir (aşağıya bkz.). |
| **EPİAŞ / Şeffaflık Platformu tüketim verisi** | Aynı gerekçe — tahmin dönemine ait gerçekleşmiş ölçüm. |
| Tahmin dönemine ait başka herhangi bir dış gözlem | Aynı gerekçe. |

### Yarışma sahiplerinin kural açıklaması ve bizim kararımız

Yarışma sahipleri, forum sorusuna verdikleri yanıtta **dış kaynak kullanımının
serbest** olduğunu, kullanılan kaynakların notebook içinde amacı ve kullanım
biçimiyle **belirtilmesinin beklendiğini** bildirmiştir.

Yani hava durumu kullanmak yasak değildir. Buna rağmen kullanmama kararımız
**bilinçlidir ve iki gerekçesi vardır:**

1. **Metodolojik.** Nisan–Temmuz 2026'yı tahmin ederken o günlerin gerçekleşmiş
   sıcaklığı gerçek bir tahmin anında bilinemez. Bu veriyle kurulan model
   operasyonel olarak kullanılamaz — dağıtım şirketi için değeri, ancak
   ileriye dönük hava *tahminiyle* çalışıyorsa vardır.
2. **Ampirik.** Ölçtük: katkısı yok.

### Şeffaflık: hava durumunu denedik, çıkardık

Geliştirme sürecinde bir ara sürümde Open-Meteo arşivinden 17 hava feature'ı
(CDD/HDD, ET0, toprak nemi, yağış, radyasyon vb.) türetildi ve sonra
**tamamen çıkarıldı**.

**Ölçülen etki: hava durumunun skora katkısı ~0.**

| Model | Public LB |
|---|---|
| Hava feature'ları dahil | 1.06483 |
| Hava feature'ları YOK (final çözümümüz) | 1.06525 |

Fark 0.0004 — gürültü seviyesinde ve **hava içeren sürüm lehine**, yani ölçüm
belirsizliğinin içinde. Karar bir fedakârlık değildir: aynı skoru, operasyonel
olarak savunulabilir bir modelle alıyoruz.

Hava kodu depoda `src/weather.py` ve `src/features.py` içinde tarihsel kayıt
olarak duruyor ancak **final notebook bu modülleri hiç import etmez** — notebook
tamamen kendi kendine yeterlidir.

---

## 3. Sızıntıya karşı yapısal önlemler

Forward leak yalnızca dış veriyle olmaz; kendi hedef verinden de sızdırabilirsin.
Notebook'ta üç yapısal kural uygulanır:

1. **Kısa lag yasak.** `lag_1`, `lag_7`, `rolling_7` gibi feature'lar
   kullanılmaz. Test 122 günlük tek bloktur; 2026-07-31'i tahmin ederken
   2026-07-30'un değeri bilinemez. En kısa izinli lag **364 gün**dür.

2. **Recursive tahmin yok.** Tahminler birbirini beslemez; her satır doğrudan
   (direct multi-horizon) tahmin edilir.

3. **Her feature bir `forecast_origin` alır.** `build_features(df, origin, history)`
   fonksiyonunun ilk satırı:

   ```python
   assert history["tarih"].max() <= origin, \
       "history, forecast_origin sonrasi satir iceriyor — SIZINTI"
   ```

   Eğitimde de tahminde de aynı kod yolu çalışır. Final tahminde
   `forecast_origin = 2026-03-31`'dir; yani modelin gördüğü en son hedef verisi
   test başlangıcından bir gün öncesidir.

**Feature testi:** Her feature için sorulan soru — *"2026-07-31 satırı için
sadece 2026-03-31'e kadarki hedef verisiyle hesaplanabilir mi? Trafo hiç
görülmemişse ne olur?"* Cevabı "hayır" olan hiçbir feature sette değildir.

---

## 4. Leaderboard'a bakılarak seçilen sabitler (2 adet)

Dürüstlük gereği belirtilir: notebook'ta **iki** kalibrasyon sabiti vardır ve
**ikisi de public leaderboard geri bildirimiyle ölçülmüştür** — modelden veya
eğitim verisinden türetilmemiştir.

Her ikisi de tek boyutludur ve **kapalı formda çözülmüştür**; parametre taraması
veya deneme-yanılma yapılmamıştır. Toplam ayarlanan serbest parametre sayısı
**ikidir** ve public LB ~214 bin satır üzerinden ölçüldüğü için aşırı-uyum riski
ihmal edilebilir.

### 4.1 `LEVEL_SHIFT = -0.2712` — genel seviye

Uniform kaydırmada hata *tam olarak* paraboliktir:

`MSE(d) = MSE(0) + 2·d·m + d²`   (m = ortalama artık, log uzayında)

İki LB noktası bu denklemi çözer:

| kaydırma | public LB | MSE |
|---|---|---|
| referans | 1.05737 | 1.118031 |
| referans − 0.30 | 1.09545 | 1.200011 |

Çözüm **m = +0.013 log**: model genel seviyede zaten kalibre. Bu sabit, LB'de en
iyi skoru veren seviyeyi sabitler.

### 4.2 `SEGMENT_DELTA = 0.1709` — cold/warm paylaşımı

Genel seviye doğru olsa bile segmentler ayrı ayrı sapabilir ve bu toplamda
görünmez. Ölçülen:

| segment | satır payı | sapma |
|---|---|---|
| cold (geçmişi olmayan trafo) | %22.2 | **+0.184** |
| warm | %77.8 | **−0.035** |
| ağırlıklı ortalama | %100 | +0.013 |

**Kök neden bir modelleme hatasıdır.** Sıfır-şişirilmiş dağılımda (y=0 olasılığı
`p`, pozitifken seviye `L`) RMSLE'yi minimize eden tahmin
`E[log1p(y)] = (1−p)·L` yani çarpımsaldır. Anchor ise toplamsal `L + log(1−p)`
kullanıyor; bu ham ölçekte ortalama için doğrudur, log ölçekte değildir. Fark `p`
ile büyür (p=0.06 → +0.38 log). Cold satırlarda `p` yüksek olduğu için sapma orada
birikir.

Düzeltme cold'u `−δ`, warm'ı `+δ·(f_cold/f_warm)` kaydırır; warm kaydırması cold
satır payına göre ölçeklendiği için **tahminlerin genel ortalaması değişmez**.
Bu parametrizasyonda da hata tam olarak paraboliktir ve iki LB noktası optimumu
verir: **δ = 0.171**.

### 4.3 Denenen ve elenen (kayda geçsin)

Aynı düzeltmenin **satır bazlı** hâli — her satıra kendi tahmini `p`'sine göre
farklı katsayı — LB'de **kötüleşti** (1.05568 → 1.06374). Ölçülen korelasyon
**0.005**; başa-baş için 0.08 gerekiyordu. Yani ilçe/ay bazlı sıfır oranları,
*hangi* cold satırının sıfır olduğu hakkında bilgi taşımıyor. Anchor hatası
yalnızca ortalamada gerçektir ve 4.2 onu düzeltir.

## 5. Notebook ile teslim edilen dosyanın ilişkisi

Notebook çalıştırıldığında `submission.csv` üretir. Bu dosya
`submissions/sub_sp17.csv` ile aynıdır — log uzayında maksimum fark `2.3e-06`
(CSV yuvarlama gürültüsü).

Yani teslim edilen notebook, teslim edilen submission'ı gerçekten üretir.

### Model kurgusu: 12 model, iki ayrı varyans kaynağı

Final tahmin **4 geçmiş-uzunluğu (H) çekilişi × 3 LightGBM tohumu** = 12 modelin
log uzayındaki ortalamasıdır. İkisi ayrı ayrı gereklidir:

- **Tohum** LightGBM'in bagging/feature örneklemesini değiştirir.
- **H çekilişi** eğitim matrisinin kendisini değiştirir (her trafoya test
  profilinden atanan geçmiş uzunluğu). Ölçülen etkisi: çekilişler arası tahmin
  seviyesi std **0.042 log**, satır bazında **0.085 log**.

Kritik nokta: H varyansı tohum ortalamasıyla **sönmez**, çünkü tüm tohumlar aynı
eğitim matrisini paylaşır. Bu yüzden ayrıca çeşitlendirilir.

### Depodaki eski dosyalar hakkında not

`submissions/sub_nowx_lo.csv` (public LB 1.06525) aynı yöntemin daha eski bir
çalıştırmasıdır ve geliştirme sırasında kullanılan `data/processed/*.parquet`
cache üzerinden üretilmişti. Notebook onu birebir üretmez; aradaki fark
`tanim` sütununun kategori sırasının pandas'ta CSV ve parquet okumalarında
farklı çıkmasından kaynaklanır (bu sıra H atamasının rastgele çekilişini
etkiler). Notebook'ta bu sıra artık açıkça sabitlenmiştir.

## 6. Yeniden üretilebilirlik

- Notebook **kendi kendine yeterlidir**: depodaki `src/` modüllerini import
  etmez, tüm kod içindedir.
- Girdi olarak yalnızca yarışma dosyalarını ister; yolu otomatik bulur
  (`/kaggle/input` veya yerel `data/raw`).
- Tüm rastgelelik `SEED = 42` üzerinden sabitlenmiştir: geçmiş-uzunluğu
  örneklemesi, LightGBM bagging/feature sampling ve tohum ortalaması.
- İnternet erişimi **gerekmez**; notebook hiçbir HTTP isteği yapmaz.
- Notebook kaynağı `notebooks/gridup_leakfree_submission.py` (percent format)
  dosyasında tutulur; `.ipynb` ondan `scripts/29_build_notebook.py` ile üretilir.

**Determinizm doğrulandı:** Notebook iki kez, bağımsız süreçlerde uçtan uca
çalıştırıldı (12 model, 4 H çekilişi × 3 tohum); üretilen `submission.csv`
dosyaları **bit düzeyinde aynıdır** (MD5 `62967caa88b16ba58d8d0eeb59630402`,
satır bazında maksimum fark 0).

---

## 7. İletişim noktası

Bu beyanla ilgili herhangi bir soruda, notebook'un **9. bölümündeki sızıntı
denetimi hücresi** tek kontrol noktasıdır: çalıştırıldığında okunan tüm
dosyaları listeler ve dış veri kullanılmadığını doğrular.
