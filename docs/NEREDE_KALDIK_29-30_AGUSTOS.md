# Nerede Kaldık — 29-30 Ağustos

**Yarışma bitişi:** 1 Eylül · Bu not, `NEREDE_KALDIK_26-27_AGUSTOS.md`'nin devamıdır.

Bu iki günde **skor gerçekten iyileşti** (1.06483 → 1.05568) ve iyileşmenin sebebi
yeni bir model değil, **eski bir ölçüm hatasının bulunması** oldu. Aşağıda basit
dille anlatıyorum.

---

## 1. Kısa özet (bu iki günde ne oldu?)

- **Teslim notebook'u kuruldu.** Hava kullanmayan, kendi kendine yeterli, tek
  dosyada çalışan bir notebook. Host ilk 20'nin notebook'unu inceleyecek, buna
  hazırız artık.
- **Skor 3 kez kırıldı:** 1.06483 → 1.05764 → 1.05737 → **1.05568**
- **En değerli bulgu:** Tahminlerimiz cold trafolarda sistematik olarak **yüksekti**
  ama bunu aylardır göremiyorduk, çünkü warm'daki ters yönlü hata onu gizliyordu.
- **İki deney başarısız oldu** ve ikisi de öğretici: seviye kaydırma (benim hesap
  hatam) ve satır-bazlı sıfır düzeltmesi (fikir doğruydu, veri desteklemedi).

---

## 2. Teslim notebook'u — neden gerekliydi?

### Sorun
En iyi submission'ımız `sub_nowx_lo.csv`'ydi ama onu üreten **tek bir script yoktu**.
Bir script çalıştırılmış, sonra elle 0.2054 kaydırılmış, dosya öyle kaydedilmişti.
Host "notebook'unuzu çalıştıralım" dese, o dosya çıkmazdı.

### Yapılan
`notebooks/gridup_leakfree_submission.ipynb` kuruldu:
- **`src/` klasörünü hiç kullanmaz** — bütün kod içinde, Kaggle'a atınca çalışır
- Sadece 3 yarışma dosyasını okur (+ Türkiye tatil takvimi paketi)
- Son hücrede **sızıntı denetimi** var: açılan tüm dosyaları listeler, yarışma
  dışı bir şey okunmadığını otomatik kontrol eder
- İki kez baştan çalıştırıldı, **bit düzeyinde aynı** dosyayı üretti

Beraberinde `docs/VERI_KAYNAKLARI.md` yazıldı — host'a "hangi veriyi neden
kullandık/kullanmadık" diye anlatan belge.

**Ders:** Skoru üreten şey ile teslim edilen şey aynı olmalı. Aksi halde iyi skor
işe yaramaz.

---

## 3. Skor nasıl iyileşti? (üç adım)

### Adım 1: Notebook'un kendi çıktısı → 1.05764
Notebook'u yazınca çıktısını gönderdik. Beklenmedik şekilde eski en iyiden
(1.06483) **daha iyi** çıktı.

Sebebini araştırdık: kod aynı, özellikler aynı, parametreler aynı. Tek fark,
eğitim sırasında her trafoya rastgele atanan "geçmiş uzunluğu" (H) çekilişiydi.

**Ders (önemli):** Aynı modelin iki farklı kurulumu arasında **0.0076 skor farkı**
olabiliyor. Yani geçmişte "CV'de 0.003 iyileşti" diye alınan kararların bir kısmı
aslında gürültüymüş.

### Adım 2: H çekilişi topluluğu → 1.05737
Yukarıdaki gürültüyü söndürmek için 4 farklı H çekilişi × 3 tohum = **12 model**
eğitip ortalamasını aldık.

Kazanç: **0.0003.** Yani neredeyse hiç. Ama bir şey öğretti — bu da bir sonraki
adımın anahtarı oldu.

### Adım 3: Cold/warm ayrı kalibrasyon → 1.05568 ⭐
Bu iki günün en değerli bulgusu. Ayrı başlık hak ediyor, aşağıda.

---

## 4. ⭐ EN ÖNEMLİ BULGU: gizlenen kalibrasyon hatası

### Aylardır ne sanıyorduk?
"Tahminlerimizin genel seviyesi doğru." Bunu ölçmüştük de: ortalama sapma sadece
**+0.013** çıkıyordu. Yani model ne yüksek ne alçak, tam yerinde.

### Gerçek neydi?
Segmentleri **ayrı ayrı** ölçünce:

| segment | satır payı | sapma |
|---|---|---|
| **cold** (geçmişi olmayan trafo) | %22 | **+0.184** ← çok yüksek tahmin |
| warm (geçmişi olan trafo) | %78 | **−0.035** ← hafif düşük tahmin |
| **ağırlıklı ortalama** | %100 | **+0.013** ← "sorun yok" diyen sayı |

İki hata birbirini götürüyordu. Genel ortalamaya bakınca her şey yolunda
görünüyordu ama içeride cold'u 0.18 fazla tahmin ediyorduk.

### Nasıl bulundu?
Genel ortalamayı **sabit tutup** sadece dağılımı değiştiren bir deney kurduk:
cold'u biraz aşağı, warm'ı bunu dengeleyecek kadar yukarı. Genel ortalama hiç
değişmedi, sadece iki segment arasındaki paylaşım değişti.

İki deneme gönderdik, sonuçlar bir parabol denklemini tam olarak çözdü ve optimal
ayarı verdi.

### Kök neden — matematik hatası
Anchor (fiziksel çapa) dediğimiz başlangıç tahmininde sıfır düzeltmesi şöyleydi:

```
anchor = L + log(1-p)          <- bizim kullandığımız
```

Doğrusu ise:

```
anchor = (1-p) * L             <- RMSLE için doğru olan
```

(`p` = trafonun sıfır tüketim yapma olasılığı, `L` = sıfır olmadığındaki seviye)

`log(1-p)` **ham ölçekte ortalamayı** düzeltmek için doğrudur. Ama yarışma metriği
RMSLE **log ölçekte** çalışıyor, orada düzeltme çarpımsal olmalı. Fark p büyüdükçe
patlıyor: p=%6'da +0.38, p=%20'de +1.26 log fazla tahmin.

**Ders:** Metriğin hangi ölçekte çalıştığına dikkat. Log ölçekli bir metrikte ham
ölçek formülü kullanmak sessizce sistematik hata üretiyor — ve toplamda görünmüyor.

---

## 5. Başarısız deney 1: seviye kaydırma (benim hesap hatam)

### Ne yaptım?
İki submission'ın skoruna bakıp "tahminlerimiz 0.31 log yüksek" diye hesapladım ve
0.30 aşağı kaydırıp göndermeyi önerdim. Öngörüm 1.012 idi.

### Sonuç
**1.09545 — belirgin şekilde kötü.**

### Hata neredeydi?
Karşılaştırdığım iki dosyanın "sadece seviyede farklı" olduğunu varsaymıştım.
Değillermiş — farklı H çekilişi, farklı kategori kodlaması, farklı kod yolu
taşıyorlardı. Yani seviye farkı sandığım şey aslında **model farkıydı.**

Aynı modelden iki nokta alınca gerçek cevap çıktı: ortalama sapma sadece **+0.013**.

**Ders:** Bir etkiyi ölçmek için **tek değişken** değişmeli. Farklı modellerin skor
farkını seviyeye bağlamak yanlış. Sonraki bütün deneyler bu kurala göre kuruldu ve
hepsi düzgün sonuç verdi.

---

## 6. Başarısız deney 2: satır-bazlı sıfır düzeltmesi

### Fikir
Madde 4'teki matematik hatası p ile büyüyor. O halde her satıra kendi p'sine göre
farklı düzeltme uygularsak, sıfır olma ihtimali yüksek satırlar çok daha aşağı
iner ve büyük kazanç geliriz. Hatanın **%56'sı** tam olarak "cold + gerçek sıfır"
satırlarında (verinin sadece %1.6'sı).

### Deney
Segment ortalamaları birebir aynı tutuldu, sadece satır-bazlı yapı eklendi. Yani
tek fark "hangi satır ne kadar iniyor".

### Sonuç
**1.06374 — kötüleşti.**

Hesapladık: düzeltme ile gerçek hata arasındaki korelasyon **0.005**. Sıfır.
Başa-baş için 0.08 gerekiyordu.

### Bu ne demek?
İlçe/ay bazlı sıfır oranları, **hangi cold trafonun sıfır olduğu** hakkında hiçbir
şey söylemiyor. Mantıklı: sıfırlık trafoya özgü bir şey, cold trafonun ise geçmişi
yok. Bilinemez.

**Ders:** Matematiği doğru olan bir fikir, veri o bilgiyi taşımıyorsa işe yaramaz.
Anchor hatası sadece **ortalamada** gerçekti; ortalamayı da zaten düzelttik.

---

## 7. Lider neden bizden iyi? Biz neyi kaçırdık?

Lider **0.99046**, biz **1.05568**. Bu soruyu ölçerek cevapladık.

### Önce: 0.99 sihirli bir sayı değil
"Geleceği bilen" kâhin tahminciler kurup ulaşılabilir tabanı hesapladık:

| tahminci | RMSLE |
|---|---|
| Kâhin: her trafonun kendi aylık ortalamasını bilse | 0.404 |
| Cold tavanı: sadece grup ortalaması (geçmiş yok) | 1.477 |
| **Test kompozisyonuyla ulaşılabilir taban** | **0.782** |
| Lider | 0.990 |
| Biz | 1.056 |

Yani lider imkânsız bir şey yapmıyor. Forward-leak şart değil, temiz veriyle
0.99 ulaşılabilir bir skor.

### Hata nerede birikiyor?

| kesim | satır payı | RMSLE | **hata payı** |
|---|---|---|---|
| **cold + gerçek sıfır** | %1.6 | 6.66 | **%56** |
| cold + pozitif | %20.6 | 1.08 | %19 |
| warm + pozitif | %75.0 | 0.54 | %18 |
| warm + sıfır | %2.8 | 1.80 | %7 |

Hatanın yarısından fazlası, satırların %1.6'sında.

### Bizim üç eksiğimiz

1. **Emek yanlış yere gitti.** Aylardır yapılan feature/model çalışmasının neredeyse
   tamamı **warm**'ı iyileştirdi. Warm hatanın %18'i. Cold'da ise trivial baseline'ı
   (`guc × 24 × yük faktörü`) sadece **%1.7** geçiyoruz.

2. **Kalibrasyon hatası aylarca gizlendi** (madde 4). Segment ayrımı yapılmadığı
   için "seviye doğru" sanıldı.

3. **Cold için trafo-bazlı bilgi bulamadık.** Denenen ve ölçülen her şey boş çıktı:
   `tanim` ID komşuluğu (korelasyon 0.094, ilçe zaten 0.179 veriyor), rampa etkisi
   (yok), hava durumu (katkı ~0), satır-bazlı sıfır yapısı (korelasyon 0.005).

**Dürüst değerlendirme:** Lider muhtemelen cold trafolar için bizim bulamadığımız
bir sinyal buldu ya da sıfır problemini yapısal olarak daha iyi çözdü. Ama ne
olduğunu bilmiyoruz — tahmin yürütmüyorum.

---

## 8. Kapanan kaldıraçlar (hepsi ölçüldü, hiçbiri varsayım değil)

| kaldıraç | ölçülen değer | durum |
|---|---|---|
| **Cold/warm ayrı kalibrasyon** | cold +0.184 / warm −0.035 | ✅ **kazandırdı** |
| Global seviye kaydırma | ortalama sapma +0.013 | ❌ kapalı |
| Satır-bazlı sıfır yapısı | korelasyon 0.005 | ❌ sinyal yok |
| H çekilişi topluluğu | 0.0003 | ❌ ihmal edilebilir |
| `tanim` ID komşuluğu | 0.094 (ilçe 0.179) | ❌ ilçenin altında |
| Yeni trafo rampası | yok (yaş 1-7 gün: −0.019) | ❌ mevcut değil |
| Hava durumu | katkı ~0 | ❌ zaten çıkarıldı |
| Hurdle sıfır matematiği | `expm1((1-p)·L)` — doğruymuş | ❌ hata yok |

---

## 9. Sonraki adımlar

### Yarın ilk iş
**`submissions/sub_sp17.csv` gönder** — öngörü **1.05343** (mevcut en iyi 1.05568).
Parabol iki ölçülmüş noktadan tam oturduğu için bu neredeyse kesin bir kazanç.

### Kapatılması gereken açık ⚠️
Notebook şu an `sub_hens_lo`'yu (1.05737) üretiyor, **segment kaydırmasını
içermiyor**. `sub_sp17`'yi final seçersek notebook onu üretmiyor olacak — yani
madde 2'de kapattığımız açık yeniden açılır.

Yapılacak: segment kaydırmasını notebook'a taşı, `docs/VERI_KAYNAKLARI.md`'ye
**ikinci LB-kalibreli parametre** olarak dürüstçe yaz (artık 1 değil 2 parametre
public leaderboard'dan geliyor).

### Denenebilecek son fikir
Cold'u ikiye ayır: **11 Mayıs toplu girişi** (1.326 trafo) vs diğer cold'lar. İdari
olarak farklı popülasyonlar, ayrı sapmaları olabilir. Aynı ortalama-koruyan yöntem.
Beklenen kazanç ~0.002 — sıçrama değil.

### Final stratejisi (1 Eylül)
Kaggle 2 submission final seçtiriyor:
- **1. final:** en iyi leak'siz skor (`sub_sp17` ya da o gün en iyisi)
- **2. final:** notebook'un birebir ürettiği dosya (teslim edilebilirlik garantisi)

**Genel durum:** ~1.053'te bitiyoruz. İlk 20 (1.015) skorla ulaşılamıyor. Kalan
gerçek şans hâlâ aynı: private'da forward-leak kullananların elenmesi — ve o
senaryoya tam hazırız (leak'siz, kendi kendine yeterli, deterministik,
denetlenebilir notebook + şeffaflık belgesi).

---

## Hatırlatma

- En iyi skor: **1.05568** (`submissions/sub_sp30.csv`)
- Yarın gönderilecek: `submissions/sub_sp17.csv` (öngörü 1.05343)
- Teslim notebook'u: `notebooks/gridup_leakfree_submission.ipynb`
- Veri beyanı: `docs/VERI_KAYNAKLARI.md`
- Deney raporları: `reports/hdraw_ensemble.md`, `reports/zero_anchor.md`
- Bir önceki not: `docs/NEREDE_KALDIK_26-27_AGUSTOS.md`
- Tam bağlam: `docs/PROJE_HANDOFF.md`

**Yarın ilk iş:** Bu notu oku → `sub_sp17`'yi gönder → notebook'a segment
kaydırmasını taşı → finalleri seç.
