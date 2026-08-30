# Nerede Kaldık — 26-27 Ağustos

**Yarışma bitişi:** 1 Eylül · Bu not, bir önceki `NEREDE_KALDIK.md`'nin devamıdır.

Bu iki günün en önemli olayı bir **skor gelişmesi değil, bir KURAL sorunu** oldu.
Aşağıda basit dille anlatıyorum.

---

## 1. Kısa özet (bu iki günde ne oldu?)

- optuna modelini derinleştirmeyi denedik → **işe yaramadı**
- Farklı modelleri karıştırmayı (ensemble) denedik → **işe yaramadı**
- **EN ÖNEMLİSİ:** Yarışma sahipleri "gerçekleşmiş hava durumu kullanmak uygun değil"
  dedi. Bizim en iyi modelimiz hava durumu kullanıyor → **risk altında.** Bu yüzden
  hava kullanmayan "güvenli" bir model kurmaya başladık.

---

## 2. optuna derinleştirme — neden işe yaramadı?

Arkadaşın en iyi modeli: **29 özellik + 25 deneme (trial)** → Kaggle 1.0648.
Biz denedik: **75 özellik (hava dahil) + 60 deneme.**

Sonuç: neredeyse **aynı** (bizimki 1.1079, arkadaşınki 1.1110 — kendi test sınavımızda).
Yani fazladan 46 özellik hiçbir şey katmadı. **Arkadaşın 29 özellik seçimi zaten en iyisiymiş.**

**Ders:** Daha çok özellik / daha çok deneme = daha iyi model DEĞİL. Model zaten
verideki sinyali yakalamış.

---

## 3. Ensemble (model karıştırma) — neden işe yaramadı?

"Birden fazla modeli ortalarsak birbirini tamamlar mı?" diye denedik (optuna + Tweedie
+ CatBoost). Ama:
- Modeller **birbirine çok benziyor** (%97-99 aynı tahmin)
- Karıştırınca tahmin seviyesi **bozuluyor** (yukarı kayıyor, bu da kötü)

**Ders:** Bizim modeller farklı bilgi görmüyor, aynı şeyi tekrar ediyor. Karıştırmanın
faydası yok.

---

## 4. ⚠️ EN KRİTİK KONU: Hava durumu = "forward leak" (ileriye bakma)

### Sorun ne?
Biz Nisan-Temmuz 2026 dönemini tahmin ederken, o dönemin **gerçekleşmiş (olmuş bitmiş)
hava durumunu** internetten (Open-Meteo arşivi) çekip kullandık.

**Neden sorun:** Gerçek hayatta Nisan 2026'yı tahmin ederken, o günün gerçek sıcaklığını
bilemezsin — henüz yaşanmadı. Biz Ağustos'ta olduğumuz için "geleceğe baktık". Buna
**forward leak** (ileriye sızma) deniyor.

### Yarışma sahipleri ne dedi?
- Önce: "tüm açık kaynak verileri kullanabilirsiniz"
- Sonra: "**gerçekleşmiş hava durumu verisi kullanmak uygun olmayacaktır**"
- Şimdi: "ilk 20 takımın notebook'u incelenecek, hangi veriyi nasıl kullandığınız
  kontrol edilecek"
- Net "diskalifiye edilir mi?" sorusuna **kesin cevap vermediler.**

### Bu bizi nasıl etkiliyor?
- En iyi skorumuz **optuna (1.0648) hava durumu içeriyor** (6 tane hava özelliği).
- Eğer hava durumu yasaksa, bu model notebook incelemesinde **elenebilir.**

### Şüphe: lider forward-leak kullanıyor olabilir
Lider şu an **0.994** skorunda. Ama "cold + ölü trafo" satırları (hatanın %56'sı) neredeyse
tahmin edilemez — bu kadar düşük skor şüpheli. Muhtemelen lider de gerçekleşmiş hava veya
EPİAŞ gibi ileriye-bakan veri kullanıyor. **Eğer sahipler bunları elerse, temiz (hava
kullanmayan) modeller öne çıkar.**

---

## 5. Şu an ne yapıyoruz?

**Hava durumu OLMADAN** en iyi modeli kuruyoruz (`sub_nowx.csv`). Bu:
- Diskalifiye riski **sıfır** (sadece geçmiş tüketim + takvim + grup istatistikleri kullanır)
- Notebook'ta şeffafça açıklanabilir: "dış hava verisi kullanmadık"
- Eğitim şu an arka planda çalışıyor (script: `27_optuna_nowx.py`)

Bittiğinde göreceğiz: hava durumunu çıkarmanın skora maliyeti ne kadar?

---

## 6. Sonraki adımlar / Final stratejisi

### Önce (senin yapman gerekenler):
1. **Yarışma sahiplerinden NET cevap iste:** "Gerçekleşmiş hava kullanan diskalifiye
   edilir mi? Private sıralamada bu nasıl değerlendirilecek?" Bu belirsizlik en büyük risk.
2. **Notebook'ta veri kaynaklarını açıkça yaz** (sahipler bunu istiyor).

### Final günü (1 Eylül) planı:
Kaggle genelde **2 submission** final seçtirir. İkisini şöyle seçelim:
- **1. final:** `sub_nowx` (hava YOK) → **güvenli**, diskalifiye riski sıfır
- **2. final:** wx'li optuna (1.0648) → eğer hava serbest kalırsa avantaj

Böylece iki senaryoya da hazırız.

### Genel durum:
- En iyi skorumuz: **1.0648** (ama hava içeriyor = riskli)
- Hava'sız güvenli skor: `sub_nowx` bitince belli olacak
- Lidere fark kapanmıyor (plato ~1.065) — ama forward-leak elenirse durum değişebilir

---

## Hatırlatma
- En iyi (riskli) model: `submissions/sub_s2_optuna.csv`
- Güvenli model: `submissions/sub_nowx.csv` (hazırlanıyor)
- Tüm kararlar: `docs/DECISIONS.md`
- Bir önceki not: `docs/NEREDE_KALDIK.md`

**Yarın ilk iş:** Bu notu oku → `sub_nowx` sonucunu kontrol et → sahiplerden hava
konusunda net cevap geldiyse ona göre final seç.
