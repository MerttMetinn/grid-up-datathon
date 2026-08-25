# Nerede Kaldık — Çalışma Notu

**Son güncelleme:** 25 Ağustos 2026 · Yarışma bitişi: 1 Eylül (1 hafta var)

Bu not, günlerce çalıştıktan sonra "nerede kaldık, ne öğrendik, yarın ne yapacağız"
sorularının basit cevabıdır. Teknik detay değil, **karar özeti.**

---

## 1. Şu an neredeyiz? (Kaggle skorları — düşük = iyi)

| Model | Kaggle skoru | Not |
|---|---|---|
| Basit baseline (b6) | 1.36728 | başlangıç noktamız |
| Bizim hurdle modeli | 1.08943 | ölü-trafo ayrımı |
| Bizim hava modeli | 1.08974 | hava durumu ekli |
| **Arkadaşın optuna modeli** | **1.06483** | **EN İYİ skorumuz** |
| Bizim son deneme (sub_final) | 1.07941 | ❌ işe yaramadı |
| **Yarışma lideri** | **1.00635** | hedef |

**Özet:** Başlangıçtan (1.37) buraya (1.065) çok yol kat ettik. Ama son birkaç gündür
**1.065 civarında sıkıştık** (buna "plato" diyoruz — düz duvar). Lidere fark: 0.06.

---

## 2. Bugün ne öğrendik? (en değerli kısım)

### Ders 1: En basit model kazanıyor 🏆
Biz günlerce karmaşık şeyler ekledik (ölü-trafo modeli, hava durumu, CatBoost, Tweedie,
seviye kaydırma). **Hiçbiri arkadaşın basit optuna modelini geçemedi.** Arkadaşın modeli:
düz LightGBM + iyi seçilmiş 29 özellik + iyi ayarlanmış parametreler. Karmaşıklık burada
işe yaramadı çünkü model zaten verideki sinyalin çoğunu yakalamış.

### Ders 2: Kendi testimiz (CV) bizi cold trafolarda yanılttı ⚠️
En önemli ders bu. "Cross-validation" (CV) = Kaggle'a yüklemeden önce kendi kendimize
yaptığımız deneme sınavı. **Bu deneme sınavı cold (yeni) trafolarda gerçeği yansıtmıyor.**
- Bizim taklit ettiğimiz cold trafo: eski bir trafonun geçmişini sildik
- Gerçek cold trafo: Mayıs'ta yeni kurulan fiziksel trafo
- Bunlar farklı davranıyor. Bu yüzden CV'de "iyi" görünen cold ayarları Kaggle'da kötü çıktı.
- **Sonuç: Cold trafolarla ilgili kararları artık sadece Kaggle skoruyla veriyoruz, kendi
  testimizle değil.**

### Ders 3: Tahmini yukarı/aşağı kaydırmak artık işe yaramıyor
Bir ara "tahminleri biraz düşürürsek skor iyileşir mi?" diye denedik. Küçük iyileşme
umuyorduk ama **skor neredeyse hiç değişmedi** (1.06486 vs 1.06483). Demek ki optuna modeli
zaten doğru seviyede tahmin ediyor. Bu yolu kapattık.

### Ders 4: Farklı algoritma da çeşitlilik getirmedi
"Farklı bir algoritma (CatBoost, Tweedie) eklersek, birbirini tamamlar mı?" diye denedik.
Ama hepsi **birbirine çok benzer tahminler** yaptı (%97-99 aynı). Çünkü hepsi aynı özellikleri
ve aynı fiziksel formülü kullanıyor. Birbirini tamamlamadılar.

### Ders 5: Hatanın kaynağını bulduk — ama çözmesi zor
Hata analizinde gördük ki **toplam hatamızın %56'sı**, verilerin sadece %1.6'sını oluşturan
"**cold + ölü trafo**" satırlarından geliyor. Yani:
- Yeni kurulmuş bir trafo (geçmişi yok)
- Ve o gün gerçekte sıfır elektrik çekiyor (kapalı/ölü)
- Model onu "ortalama" tahmin ediyor → büyük hata

**Sorun:** Bu trafonun geçmişi olmadığı için ölü olduğunu önceden bilmek neredeyse imkânsız.
Bu muhtemelen **herkes için bir tavan** — lider bile bu satırları tam çözemez.

### Ders 6: `tanim` (trafo kimliği) işe yaramaz
Trafo ID'lerinde (`70122340` gibi) gizli bir anlam var mı diye baktık. Yok — sadece rastgele
numaralar. Bu yolu da eledik.

---

## 3. Ne denedik, ne oldu? (özet tablo)

| Deneme | Sonuç |
|---|---|
| Seviye kaydırma (tahminleri düşür) | ❌ değişmedi |
| CatBoost (farklı algoritma) | ❌ çok benzer, çeşitlilik yok |
| Tweedie (farklı kayıp fonksiyonu) | ❌ genel kötü (sadece yaz fold'unda iyi) |
| Cold b5 ayarı (W=1.0) | ❌ CV'de iyi, Kaggle'da kötü |
| `tanim` pattern analizi | ❌ anlamsız ID |

---

## 4. Yarın nereden devam edelim? (sonraki adımlar)

**Genel gerçek:** Artık büyük sıçrama değil, **küçük kazançlar + risk yönetimi** oyunundayız.
Final sıralama test verisinin **%70'lik gizli kısmıyla** (private) belirlenecek. Şu an gördüğümüz
skor sadece %30'luk (public) kısım. Yani "public'e aşırı uyan" bir model private'da çökebilir.

### Öncelik sırası:

**A) optuna modelini derinleştir** (en umutlu somut adım)
- Arkadaş 29 özellik + 25 deneme (trial) kullandı
- Denenecek: tüm 75 özellik + 100 deneme ile daha uzun optuna
- Beklenti: belki −0.005 (küçük ama gerçek)

**B) Güvenli ensemble (private için sigorta)**
- optuna (ana model) + Tweedie (yaz'da farklı davranıyor) → %80/%20 karışım
- Amaç skoru çok düşürmek değil, private'da **sağlam** kalmak (tek modele bağımlı olmamak)

**C) Recency ağırlığı** (denenebilir)
- Son aylara daha çok ağırlık ver — 2026 tüketimi 2025'ten farklı olduğu için

### Final günü stratejisi (1 Eylül'e doğru):
- En iyi 2 submission'ı "final" olarak işaretle (Kaggle genelde 2 final seçtirir)
- **Bir tanesi:** en yüksek public skor (optuna, 1.0648)
- **Bir tanesi:** en sağlam/çeşitli (ensemble) — private güvencesi
- Böylece public overfit riskine karşı korunmuş oluruz

---

## Hatırlatma: elimizde ne var?
- **En iyi submission:** `submissions/sub_s2_optuna.csv` (Kaggle 1.0648)
- Tüm modeller `models/` klasöründe kayıtlı
- Optuna kodu ve sonuçları `sena` branch'inde (arkadaşın çalışması)
- Tüm kararların detayı: `docs/DECISIONS.md`

**Yarın ilk iş:** Bu notu oku, sonra A maddesinden (optuna derinleştirme) başla.
