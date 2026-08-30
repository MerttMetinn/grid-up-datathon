# H-çekilişi topluluğu — eğitim matrisi varyansının söndürülmesi

Üretim: `scripts/30_hdraw_ensemble.py` · 2026-08-29 14:07
- 4 H çekilişi × 3 tohum = 12 model
- Taban: sub_notebook.csv (tek çekiliş, 3 tohum) LB **1.05764**

## 1. Çekilişler arası yayılım (kaydırma öncesi, log1p ortalaması)

| çekiliş | fold_i | ortalama log1p |
|---|---|---|
| 1 | 9 | 6.8020 |
| 2 | 19 | 6.8998 |
| 3 | 29 | 6.9091 |
| 4 | 39 | 6.8577 |

- Çekilişler arası ortalama-seviye std: **0.0423** log
- Satır bazında çekilişler arası std (medyan): **0.0850** log

Bu yayılım tek çekilişli modelde tamamen tahmine geçiyor; topluluk onu söndürür.

## 2. sub_hens.csv

- yazıldı · 714,688 satır
- sub_notebook'a göre: ortalama fark **+0.0658** · MAE 0.1075 · korelasyon 0.99682

| ay | sub_notebook | sub_hens |
|---|---|---|
| 2026-04 | 6.2032 | 6.2525 |
| 2026-05 | 6.3018 | 6.3521 |
| 2026-06 | 6.6597 | 6.7173 |
| 2026-07 | 7.0185 | 7.1149 |

**Beklenti:** varyans azaltma; seviye neredeyse aynı kalmalı, satır bazında gürültü düşmeli. LB ile doğrulanmalı.

---

## 3. CONFOUND ve düzeltilmiş deney tasarımı

`sub_notebook`'un kullandığı çekiliş (fold_i=9) dört çekilişin **en düşük
seviyelisi** (6.8020 vs 6.86–6.91). Dolayısıyla `sub_hens` ham haliyle
`sub_notebook`'tan 0.0658 log **yukarıda** kalıyor.

Bu bir confound: LB'de düşük seviye kazanıyordu (sub_notebook 1.05764 <
sub_nowx_lo 1.06525, aralarında 0.027 seviye farkı vardı). `sub_hens`'i ham
gönderirsek varyans azaltmanın kazancı ile seviye yükselmesinin zararı
birbirine karışır ve sonuç yorumlanamaz.

**Ayrıştırılmış iki dosya üretildi:**

| dosya | ort log1p | ne ölçer |
|---|---|---|
| `sub_notebook.csv` | 6.5991 | taban — LB 1.05764 |
| `sub_hens_lo.csv` | 6.6000 | **saf varyans azaltma** (seviye tabanla aynı) |
| `sub_hens_lo2.csv` | 6.5409 | seviye eğrisi probu (aynı topluluk, −0.06) |

**Okuma kuralı:**
- `sub_hens_lo` < 1.05764 → H-çekilişi topluluğu gerçekten kazandırıyor, kalıcı olsun
- `sub_hens_lo` ≈ 1.05764 → varyans azaltma etkisiz, tek çekiliş yeterli
- `sub_hens_lo2` < `sub_hens_lo` → seviye daha da düşürülmeli, eğri henüz dibe oturmadı

Ham `sub_hens.csv` bilerek gönderilmiyor — iki etkiyi karıştırdığı için
bilgi değeri düşük.

---

## 4. SEVİYE DENEYİ — kapandı (29 Ağustos)

`sub_hens_lo` tabanından uniform kaydırma testi. Uniform kaydırmada MSE **tam
olarak** parabolik olduğu için iki nokta kapalı formda çözer:

`MSE(d) = MSE(0) + 2·d·m + d²`   (m = ortalama artık, log uzayında)

| kaydırma | LB skoru | MSE |
|---|---|---|
| 0.00 | 1.05737 | 1.118031 |
| −0.30 | 1.09545 | 1.200011 |

**Çözüm: m = +0.0134 log.**

- optimal ek kaydırma: **−0.013**
- optimumdaki skor: **1.05729** → kazanç **0.0001**

**SONUÇ: seviye kalibrasyonu kapandı.** Tahminler 0.013 log içinde doğru.
Daha fazla kaydırma denemesi yapılmayacak.

### Yapılan hata (kayda geçsin)

Önceki uyum `sub_nowx_lo` (1.06525) ile `sub_hens_lo` (1.05737) noktalarını
kullanıp −0.306 kaydırma ve 1.012 skor öngörmüştü. Hata: bu iki dosya **sadece
seviyede farklı değil** — farklı H çekilişi, farklı kategori kodlaması ve farklı
kod yolu (src/parquet vs notebook/CSV) taşıyorlar. Çekiliş etkisini
`sub_notebook` vs `sub_hens_lo` ile ölçüp (0.0003) ihmal edilebilir saymıştım,
ama o karşılaştırma kategori sırasını ve kod yolunu sabit tuttuğu için
`sub_nowx_lo`'yu ayıran farkı kontrol etmiyordu.

**Ders: seviye gradyanı yalnızca AYNI modelin kaydırılmış sürümlerinden
ölçülebilir.** Farklı modeller arası skor farkı seviyeye atfedilemez.

**İkinci ders:** eşleşmeli kohort ölçümü (ilçe × güç × ay, cold +0.09 / warm
+0.05) baştan doğruydu; LB ekstrapolasyonu onunla çeliştiğinde ekstrapolasyona
güvenmek yanlıştı.

### Model-arası gürültü büyüklüğü

`sub_nowx_lo` → `sub_notebook` farkı (0.0076) seviye DEĞİLmiş. Aynı feature,
aynı parametre, farklı kurulum = 0.0076 skor oynaması. Bu, geçmişte CV'de
ölçülen 0.003–0.005'lik "iyileşme"lerin gürültü içinde kaldığı anlamına gelir.
