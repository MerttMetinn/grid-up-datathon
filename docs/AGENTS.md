# Çok-Ajanlı Çalışma Düzeni

Ana fikir: **Ajanları fazlara değil, "paylaşılan koda dokunup dokunmadığına" göre ayır.**

Datathon'da darboğaz kod yazma hızı değil, **sızıntısız ve tutarlı tek bir
feature/validasyon pipeline'ıdır.** Aynı `features.py` dosyasına paralel yazan iki ajan
size hız değil, sessiz sızıntı ve çakışma üretir.

---

## Ajanları ikiye ayır

### 🟢 Paralel çalışabilir — kendi dosyasının tek sahibi, `features.py`/`train.py`'a dokunmaz

| Ajan | Sahip olduğu dosya | Görev |
|---|---|---|
| **weather-agent** | `src/weather.py`, `data/external/` | İlçe→koordinat eşleme, Open-Meteo çekme, cache, CDD/HDD türetme. Çıktı: temiz parquet + kolon sözlüğü |
| **calendar-agent** | `src/calendar_tr.py` | Resmî/dinî tatil, köprü günü, okul takvimi, turizm sezonu. Çıktı: tarih indeksli feature tablosu |
| **eda-agent** | `notebooks/01_eda.ipynb` (**read-only kod**) | ROADMAP Faz 1 checklist'i. Çıktı: `reports/eda_findings.md` — bulgular + önerilen feature listesi |
| **error-analysis-agent** | `notebooks/02_error_analysis.ipynb` (**read-only**) | En kötü tahminleri incele. Çıktı: `reports/error_findings.md` |

Bu dördü birbirini bloklamaz, çünkü **ortak dosya yok**.

### 🔴 Seri çalışmalı — tek sahip, tek oturum

| Ajan | Görev |
|---|---|
| **core-agent** | `src/config.py`, `data.py`, `validation.py`, `features.py`, `train.py`, `predict.py` |

Bu dosyalar birbirine sıkı bağlı. Bölersen sözleşmeler kayar, sızıntı fark edilmez.
`features.py`'ın **tek sahibi** olsun.

---

## Sıralama

```
1. core-agent    → config + data + validation + RMSLE + baseline'lar + submission pipeline
                   (ÖNCE BU. Ölçemiyorsan hiçbir şey yapmıyorsun.)
                          │
        ┌─────────────────┼─────────────────┬──────────────────┐
        ▼                 ▼                 ▼                  ▼
2.  weather-agent   calendar-agent      eda-agent       (paralel, bloklamaz)
        └─────────────────┴─────────────────┘
                          ▼
3. core-agent    → feature'ları entegre et, LightGBM, deney döngüsü
                          ▼
4. error-analysis-agent  → bulgular
                          ▼
5. core-agent    → hedefli feature ekleme, CatBoost, ensemble, final
```

---

## Ajan brief şablonu

Her ajanı açarken şu formatta görev ver:

```markdown
## ROL
Sen <ajan adı>'sın. SADECE şu dosyalardan sorumlusun: <dosya listesi>
Başka hiçbir dosyayı DEĞİŞTİRME. Gerekiyorsa oku, ama yazma.

## ÖNCE OKU
- CLAUDE.md  (mutlak kurallar + sözleşmeler)
- ROADMAP.md, bölüm <N>

## GÖREV
<net, tek cümlelik hedef>

## ÇIKTI SÖZLEŞMESİ
Fonksiyon imzası: <imza>
Dönen tablonun anahtarları: <key'ler>
Kolon adlandırma kuralı: <prefix, örn. wx_*, cal_*>

## KABUL KRİTERLERİ
- [ ] <ölçülebilir kriter 1>
- [ ] <ölçülebilir kriter 2>
- [ ] Sızıntı kontrolü: test döneminde tüm kolonlar dolu mu?

## YASAK
- CLAUDE.md'deki mutlak kuralları ihlal etmek
- src/features.py veya src/train.py'ı değiştirmek
```

---

## Çakışmayı önleyen 5 pratik

1. **Kolon namespace'i.** Her ajan kendi prefix'ini kullanır:
   `wx_*` (hava), `cal_*` (takvim), `sta_*` (statik), `lvl_*` (level), `seas_*` (mevsimsel).
   Entegrasyonda çakışma olmaz, feature importance okuması kolaylaşır.

2. **Git branch per ajan.** `feat/weather`, `feat/calendar`. Merge'ü core-agent yapar.

3. **Sözleşme önce, kod sonra.** Paralel ajanları başlatmadan önce `CLAUDE.md`'deki
   imzaları kilitle. Bir ajan imzayı değiştirmek isterse sana sorar, tek başına değiştirmez.

4. **`reports/` klasörü ajanlar arası mesaj kutusudur.** eda-agent bulgularını
   `reports/eda_findings.md`'ye yazar; core-agent oradan okur. Konuşma geçmişi taşınmaz,
   dosya taşınır.

5. **Sızıntı kontrolü ayrı bir adımdır, ajanın kendi işi değildir.**
   `src/leak_check.py` yaz: her feature için train/test dağılım karşılaştırması,
   test'te NaN oranı, ve "bu feature tek başına ana fold'da RMSLE'yi 0.05'ten fazla
   iyileştiriyorsa şüphelen" uyarısı.

---

## Verimlilik için asıl belirleyici şeyler

Ajan sayısından çok daha önemli olanlar:

**Önce ölçüm altyapısı.** `validation.py` + `rmsle` + `evaluate` + uçtan uca submission
üretimi ilk 2–3 saatte bitsin. Bundan sonra her fikir 5 dakikada test edilebilir hale gelir.

**Tek komutluk deney döngüsü.**
```bash
python -m src.train --exp-id 017 --features base+weather --model lgbm
# → 3 fold skoru + experiments/log.csv'ye satır + submissions/sub_017.csv
```
Bu varsa Claude'a "şu feature'ı ekle ve çalıştır, sonucu logla" demek yeterli olur.
Yoksa her deneme manuel debug'a döner.

**Bağlamı dosyaya yaz, sohbete değil.** Uzun sohbet geçmişi ajanlar arası taşınmaz.
`CLAUDE.md` + `reports/*.md` + `experiments/log.csv` üçlüsü kalıcı hafızandır.

**Küçük, doğrulanabilir adımlar iste.** "Feature pipeline'ını yaz" yerine
"lag_364 ailesini `seas_` prefix'iyle ekle, ana fold'da çalıştır, önceki skorla karşılaştır."
Her adımın sonunda bir sayı görmelisin.

**Sen mimar kal, ajan uygulayıcı olsun.** Validasyon tasarımı, hangi feature'ın mantıklı
olduğu, hangi sonucun şüpheli olduğu — bunlar senin kararların. Jüri sunumunda da
savunacağın şey bu kararlar olacak, kodun kendisi değil.
