# Cold Popülasyon Kontrolü

Üretim: `scripts/05_cold_population.py` · 2026-08-22 16:33 · SEED=42

## a–b. Sıfır profili — dört grup yan yana

| grup | satır | trafo | sıfır satır oranı | %75+ sıfırlı trafo payı | %100 sıfır trafo |
|---|---|---|---|---|---|
| GENEL (tüm train) | 1,226,237 | 5,344 | %4.69 | %6.08 | 298 |
| YENİ GİREN (3,285 trafo, tüm satırları) | 381,516 | 3,285 | %5.91 | %6.42 | 201 |
| TOPLU GİREN (15 toplu günde başlayan) | 104,388 | 1,233 | %6.17 | %4.95 | 57 |
| F1 SİMÜLE COLD (valid satırları) | 66,490 | 1,620 | %7.05 | %5.62 | 91 |

## c. Yeni girenler: ilk 90 gün vs sonrası

| pencere | satır | trafo | sıfır oranı |
|---|---|---|---|
| ilk 90 gün (tüm yeni girenler) | 204,354 | 3,285 | %6.56 |
| ilk 90 gün (90+ güne ulaşanlar) | 161,803 | 1,982 | %7.20 |
| 90. günden sonrası (aynı trafolar) | 177,162 | 1,982 | %5.16 |

- Aynı trafolarda sıfır oranı ilk 90 günde %7.20, sonrasında %5.16 → yeni trafolar zamanla ölmüyor, tersine oturuyor.
- 'Ölü doğan' (tüm satırları sıfır) yeni giren trafo: 201 / 3,285 (%6.12)

## d. Karar

- Yeni giren − genel: **+1.22 puan** (%5.91 vs %4.69)
- Toplu giren − genel: +1.48 puan (%6.17)
- F1 simüle cold − toplu giren (test proxy'si): **+0.88 puan** (%7.05 vs %6.17)

**KARAR:** Fark eşik altında: simüle cold seti, test cold'unun en iyi proxy'siyle uyumlu. make_folds DEĞİŞMEZ.

- Taban (F1 simüle cold p=0.0705, L=6.968): RMSLE **1.7841**
- Taban (toplu-giren proxy p=0.0617, L=6.968): RMSLE **1.6770**

> **Sonuç:** Yukarıdaki tabloda dört grubun sıfır profili; karar satırı net — Fark eşik altında: simüle cold seti, test cold'unun en iyi proxy'siyle uyumlu. make_folds DEĞİŞMEZ.
