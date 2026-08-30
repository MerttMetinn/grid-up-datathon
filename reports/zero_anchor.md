# Anchor sıfır düzeltmesi — RMSLE-doğru forma çevrildi

Üretim: `scripts/31_zero_anchor.py` · 2026-08-30 15:40
- 4 H çekilişi × 3 tohum (taban ile aynı kurgu)
- Taban: `sub_hens_lo` LB **1.05737** · üniform cold kaydırması `sub_sp15` öngörü 1.0535

## 1. Değişikliğin büyüklüğü (test satırlarında)

| kesim | n | eski anchor | yeni anchor | fark |
|---|---|---|---|---|
| cold | 158,369 | 7.2294 | 6.9016 | -0.3277 |
| warm | 556,319 | 6.4027 | 6.4027 | +0.0000 |

- cold farkının dağılımı: p10 -0.718 · medyan -0.251 · p90 -0.014
- Üniform kaydırmadan farkı: düzeltme **satır bazlı**; sıfır olasılığı yüksek satırlar çok daha aşağı iniliyor.

## 2. sub_zanch.csv

- yazıldı · 714,688 satır
- `sub_hens_lo`'a göre: cold ortalama -0.3277 · warm +0.0000 · genel -0.0726
- genel ortalama log1p: 6.5274 (taban 6.6000)

**LB ile doğrulanmalı.** Beklenti: üniform kaydırmanın (sub_sp15) üstüne, satır bazlı düzeltmenin katkısı kadar iyileşme.
