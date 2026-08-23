# Model Kartı — s2+wx (hava durumlu final)

Üretim: `scripts/17_save_wx_model.py` · 2026-08-23 12:31 · SEED=42

## Ne bu model
s2 kurgusu + hava durumu (wx_) feature ailesi. Trafo bazlı günlük tüketim tahmini.

## Mimari
- Hedef `log1p(tuketim)`, tahmin `expm1(model + init_score)`, `clip(0)`.
- init_score: mevsim-farkındalıklı anchor, α=0.4, cold sıfır düzeltmeli.
- Ana model: LightGBM, 75 feature (static/cal/lvl/grp/seas/**wx**),
  126 tur, 3 seed log ortalaması.
- Cold model: cold örneklerle, 50 feature (static+cal+grp+wx),
  73 tur. Cold satırlarda w=0.45 ile b5 harmanı.
- **wx_ (17 feature):** CDD/CDD²/CDD³/HDD, sıcaklık/apparent/nem, tarımsal
  (ET0 7g, yağış açığı 30g, toprak nemi), termal kütle (CDD 7g MA), ilk-sıcak-gün anomalisi.
  Kaynak: Open-Meteo arşivi (test dönemi gerçek gözlem). Cache: data/external/weather.parquet.

## Artifact'lar
| dosya | içerik |
|---|---|
| `wx_main_seed{0,1,2}.txt` | ana booster (wx dahil) |
| `wx_cold_seed{0,1,2}.txt` | cold-only booster (wx dahil) |

## Yükleme
`scripts/17_save_wx_model.py` akışıyla aynı: build_features (wx cache'ten) + anchor kur,
booster'ı `lgb.Booster(model_file=...)` ile yükle, `expm1(predict + init_score)`.
weather.parquet gerekli (yoksa `scripts/15_fetch_weather.py`).

## Yeniden üretilebilirlik
SEED=42, sabit tur. sub_wx.csv ile birebir.
