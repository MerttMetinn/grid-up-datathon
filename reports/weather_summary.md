# Hava Verisi Özeti (Open-Meteo arşivi)

Üretim: `scripts/15_fetch_weather.py` · 2026-08-23 12:21
Kaynak: Open-Meteo Archive API · aralık 2024-12-01 → 2026-08-01

- İlçe sayısı: 47 · satır: 28,623 · gün/ilçe: 609

### Koordinat kaynağı

- ilce: 47

### Değişken kapsamı (NaN oranı)

| değişken | NaN | min | medyan | max |
|---|---|---|---|---|
| temperature_2m_max | %0.0 | -2.8 | 21.3 | 45.5 |
| temperature_2m_min | %0.0 | -10.3 | 11.4 | 29.3 |
| temperature_2m_mean | %0.0 | -5.5 | 15.9 | 36.4 |
| apparent_temperature_max | %0.0 | -8.4 | 20.4 | 45.1 |
| apparent_temperature_mean | %0.0 | -10.0 | 14.6 | 36.3 |
| relative_humidity_2m_mean | %0.0 | 15.0 | 64.0 | 98.0 |
| precipitation_sum | %0.0 | 0.0 | 0.0 | 108.3 |
| wind_speed_10m_max | %0.0 | 1.8 | 15.0 | 66.6 |
| shortwave_radiation_sum | %0.0 | 0.6 | 18.0 | 31.9 |
| sunshine_duration | %0.0 | 0.0 | 39446.7 | 51596.0 |
| et0_fao_evapotranspiration | %0.0 | 0.2 | 3.1 | 12.1 |
| soil_moisture_0_to_7cm_mean | %0.0 | 0.0 | 0.2 | 0.4 |

### Test dönemi aylık ort. sıcaklık (yaz rampası kanıtı)

| ay | T_mean | T_max | ET0 | yağış toplamı |
|---|---|---|---|---|
| 2026-04 | 14.4 | 19.9 | 3.51 | 2.36 |
| 2026-05 | 18.5 | 23.8 | 4.61 | 1.88 |
| 2026-06 | 25.4 | 31.5 | 6.54 | 0.44 |
| 2026-07 | 27.5 | 33.7 | 7.39 | 0.11 |

- Temmuz T_max en serin: DEMİRCİ (28.9°C) · en sıcak: AHMETLİ (36.4°C)

> Test dönemi gerçek gözlem (tahmin değil) — yaz rampası ve ilçeler arası sıcaklık farkı wx_ feature'larıyla doğrudan modele verilebilir.
