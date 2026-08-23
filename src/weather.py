# -*- coding: utf-8 -*-
"""Open-Meteo arşiv hava verisi — çekme + cache. Sözleşme: get_weather(coords, start, end).

Bugün Ağustos 2026 → Nisan–Temmuz 2026 (test dönemi) için *gerçek gözlemlenmiş* hava
verisi arşivde mevcut (tahmin değil). Bu büyük avantaj: yaz rampası gerçek sıcaklıkla kurulur.

Cache: data/external/weather.parquet — her çalıştırmada API'ye GİDİLMEZ (CLAUDE.md kuralı).
Kaynak: https://archive-api.open-meteo.com/v1/archive (kural gereği notebook'ta belirtilir).
"""
import time

import numpy as np
import pandas as pd
import requests

from src.config import EXTERNAL_DIR

_CACHE = EXTERNAL_DIR / "weather.parquet"
_API = "https://archive-api.open-meteo.com/v1/archive"

# Günlük değişkenler — meteorolojik + tarımsal (sulama sürücüsü)
_DAILY = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_mean",
    "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_10m_max",
    "shortwave_radiation_sum", "sunshine_duration",
    "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm_mean",
]

# İl merkezi fallback koordinatları
_IL_MERKEZ = {"İZMİR": (38.42, 27.14), "MANİSA": (38.61, 27.43)}

# İlçe merkez koordinatları (Open-Meteo en yakın grid'e snap eder, ±birkaç km önemsiz)
ILCE_COORDS = {
    # İzmir
    "BALÇOVA": (38.39, 27.05), "BAYINDIR": (38.22, 27.65), "BAYRAKLI": (38.46, 27.17),
    "BERGAMA": (39.12, 27.18), "BEYDAĞ": (38.08, 28.21), "BORNOVA": (38.47, 27.22),
    "BUCA": (38.39, 27.18), "ÇEŞME": (38.32, 26.30), "ÇİĞLİ": (38.50, 27.07),
    "DİKİLİ": (39.07, 26.89), "FOÇA": (38.67, 26.76), "GAZİEMİR": (38.32, 27.13),
    "GÜZELBAHÇE": (38.36, 26.88), "KARABAĞLAR": (38.39, 27.13),
    "KARABURUN": (38.64, 26.51), "KARŞIYAKA": (38.46, 27.11),
    "KEMALPAŞA": (38.42, 27.42), "KINIK": (39.09, 27.38), "KİRAZ": (38.23, 28.20),
    "KONAK": (38.42, 27.14), "MENDERES": (38.25, 27.13), "MENEMEN": (38.60, 27.07),
    "NARLIDERE": (38.39, 27.00), "ÖDEMİŞ": (38.23, 27.97),
    "SEFERİHİSAR": (38.20, 26.84), "SELÇUK": (37.95, 27.37), "TİRE": (38.09, 27.74),
    "TORBALI": (38.16, 27.36), "URLA": (38.32, 26.77), "ALİAĞA": (38.80, 26.97),
    # Manisa
    "AHMETLİ": (38.52, 28.00), "AKHİSAR": (38.92, 27.84), "ALAŞEHİR": (38.35, 28.52),
    "DEMİRCİ": (39.04, 28.66), "GÖLMARMARA": (38.72, 27.95), "GÖRDES": (38.93, 28.29),
    "KIRKAĞAÇ": (39.10, 27.67), "KULA": (38.55, 28.65), "KÖPRÜBAŞI": (38.74, 28.41),
    "SALİHLİ": (38.48, 28.14), "SARIGÖL": (38.24, 28.69), "SARUHANLI": (38.73, 27.57),
    "SELENDİ": (38.74, 28.87), "SOMA": (39.19, 27.61), "ŞEHZADELER": (38.62, 27.43),
    "TURGUTLU": (38.49, 27.70), "YUNUSEMRE": (38.62, 27.40),
}


def build_coords(ilce_keys) -> pd.DataFrame:
    """ilce_key ('İL>İLÇE') listesinden koordinat tablosu. Fallback: ilçe → il merkezi."""
    rows = []
    for key in sorted(set(ilce_keys)):
        il = key.split(">")[0].strip()
        ilce = key.split(">")[-1].strip()
        if ilce in ILCE_COORDS:
            lat, lon = ILCE_COORDS[ilce]
            src = "ilce"
        elif il in _IL_MERKEZ:
            lat, lon = _IL_MERKEZ[il]
            src = "il_merkez"
        else:
            lat, lon = _IL_MERKEZ["İZMİR"]
            src = "genel_ege"
        rows.append({"ilce_key": key, "il": il, "ilce": ilce,
                     "lat": lat, "lon": lon, "coord_src": src})
    return pd.DataFrame(rows)


_BATCH = 25          # tek istekte lokasyon sayısı (rate-limit dostu)


def _fetch_batch(lats, lons, start, end, retries=4):
    """Çoklu lokasyon tek istekte. Dönen list input sırasıyla hizalı."""
    params = {"latitude": ",".join(f"{x:.4f}" for x in lats),
              "longitude": ",".join(f"{x:.4f}" for x in lons),
              "start_date": start, "end_date": end,
              "daily": ",".join(_DAILY), "timezone": "Europe/Istanbul"}
    for attempt in range(retries):
        r = requests.get(_API, params=params, timeout=120)
        if r.status_code == 200:
            js = r.json()
            return js if isinstance(js, list) else [js]
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))   # rate-limit: uzun backoff
        else:
            time.sleep(3 * (attempt + 1))
    r.raise_for_status()


def get_weather(coords: pd.DataFrame, start: str, end: str,
                refresh: bool = False) -> pd.DataFrame:
    """Her ilçe için günlük hava verisi. İlk çağrıda çeker + parquet'e cache'ler.

    Dönen: uzun format — kolonlar [ilce_key, tarih, wx_ham_*...].
    coords: build_coords() çıktısı (ilce_key, lat, lon).
    """
    if _CACHE.exists() and not refresh:
        cached = pd.read_parquet(_CACHE)
        have = set(cached["ilce_key"].unique())
        need = set(coords["ilce_key"])
        cmin, cmax = cached["tarih"].min(), cached["tarih"].max()
        if need <= have and cmin <= pd.Timestamp(start) and cmax >= pd.Timestamp(end):
            return cached[cached["ilce_key"].isin(need)].reset_index(drop=True)

    parts = []
    coords = coords.reset_index(drop=True)
    for i in range(0, len(coords), _BATCH):
        chunk = coords.iloc[i:i + _BATCH]
        res = _fetch_batch(chunk["lat"].tolist(), chunk["lon"].tolist(), start, end)
        for (_, r), loc in zip(chunk.iterrows(), res):
            d = pd.DataFrame(loc["daily"]).rename(columns={"time": "tarih"})
            d["tarih"] = pd.to_datetime(d["tarih"])
            d.insert(0, "ilce_key", r["ilce_key"])
            parts.append(d)
        time.sleep(1.5)   # batch'ler arası nazik bekleme
    wx = pd.concat(parts, ignore_index=True)
    # ham kolonlara wx_ham_ önekı (feature türetme features.py'de)
    ren = {c: f"wx_ham_{c}" for c in _DAILY}
    wx = wx.rename(columns=ren)
    for c in ren.values():
        wx[c] = wx[c].astype("float32")

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(_CACHE)
    return wx


RAW_WX_COLS = [f"wx_ham_{c}" for c in _DAILY]
