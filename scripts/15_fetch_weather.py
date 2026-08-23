# -*- coding: utf-8 -*-
"""
15_fetch_weather.py — Open-Meteo arşivinden 47 ilçe için hava verisi çeker + cache.

Aralık: 2024-12-01 → 2026-08-01 (test dönemi Nis–Tem 2026 gerçek gözlem dahil;
lag/hareketli-ortalama için train başından öncesi de kapsanır).
Çıktı: data/external/weather.parquet · reports/weather_summary.md
Kullanım: python scripts/15_fetch_weather.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR  # noqa: E402
from src.data import load_test, load_train  # noqa: E402
from src.weather import RAW_WX_COLS, build_coords, get_weather  # noqa: E402

START, END = "2024-12-01", "2026-08-01"

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def main():
    tr, te = load_train(), load_test()
    ilce_keys = pd.concat([tr["ilce_key"], te["ilce_key"]]).dropna().unique()
    coords = build_coords(ilce_keys)

    print(f"{len(coords)} ilçe için hava çekiliyor ({START} → {END}) ...")
    wx = get_weather(coords, START, END)

    w("# Hava Verisi Özeti (Open-Meteo arşivi)")
    w()
    w(f"Üretim: `scripts/15_fetch_weather.py` · {datetime.now():%Y-%m-%d %H:%M}")
    w(f"Kaynak: Open-Meteo Archive API · aralık {START} → {END}")
    w()
    w(f"- İlçe sayısı: {coords['ilce_key'].nunique()} · satır: {len(wx):,} · "
      f"gün/ilçe: {wx.groupby('ilce_key').size().median():.0f}")
    w()
    w("### Koordinat kaynağı")
    w()
    vc = coords["coord_src"].value_counts()
    for k, v in vc.items():
        w(f"- {k}: {v}")
    w()

    # veri kalitesi
    w("### Değişken kapsamı (NaN oranı)")
    w()
    w("| değişken | NaN | min | medyan | max |")
    w("|---|---|---|---|---|")
    for c in RAW_WX_COLS:
        s = wx[c]
        w(f"| {c.replace('wx_ham_','')} | %{100*s.isna().mean():.1f} | "
          f"{s.min():.1f} | {s.median():.1f} | {s.max():.1f} |")
    w()

    # yaz doğrulaması: test dönemi aylık ortalama sıcaklık
    wx["ay"] = wx["tarih"].dt.to_period("M")
    w("### Test dönemi aylık ort. sıcaklık (yaz rampası kanıtı)")
    w()
    w("| ay | T_mean | T_max | ET0 | yağış toplamı |")
    w("|---|---|---|---|---|")
    for m in ["2026-04", "2026-05", "2026-06", "2026-07"]:
        sub = wx[wx["ay"] == pd.Period(m)]
        w(f"| {m} | {sub['wx_ham_temperature_2m_mean'].mean():.1f} | "
          f"{sub['wx_ham_temperature_2m_max'].mean():.1f} | "
          f"{sub['wx_ham_et0_fao_evapotranspiration'].mean():.2f} | "
          f"{sub['wx_ham_precipitation_sum'].mean():.2f} |")
    w()
    # ilçe farkı: en sıcak/serin Temmuz
    jul = wx[wx["ay"] == pd.Period("2026-07")].groupby("ilce_key")[
        "wx_ham_temperature_2m_max"].mean().sort_values()
    w(f"- Temmuz T_max en serin: {jul.index[0].split('>')[-1]} ({jul.iloc[0]:.1f}°C) · "
      f"en sıcak: {jul.index[-1].split('>')[-1]} ({jul.iloc[-1]:.1f}°C)")
    w()
    w("> Test dönemi gerçek gözlem (tahmin değil) — yaz rampası ve ilçeler arası "
      "sıcaklık farkı wx_ feature'larıyla doğrudan modele verilebilir.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "weather_summary.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nCache: data/external/weather.parquet · Rapor: reports/weather_summary.md")


if __name__ == "__main__":
    main()
