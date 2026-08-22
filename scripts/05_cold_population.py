# -*- coding: utf-8 -*-
"""
05_cold_population.py — Simüle cold seti, test cold'unun en iyi proxy'sine
(yeni giren / toplu giren trafolar) sıfır profili açısından benziyor mu?

Gruplar:
  GENEL      : tüm train satırları / trafoları
  YENİ GİREN : train'de ilk kez 2025-01-01 sonrası görülen trafolar (tüm satırları)
  TOPLU GİREN: yeni girenlerden, ≥30 trafo/gün olan toplu giriş günlerinde başlayanlar
  F1 COLD    : make_folds'un F1'de simüle ettiği cold trafoların valid satırları

Çıktı: reports/cold_population.md
Kullanım: python scripts/05_cold_population.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR, SEED  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.validation import make_folds  # noqa: E402

BULK_THRESHOLD = 30

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")


def pct(x):
    return f"%{100 * x:.2f}"


def zero_stats(rows: pd.DataFrame) -> dict:
    """Satır bazlı sıfır oranı + trafo bazlı ölülük metrikleri."""
    zr = rows.groupby("tanim", observed=True)["tuketim"].agg(
        n="size", z=lambda s: (s == 0).mean())
    return {
        "satir": len(rows),
        "trafo": len(zr),
        "sifir_satir_orani": (rows["tuketim"] == 0).mean(),
        "p75_trafo_payi": (zr["z"] >= 0.75).mean(),
        "p100_trafo": int((zr["z"] == 1.0).sum()),
    }


def main() -> None:
    df = load_train()
    profile = load_profile()

    w("# Cold Popülasyon Kontrolü")
    w()
    w(f"Üretim: `scripts/05_cold_population.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    first = df.groupby("tanim", observed=True)["tarih"].min()
    new_tx = set(first[first > df["tarih"].min()].index)
    daily = first[first > df["tarih"].min()].value_counts()
    bulk_days = set(daily[daily >= BULK_THRESHOLD].index)
    bulk_tx = set(first[first.isin(bulk_days)].index)

    folds = make_folds(df, profile, seed=SEED)
    f1 = folds[0]
    v1 = df.loc[f1["valid_idx"]]
    f1_cold_rows = v1[v1["tanim"].isin(f1["cold_tx"])]

    groups = {
        "GENEL (tüm train)": df,
        "YENİ GİREN (3,285 trafo, tüm satırları)": df[df["tanim"].isin(new_tx)],
        "TOPLU GİREN (15 toplu günde başlayan)": df[df["tanim"].isin(bulk_tx)],
        "F1 SİMÜLE COLD (valid satırları)": f1_cold_rows,
    }

    # ------------------------------------------------ a-b. karşılaştırma tablosu
    w("## a–b. Sıfır profili — dört grup yan yana")
    w()
    w("| grup | satır | trafo | sıfır satır oranı | %75+ sıfırlı trafo payı | %100 sıfır trafo |")
    w("|---|---|---|---|---|---|")
    stats = {}
    for name, rows in groups.items():
        s = zero_stats(rows)
        stats[name] = s
        w(f"| {name} | {s['satir']:,} | {s['trafo']:,} | "
          f"{pct(s['sifir_satir_orani'])} | {pct(s['p75_trafo_payi'])} | "
          f"{s['p100_trafo']:,} |")
    w()

    # ------------------------------------------------ c. zamanla ölüm?
    w("## c. Yeni girenler: ilk 90 gün vs sonrası")
    w()
    sub = df[df["tanim"].isin(new_tx)].copy()
    dse = (sub["tarih"] - sub["tanim"].map(first)).dt.days
    early = sub[dse <= 90]
    late = sub[dse > 90]
    # adil kıyas: sadece 90+ güne ulaşan trafolar
    survivors = set(late["tanim"].unique())
    early_s = early[early["tanim"].isin(survivors)]
    w("| pencere | satır | trafo | sıfır oranı |")
    w("|---|---|---|---|")
    for name, rows in [("ilk 90 gün (tüm yeni girenler)", early),
                       ("ilk 90 gün (90+ güne ulaşanlar)", early_s),
                       ("90. günden sonrası (aynı trafolar)", late)]:
        w(f"| {name} | {len(rows):,} | {rows['tanim'].nunique():,} | "
          f"{pct((rows['tuketim'] == 0).mean())} |")
    w()
    e_rate = (early_s["tuketim"] == 0).mean()
    l_rate = (late["tuketim"] == 0).mean()
    # ölü doğanlar: yeni girip hiç tüketmeyenler
    zr_new = sub.groupby("tanim", observed=True)["tuketim"].apply(lambda s: (s == 0).mean())
    dead_born = int((zr_new == 1.0).sum())
    w(f"- Aynı trafolarda sıfır oranı ilk 90 günde {pct(e_rate)}, sonrasında {pct(l_rate)} "
      f"→ yeni trafolar zamanla {'ölmüyor, tersine oturuyor' if l_rate < e_rate else 'ölüyor'}.")
    w(f"- 'Ölü doğan' (tüm satırları sıfır) yeni giren trafo: {dead_born:,} / {len(new_tx):,} "
      f"({pct(dead_born / len(new_tx))})")
    w()

    # ------------------------------------------------ d. karar
    w("## d. Karar")
    w()
    gen = stats["GENEL (tüm train)"]["sifir_satir_orani"]
    new = stats["YENİ GİREN (3,285 trafo, tüm satırları)"]["sifir_satir_orani"]
    bulk = stats["TOPLU GİREN (15 toplu günde başlayan)"]["sifir_satir_orani"]
    sim = stats["F1 SİMÜLE COLD (valid satırları)"]["sifir_satir_orani"]
    diff_new_gen = new - gen
    diff_sim_bulk = sim - bulk

    w(f"- Yeni giren − genel: **{100*diff_new_gen:+.2f} puan** "
      f"({pct(new)} vs {pct(gen)})")
    w(f"- Toplu giren − genel: {100*(bulk-gen):+.2f} puan ({pct(bulk)})")
    w(f"- F1 simüle cold − toplu giren (test proxy'si): **{100*diff_sim_bulk:+.2f} puan** "
      f"({pct(sim)} vs {pct(bulk)})")
    w()
    if diff_new_gen < -0.02:
        karar = ("Yeni giren grubun sıfır oranı genelden 2+ puan DÜŞÜK → make_folds "
                 "cold seçimi bu alt popülasyondan stratified yapılmalı.")
    elif abs(diff_sim_bulk) > 0.02:
        karar = ("Yeni girenler genelden DÜŞÜK değil; ama simüle cold ile test "
                 "proxy'si (toplu girenler) arasında 2+ puan fark var → make_folds "
                 "cold seçimi gözden geçirilmeli.")
    else:
        karar = ("Fark eşik altında: simüle cold seti, test cold'unun en iyi "
                 "proxy'siyle uyumlu. make_folds DEĞİŞMEZ.")
    w(f"**KARAR:** {karar}")
    w()

    # taban yeniden hesabı — proxy p ile
    for label, p in [("F1 simüle cold p", sim), ("toplu-giren proxy p", bulk)]:
        nz = f1_cold_rows[f1_cold_rows["tuketim"] > 0]
        L = float(np.log1p(nz["tuketim"]).mean())
        floor = float(np.sqrt(p * (1 - p) * L * L))
        w(f"- Taban ({label}={p:.4f}, L={L:.3f}): RMSLE **{floor:.4f}**")
    w()
    w(f"> **Sonuç:** Yukarıdaki tabloda dört grubun sıfır profili; karar satırı "
      f"net — {karar}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cold_population.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"Rapor: {REPORTS_DIR / 'cold_population.md'}")


if __name__ == "__main__":
    main()
