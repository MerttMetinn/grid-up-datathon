# -*- coding: utf-8 -*-
"""
18_build_dataset.py — Optuna/feature-selection için materialize edilmiş dataset paketi.

Tüm feature'lar (75) önceden hesaplanmış, hava dahil. data/dataset/ altına yazar.
Hava güncellenince (scripts/15) bu scripti tekrar koş → dataset tazelenir.

Çıktı: data/dataset/*.parquet + reports/dataset_summary.md
Kullanım: python scripts/18_build_dataset.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR  # noqa: E402
from src.dataset import (CAT_COLS, DATASET_DIR, FEATURE_COLS,
                         build_datasets, load_dataset)  # noqa: E402
from src.features import FEATURE_GROUPS  # noqa: E402


def main():
    written = build_datasets()

    out = io.StringIO()

    def w(line=""):
        out.write(line + "\n")
        print(line)

    w("# Dataset Paketi Özeti")
    w()
    w(f"Üretim: `scripts/18_build_dataset.py` · {datetime.now():%Y-%m-%d %H:%M}")
    w()
    w(f"Konum: `data/dataset/` · toplam {len(FEATURE_COLS)} feature (hava dahil)")
    w()
    w("### Üretilen dosyalar")
    w()
    for name in written:
        w(f"- {name}")
    w()
    w("### Feature grupları")
    w()
    for g, cols in FEATURE_GROUPS.items():
        w(f"- `{g}_` : {len(cols)} feature")
    w()
    w(f"- Kategorik feature'lar: {CAT_COLS}")
    w()

    # örnek: test dataseti kolonları
    test = load_dataset("test")
    meta_cols = [c for c in test.columns if c not in FEATURE_COLS]
    w("### Feature dışı kolonlar (her dosyada)")
    w()
    w(f"{meta_cols}")
    w()
    w("- `y_log1p` : model hedefi (log1p tüketim) — train/valid'de")
    w("- `tuketim` : orijinal ölçek gerçek değer — train/valid'de (skorlama için)")
    w("- `init_score` : s2 fiziksel çapası (α=0.4). LightGBM'e init_score olarak ver.")
    w("- `anc_base/anc_dev/anc_zero` : çapa bileşenleri — kendi α'nı denemek için")
    w("  `init_score = anc_base + α*anc_dev + anc_zero`")
    w("- `is_cold` : trafo train'de görülmemiş mi (cold-start)")
    w()
    w("Kullanım: `from src.dataset import load_dataset; df = load_dataset('f1_train')`")
    w()
    w("Detaylı kullanım + UYARILAR: `docs/DATASET.md`")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dataset_summary.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nDataset: {DATASET_DIR} · Rapor: reports/dataset_summary.md")


if __name__ == "__main__":
    main()
