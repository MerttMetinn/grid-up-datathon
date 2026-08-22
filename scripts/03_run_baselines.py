# -*- coding: utf-8 -*-
"""
03_run_baselines.py — 3 fold × 6 baseline + fold doğrulama.

Çıktılar:
  - stdout: verify_fold tablosu + warm/cold kırılımlı skor tablosu
  - experiments/log.csv (append)
  - reports/baseline_results.md

Kullanım: python scripts/03_run_baselines.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import BASELINES, b5_guc_lf  # noqa: E402
from src.config import (COLD_ROW_SHARE, EXPERIMENTS_DIR, H_MEDIAN,
                        LAG364_COV_PM7, REPORTS_DIR, SEED)  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.validation import (add_eval_columns, evaluate, make_folds,
                            verify_fold)  # noqa: E402

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    # ---------------------------------------------------------- fold doğrulama
    w("# Baseline sonuçları")
    w()
    w(f"Üretim: `scripts/03_run_baselines.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()
    w("## 1. Fold doğrulaması (verify_fold)")
    w()
    w("| fold | cold_row_share | hedef | lag364_cov_pm7 | hedef | h_median | hedef | durum |")
    w("|---|---|---|---|---|---|---|---|")
    verify_results = []
    for fold in folds:
        v = verify_fold(fold, df)
        verify_results.append(v)
        durum = "OK" if not v["warnings"] else "UYARI: " + "; ".join(v["warnings"])
        if v["lag364_structural_na"]:
            durum += " · lag364: yapısal N/A"
        w(f"| {v['fold']} | {v['cold_row_share']:.4f} | {COLD_ROW_SHARE} | "
          f"{v['lag364_cov_pm7']:.4f} | {LAG364_COV_PM7} | "
          f"{v['h_median']:.0f} | {H_MEDIAN} | {durum} |")
    w()

    # ---------------------------------------------------------- baseline'lar
    w("## 2. Baseline skorları (RMSLE)")
    w()
    scores = {}   # (fold, baseline) -> {"all","warm","cold","blend"}
    eval_tables = {}
    for fold in folds:
        train = df.loc[fold["train_idx"]]
        valid = df.loc[fold["valid_idx"]]
        valid = add_eval_columns(valid, fold, df)
        for bname, bfunc in BASELINES.items():
            valid["_pred"] = bfunc(train, valid)
            ev = evaluate(valid, "tuketim", "_pred")
            eval_tables[(fold["name"], bname)] = ev
            get = lambda k, s: ev.loc[(ev["kirilim"] == k) & (ev["seviye"] == s),
                                      "rmsle"]
            allv = float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0])
            warm = get("warm_cold", "warm")
            cold = get("warm_cold", "cold")
            blend = ev.loc[ev["kirilim"] == "blend", "rmsle"]
            scores[(fold["name"], bname)] = {
                "all": allv,
                "warm": float(warm.iloc[0]) if len(warm) else np.nan,
                "cold": float(cold.iloc[0]) if len(cold) else np.nan,
                "blend": float(blend.iloc[0]) if len(blend) else np.nan,
            }

    for fold in folds:
        fn = fold["name"]
        w(f"### {fn}  (train_end={fold['spec']['train_end']}, "
          f"valid={fold['spec']['valid_start']}..{fold['spec']['valid_end']})")
        w()
        w("| baseline | all | warm | cold | blend |")
        w("|---|---|---|---|---|")
        for bname in BASELINES:
            s = scores[(fn, bname)]
            w(f"| {bname} | {s['all']:.4f} | {s['warm']:.4f} | "
              f"{s['cold']:.4f} | {s['blend']:.4f} |")
        w()

    # ---------------------------------------------------------- b5 fallback teşhisi
    w("## 3. b5 fallback seviyesi kullanımı (F1)")
    w()
    f1 = folds[0]
    train1 = df.loc[f1["train_idx"]]
    valid1 = add_eval_columns(df.loc[f1["valid_idx"]], f1, df)
    _, level = b5_guc_lf(train1, valid1, return_level=True)
    lv = level.value_counts()
    w("| seviye | satır | pay |")
    w("|---|---|---|")
    for k, v in lv.items():
        w(f"| {k} | {v:,} | %{100*v/len(level):.2f} |")
    w()

    # ---------------------------------------------------------- kabul kriterleri
    w("## 4. Kabul kriterleri")
    w()
    ok = True
    for v in verify_results:
        if v["warnings"]:
            ok = False
    w(f"- verify_fold üç metrikte hedefe yakın: "
      f"{'SAĞLANDI' if ok else 'SAĞLANMADI'}")
    f1s = {b: scores[("F1", b)] for b in BASELINES}
    c1 = f1s["b5_guc_lf"]["cold"] < min(f1s["b1_global"]["cold"],
                                        f1s["b2_trafo"]["cold"],
                                        f1s["b3_trafo_ay"]["cold"])
    w(f"- b5 cold satırlarda b1/b2/b3'ten iyi (F1): "
      f"{'SAĞLANDI' if c1 else 'SAĞLANMADI'} "
      f"(b5={f1s['b5_guc_lf']['cold']:.4f} vs b1={f1s['b1_global']['cold']:.4f}, "
      f"b2={f1s['b2_trafo']['cold']:.4f}, b3={f1s['b3_trafo_ay']['cold']:.4f})")
    c2 = f1s["b6_hibrit"]["all"] < min(f1s[b]["all"] for b in BASELINES
                                       if b != "b6_hibrit")
    w(f"- b6 global olarak hepsinden iyi (F1): "
      f"{'SAĞLANDI' if c2 else 'SAĞLANMADI'} "
      f"(b6={f1s['b6_hibrit']['all']:.4f})")
    w()

    # ---------------------------------------------------------- experiments/log.csv
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    cols = ["timestamp", "exp_id", "feature_set", "model",
            "f1_all", "f1_warm", "f1_cold", "f1_blend", "f2_all", "f3_all",
            "lb", "note"]
    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for bname in BASELINES:
        rows.append({
            "timestamp": ts, "exp_id": f"baseline_{bname}",
            "feature_set": "none", "model": bname,
            "f1_all": round(scores[("F1", bname)]["all"], 4),
            "f1_warm": round(scores[("F1", bname)]["warm"], 4),
            "f1_cold": round(scores[("F1", bname)]["cold"], 4),
            "f1_blend": round(scores[("F1", bname)]["blend"], 4),
            "f2_all": round(scores[("F2", bname)]["all"], 4),
            "f3_all": round(scores[("F3", bname)]["all"], 4),
            "lb": "", "note": "03_run_baselines",
        })
    log_df = pd.DataFrame(rows, columns=cols)
    header = not log_path.exists()
    log_df.to_csv(log_path, mode="a", header=header, index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baseline_results.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'baseline_results.md'}")


if __name__ == "__main__":
    main()
