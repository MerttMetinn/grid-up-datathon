# -*- coding: utf-8 -*-
"""
07_train_multi.py — Çok-origin eğitim: 3 fold × 3 varyant (m1/m2/m3).

Çıktılar: reports/model_v2.md · experiments/log.csv (append)
Kullanım: python scripts/07_train_multi.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EXPERIMENTS_DIR, REPORTS_DIR, SEED, TARGET_BLEND  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.features import ALL_FEATURES, FEATURE_GROUPS, build_features  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, align_categories,
                       build_training_set, fit_lgbm)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

REFS = {"b6": 1.2692, "v3": 1.2665, "cita": TARGET_BLEND,
        "cold_taban": 1.78, "b5_nz_cold": 1.102}
V3_F2, V3_F3 = 1.2246, 1.2207
VARIANTS = ["m1", "m2", "m3"]

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model v2 — çok-origin LightGBM")
    w()
    w(f"Üretim: `scripts/07_train_multi.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    scores, iters, checks = {}, {}, {}
    imp_main = None

    for fi, fold in enumerate(folds):
        fn = fold["name"]
        print(f"[{fn}] cok-origin egitim seti ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)

        valid_rows = df.loc[fold["valid_idx"]]
        train_end = fold["spec"]["train_end"]
        X_va = build_features(valid_rows, train_end, df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        y_va = np.log1p(valid_rows["tuketim"])
        valid = add_eval_columns(valid_rows, fold, df)

        # zorunlu kontrol 1-2
        cold_share = float(meta["is_cold_example"].mean())
        lvl_nan_tr = float(X_tr["lvl_mean_log_90d"].isna().mean())
        lvl_nan_va = float(X_va["lvl_mean_log_90d"].isna().mean())
        checks[fn] = {"cold_share": cold_share, "lvl_nan_tr": lvl_nan_tr,
                      "lvl_nan_va": lvl_nan_va, "n_rows": len(X_tr)}

        init_tr = np.log(meta["guc"].to_numpy() * 24.0)
        init_va = np.log(valid_rows["guc"].to_numpy() * 24.0)

        preds = {}
        print(f"[{fn}] m1 ...")
        b1, preds["m1"], it1 = fit_lgbm(X_tr, y_tr, X_va, y_va, ALL_FEATURES)
        print(f"[{fn}] m2 ...")
        b2, preds["m2"], it2 = fit_lgbm(X_tr, y_tr, X_va, y_va, ALL_FEATURES,
                                        init_tr, init_va)
        print(f"[{fn}] m3 cold modeli ...")
        bc, pred_cold, itc = fit_lgbm(X_tr, y_tr, X_va, y_va,
                                      COLD_MODEL_FEATURES, init_tr, init_va)
        is_cold_va = valid["is_cold"].to_numpy()
        preds["m3"] = np.where(is_cold_va, pred_cold, preds["m2"])
        iters[fn] = {"m1": it1, "m2": it2, "m3_cold": itc}

        if fn == "F1":
            imp_main = pd.Series(b2.feature_importance("gain"),
                                 index=ALL_FEATURES)

        for var in VARIANTS:
            valid["_pred"] = pd.Series(preds[var], index=valid.index)
            ev = evaluate(valid, "tuketim", "_pred")
            e2 = (np.log1p(valid["_pred"].clip(0)) -
                  np.log1p(valid["tuketim"])) ** 2
            nz = valid["tuketim"] > 0
            get = lambda k, s: float(ev.loc[
                (ev["kirilim"] == k) & (ev["seviye"] == s), "rmsle"].iloc[0])
            scores[(fn, var)] = {
                "all": float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0]),
                "warm": get("warm_cold", "warm"), "cold": get("warm_cold", "cold"),
                "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
                "nz_warm": float(np.sqrt(e2[nz & ~valid["is_cold"]].mean())),
                "nz_cold": float(np.sqrt(e2[nz & valid["is_cold"]].mean())),
            }

    # ------------------------------------------------------------ 1. skorlar
    w("## 1. Skorlar")
    w()
    for fn in ["F1", "F2", "F3"]:
        w(f"### {fn}")
        w()
        w("| varyant | all | warm | cold | blend | nz_warm | nz_cold | best_iter |")
        w("|---|---|---|---|---|---|---|---|")
        it = iters[fn]
        for var in VARIANTS:
            s = scores[(fn, var)]
            bi = it["m1"] if var == "m1" else (
                it["m2"] if var == "m2" else f"{it['m2']}/{it['m3_cold']}c")
            w(f"| {var} | {s['all']:.4f} | {s['warm']:.4f} | {s['cold']:.4f} | "
              f"**{s['blend']:.4f}** | {s['nz_warm']:.4f} | {s['nz_cold']:.4f} | {bi} |")
        w()

    best_var = min(VARIANTS, key=lambda v: scores[("F1", v)]["blend"])
    best = scores[("F1", best_var)]

    # ------------------------------------------------------------ 2. importance
    w("## 2. F1 feature importance (m2, gain) — ilk 25 + grup toplamları")
    w()
    imp = imp_main.sort_values(ascending=False)
    tot = imp.sum()
    w("| # | feature | gain payı |")
    w("|---|---|---|")
    for i, (f, v) in enumerate(imp.head(25).items(), 1):
        w(f"| {i} | {f} | %{100*v/tot:.2f} |")
    w()
    w("| grup | toplam gain payı |")
    w("|---|---|")
    grp_gains = {}
    for grp, cols in FEATURE_GROUPS.items():
        share = 100 * imp_main.reindex(cols).fillna(0).sum() / tot
        grp_gains[grp] = share
        w(f"| {grp}_ | %{share:.2f} |")
    w()

    # ------------------------------------------------------------ 3. kontroller
    w("## 3. Zorunlu kontroller")
    w()
    w("| fold | eğitim satırı | cold satır payı (hedef ~%22) | lvl_ NaN eğitim | lvl_ NaN valid |")
    w("|---|---|---|---|---|")
    for fn in ["F1", "F2", "F3"]:
        c = checks[fn]
        w(f"| {fn} | {c['n_rows']:,} | %{100*c['cold_share']:.1f} | "
          f"%{100*c['lvl_nan_tr']:.1f} | %{100*c['lvl_nan_va']:.1f} |")
    w()
    low_iters = [(fn, k, v) for fn in iters for k, v in iters[fn].items() if v < 150]
    if low_iters:
        w(f"- **best_iter < 150 uyarısı:** {low_iters} — leakage tamamen gitmemiş "
          f"olabilir veya model erken doyuyor.")
    else:
        w("- best_iter kontrolü: tüm eğitimler ≥150 iterasyon — erken duruş sorunu yok.")
    w(f"- lvl_ ailesi toplam gain: **%{grp_gains['lvl']:.1f}** (önceki model: %77+) — "
      + ("belirgin düştü." if grp_gains["lvl"] < 65 else
         "**yeterince düşmedi, rapor ediliyor.**"))
    w()

    # ------------------------------------------------------------ 4. referans + kabul
    w("## 4. Referanslar ve kabul")
    w()
    w(f"- Referanslar: b6={REFS['b6']} · v3={REFS['v3']} · çıta={REFS['cita']} · "
      f"cold tabanı={REFS['cold_taban']} · b5 nz_cold={REFS['b5_nz_cold']}")
    w(f"- En iyi varyant (F1 blend): **{best_var} = {best['blend']:.4f}**")
    w()
    ka = best["blend"] <= 1.15
    kb = (scores[("F2", best_var)]["blend"] < V3_F2 and
          scores[("F3", best_var)]["blend"] < V3_F3)
    kc = best["nz_cold"] <= 1.10
    w(f"- a) F1 blend ≤ 1.15: {best['blend']:.4f} → {'SAĞLANDI' if ka else 'SAĞLANMADI'}")
    w(f"- b) F2 < {V3_F2} ve F3 < {V3_F3}: "
      f"{scores[('F2', best_var)]['blend']:.4f} / {scores[('F3', best_var)]['blend']:.4f} "
      f"→ {'SAĞLANDI' if kb else 'SAĞLANMADI'}")
    w(f"- c) nz_cold ≤ 1.10: {best['nz_cold']:.4f} → {'SAĞLANDI' if kc else 'SAĞLANMADI'}")
    w(f"- **SONUÇ: {'KABUL' if (ka and kb and kc) else 'KRİTER DÜŞTÜ — DUR'}**")
    w()

    # ------------------------------------------------------------ log
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for var in VARIANTS:
        rows.append({
            "timestamp": ts, "exp_id": f"lgbm_multiorigin_{var}",
            "feature_set": "static+cal+lvl+grp+seas+horizon",
            "model": f"lgbm_{var}",
            "f1_all": round(scores[("F1", var)]["all"], 4),
            "f1_warm": round(scores[("F1", var)]["warm"], 4),
            "f1_cold": round(scores[("F1", var)]["cold"], 4),
            "f1_blend": round(scores[("F1", var)]["blend"], 4),
            "f2_all": round(scores[("F2", var)]["all"], 4),
            "f3_all": round(scores[("F3", var)]["all"], 4),
            "lb": "", "note": "07_train_multi cok-origin",
        })
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not log_path.exists(), index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_v2.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_v2.md'}")


if __name__ == "__main__":
    main()
