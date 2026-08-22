# -*- coding: utf-8 -*-
"""
08_train_multi2.py — Origin yayma + grp_ ayrıştırması: n1/n2/n3.

  n1: çok-origin (yayılmış) + yeni grp_, tek model, init_score=log(guc*24)
  n2: n1 + SADECE cold örneklerle eğitilmiş cold modeli (static_+cal_+grp_)
  n3: n2 + cold satırlarda log-uzayı harmanı: w·model_cold + (1−w)·b5
      (w F2'de optimize edilir, F1/F3'te doğrulanır)

Çıktılar: reports/model_v3.md · experiments/log.csv (append)
Kullanım: python scripts/08_train_multi2.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import b5_guc_lf  # noqa: E402
from src.config import (EXPERIMENTS_DIR, REPORTS_DIR, SEED, TEST_N_DAYS)  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.features import ALL_FEATURES, FEATURE_GROUPS, build_features  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, ORIGINS, align_categories,
                       build_training_set, fit_lgbm)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

B6_BLEND = {"F1": 1.2692, "F2": 1.2654, "F3": 1.3055}
VARIANTS = ["n1", "n2", "n3"]

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def logblend(p_model, p_b5, wgt):
    return np.expm1(wgt * np.log1p(p_model) + (1 - wgt) * np.log1p(p_b5))


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model v3 — origin yayma + grp_ ayrıştırması")
    w()
    w(f"Üretim: `scripts/08_train_multi2.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    store = {}   # fold -> dict(valid, preds_n1, pred_cold_model, pred_b5_cold, iters...)
    imp_f2 = None
    checks = {}

    for fi, fold in enumerate(folds):
        fn = fold["name"]
        print(f"[{fn}] egitim seti ({len(ORIGINS[fn])} origin) ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)

        valid_rows = df.loc[fold["valid_idx"]]
        train_end = fold["spec"]["train_end"]
        X_va = build_features(valid_rows, train_end, df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        y_va = np.log1p(valid_rows["tuketim"])
        valid = add_eval_columns(valid_rows, fold, df)
        is_cold_va = valid["is_cold"].to_numpy()

        checks[fn] = {
            "n_rows": len(X_tr),
            "cold_share": float(meta["is_cold_example"].mean()),
            "lvl_nan_tr": float(X_tr["lvl_mean_log_90d"].isna().mean()),
            "lvl_nan_va": float(X_va["lvl_mean_log_90d"].isna().mean()),
        }

        init_tr = np.log(meta["guc"].to_numpy() * 24.0)
        init_va = np.log(valid_rows["guc"].to_numpy() * 24.0)

        print(f"[{fn}] n1 ...")
        b_main, pred_n1, it1 = fit_lgbm(X_tr, y_tr, X_va, y_va, ALL_FEATURES,
                                        init_tr, init_va)

        # cold modeli: SADECE cold örnekler, static_+cal_+grp_
        cold_mask = meta["is_cold_example"].to_numpy()
        Xc, yc = X_tr[cold_mask], y_tr[cold_mask]
        ic = init_tr[cold_mask]
        # early stopping valid'i: fold valid'inin cold satırları
        Xvc = X_va.loc[valid.index[is_cold_va]]
        yvc = y_va[is_cold_va]
        ivc = init_va[is_cold_va]
        print(f"[{fn}] cold modeli ({cold_mask.sum():,} satır) ...")
        b_cold, pred_cold_c, itc = fit_lgbm(Xc, yc, Xvc, yvc,
                                            COLD_MODEL_FEATURES, ic, ivc)

        # b5 cold tahmini (fold train'inden)
        pred_b5_cold = b5_guc_lf(df.loc[fold["train_idx"]],
                                 valid_rows[is_cold_va]).to_numpy()

        store[fn] = {
            "valid": valid, "pred_n1": pred_n1, "is_cold": is_cold_va,
            "pred_cold_model": pred_cold_c, "pred_b5_cold": pred_b5_cold,
            "y_cold": valid_rows.loc[is_cold_va, "tuketim"].to_numpy(),
            "iters": {"n1": it1, "cold": itc},
        }
        if fn == "F2":
            imp_f2 = pd.Series(b_main.feature_importance("gain"),
                               index=ALL_FEATURES)

    # ---------------------------------------------------------- w optimizasyonu (F2)
    s2 = store["F2"]
    grid = np.arange(0, 1.0001, 0.05)
    w_scores = [rmsle(s2["y_cold"],
                      logblend(s2["pred_cold_model"], s2["pred_b5_cold"], g))
                for g in grid]
    w_opt = float(grid[int(np.argmin(w_scores))])

    # ---------------------------------------------------------- skorlar
    scores = {}
    for fn in ["F1", "F2", "F3"]:
        s = store[fn]
        valid = s["valid"]
        variants_pred = {}
        variants_pred["n1"] = s["pred_n1"]
        p2 = np.array(s["pred_n1"], dtype="float64")
        p2[s["is_cold"]] = s["pred_cold_model"]
        variants_pred["n2"] = p2
        p3 = p2.copy()
        p3[s["is_cold"]] = logblend(s["pred_cold_model"], s["pred_b5_cold"], w_opt)
        variants_pred["n3"] = p3
        for var, p in variants_pred.items():
            valid["_pred"] = pd.Series(p, index=valid.index)
            ev = evaluate(valid, "tuketim", "_pred")
            e2 = (np.log1p(valid["_pred"].clip(0)) -
                  np.log1p(valid["tuketim"])) ** 2
            nz = valid["tuketim"] > 0
            get = lambda k, sv: float(ev.loc[
                (ev["kirilim"] == k) & (ev["seviye"] == sv), "rmsle"].iloc[0])
            scores[(fn, var)] = {
                "all": float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0]),
                "warm": get("warm_cold", "warm"), "cold": get("warm_cold", "cold"),
                "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
                "nz_warm": float(np.sqrt(e2[nz & ~valid["is_cold"]].mean())),
                "nz_cold": float(np.sqrt(e2[nz & valid["is_cold"]].mean())),
            }

    # ---------------------------------------------------------- 1-2. skor tabloları
    w("## 1. Skorlar (referans satırı: b6 blend)")
    w()
    for fn in ["F1", "F2", "F3"]:
        it = store[fn]["iters"]
        w(f"### {fn}  (b6 blend = {B6_BLEND[fn]})")
        w()
        w("| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |")
        w("|---|---|---|---|---|---|---|---|---|")
        for var in VARIANTS:
            s = scores[(fn, var)]
            bi = it["n1"] if var == "n1" else f"{it['n1']}/{it['cold']}c"
            w(f"| {var} | {s['all']:.4f} | {s['warm']:.4f} | {s['cold']:.4f} | "
              f"**{s['blend']:.4f}** | {s['blend']-B6_BLEND[fn]:+.4f} | "
              f"{s['nz_warm']:.4f} | {s['nz_cold']:.4f} | {bi} |")
        w()

    # ---------------------------------------------------------- 3. F2 importance
    w("## 2. F2 feature importance (n1, gain) — ilk 25 + grup toplamları")
    w()
    imp = imp_f2.sort_values(ascending=False)
    tot = imp.sum()
    w("| # | feature | gain payı |")
    w("|---|---|---|")
    for i, (f, v) in enumerate(imp.head(25).items(), 1):
        w(f"| {i} | {f} | %{100*v/tot:.2f} |")
    w()
    grp_gains = {}
    w("| grup | toplam gain payı |")
    w("|---|---|")
    for grp, cols in FEATURE_GROUPS.items():
        share = 100 * imp_f2.reindex(cols).fillna(0).sum() / tot
        grp_gains[grp] = share
        w(f"| {grp}_ | %{share:.2f} |")
    w()

    # ---------------------------------------------------------- 4. kapsam matrisi
    w("## 3. (origin ayı × hedef ayı) kapsam matrisi")
    w()
    valid_months = {"F1": [1, 2, 3], "F2": [4, 5, 6, 7], "F3": [9, 10, 11, 12]}
    for fn in ["F1", "F2", "F3"]:
        train_end = pd.Timestamp(next(f["spec"]["train_end"] for f in folds
                                      if f["name"] == fn))
        pairs = {}
        for o in ORIGINS[fn]:
            ot = pd.Timestamp(o)
            he = min(ot + pd.Timedelta(days=TEST_N_DAYS), train_end)
            months = sorted(set(pd.date_range(ot + pd.Timedelta(days=1), he,
                                              freq="D").month))
            pairs[ot.month] = months
        covered = sorted({m for ms in pairs.values() for m in ms})
        missing = [m for m in valid_months[fn] if m not in covered]
        w(f"### {fn} — valid ayları: {valid_months[fn]}")
        w()
        w("| origin ayı | hedef ayları |")
        w("|---|---|")
        for om, ms in pairs.items():
            w(f"| {om:02d} | {', '.join(f'{m:02d}' for m in ms)} |")
        w(f"- Eğitimde hedef olarak görülen aylar: {covered}")
        w(f"- **Valid'de olup eğitim hedefinde HİÇ görülmeyen aylar: "
          f"{missing if missing else 'YOK'}**")
        w()

    # ---------------------------------------------------------- 5-6. kontroller + w
    w("## 4. Kontroller")
    w()
    w("| fold | eğitim satırı | cold payı | lvl_ NaN eğitim | lvl_ NaN valid | best_iter n1/cold |")
    w("|---|---|---|---|---|---|")
    for fn in ["F1", "F2", "F3"]:
        c, it = checks[fn], store[fn]["iters"]
        w(f"| {fn} | {c['n_rows']:,} | %{100*c['cold_share']:.1f} | "
          f"%{100*c['lvl_nan_tr']:.1f} | %{100*c['lvl_nan_va']:.1f} | "
          f"{it['n1']}/{it['cold']} |")
    w()
    w(f"- grp_ gain payı (F2): **%{grp_gains['grp']:.1f}** (önceki %2.3, hedef ≥%8)")
    w(f"- n3 harman ağırlığı: **w = {w_opt:.2f}** (F2'de optimize; "
      f"w=1 saf cold modeli, w=0 saf b5)")
    w()

    # ---------------------------------------------------------- kabul
    best_var = min(VARIANTS, key=lambda v: scores[("F2", v)]["blend"])
    w("## 5. Kabul kriterleri")
    w()
    ka = scores[("F2", best_var)]["blend"] <= 1.17
    kb = scores[("F3", best_var)]["blend"] < B6_BLEND["F3"]
    kc = scores[("F1", best_var)]["blend"] <= 1.13
    kd = grp_gains["grp"] >= 8.0
    w(f"- En iyi varyant (F2 blend'e göre): **{best_var}**")
    w(f"- a) F2 blend ≤ 1.17: {scores[('F2', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if ka else 'SAĞLANMADI'}")
    w(f"- b) F3 blend < {B6_BLEND['F3']}: {scores[('F3', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kb else 'SAĞLANMADI'}")
    w(f"- c) F1 blend ≤ 1.13: {scores[('F1', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kc else 'SAĞLANMADI'}")
    w(f"- d) grp_ gain ≥ %8: %{grp_gains['grp']:.1f} → "
      f"{'SAĞLANDI' if kd else 'SAĞLANMADI'}")
    w(f"- **SONUÇ: {'KABUL' if all([ka, kb, kc, kd]) else 'KRİTER DÜŞTÜ — DUR'}**")
    w()

    # ---------------------------------------------------------- log
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for var in VARIANTS:
        rows.append({
            "timestamp": ts, "exp_id": f"lgbm_spread_{var}",
            "feature_set": "static+cal+lvl+grp2+seas+horizon",
            "model": f"lgbm_{var}",
            "f1_all": round(scores[("F1", var)]["all"], 4),
            "f1_warm": round(scores[("F1", var)]["warm"], 4),
            "f1_cold": round(scores[("F1", var)]["cold"], 4),
            "f1_blend": round(scores[("F1", var)]["blend"], 4),
            "f2_all": round(scores[("F2", var)]["all"], 4),
            "f3_all": round(scores[("F3", var)]["all"], 4),
            "lb": "", "note": f"08 origin-yayma grp2 w={w_opt:.2f}",
        })
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not log_path.exists(), index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_v3.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_v3.md'}")


if __name__ == "__main__":
    main()
