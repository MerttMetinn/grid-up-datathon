# -*- coding: utf-8 -*-
"""
11_train_season.py — Mevsim düzeltmesi: q1/q2.

  q1: p3 kurgusu + lvl_median_log_full, lvl_lf_median_364d,
      lvl_season_adjusted_90d/28d, lvl_season_gap (cold modele de girer)
  q2: q1 + warm sigorta harmanı: w_warm·model + (1−w_warm)·b2 (log uzayında),
      w_warm üç fold ortalamasında optimize

Ek: F3 H181-300 kesimi · q1/q2 tam-eğitim Temmuz/Mayıs sağlık oranı (1 seed).
Çıktılar: reports/model_v5.md · experiments/log.csv (append)
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import b2_trafo_median, b5_guc_lf, b6_hybrid  # noqa: E402
from src.config import (EXPERIMENTS_DIR, REPORTS_DIR, SEED, TRAIN_END)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import ALL_FEATURES, build_features  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set, fit_lgbm)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

B6_BLEND = {"F1": 1.2692, "F2": 1.2654, "F3": 1.3055}
W_COLD = 0.45
SEEDS = [0, 1, 2]
SEASON_FEATS = ["lvl_season_adjusted_90d", "lvl_season_adjusted_28d",
                "lvl_season_gap", "lvl_median_log_full", "lvl_lf_median_364d"]
FINAL_ROUNDS_MAIN, FINAL_ROUNDS_COLD = 126, 73

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def logmean(preds):
    return np.expm1(np.mean([np.log1p(p) for p in preds], axis=0))


def logblend(pm, pb, wgt):
    return np.expm1(wgt * np.log1p(pm) + (1 - wgt) * np.log1p(pb))


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model v5 — mevsim düzeltmesi (q1/q2)")
    w()
    w(f"Üretim: `scripts/11_train_season.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    store, imps = {}, {}
    for fi, fold in enumerate(folds):
        fn = fold["name"]
        print(f"[{fn}] egitim seti ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)
        valid_rows = df.loc[fold["valid_idx"]]
        X_va = build_features(valid_rows, fold["spec"]["train_end"],
                              df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        y_va = np.log1p(valid_rows["tuketim"])
        valid = add_eval_columns(valid_rows, fold, df)
        is_cold_va = valid["is_cold"].to_numpy()

        init_tr = np.log(meta["guc"].to_numpy() * 24.0)
        init_va = np.log(valid_rows["guc"].to_numpy() * 24.0)
        cold_mask = meta["is_cold_example"].to_numpy()
        Xvc = X_va.loc[valid.index[is_cold_va]]

        mains, colds = [], []
        for so in SEEDS:
            print(f"[{fn}] seed+{so} ...")
            bm, pm, _ = fit_lgbm(X_tr, y_tr, X_va, y_va, ALL_FEATURES,
                                 init_tr, init_va, seed_offset=so)
            bc, pc, _ = fit_lgbm(X_tr[cold_mask], y_tr[cold_mask],
                                 Xvc, y_va[is_cold_va], COLD_MODEL_FEATURES,
                                 init_tr[cold_mask], init_va[is_cold_va],
                                 seed_offset=so)
            mains.append(np.asarray(pm))
            colds.append(np.asarray(pc))
            if so == 0:
                imps[fn] = pd.Series(bm.feature_importance("gain"),
                                     index=ALL_FEATURES)

        store[fn] = {
            "fold": fold, "valid": valid, "valid_rows": valid_rows,
            "is_cold": is_cold_va, "mains": mains, "colds": colds,
            "b5_cold": b5_guc_lf(df.loc[fold["train_idx"]],
                                 valid_rows[is_cold_va]).to_numpy(),
            "b2": b2_trafo_median(df.loc[fold["train_idx"]],
                                  valid_rows).to_numpy(),
            "y": valid_rows["tuketim"].to_numpy(),
        }

    # ---------------------------------------------------------- q1 tahminleri
    def q1_pred(fn):
        s = store[fn]
        p = np.array(logmean(s["mains"]), dtype="float64")
        p[s["is_cold"]] = logblend(logmean(s["colds"]), s["b5_cold"], W_COLD)
        return p

    q1 = {fn: q1_pred(fn) for fn in store}

    # w_warm: üç fold ortalamasında optimize (sadece warm satırlar)
    grid = np.arange(0.5, 1.0001, 0.05)

    def warm_score(fn, wgt):
        s = store[fn]
        m = ~s["is_cold"]
        pw = logblend(q1[fn][m], s["b2"][m], wgt)
        return rmsle(s["y"][m], pw)

    w_warm = float(grid[int(np.argmin(
        [np.mean([warm_score(fn, g) for g in [g] for fn in store]) for g in grid]))])

    def q2_pred(fn):
        s = store[fn]
        p = q1[fn].copy()
        m = ~s["is_cold"]
        p[m] = logblend(q1[fn][m], s["b2"][m], w_warm)
        return p

    q2 = {fn: q2_pred(fn) for fn in store}

    # ---------------------------------------------------------- skorlar
    scores = {}
    for fn in ["F1", "F2", "F3"]:
        valid = store[fn]["valid"]
        for var, p in [("q1", q1[fn]), ("q2", q2[fn])]:
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

    w("## 1. Skorlar")
    w()
    for fn in ["F1", "F2", "F3"]:
        w(f"### {fn}  (b6 blend = {B6_BLEND[fn]})")
        w()
        w("| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold |")
        w("|---|---|---|---|---|---|---|---|")
        for var in ["q1", "q2"]:
            s = scores[(fn, var)]
            w(f"| {var} | {s['all']:.4f} | {s['warm']:.4f} | {s['cold']:.4f} | "
              f"**{s['blend']:.4f}** | {s['blend']-B6_BLEND[fn]:+.4f} | "
              f"{s['nz_warm']:.4f} | {s['nz_cold']:.4f} |")
        w()

    # ---------------------------------------------------------- 2. gain payları
    w("## 2. F1 mevsim feature gain payları (ana model)")
    w()
    imp1 = imps["F1"]
    tot = imp1.sum()
    w("| feature | gain payı |")
    w("|---|---|")
    for f in SEASON_FEATS:
        w(f"| {f} | %{100*imp1.get(f, 0)/tot:.2f} |")
    w()

    # ---------------------------------------------------------- 3. F3 H181-300
    w("## 3. F3 H 181-300 kesimi (önceki: +0.2363)")
    w()
    s3 = store["F3"]
    valid3 = s3["valid"]
    p_best = q2["F3"] if scores[("F3", "q2")]["blend"] <= scores[("F3", "q1")]["blend"] \
        else q1["F3"]
    p_b6 = b6_hybrid(df.loc[s3["fold"]["train_idx"]], s3["valid_rows"]).to_numpy()
    e_m = (np.log1p(np.clip(p_best, 0, None)) - np.log1p(s3["y"])) ** 2
    e_b = (np.log1p(np.clip(p_b6, 0, None)) - np.log1p(s3["y"])) ** 2
    mask = (valid3["H_bucket"] == "181-300").to_numpy()
    h_diff = float(np.sqrt(e_m[mask].mean()) - np.sqrt(e_b[mask].mean()))
    w(f"- n={int(mask.sum()):,} · model {np.sqrt(e_m[mask].mean()):.4f} · "
      f"b6 {np.sqrt(e_b[mask].mean()):.4f} → fark **{h_diff:+.4f}**")
    w()

    # ---------------------------------------------------------- 4. w_warm
    q1_f1, q2_f1 = scores[("F1", "q1")]["blend"], scores[("F1", "q2")]["blend"]
    w("## 4. w_warm")
    w()
    w(f"- w_warm = **{w_warm:.2f}** (0.50–1.00 grid, 3-fold warm ortalaması)")
    w(f"- F1 maliyeti: q1 {q1_f1:.4f} → q2 {q2_f1:.4f} ({q2_f1-q1_f1:+.4f})")
    w()

    # ---------------------------------------------------------- 5. sağlık oranı
    w("## 5. Tam-eğitim Temmuz/Mayıs sağlık oranı (1 seed)")
    w()
    print("[FULL] saglik kontrolu icin tam egitim ...")
    tr, te = df, load_test()
    ORIGINS["FULL"] = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                       "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                       "2025-10-31", "2025-11-30"]
    pseudo = {"name": "FULL", "train_idx": tr.index,
              "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(tr, pseudo, profile, fold_i=9)
    Xt = build_features(te, TRAIN_END, tr)
    align_categories([Xf, Xt])
    init_f = np.log(metaf["guc"].to_numpy() * 24.0)
    init_t = np.log(te["guc"].to_numpy() * 24.0)
    cold_f = metaf["is_cold_example"].to_numpy()
    is_cold_te = ~te["tanim"].isin(set(tr["tanim"].unique())).to_numpy()

    import lightgbm as lgb
    ds = lgb.Dataset(Xf[ALL_FEATURES], label=yf, init_score=init_f,
                     categorical_feature=[c for c in Xf.columns
                                          if str(Xf[c].dtype) == "category"])
    bm = lgb.train(LGB_PARAMS, ds, num_boost_round=FINAL_ROUNDS_MAIN)
    p_main = np.clip(np.expm1(bm.predict(Xt[ALL_FEATURES]) + init_t), 0, None)
    dsc = lgb.Dataset(Xf.loc[cold_f, COLD_MODEL_FEATURES], label=yf[cold_f],
                      init_score=init_f[cold_f],
                      categorical_feature=[c for c in COLD_MODEL_FEATURES
                                           if str(Xf[c].dtype) == "category"])
    bc = lgb.train(LGB_PARAMS, dsc, num_boost_round=FINAL_ROUNDS_COLD)
    p_cold = np.clip(np.expm1(
        bc.predict(Xt.loc[is_cold_te, COLD_MODEL_FEATURES])
        + init_t[is_cold_te]), 0, None)
    p_b5t = b5_guc_lf(tr, te[is_cold_te]).to_numpy()
    pq1 = p_main.copy()
    pq1[is_cold_te] = logblend(p_cold, p_b5t, W_COLD)
    p_b2t = b2_trafo_median(tr, te).to_numpy()
    pq2 = pq1.copy()
    pq2[~is_cold_te] = logblend(pq1[~is_cold_te], p_b2t[~is_cold_te], w_warm)

    te_ay = te["tarih"].dt.to_period("M")
    ratios = {}
    for name, p in [("q1", pq1), ("q2", pq2)]:
        lp = np.log1p(p)
        jul = lp[te_ay == pd.Period("2026-07")].mean()
        may = lp[te_ay == pd.Period("2026-05")].mean()
        ratios[name] = float(np.expm1(jul) / np.expm1(may))
        w(f"- {name}: Temmuz/Mayıs = **{ratios[name]:.2f}×** (beklenen ~1.86×, eşik ≥1.6)")
    w()

    # ---------------------------------------------------------- kabul
    best_var = min(["q1", "q2"], key=lambda v: scores[("F1", v)]["blend"])
    best = scores[("F1", best_var)]
    w("## 6. Kabul kriterleri")
    w()
    ka = best["blend"] <= 1.11
    kb = scores[("F3", best_var)]["blend"] <= 1.3155
    kc = h_diff <= 0.10
    kd = ratios[best_var] >= 1.6
    w(f"- En iyi varyant (F1 blend): **{best_var}**")
    w(f"- a) F1 blend ≤ 1.11: {best['blend']:.4f} → {'SAĞLANDI' if ka else 'SAĞLANMADI'}")
    w(f"- b) F3 blend ≤ 1.3155: {scores[('F3', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kb else 'SAĞLANMADI'}")
    w(f"- c) F3 H181-300 farkı ≤ +0.10: {h_diff:+.4f} → "
      f"{'SAĞLANDI' if kc else 'SAĞLANMADI'}")
    w(f"- d) Temmuz/Mayıs ≥ 1.6: {ratios[best_var]:.2f}× → "
      f"{'SAĞLANDI' if kd else 'SAĞLANMADI'}")
    w(f"- **SONUÇ: {'KABUL' if all([ka, kb, kc, kd]) else 'KRİTER DÜŞTÜ — DUR'}**")
    w()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for var in ["q1", "q2"]:
        rows.append({
            "timestamp": ts, "exp_id": f"lgbm_season_{var}",
            "feature_set": "static+cal+lvl_season+grp2+seas+horizon",
            "model": f"lgbm_{var}",
            "f1_all": round(scores[("F1", var)]["all"], 4),
            "f1_warm": round(scores[("F1", var)]["warm"], 4),
            "f1_cold": round(scores[("F1", var)]["cold"], 4),
            "f1_blend": round(scores[("F1", var)]["blend"], 4),
            "f2_all": round(scores[("F2", var)]["all"], 4),
            "f3_all": round(scores[("F3", var)]["all"], 4),
            "lb": "", "note": f"11 season w_warm={w_warm:.2f} "
                              f"julmay={ratios[var]:.2f}",
        })
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not log_path.exists(), index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_v5.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_v5.md'}")


if __name__ == "__main__":
    main()
