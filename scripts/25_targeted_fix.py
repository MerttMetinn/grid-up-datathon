# -*- coding: utf-8 -*-
"""
25_targeted_fix.py — Teşhis-hedefli düzeltme. Gap: cold+ölü %56 + warm felaket.

hurdle birleştirme (1-p)*L, büyük trafolarda classifier yanılınca felaket:
  canlı dev trafo → sıfır (161K→3)  ·  ölü trafo → dev (0→112K)
Düzeltmeler (F1 valid'de grid, post-hoc → hızlı):
  p_hi: classifier üst-clip (min(p,p_hi)) — canlı→sıfır felaketini önle
  W:    cold b5 harman ağırlığı — cold+ölü aşırı-tahminini bastır

En iyi config → full eğit + submission (sub_fix.csv).
Çıktı: reports/targeted_fix.md
"""
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

from src.baselines import b5_guc_lf  # noqa: E402
from src.config import (REPORTS_DIR, SEED, SUBMISSIONS_DIR, TRAIN_END, YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.train import ORIGINS, align_categories, build_training_set  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

ALPHA = 0.4
SEEDS = [0, 1, 2]
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]
MAIN = json.loads((ROOT / "data" / "feature-selection-results"
                   / "model_features.json").read_text(encoding="utf-8"))
_LS = set(FEATURE_GROUPS["lvl"]) | set(FEATURE_GROUPS["seas"])
COLD = [c for c in MAIN if c not in _LS]
OPT = {"objective": "regression", "verbose": -1, "learning_rate": 0.0210857,
       "num_leaves": 113, "min_data_in_leaf": 167, "feature_fraction": 0.666845,
       "bagging_fraction": 0.738261, "bagging_freq": 1, "lambda_l1": 0.887017,
       "lambda_l2": 2.07964, "seed": SEED}
CLF = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
       "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
       "bagging_freq": 1, "verbose": -1, "seed": SEED}
CATS = [c for c in CATEGORICAL_FEATURES if c in MAIN]
CATS_C = [c for c in CATEGORICAL_FEATURES if c in COLD]

# grid
P_HI = [1.0, 0.90, 0.80, 0.70]
W_GRID = [0.70, 0.85, 1.0]

out = io.StringIO()


def w(line=""):
    out.write(line + "\n"); print(line)


def asm(b, d, z):
    return (b + ALPHA * d + z).to_numpy()


def fit_reg(X, y, feats, init, rounds, so):
    p = dict(OPT); p["seed"] = SEED + so
    return lgb.train(p, lgb.Dataset(X[feats], label=y, init_score=init,
                     categorical_feature=[c for c in CATS if c in feats]),
                     num_boost_round=rounds)


def fit_clf(X, z, so):
    p = dict(CLF); p["seed"] = SEED + so
    return lgb.train(p, lgb.Dataset(X[MAIN], label=z.astype(int),
                     categorical_feature=CATS), num_boost_round=300)


def combine(p, reg_l, cold_l, b5c, ic, p_hi, W):
    pp = np.minimum(p, p_hi)
    pred = np.clip(np.expm1((1 - pp) * reg_l), 0, None)
    cpred = np.clip(np.expm1((1 - pp[ic]) * cold_l), 0, None)
    pred[ic] = np.expm1(W * np.log1p(cpred) + (1 - W) * np.log1p(b5c))
    return pred


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    fold = make_folds(df, profile, seed=SEED)[0]     # F1
    print("[F1] egitim ...")
    X_tr, y_tr, meta = build_training_set(df, fold, profile, 0)
    vr = df.loc[fold["valid_idx"]]
    X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    align_categories([X_tr, X_va])
    valid = add_eval_columns(vr, fold, df)
    ic = valid["is_cold"].to_numpy()
    comp = anchor_components(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    a_tr, a_va = asm(meta["anc_base"], meta["anc_dev"], meta["anc_zero"]), \
        asm(comp["base"], comp["season_dev"], comp["zero_adj"])
    cm = meta["is_cold_example"].to_numpy()
    vci = valid.index[ic]
    b5c = b5_guc_lf(df.loc[fold["train_idx"]], vr[ic]).to_numpy()
    nz = meta["tuketim"].to_numpy() > 0

    p_l, warm_l, cold_l = [], [], []
    for so in SEEDS:
        p_l.append(fit_clf(X_tr, pd.Series(~nz), so).predict(X_va[MAIN]))
        warm_l.append(fit_reg(X_tr[nz], y_tr[nz], MAIN, a_tr[nz], 340, so).predict(X_va[MAIN]) + a_va)
        cnz = cm & nz
        cold_l.append(fit_reg(X_tr[cnz], y_tr[cnz], COLD, a_tr[cnz], 200, so)
                      .predict(X_va.loc[vci][COLD]) + a_va[ic])
    p = np.clip(np.mean(p_l, axis=0), 0, 1)
    reg_l, cold_reg = np.mean(warm_l, axis=0), np.mean(cold_l, axis=0)
    yv = vr["tuketim"].to_numpy()

    w("# Hedefli Düzeltme — cold+ölü + warm felaket (F1 valid grid)")
    w()
    w(f"Üretim: `scripts/25_targeted_fix.py` · {datetime.now():%Y-%m-%d %H:%M}")
    w()
    w("## 1. Grid — p_hi (classifier üst-clip) × W (cold b5)")
    w()
    w("| p_hi | W | blend | cold RMSLE | felaket(e²>50) satır |")
    w("|---|---|---|---|---|")
    results = {}
    for p_hi in P_HI:
        for W in W_GRID:
            pred = combine(p, reg_l, cold_reg, b5c, ic, p_hi, W)
            vv = valid.copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
            ev = evaluate(vv, "tuketim", "_pred")
            blend = float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])
            e2 = (np.log1p(pred) - np.log1p(yv)) ** 2
            results[(p_hi, W)] = blend
            w(f"| {p_hi:.2f} | {W:.2f} | **{blend:.4f}** | "
              f"{rmsle(yv[ic], pred[ic]):.3f} | {int((e2 > 50).sum())} |")
    w()
    best = min(results, key=results.get)
    w(f"- **En iyi config: p_hi={best[0]}, W={best[1]} → blend {results[best]:.4f}** "
      f"(mevcut p_hi=1.0/W=0.70: {results[(1.0, 0.70)]:.4f})")
    w()

    # full eğit + submission (en iyi config)
    print(f"[FULL] en iyi config p_hi={best[0]} W={best[1]} ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    Xt = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xt])
    comp_t = anchor_components(te, TRAIN_END, df)
    a_f, a_t = asm(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"]), \
        asm(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"])
    cold_f = metaf["is_cold_example"].to_numpy()
    ict = ~te["tanim"].isin(set(df["tanim"].unique())).to_numpy()
    b5t = b5_guc_lf(df, te[ict]).to_numpy()
    nzf = metaf["tuketim"].to_numpy() > 0

    p_l, warm_l, cold_l = [], [], []
    for so in SEEDS:
        p_l.append(fit_clf(Xf, pd.Series(~nzf), so).predict(Xt[MAIN]))
        warm_l.append(fit_reg(Xf[nzf], yf[nzf], MAIN, a_f[nzf], 340, so).predict(Xt[MAIN]) + a_t)
        cnz = cold_f & nzf
        cold_l.append(fit_reg(Xf[cnz], yf[cnz], COLD, a_f[cnz], 200, so)
                      .predict(Xt.loc[ict][COLD]) + a_t[ict])
    p = np.clip(np.mean(p_l, axis=0), 0, 1)
    pred = combine(p, np.mean(warm_l, axis=0), np.mean(cold_l, axis=0),
                   b5t, ict, best[0], best[1])
    pd.DataFrame({"id": te["id"], "tuketim": pred}).to_csv(
        SUBMISSIONS_DIR / "sub_fix.csv", index=False)

    te_ay = te["tarih"].dt.to_period("M")
    tr_s = df[df["tanim"].isin(set(te["tanim"].unique()))].copy()
    tr_s["ay_p"] = tr_s["tarih"].dt.to_period("M")
    ay = [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]
    cnt = tr_s[tr_s["ay_p"].isin(ay)].groupby("tanim", observed=True)["tarih"].nunique()
    cov = set(cnt[cnt >= 110].index)
    base_cov = {m: float(np.log1p(tr_s.loc[(tr_s["ay_p"] == pd.Period(f"2025-{m:02d}"))
                & (tr_s["tanim"].isin(cov)), "tuketim"]).mean()) for m in (4, 5, 6, 7)}
    devs = [float(np.log1p(pred[(te_ay == pd.Period(f'2026-{m:02d}')).to_numpy()]).mean())
            - base_cov[m] - YOY_DRIFT for m in (4, 5, 6, 7)]
    w("## 2. sub_fix.csv (en iyi config) — kalibrasyon")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")
    w()
    w(f"- **sub_fix.csv yazıldı** (p_hi={best[0]}, W={best[1]}).")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "targeted_fix.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'targeted_fix.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "targeted_fix.md").write_text(out.getvalue(), encoding="utf-8")
