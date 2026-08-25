# -*- coding: utf-8 -*-
"""
22_cold_b5_test.py — Cold b5 harmanı hipotezini test et.

Hipotez: cold satırlarda b5 baseline harmanı (w=0.45) tahminleri YUKARI çekiyor;
"düşük tahmin = iyi" rejiminde zararlı olabilir. Arkadaşın modeli (b5 YOK) bizden
düşük tahmin edip LB'de iyi (1.0648).

Test: hurdle+optuna mimarisi, cold harman ağırlığı W ∈ {0, 0.25, 0.45, 0.70} grid.
Her W için: fold CV cold skoru + tam-eğitim submission + kalibrasyon.
W=0 → cold saf model (b5 yok). W=0.45 → mevcut.

Çıktı: reports/cold_b5_test.md · submissions/sub_w{00,25,45,70}.csv
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
from src.config import (REPORTS_DIR, SEED, SUBMISSIONS_DIR, TRAIN_END,
                        YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.train import (ORIGINS, align_categories, build_training_set)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

ALPHA = 0.4
W_GRID = [0.0, 0.25, 0.45, 0.70]
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
RM, RC = 340, 200
CLF = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
       "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
       "bagging_freq": 1, "verbose": -1, "seed": SEED}

out = io.StringIO()


def w(line=""):
    out.write(line + "\n"); print(line)


def asm(base, dev, zero):
    return (base + ALPHA * dev + zero).to_numpy()


def fit_reg(X, y, feats, init, rounds, so):
    p = dict(OPT); p["seed"] = SEED + so
    ds = lgb.Dataset(X[feats], label=y, init_score=init,
                     categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feats])
    return lgb.train(p, ds, num_boost_round=rounds)


def fit_clf(X, z, feats, so):
    p = dict(CLF); p["seed"] = SEED + so
    ds = lgb.Dataset(X[feats], label=z.astype(int),
                     categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feats])
    return lgb.train(p, ds, num_boost_round=300)


def blend(pm, pb, wg):
    return np.expm1(wg * np.log1p(pm) + (1 - wg) * np.log1p(pb))


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Cold b5 harmanı testi — W grid")
    w()
    w(f"Üretim: `scripts/22_cold_b5_test.py` · {datetime.now():%Y-%m-%d %H:%M}")
    w(f"- W=0 cold saf model (b5 yok) · W=0.45 mevcut")
    w()

    # fold CV: her W için blend skoru
    fold_cold = {fn: {} for fn in ["F1", "F2", "F3"]}
    fold_blend = {fn: {} for fn in ["F1", "F2", "F3"]}
    for fold in folds:
        fn = fold["name"]
        print(f"[{fn}] egitim ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile,
                                              {"F1": 0, "F2": 1, "F3": 2}[fn])
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
            bc = fit_clf(X_tr, pd.Series(~nz), MAIN, so)
            p_l.append(bc.predict(X_va[MAIN]))
            br = fit_reg(X_tr[nz], y_tr[nz], MAIN, a_tr[nz], RM, so)
            warm_l.append(br.predict(X_va[MAIN]) + a_va)
            cnz = cm & nz
            bcold = fit_reg(X_tr[cnz], y_tr[cnz], COLD, a_tr[cnz], RC, so)
            cold_l.append(bcold.predict(X_va.loc[vci][COLD]) + a_va[ic])
        p = np.clip(np.mean(p_l, axis=0), 0, 1)
        reg_l, cold_reg = np.mean(warm_l, axis=0), np.mean(cold_l, axis=0)
        base_pred = np.clip(np.expm1((1 - p) * reg_l), 0, None)
        cold_model_pred = np.clip(np.expm1((1 - p[ic]) * cold_reg), 0, None)
        yv = vr["tuketim"].to_numpy()
        for W in W_GRID:
            pred = base_pred.copy()
            pred[ic] = blend(cold_model_pred, b5c, W)
            vv = valid.copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
            ev = evaluate(vv, "tuketim", "_pred")
            fold_blend[fn][W] = float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])
            fold_cold[fn][W] = rmsle(yv[ic], pred[ic])

    w("## 1. Fold CV — W etkisi (cold RMSLE / blend)")
    w()
    for fn in ["F1", "F2", "F3"]:
        w(f"### {fn}")
        w("| W | cold RMSLE | blend |")
        w("|---|---|---|")
        for W in W_GRID:
            w(f"| {W:.2f} | {fold_cold[fn][W]:.4f} | {fold_blend[fn][W]:.4f} |")
        w()

    # tam eğitim: her W için submission + kalibrasyon
    print("[FULL] egitim ...")
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
        bc = fit_clf(Xf, pd.Series(~nzf), MAIN, so)
        p_l.append(bc.predict(Xt[MAIN]))
        br = fit_reg(Xf[nzf], yf[nzf], MAIN, a_f[nzf], RM, so)
        warm_l.append(br.predict(Xt[MAIN]) + a_t)
        cnz = cold_f & nzf
        bcold = fit_reg(Xf[cnz], yf[cnz], COLD, a_f[cnz], RC, so)
        cold_l.append(bcold.predict(Xt.loc[ict][COLD]) + a_t[ict])
    p = np.clip(np.mean(p_l, axis=0), 0, 1)
    base_pred = np.clip(np.expm1((1 - p) * np.mean(warm_l, axis=0)), 0, None)
    cold_model_pred = np.clip(np.expm1((1 - p[ict]) * np.mean(cold_l, axis=0)), 0, None)

    te_ay = te["tarih"].dt.to_period("M")
    tr_s = df[df["tanim"].isin(set(te["tanim"].unique()))].copy()
    tr_s["ay_p"] = tr_s["tarih"].dt.to_period("M")
    ay = [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]
    cnt = tr_s[tr_s["ay_p"].isin(ay)].groupby("tanim", observed=True)["tarih"].nunique()
    cov = set(cnt[cnt >= 110].index)
    base_cov = {m: float(np.log1p(tr_s.loc[(tr_s["ay_p"] == pd.Period(f"2025-{m:02d}"))
                & (tr_s["tanim"].isin(cov)), "tuketim"]).mean()) for m in (4, 5, 6, 7)}

    w("## 2. Tam-eğitim submission + kalibrasyon (her W)")
    w()
    w("| W | Nis | May | Haz | Tem | cold ort. tahmin log |")
    w("|---|---|---|---|---|---|")
    for W in W_GRID:
        pred = base_pred.copy()
        pred[ict] = blend(cold_model_pred, b5t, W)
        tag = f"{int(W*100):02d}"
        pd.DataFrame({"id": te["id"], "tuketim": pred}).to_csv(
            SUBMISSIONS_DIR / f"sub_w{tag}.csv", index=False)
        devs = [np.log1p(pred[(te_ay == pd.Period(f'2026-{m:02d}')).to_numpy()]).mean()
                - base_cov[m] - YOY_DRIFT for m in (4, 5, 6, 7)]
        cold_mean = float(np.log1p(pred[ict]).mean())
        w(f"| {W:.2f} | " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {cold_mean:.3f} |")
    w()
    w("- Submissionlar: sub_w00 (b5 yok), sub_w25, sub_w45 (mevcut), sub_w70.")
    w("- Hipotez: W küçüldükçe cold tahmini düşer. LB 'düşük iyi' ise sub_w00 < sub_w45.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cold_b5_test.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'cold_b5_test.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "cold_b5_test.md").write_text(out.getvalue(), encoding="utf-8")
