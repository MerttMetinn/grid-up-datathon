# -*- coding: utf-8 -*-
"""
20_hurdle_optuna.py — İki avantajı birleştir: hurdle mimarisi (bizim) +
feature selection (29 feature) + optuna hiperparametreleri (arkadaşın, sena branch).

Referanslar (LB): hurdle 1.08943 · optuna-düz 1.06483. Hedef: ikisinden iyi.
  - regresör: 29 seçili feature + anchor init_score, optuna birleştirilmiş params, nonzero
  - classifier: ölü-trafo (P(zero)), 29 feature
  - hurdle birleştirme: expm1((1-p)*L), cold'da b5 harmanı

Çıktı: reports/model_hurdle_opt.md · submissions/sub_hurdle_opt.csv
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.baselines import b5_guc_lf  # noqa: E402
from src.config import (REPORTS_DIR, SEED, SUBMISSIONS_DIR, TRAIN_END,
                        YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.predict import write_submission  # noqa: E402
from src.train import (ORIGINS, align_categories, build_training_set)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

# referans: bizim hurdle (75 feat, default param) LB 1.08943
HURDLE = {"F1": 1.1285, "F2": 1.2327, "F3": 1.2481}
ALPHA, W_COLD = 0.4, 0.45
SEEDS = [0, 1, 2]
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]

# arkadaşın feature selection çıktısı (29 feature)
MAIN = json.loads((ROOT / "data" / "feature-selection-results"
                   / "model_features.json").read_text(encoding="utf-8"))
_LVL_SEAS = set(FEATURE_GROUPS["lvl"]) | set(FEATURE_GROUPS["seas"])
COLD = [c for c in MAIN if c not in _LVL_SEAS]      # cold'da lvl/seas NaN → çıkar

# arkadaşın optuna birleştirilmiş parametreleri (reports/optuna_summary.md)
OPT_PARAMS = {
    "objective": "regression", "metric": "rmse", "verbose": -1,
    "learning_rate": 0.0210857, "num_leaves": 113, "min_data_in_leaf": 167,
    "feature_fraction": 0.666845, "bagging_fraction": 0.738261, "bagging_freq": 1,
    "lambda_l1": 0.887017, "lambda_l2": 2.07964,
    "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
}
REG_ROUNDS = 340        # optuna F1 best_iter 306 × ~1.1
CLF_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": SEED}
CLF_ROUNDS = 300

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def assemble(base, dev, zero):
    return (base + ALPHA * dev + zero).to_numpy()


def fit_reg(X, y, feats, init, so):
    p = dict(OPT_PARAMS)
    for k in ("seed", "feature_fraction_seed", "bagging_seed"):
        p[k] = OPT_PARAMS[k] + so
    ds = lgb.Dataset(X[feats], label=y, init_score=init,
                     categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feats])
    return lgb.train(p, ds, num_boost_round=REG_ROUNDS)


def fit_clf(X, is_zero, feats, so):
    p = dict(CLF_PARAMS); p["seed"] = CLF_PARAMS["seed"] + so
    ds = lgb.Dataset(X[feats], label=is_zero.astype(int),
                     categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feats])
    return lgb.train(p, ds, num_boost_round=CLF_ROUNDS)


def reg_log(b, X, feats, init):
    return b.predict(X[feats]) + init


def logblend(pm, pb, wg):
    return np.expm1(wg * np.log1p(pm) + (1 - wg) * np.log1p(pb))


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model hurdle+optuna — mimari + feature selection + optuna params")
    w()
    w(f"Üretim: `scripts/20_hurdle_optuna.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()
    w(f"- Feature: {len(MAIN)} (arkadaşın seçimi) · cold model: {len(COLD)}")
    w()

    scores, aucs = {}, {}
    for fold in folds:
        fn = fold["name"]
        print(f"[{fn}] feature build ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile,
                                              {"F1": 0, "F2": 1, "F3": 2}[fn])
        vr = df.loc[fold["valid_idx"]]
        X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        valid = add_eval_columns(vr, fold, df)
        is_cold = valid["is_cold"].to_numpy()
        comp_va = anchor_components(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
        itr = assemble(meta["anc_base"], meta["anc_dev"], meta["anc_zero"])
        iva = assemble(comp_va["base"], comp_va["season_dev"], comp_va["zero_adj"])
        cm = meta["is_cold_example"].to_numpy()
        vc_idx = valid.index[is_cold]
        b5c = b5_guc_lf(df.loc[fold["train_idx"]], vr[is_cold]).to_numpy()
        y_orig = meta["tuketim"].to_numpy()
        nz = y_orig > 0

        print(f"[{fn}] classifier + nonzero regresör (3 seed) ...")
        p_list, warm_list, cold_list = [], [], []
        for so in SEEDS:
            bc = fit_clf(X_tr, pd.Series(~nz), MAIN, so)
            p_list.append(bc.predict(X_va[MAIN]))
            br = fit_reg(X_tr[nz], y_tr[nz], MAIN, itr[nz], so)
            warm_list.append(reg_log(br, X_va, MAIN, iva))
            cnz = cm & nz
            bcold = fit_reg(X_tr[cnz], y_tr[cnz], COLD, itr[cnz], so)
            cold_list.append(reg_log(bcold, X_va.loc[vc_idx], COLD, iva[is_cold]))

        p = np.clip(np.mean(p_list, axis=0), 0, 1)
        reg_l = np.mean(warm_list, axis=0)
        cold_l = np.mean(cold_list, axis=0)
        aucs[fn] = roc_auc_score((vr["tuketim"].to_numpy() == 0).astype(int), p)

        pred = np.clip(np.expm1((1 - p) * reg_l), 0, None)
        cold_pred = np.clip(np.expm1((1 - p[is_cold]) * cold_l), 0, None)
        pred[is_cold] = logblend(cold_pred, b5c, W_COLD)

        vv = valid.copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
        ev = evaluate(vv, "tuketim", "_pred")
        e2 = (np.log1p(vv["_pred"].clip(0)) - np.log1p(vv["tuketim"])) ** 2
        nzv = vv["tuketim"] > 0
        g = lambda k, s: float(ev.loc[(ev["kirilim"] == k) & (ev["seviye"] == s), "rmsle"].iloc[0])
        scores[fn] = {
            "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
            "warm": g("warm_cold", "warm"), "cold": g("warm_cold", "cold"),
            "nz_warm": float(np.sqrt(e2[nzv & ~vv["is_cold"]].mean())),
            "nz_cold": float(np.sqrt(e2[nzv & vv["is_cold"]].mean()))}

    w("## 1. Skorlar — hurdle+optuna vs bizim hurdle")
    w()
    w("| fold | hurdle+opt | eski hurdle | Δ | warm | cold | AUC |")
    w("|---|---|---|---|---|---|---|")
    for fn in ["F1", "F2", "F3"]:
        s = scores[fn]; d = s["blend"] - HURDLE[fn]
        w(f"| {fn} | **{s['blend']:.4f}** | {HURDLE[fn]:.4f} | {d:+.4f} | "
          f"{s['warm']:.4f} | {s['cold']:.4f} | {aucs[fn]:.3f} |")
    w()

    # tam eğitim + submission
    print("[FULL] tam egitim ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    Xt = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xt])
    comp_t = anchor_components(te, TRAIN_END, df)
    itr = assemble(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"])
    itt = assemble(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"])
    cold_f = metaf["is_cold_example"].to_numpy()
    is_cold_te = ~te["tanim"].isin(set(df["tanim"].unique())).to_numpy()
    b5t = b5_guc_lf(df, te[is_cold_te]).to_numpy()
    nzf = metaf["tuketim"].to_numpy() > 0

    p_list, warm_list, cold_list = [], [], []
    for so in SEEDS:
        bc = fit_clf(Xf, pd.Series(~nzf), MAIN, so)
        p_list.append(bc.predict(Xt[MAIN]))
        br = fit_reg(Xf[nzf], yf[nzf], MAIN, itr[nzf], so)
        warm_list.append(reg_log(br, Xt, MAIN, itt))
        cnz = cold_f & nzf
        bcold = fit_reg(Xf[cnz], yf[cnz], COLD, itr[cnz], so)
        cold_list.append(reg_log(bcold, Xt.loc[is_cold_te], COLD, itt[is_cold_te]))
    p = np.clip(np.mean(p_list, axis=0), 0, 1)
    pred = np.clip(np.expm1((1 - p) * np.mean(warm_list, axis=0)), 0, None)
    cold_pred = np.clip(np.expm1((1 - p[is_cold_te]) * np.mean(cold_list, axis=0)), 0, None)
    pred[is_cold_te] = logblend(cold_pred, b5t, W_COLD)

    # kalibrasyon
    te_ay = te["tarih"].dt.to_period("M")
    tr_s = df[df["tanim"].isin(set(te["tanim"].unique()))].copy()
    tr_s["ay_p"] = tr_s["tarih"].dt.to_period("M")
    aylar = [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]
    cnt = tr_s[tr_s["ay_p"].isin(aylar)].groupby("tanim", observed=True)["tarih"].nunique()
    cov = set(cnt[cnt >= 110].index)
    base_cov = {m: float(np.log1p(tr_s.loc[
        (tr_s["ay_p"] == pd.Period(f"2025-{m:02d}")) & (tr_s["tanim"].isin(cov)),
        "tuketim"]).mean()) for m in (4, 5, 6, 7)}
    devs = [float(np.log1p(pred[(te_ay == pd.Period(f'2026-{m:02d}')).to_numpy()]).mean())
            - base_cov[m] - YOY_DRIFT for m in (4, 5, 6, 7)]
    w("## 2. Kohort-eş aylık kalibrasyon")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")
    w()

    sub = pd.DataFrame({"id": te["id"], "tuketim": pred})
    write_submission(sub, SUBMISSIONS_DIR / "sub_hurdle_opt.csv")
    w(f"- submissions/sub_hurdle_opt.csv yazıldı.")
    w(f"- F1 blend {scores['F1']['blend']:.4f} (eski hurdle 1.1285, optuna-düz LB 1.065)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_hurdle_opt.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_hurdle_opt.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_hurdle_opt.md").write_text(out.getvalue(), encoding="utf-8")
