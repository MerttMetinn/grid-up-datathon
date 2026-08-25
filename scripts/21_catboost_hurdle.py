# -*- coding: utf-8 -*-
"""
21_catboost_hurdle.py — CatBoost hurdle: LightGBM'e farklı algoritma çeşitliliği.

Tüm modellerimiz LightGBM (%98 korele → aynı hata). CatBoost farklı ağaç kurma
(symmetric trees, ordered boosting) → korelasyonu kırar → ensemble gerçek kazanç.

Aynı hurdle mimarisi: classifier P(zero) + nonzero regresör, birleştirme (1-p)*L,
cold'da b5 harmanı. Anchor init_score, CatBoost'ta "residual hedef" ile
(y_log1p - anchor öğrenilir, tahminde geri eklenir — LightGBM init_score ile eşdeğer).
29 feature (arkadaşın seçimi).

Çıktı: reports/model_catboost.md · submissions/sub_cat.csv + ensemble
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

from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # noqa: E402
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

# referans: LightGBM hurdle+optuna (scripts/20) yerel foldları
LGB = {"F1": 1.1148, "F2": 1.2401, "F3": 1.2482}
ALPHA, W_COLD = 0.4, 0.45
SEEDS = [0, 1, 2]
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]

MAIN = json.loads((ROOT / "data" / "feature-selection-results"
                   / "model_features.json").read_text(encoding="utf-8"))
_LVL_SEAS = set(FEATURE_GROUPS["lvl"]) | set(FEATURE_GROUPS["seas"])
COLD = [c for c in MAIN if c not in _LVL_SEAS]
CATS_MAIN = [c for c in CATEGORICAL_FEATURES if c in MAIN]
CATS_COLD = [c for c in CATEGORICAL_FEATURES if c in COLD]

REG_PARAMS = dict(iterations=700, learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
                  loss_function="RMSE", random_seed=SEED, verbose=0)
CLF_PARAMS = dict(iterations=500, learning_rate=0.05, depth=7, l2_leaf_reg=3.0,
                  loss_function="Logloss", random_seed=SEED, verbose=0)

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def assemble(base, dev, zero):
    return (base + ALPHA * dev + zero).to_numpy()


def prep(X, feats, cats):
    """CatBoost için: seçili feature'lar, kategorikler string (NaN→'NA').

    CatBoost kategorikte NaN kabul etmez; object'e çevirip NaN'ı 'NA' string yaparız
    (static_bolge, Manisa 2-parçalı lokasyonlarda boş)."""
    Z = X[feats].copy()
    for c in cats:
        col = Z[c].astype("object")
        Z[c] = col.where(pd.notna(col), "NA").astype(str)
    return Z


def fit_reg(X, y_resid, feats, cats, so):
    p = dict(REG_PARAMS); p["random_seed"] = SEED + so
    m = CatBoostRegressor(**p)
    m.fit(Pool(prep(X, feats, cats), label=y_resid, cat_features=cats))
    return m


def fit_clf(X, is_zero, feats, cats, so):
    p = dict(CLF_PARAMS); p["random_seed"] = SEED + so
    m = CatBoostClassifier(**p)
    m.fit(Pool(prep(X, feats, cats), label=is_zero.astype(int), cat_features=cats))
    return m


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model CatBoost hurdle — algoritma çeşitliliği (ensemble için)")
    w()
    w(f"Üretim: `scripts/21_catboost_hurdle.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w(f"- {len(MAIN)} feature · cold {len(COLD)} · kategorik {CATS_MAIN}")
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
        a_tr = assemble(meta["anc_base"], meta["anc_dev"], meta["anc_zero"])
        a_va = assemble(comp_va["base"], comp_va["season_dev"], comp_va["zero_adj"])
        cm = meta["is_cold_example"].to_numpy()
        vc_idx = valid.index[is_cold]
        b5c = b5_guc_lf(df.loc[fold["train_idx"]], vr[is_cold]).to_numpy()
        y_orig = meta["tuketim"].to_numpy()
        nz = y_orig > 0
        resid = y_tr.to_numpy() - a_tr           # anchor-residual hedef

        print(f"[{fn}] CatBoost classifier + nonzero regresör (3 seed) ...")
        p_list, warm_list, cold_list = [], [], []
        for so in SEEDS:
            bc = fit_clf(X_tr, pd.Series(~nz), MAIN, CATS_MAIN, so)
            p_list.append(bc.predict_proba(prep(X_va, MAIN, CATS_MAIN))[:, 1])
            br = fit_reg(X_tr[nz], resid[nz], MAIN, CATS_MAIN, so)
            warm_list.append(br.predict(prep(X_va, MAIN, CATS_MAIN)) + a_va)
            cnz = cm & nz
            bcold = fit_reg(X_tr[cnz], resid[cnz], COLD, CATS_COLD, so)
            cold_list.append(bcold.predict(prep(X_va.loc[vc_idx], COLD, CATS_COLD))
                             + a_va[is_cold])

        p = np.clip(np.mean(p_list, axis=0), 0, 1)
        reg_l = np.mean(warm_list, axis=0)
        cold_l = np.mean(cold_list, axis=0)
        aucs[fn] = roc_auc_score((vr["tuketim"].to_numpy() == 0).astype(int), p)

        pred = np.clip(np.expm1((1 - p) * reg_l), 0, None)
        cold_pred = np.clip(np.expm1((1 - p[is_cold]) * cold_l), 0, None)
        pred[is_cold] = np.expm1(W_COLD * np.log1p(cold_pred) + (1 - W_COLD) * np.log1p(b5c))

        vv = valid.copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
        ev = evaluate(vv, "tuketim", "_pred")
        g = lambda k, s: float(ev.loc[(ev["kirilim"] == k) & (ev["seviye"] == s), "rmsle"].iloc[0])
        scores[fn] = {
            "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
            "warm": g("warm_cold", "warm"), "cold": g("warm_cold", "cold")}

    w("## 1. CatBoost hurdle skorları (vs LightGBM hurdle+opt)")
    w()
    w("| fold | catboost | lgb hurdle+opt | Δ | warm | cold | AUC |")
    w("|---|---|---|---|---|---|---|")
    for fn in ["F1", "F2", "F3"]:
        s = scores[fn]; d = s["blend"] - LGB[fn]
        w(f"| {fn} | **{s['blend']:.4f}** | {LGB[fn]:.4f} | {d:+.4f} | "
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
    a_f = assemble(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"])
    a_t = assemble(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"])
    cold_f = metaf["is_cold_example"].to_numpy()
    is_cold_te = ~te["tanim"].isin(set(df["tanim"].unique())).to_numpy()
    b5t = b5_guc_lf(df, te[is_cold_te]).to_numpy()
    nzf = metaf["tuketim"].to_numpy() > 0
    residf = yf.to_numpy() - a_f

    p_list, warm_list, cold_list = [], [], []
    for so in SEEDS:
        bc = fit_clf(Xf, pd.Series(~nzf), MAIN, CATS_MAIN, so)
        p_list.append(bc.predict_proba(prep(Xt, MAIN, CATS_MAIN))[:, 1])
        br = fit_reg(Xf[nzf], residf[nzf], MAIN, CATS_MAIN, so)
        warm_list.append(br.predict(prep(Xt, MAIN, CATS_MAIN)) + a_t)
        cnz = cold_f & nzf
        bcold = fit_reg(Xf[cnz], residf[cnz], COLD, CATS_COLD, so)
        cold_list.append(bcold.predict(prep(Xt.loc[is_cold_te], COLD, CATS_COLD))
                         + a_t[is_cold_te])
    p = np.clip(np.mean(p_list, axis=0), 0, 1)
    pred = np.clip(np.expm1((1 - p) * np.mean(warm_list, axis=0)), 0, None)
    cold_pred = np.clip(np.expm1((1 - p[is_cold_te]) * np.mean(cold_list, axis=0)), 0, None)
    pred[is_cold_te] = np.expm1(W_COLD * np.log1p(cold_pred) + (1 - W_COLD) * np.log1p(b5t))

    sub = pd.DataFrame({"id": te["id"], "tuketim": pred})
    write_submission(sub, SUBMISSIONS_DIR / "sub_cat.csv")
    w("## 2. Submission + ensemble")
    w()
    w("- submissions/sub_cat.csv yazıldı.")

    # LightGBM hurdle+opt ile korelasyon + ensemble
    lgb_path = SUBMISSIONS_DIR / "sub_hurdle_opt.csv"
    if lgb_path.exists():
        lgbp = pd.read_csv(lgb_path)
        lc, cc = np.log1p(lgbp["tuketim"]), np.log1p(pred)
        corr = np.corrcoef(lc, cc)[0, 1]
        w(f"- CatBoost ↔ LightGBM(hurdle+opt) korelasyon: **{corr:.4f}** "
          f"({'düşük→ensemble güçlü' if corr < 0.97 else 'yüksek→ensemble marjinal'})")
        for wname, wc in [("50", 0.5), ("40", 0.4)]:
            ens = np.expm1(wc * cc + (1 - wc) * lc)
            pd.DataFrame({"id": te["id"], "tuketim": ens}).to_csv(
                SUBMISSIONS_DIR / f"sub_cat_lgb_{wname}.csv", index=False)
        w("- Ensemble: sub_cat_lgb_50 (50/50), sub_cat_lgb_40 (40cat/60lgb) yazıldı.")
    w()

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
    w("## 3. Kohort-eş kalibrasyon (CatBoost)")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")
    w()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_catboost.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_catboost.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_catboost.md").write_text(out.getvalue(), encoding="utf-8")
