# -*- coding: utf-8 -*-
"""
19_train_hurdle.py — Ölü-trafo cephesi: hurdle (iki-aşamalı) model.

Hatanın %57'si gerçek-sıfır satırlardan (ölü/kapalı trafolar). Tek model "sıfır mı"
ve "ne kadar"ı karıştırıyor. Ayırıyoruz:
  classifier p = P(tuketim=0)               (LightGBM binary, tüm satır)
  regresör   L = E[log1p | tuketim>0]       (LightGBM regression, YALNIZ nonzero, init=anchor)
  tahmin     = expm1((1-p) * L)             (log-L2'nin optimumu)

Cold satırlarda ayrıca b5 harmanı (s2 ile aynı, w=0.45).
Referans: s2+wx (scripts/16 t1): F1 1.1447 · F2 1.2486 · F3 1.2759.
Çıktı: reports/model_hurdle.md · (kazanırsa) submissions/sub_hurdle.csv
"""
import io
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
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.predict import write_submission  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

B6 = {"F1": 1.2692, "F2": 1.2654, "F3": 1.3055}
S2WX = {"F1": 1.1447, "F2": 1.2486, "F3": 1.2759}   # scripts/16 t1 referans
ALPHA, W_COLD = 0.4, 0.45
SEEDS = [0, 1, 2]
RM, RC = 126, 73
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]
MAIN = ALL_FEATURES
COLD = COLD_MODEL_FEATURES + FEATURE_GROUPS["wx"]

CLF_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
              "seed": SEED}
CLF_ROUNDS = 300

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def assemble(base, dev, zero):
    return (base + ALPHA * dev + zero).to_numpy()


def fit_reg(X, y, feats, init, rounds, so):
    p = dict(LGB_PARAMS)
    for k in ("seed", "feature_fraction_seed", "bagging_seed"):
        p[k] = LGB_PARAMS[k] + so
    ds = lgb.Dataset(X[feats], label=y, init_score=init,
                     categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feats])
    return lgb.train(p, ds, num_boost_round=rounds)


def fit_clf(X, is_zero, feats, so):
    p = dict(CLF_PARAMS)
    for k in ("seed",):
        p[k] = CLF_PARAMS[k] + so
    ds = lgb.Dataset(X[feats], label=is_zero.astype(int),
                     categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feats])
    return lgb.train(p, ds, num_boost_round=CLF_ROUNDS)


def reg_log(b, X, feats, init):
    return b.predict(X[feats]) + init


def logmean(ps):
    return np.mean(ps, axis=0)      # log uzayında (girdi zaten log)


def logblend(pm, pb, wg):
    return np.expm1(wg * np.log1p(pm) + (1 - wg) * np.log1p(pb))


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model hurdle — ölü-trafo iki-aşamalı model")
    w()
    w(f"Üretim: `scripts/19_train_hurdle.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
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

        # hedefler
        y_orig = meta["tuketim"].to_numpy()
        nz = y_orig > 0                       # nonzero eğitim satırları
        is_zero_tr = ~nz

        print(f"[{fn}] classifier + nonzero regresör (3 seed) ...")
        p_list, warm_list, cold_list = [], [], []
        for so in SEEDS:
            # classifier (tüm satır)
            bc = fit_clf(X_tr, pd.Series(is_zero_tr), MAIN, so)
            p_va = bc.predict(X_va[MAIN])
            p_list.append(p_va)
            # nonzero regresör (warm+cold, y>0), init=anchor
            br = fit_reg(X_tr[nz], y_tr[nz], MAIN, itr[nz], RM, so)
            warm_list.append(reg_log(br, X_va, MAIN, iva))
            # cold nonzero regresör
            cnz = cm & nz
            bcold = fit_reg(X_tr[cnz], y_tr[cnz], COLD, itr[cnz], RC, so)
            cold_list.append(reg_log(bcold, X_va.loc[vc_idx], COLD, iva[is_cold]))

        p = np.clip(np.mean(p_list, axis=0), 0, 1)
        reg_l = logmean(warm_list)                    # L (log1p)
        cold_l = logmean(cold_list)
        aucs[fn] = roc_auc_score((vr["tuketim"].to_numpy() == 0).astype(int), p)

        # hurdle birleştirme: expm1((1-p)*L)
        pred = np.clip(np.expm1((1 - p) * reg_l), 0, None)
        p_cold = p[is_cold]
        cold_pred = np.clip(np.expm1((1 - p_cold) * cold_l), 0, None)
        pred[is_cold] = logblend(cold_pred, b5c, W_COLD)

        vv = valid.copy()
        vv["_pred"] = pd.Series(pred, index=vv.index)
        ev = evaluate(vv, "tuketim", "_pred")
        e2 = (np.log1p(vv["_pred"].clip(0)) - np.log1p(vv["tuketim"])) ** 2
        nzv = vv["tuketim"] > 0
        g = lambda k, s: float(ev.loc[(ev["kirilim"] == k) & (ev["seviye"] == s), "rmsle"].iloc[0])
        scores[fn] = {
            "all": float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0]),
            "warm": g("warm_cold", "warm"), "cold": g("warm_cold", "cold"),
            "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
            "nz_warm": float(np.sqrt(e2[nzv & ~vv["is_cold"]].mean())),
            "nz_cold": float(np.sqrt(e2[nzv & vv["is_cold"]].mean()))}

    # ---- skorlar ----
    w("## 1. Skorlar — hurdle vs s2+wx referans")
    w()
    w("| fold | hurdle blend | s2+wx | Δ | warm | cold | AUC(zero) |")
    w("|---|---|---|---|---|---|---|")
    for fn in ["F1", "F2", "F3"]:
        s = scores[fn]
        d = s["blend"] - S2WX[fn]
        w(f"| {fn} | **{s['blend']:.4f}** | {S2WX[fn]:.4f} | {d:+.4f} | "
          f"{s['warm']:.4f} | {s['cold']:.4f} | {aucs[fn]:.3f} |")
    w()
    w(f"- Önceki sıfır AUC (statik hücre, reports/diagnosis): 0.56 → hurdle classifier: "
      f"{np.mean(list(aucs.values())):.3f} ort")
    w()

    # ---- tam eğitim + kalibrasyon + submission ----
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
    yf_orig = metaf["tuketim"].to_numpy()
    nzf = yf_orig > 0

    p_list, warm_list, cold_list = [], [], []
    for so in SEEDS:
        bc = fit_clf(Xf, pd.Series(~nzf), MAIN, so)
        p_list.append(bc.predict(Xt[MAIN]))
        br = fit_reg(Xf[nzf], yf[nzf], MAIN, itr[nzf], RM, so)
        warm_list.append(reg_log(br, Xt, MAIN, itt))
        cnz = cold_f & nzf
        bcold = fit_reg(Xf[cnz], yf[cnz], COLD, itr[cnz], RC, so)
        cold_list.append(reg_log(bcold, Xt.loc[is_cold_te], COLD, itt[is_cold_te]))
    p = np.clip(np.mean(p_list, axis=0), 0, 1)
    reg_l = logmean(warm_list)
    pred = np.clip(np.expm1((1 - p) * reg_l), 0, None)
    cold_pred = np.clip(np.expm1((1 - p[is_cold_te]) * logmean(cold_list)), 0, None)
    pred[is_cold_te] = logblend(cold_pred, b5t, W_COLD)

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
    w("## 2. Kohort-eş aylık kalibrasyon (hurdle, 3-seed)")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")
    w()

    # karar
    d_f1 = scores["F1"]["blend"] - S2WX["F1"]
    d_avg = np.mean([scores[f]["blend"] - S2WX[f] for f in ["F1", "F2", "F3"]])
    wins = d_f1 < -0.002 and d_avg < 0
    w("## 3. Karar")
    w()
    w(f"- F1 Δ: {d_f1:+.4f} · 3-fold ort Δ: {d_avg:+.4f}")
    w(f"- **SONUÇ: {'hurdle KABUL — submission üretildi' if wins else 'hurdle marjinal — karar kullanıcıya'}**")
    w()
    if wins:
        sub = pd.DataFrame({"id": te["id"], "tuketim": pred})
        write_submission(sub, SUBMISSIONS_DIR / "sub_hurdle.csv")
        w("- submissions/sub_hurdle.csv yazıldı.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_hurdle.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_hurdle.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_hurdle.md").write_text(out.getvalue(), encoding="utf-8")
