# -*- coding: utf-8 -*-
"""
27_optuna_nowx.py — LEAK'SİZ model: gerçekleşmiş hava (wx_) KULLANMADAN.

Host uyarısı: Nisan–Temmuz 2026 gerçekleşmiş hava verisi kullanımı "uygun olmayacak"
(forward leak). En iyi modelimiz (optuna, wx içerir) risk altında. Bu script hava
OLMADAN en iyi modeli kurar — diskalifiye riski sıfır, notebook'ta şeffaf.

Feature: 75 - 17 wx = leak'siz set. DÜZ model + anchor init_score. optuna 60 trial.
Çıktı: reports/optuna_nowx.md · submissions/sub_nowx.csv
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
import optuna  # noqa: E402

from src.config import (REPORTS_DIR, SEED, SUBMISSIONS_DIR, TRAIN_END,
                        YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.train import ORIGINS, align_categories, build_training_set  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

# LEAK'SİZ feature seti: hava (wx_) çıkarıldı
NOWX = [f for f in ALL_FEATURES if f not in FEATURE_GROUPS["wx"]]
CATS = [c for c in CATEGORICAL_FEATURES if c in NOWX]
ALPHA = 0.4
N_TRIALS = 60
SEEDS = [0, 1, 2]
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]

out = io.StringIO()


def w(line=""):
    out.write(line + "\n"); print(line)


def asm(b, d, z):
    return (b + ALPHA * d + z).to_numpy()


def fold_matrix(df, fold, profile, fi):
    X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)
    vr = df.loc[fold["valid_idx"]]
    X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    align_categories([X_tr, X_va])
    comp = anchor_components(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    return (X_tr, y_tr.to_numpy(), asm(meta["anc_base"], meta["anc_dev"], meta["anc_zero"]),
            X_va, np.log1p(vr["tuketim"].to_numpy()),
            asm(comp["base"], comp["season_dev"], comp["zero_adj"]),
            vr, add_eval_columns(vr, fold, df))


def train_predict(params, X_tr, y_tr, itr, X_va, iva, rounds=None, y_va=None):
    ds = lgb.Dataset(X_tr[NOWX], label=y_tr, init_score=itr, categorical_feature=CATS)
    cbs, valid_sets, nr = [], None, rounds or 3000
    if y_va is not None:
        valid_sets = [lgb.Dataset(X_va[NOWX], label=y_va, init_score=iva, reference=ds)]
        cbs = [lgb.early_stopping(150, verbose=False)]
    b = lgb.train(params, ds, num_boost_round=nr, valid_sets=valid_sets, callbacks=cbs)
    it = b.best_iteration if y_va is not None else nr
    return np.clip(np.expm1(b.predict(X_va[NOWX], num_iteration=it) + iva), 0, None), \
        (b.best_iteration or nr)


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# LEAK'SİZ model — hava (wx_) KULLANMADAN")
    w()
    w(f"Üretim: `scripts/27_optuna_nowx.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w(f"- {len(NOWX)} feature (hava çıkarıldı) · DÜZ model + anchor")
    w(f"- Referans: wx'li optuna LB 1.0648 (ama leak riski)")
    w()

    print("[F1] matris ...")
    f1 = fold_matrix(df, folds[0], profile, 0)
    X_tr, y_tr, itr, X_va, y_va, iva, vr1, valid1 = f1

    def objective(trial):
        params = {
            "objective": "regression", "verbose": -1, "seed": SEED, "bagging_freq": 1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 400),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 15.0, log=True),
        }
        pred, _ = train_predict(params, X_tr, y_tr, itr, X_va, iva, y_va=y_va)
        return rmsle(vr1["tuketim"].to_numpy(), pred)

    print(f"[F1] optuna {N_TRIALS} trial ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    best = dict(study.best_params)
    best.update({"objective": "regression", "verbose": -1, "seed": SEED, "bagging_freq": 1})
    w(f"## 1. En iyi F1 RMSLE: **{study.best_value:.4f}** (wx'li deep optuna: 1.1079)")
    w()
    for k, v in study.best_params.items():
        w(f"- {k}: {v:.5g}" if isinstance(v, float) else f"- {k}: {v}")
    w()

    w("## 2. Fold doğrulama (best params, blend)")
    w()
    w("| fold | blend | warm | cold |")
    w("|---|---|---|---|")
    scores = {}
    for i, fold in enumerate(folds):
        fn = fold["name"]
        Xt_, yt_, it_, Xv_, yv_, iv_, vr_, valid_ = f1 if i == 0 else fold_matrix(df, fold, profile, i)
        preds = []
        for so in SEEDS:
            p = dict(best); p["seed"] = SEED + so
            pr, _ = train_predict(p, Xt_, yt_, it_, Xv_, iv_, y_va=yv_)
            preds.append(np.log1p(pr))
        pred = np.expm1(np.mean(preds, axis=0))
        vv = valid_.copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
        ev = evaluate(vv, "tuketim", "_pred")
        g = lambda k, s: float(ev.loc[(ev["kirilim"] == k) & (ev["seviye"] == s), "rmsle"].iloc[0])
        scores[fn] = float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])
        w(f"| {fn} | {scores[fn]:.4f} | {g('warm_cold','warm'):.4f} | {g('warm_cold','cold'):.4f} |")
    w()

    print("[FULL] egitim ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    Xtest = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xtest])
    comp_t = anchor_components(te, TRAIN_END, df)
    a_f, a_t = asm(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"]), \
        asm(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"])
    preds = []
    for so in SEEDS:
        p = dict(best); p["seed"] = SEED + so
        pr, _ = train_predict(p, Xf, yf.to_numpy(), a_f, Xtest, a_t, rounds=400)
        preds.append(np.log1p(pr))
    pred = np.clip(np.expm1(np.mean(preds, axis=0)), 0, None)
    pd.DataFrame({"id": te["id"], "tuketim": pred}).to_csv(
        SUBMISSIONS_DIR / "sub_nowx.csv", index=False)

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
    w("## 3. sub_nowx.csv (LEAK'SİZ) — kalibrasyon")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")
    w()
    w(f"- **sub_nowx.csv yazıldı** — hava YOK, diskalifiye riski sıfır.")
    w(f"- F1 blend {scores['F1']:.4f} (wx'li ~1.113). Hava kaybının maliyeti = fark.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "optuna_nowx.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'optuna_nowx.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "optuna_nowx.md").write_text(out.getvalue(), encoding="utf-8")
