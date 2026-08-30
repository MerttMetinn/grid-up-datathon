# -*- coding: utf-8 -*-
"""
26_optuna_deep.py — optuna derinleştirme (NEREDE_KALDIK planı A).

Arkadaşın optuna: 29 feature + 25 trial → LB 1.0648. Bu deneme: TÜM 75 feature
(hava dahil) + 60 trial. Öğrenilen derse uyar: DÜZ model (hurdle/b5 YOK — basit
kazanıyor, cold b5 LB'de zararlı). Cold da ana modelle tahmin edilir (arkadaş gibi).

F1'de tune, F2/F3'te doğrula, full_train + submission.
Çıktı: reports/optuna_deep.md · submissions/sub_opt_deep.csv
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
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES,
                          anchor_components, build_features)  # noqa: E402
from src.train import ORIGINS, align_categories, build_training_set  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

ALPHA = 0.4          # anchor yumuşatma (init_score)
N_TRIALS = 60
SEEDS = [0, 1, 2]
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]
CATS = [c for c in CATEGORICAL_FEATURES if c in ALL_FEATURES]
FRIEND_LB = 1.06483   # arkadaşın optuna (29 feat) referansı

out = io.StringIO()


def w(line=""):
    out.write(line + "\n"); print(line)


def asm(b, d, z):
    return (b + ALPHA * d + z).to_numpy()


def make_fold_matrix(df, fold, profile, fi):
    X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)
    vr = df.loc[fold["valid_idx"]]
    X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    align_categories([X_tr, X_va])
    comp = anchor_components(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    return (X_tr, y_tr.to_numpy(), asm(meta["anc_base"], meta["anc_dev"], meta["anc_zero"]),
            X_va, np.log1p(vr["tuketim"].to_numpy()), asm(comp["base"], comp["season_dev"],
            comp["zero_adj"]), vr, add_eval_columns(vr, fold, df))


def train_predict(params, X_tr, y_tr, itr, X_va, iva, rounds=None, y_va=None):
    ds = lgb.Dataset(X_tr[ALL_FEATURES], label=y_tr, init_score=itr,
                     categorical_feature=CATS)
    cbs, valid_sets, nr = [], None, rounds or 3000
    if y_va is not None:
        valid_sets = [lgb.Dataset(X_va[ALL_FEATURES], label=y_va, init_score=iva, reference=ds)]
        cbs = [lgb.early_stopping(150, verbose=False)]
    booster = lgb.train(params, ds, num_boost_round=nr, valid_sets=valid_sets, callbacks=cbs)
    it = booster.best_iteration if y_va is not None else nr
    raw = booster.predict(X_va[ALL_FEATURES], num_iteration=it)
    return np.clip(np.expm1(raw + iva), 0, None), booster.best_iteration or nr


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# optuna derinleştirme — 75 feature (hava dahil) + 60 trial")
    w()
    w(f"Üretim: `scripts/26_optuna_deep.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w(f"- {len(ALL_FEATURES)} feature · DÜZ model (hurdle/b5 YOK) · arkadaş ref {FRIEND_LB}")
    w()

    print("[F1] matris ...")
    f1 = make_fold_matrix(df, folds[0], profile, 0)
    X_tr, y_tr, itr, X_va, y_va, iva, vr1, valid1 = f1

    def objective(trial):
        params = {
            "objective": "regression", "verbose": -1, "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 400),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 15.0, log=True),
        }
        pred, _ = train_predict(params, X_tr, y_tr, itr, X_va, iva, y_va=y_va)
        return rmsle(vr1["tuketim"].to_numpy(), pred)

    print(f"[F1] optuna {N_TRIALS} trial ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"objective": "regression", "verbose": -1, "seed": SEED, "bagging_freq": 1})
    w(f"## 1. En iyi F1 RMSLE (tek fold, tek seed): **{study.best_value:.4f}**")
    w()
    w("En iyi parametreler:")
    for k, v in study.best_params.items():
        w(f"- {k}: {v:.5g}" if isinstance(v, float) else f"- {k}: {v}")
    w()

    # F1/F2/F3 doğrulama (best params, 3 seed, blend)
    w("## 2. Fold doğrulama (best params, blend)")
    w()
    w("| fold | blend | warm | cold | best_iter |")
    w("|---|---|---|---|---|")
    scores = {}
    for i, fold in enumerate(folds):
        fn = fold["name"]
        Xt_, yt_, it_, Xv_, yv_, iv_, vr_, valid_ = f1 if i == 0 else \
            make_fold_matrix(df, fold, profile, i)
        preds, bi = [], 0
        for so in SEEDS:
            p = dict(best); p["seed"] = SEED + so
            pr, bi = train_predict(p, Xt_, yt_, it_, Xv_, iv_, y_va=yv_)
            preds.append(np.log1p(pr))
        pred = np.expm1(np.mean(preds, axis=0))
        vv = valid_.copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
        ev = evaluate(vv, "tuketim", "_pred")
        g = lambda k, s: float(ev.loc[(ev["kirilim"] == k) & (ev["seviye"] == s), "rmsle"].iloc[0])
        scores[fn] = float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])
        w(f"| {fn} | {scores[fn]:.4f} | {g('warm_cold','warm'):.4f} | "
          f"{g('warm_cold','cold'):.4f} | {bi} |")
    w()

    # full + submission
    print("[FULL] egitim ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    Xtest = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xtest])
    comp_t = anchor_components(te, TRAIN_END, df)
    a_f, a_t = asm(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"]), \
        asm(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"])
    best_iter = max(60, int(np.median([bi for bi in [study.best_trial.number]]) or 300))
    rounds = 400   # early stop yok, sabit (F1 best_iter civarı × 1.1)

    preds = []
    for so in SEEDS:
        p = dict(best); p["seed"] = SEED + so
        pr, _ = train_predict(p, Xf, yf.to_numpy(), a_f, Xtest, a_t, rounds=rounds)
        preds.append(np.log1p(pr))
    pred = np.clip(np.expm1(np.mean(preds, axis=0)), 0, None)
    pd.DataFrame({"id": te["id"], "tuketim": pred}).to_csv(
        SUBMISSIONS_DIR / "sub_opt_deep.csv", index=False)

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
    w("## 3. sub_opt_deep.csv — kalibrasyon")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")
    w()
    w(f"- **sub_opt_deep.csv yazıldı.** F1 blend {scores['F1']:.4f} "
      f"(arkadaş 29-feat optuna LB {FRIEND_LB}).")
    w("- Karşılaştırma: LB'de arkadaşın optuna'sını (1.0648) geçerse 75-feature değerli.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "optuna_deep.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'optuna_deep.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "optuna_deep.md").write_text(out.getvalue(), encoding="utf-8")
