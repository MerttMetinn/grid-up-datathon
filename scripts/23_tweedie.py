# -*- coding: utf-8 -*-
"""
23_tweedie.py — Tweedie loss: kökten farklı yaklaşım, gerçek ensemble çeşitliliği.

Tüketim sıfır-şişkin + sağa çarpık. Tweedie loss bu dağılım için doğal — sıfır ve
pozitif kuyruğu TEK modelde ele alır (hurdle classifier+regresör ayrımı gereksiz).
Farklı loss → farklı hata → düşük korelasyon → anlamlı ensemble.

Anchor init_score DEĞİL, FEATURE olarak verilir (çeşitliliği bastırmasın).
Hedef: orijinal ölçek tuketim (Tweedie log-link kendi expm1'ini yapar).
29 feature + 3 anchor bileşeni. Cold'da b5 harmanı (W=0.70, scripts/22 sonucu).

Çıktı: reports/model_tweedie.md · submissions/sub_tw.csv + ensemble
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
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

W_COLD = 0.70          # scripts/22: b5 ağırlığı 0.70 en iyi
SEEDS = [0, 1, 2]
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]
BASE = json.loads((ROOT / "data" / "feature-selection-results"
                   / "model_features.json").read_text(encoding="utf-8"))
# anchor bileşenlerini feature olarak ekle
ANCHOR_FEATS = ["anc_base", "anc_dev", "anc_zero"]
MAIN = BASE + ANCHOR_FEATS
_LS = set(FEATURE_GROUPS["lvl"]) | set(FEATURE_GROUPS["seas"])
COLD = [c for c in BASE if c not in _LS] + ANCHOR_FEATS
CATS = [c for c in CATEGORICAL_FEATURES if c in MAIN]
CATS_C = [c for c in CATEGORICAL_FEATURES if c in COLD]

# LB referansı: mevcut plato ~1.065; hurdle+opt yerel foldları
REF = {"F1": 1.1148, "F2": 1.2401, "F3": 1.2482}
TW_PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.2,
             "verbose": -1, "learning_rate": 0.03, "num_leaves": 100,
             "min_data_in_leaf": 150, "feature_fraction": 0.7,
             "bagging_fraction": 0.75, "bagging_freq": 1, "lambda_l2": 2.0,
             "seed": SEED}
ROUNDS = 500

out = io.StringIO()


def w(line=""):
    out.write(line + "\n"); print(line)


def add_anchor(X, comp):
    X = X.copy()
    X["anc_base"] = comp["base"].to_numpy().astype("float32")
    X["anc_dev"] = comp["season_dev"].to_numpy().astype("float32")
    X["anc_zero"] = comp["zero_adj"].to_numpy().astype("float32")
    return X


def add_anchor_meta(X, meta):
    X = X.copy()
    X["anc_base"] = meta["anc_base"].to_numpy().astype("float32")
    X["anc_dev"] = meta["anc_dev"].to_numpy().astype("float32")
    X["anc_zero"] = meta["anc_zero"].to_numpy().astype("float32")
    return X


def fit_tw(X, y_orig, feats, cats, so):
    p = dict(TW_PARAMS); p["seed"] = SEED + so
    ds = lgb.Dataset(X[feats], label=y_orig,
                     categorical_feature=[c for c in cats if c in feats])
    return lgb.train(p, ds, num_boost_round=ROUNDS)


def blend(pm, pb, wg):
    return np.expm1(wg * np.log1p(pm) + (1 - wg) * np.log1p(pb))


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model Tweedie — sıfır-şişkin loss, ensemble çeşitliliği")
    w()
    w(f"Üretim: `scripts/23_tweedie.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w(f"- {len(MAIN)} feature (29 + 3 anchor) · Tweedie var_power={TW_PARAMS['tweedie_variance_power']}")
    w()

    scores = {}
    for fold in folds:
        fn = fold["name"]
        print(f"[{fn}] egitim ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile,
                                              {"F1": 0, "F2": 1, "F3": 2}[fn])
        X_tr = add_anchor_meta(X_tr, meta)
        vr = df.loc[fold["valid_idx"]]
        X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
        comp = anchor_components(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
        X_va = add_anchor(X_va, comp)
        align_categories([X_tr, X_va])
        valid = add_eval_columns(vr, fold, df)
        ic = valid["is_cold"].to_numpy()
        cm = meta["is_cold_example"].to_numpy()
        vci = valid.index[ic]
        b5c = b5_guc_lf(df.loc[fold["train_idx"]], vr[ic]).to_numpy()
        y_orig = meta["tuketim"].to_numpy()   # ORİJİNAL ölçek (Tweedie)

        warm_l, cold_l = [], []
        for so in SEEDS:
            bm = fit_tw(X_tr, y_orig, MAIN, CATS, so)
            warm_l.append(np.clip(bm.predict(X_va[MAIN]), 0, None))
            cnz = cm
            bcold = fit_tw(X_tr[cnz], y_orig[cnz], COLD, CATS_C, so)
            cold_l.append(np.clip(bcold.predict(X_va.loc[vci][COLD]), 0, None))
        pred = np.mean(warm_l, axis=0)
        cold_pred = np.mean(cold_l, axis=0)
        pred[ic] = blend(cold_pred, b5c, W_COLD)

        vv = valid.copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
        ev = evaluate(vv, "tuketim", "_pred")
        g = lambda k, s: float(ev.loc[(ev["kirilim"] == k) & (ev["seviye"] == s), "rmsle"].iloc[0])
        scores[fn] = {
            "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
            "warm": g("warm_cold", "warm"), "cold": g("warm_cold", "cold")}

    w("## 1. Tweedie skorları (vs LightGBM hurdle+opt)")
    w()
    w("| fold | tweedie | hurdle+opt | Δ | warm | cold |")
    w("|---|---|---|---|---|---|")
    for fn in ["F1", "F2", "F3"]:
        s = scores[fn]; d = s["blend"] - REF[fn]
        w(f"| {fn} | **{s['blend']:.4f}** | {REF[fn]:.4f} | {d:+.4f} | {s['warm']:.4f} | {s['cold']:.4f} |")
    w()

    # tam eğitim + submission + ensemble
    print("[FULL] egitim ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    Xf = add_anchor_meta(Xf, metaf)
    Xt = build_features(te, TRAIN_END, df)
    comp_t = anchor_components(te, TRAIN_END, df)
    Xt = add_anchor(Xt, comp_t)
    align_categories([Xf, Xt])
    cold_f = metaf["is_cold_example"].to_numpy()
    ict = ~te["tanim"].isin(set(df["tanim"].unique())).to_numpy()
    b5t = b5_guc_lf(df, te[ict]).to_numpy()
    yf_orig = metaf["tuketim"].to_numpy()

    warm_l, cold_l = [], []
    for so in SEEDS:
        bm = fit_tw(Xf, yf_orig, MAIN, CATS, so)
        warm_l.append(np.clip(bm.predict(Xt[MAIN]), 0, None))
        bcold = fit_tw(Xf[cold_f], yf_orig[cold_f], COLD, CATS_C, so)
        cold_l.append(np.clip(bcold.predict(Xt.loc[ict][COLD]), 0, None))
    pred = np.mean(warm_l, axis=0)
    pred[ict] = blend(np.mean(cold_l, axis=0), b5t, W_COLD)

    sub = pd.DataFrame({"id": te["id"], "tuketim": pred})
    sub.to_csv(SUBMISSIONS_DIR / "sub_tw.csv", index=False)
    w("## 2. Submission + ensemble (KRİTİK: korelasyon)")
    w()
    w("- submissions/sub_tw.csv yazıldı.")

    # korelasyon: en iyi LightGBM (optuna) ile
    opt = pd.read_csv(SUBMISSIONS_DIR / "sub_s2_optuna.csv")
    lo, lt = np.log1p(opt["tuketim"]), np.log1p(pred)
    corr = np.corrcoef(lo, lt)[0, 1]
    w(f"- Tweedie ↔ optuna(LGB) korelasyon: **{corr:.4f}** "
      f"({'DÜŞÜK → ensemble GÜÇLÜ!' if corr < 0.95 else 'yüksek → marjinal'})")
    for wn, wt in [("30", 0.3), ("40", 0.4), ("50", 0.5)]:
        ens = np.expm1(wt * lt + (1 - wt) * lo)
        pd.DataFrame({"id": te["id"], "tuketim": ens}).to_csv(
            SUBMISSIONS_DIR / f"sub_tw_opt_{wn}.csv", index=False)
    w("- Ensemble: sub_tw_opt_30/40/50 (tweedie ağırlığı) yazıldı.")
    w()

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
    w("## 3. Kalibrasyon (Tweedie)")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_tweedie.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_tweedie.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_tweedie.md").write_text(out.getvalue(), encoding="utf-8")
