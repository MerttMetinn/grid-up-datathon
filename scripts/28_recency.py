# -*- coding: utf-8 -*-
"""
28_recency.py — LEAK'SİZ recency weighting: son aylara ağırlık (2026-uyum).

Domain shift: 2026 tüketimi 2025'ten farklı. Model tüm geçmişe eşit ağırlık veriyor.
Son aylara (train_end'e yakın) daha çok ağırlık → model güncel rejime uyar.
Ağırlık = exp(-(train_end - tarih).days / halflife). Grid: halflife ∈ {90,180,365,∞}.

Hava YOK (leak'siz), optuna-nowx params temelli. F1/F2/F3 + full + submission.
Çıktı: reports/recency.md · submissions/sub_recency.csv
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

from src.config import (REPORTS_DIR, SEED, SUBMISSIONS_DIR, TRAIN_END,
                        YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.train import ORIGINS, align_categories, build_training_set  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

NOWX = [f for f in ALL_FEATURES if f not in FEATURE_GROUPS["wx"]]
CATS = [c for c in CATEGORICAL_FEATURES if c in NOWX]
ALPHA = 0.4
SEEDS = [0, 1, 2]
HALFLIVES = [90, 180, 365, None]     # None = ağırlıksız (referans)
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]
# leak'siz optuna (scripts/27) params
PARAMS = {"objective": "regression", "verbose": -1, "seed": SEED, "bagging_freq": 1,
          "learning_rate": 0.038141, "num_leaves": 31, "min_data_in_leaf": 156,
          "feature_fraction": 0.55605, "bagging_fraction": 0.64118,
          "lambda_l1": 0.0096444, "lambda_l2": 11.065}
ROUNDS = 400

out = io.StringIO()


def w(line=""):
    out.write(line + "\n"); print(line)


def asm(b, d, z):
    return (b + ALPHA * d + z).to_numpy()


def rec_weight(tarih, train_end, halflife):
    if halflife is None:
        return None
    days = (pd.Timestamp(train_end) - pd.to_datetime(tarih)).dt.days.to_numpy()
    return np.exp(-np.clip(days, 0, None) / halflife).astype("float64")


def train_pred(X_tr, y_tr, itr, X_va, iva, sw, so):
    p = dict(PARAMS); p["seed"] = SEED + so
    ds = lgb.Dataset(X_tr[NOWX], label=y_tr, init_score=itr, weight=sw,
                     categorical_feature=CATS)
    b = lgb.train(p, ds, num_boost_round=ROUNDS)
    return np.clip(np.expm1(b.predict(X_va[NOWX]) + iva), 0, None)


def main():
    df, te, profile = load_train(), load_test(), load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# LEAK'SİZ recency weighting — son aylara ağırlık")
    w()
    w(f"Üretim: `scripts/28_recency.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w(f"- {len(NOWX)} feature (hava YOK) · halflife grid {HALFLIVES}")
    w()

    # fold matrisleri (bir kez)
    fm = {}
    for i, fold in enumerate(folds):
        print(f"[{fold['name']}] matris ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile, i)
        vr = df.loc[fold["valid_idx"]]
        X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        comp = anchor_components(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
        fm[fold["name"]] = dict(
            X_tr=X_tr, y_tr=y_tr.to_numpy(), tarih=meta["tarih"],
            itr=asm(meta["anc_base"], meta["anc_dev"], meta["anc_zero"]),
            X_va=X_va, iva=asm(comp["base"], comp["season_dev"], comp["zero_adj"]),
            vr=vr, valid=add_eval_columns(vr, fold, df),
            train_end=fold["spec"]["train_end"])

    w("## 1. Halflife grid — fold blend skorları")
    w()
    w("| halflife | F1 | F2 | F3 |")
    w("|---|---|---|---|")
    results = {}
    for hl in HALFLIVES:
        row = {}
        for fn in ["F1", "F2", "F3"]:
            d = fm[fn]
            sw = rec_weight(d["tarih"], d["train_end"], hl)
            preds = [np.log1p(train_pred(d["X_tr"], d["y_tr"], d["itr"],
                     d["X_va"], d["iva"], sw, so)) for so in SEEDS]
            pred = np.expm1(np.mean(preds, axis=0))
            vv = d["valid"].copy(); vv["_pred"] = pd.Series(pred, index=vv.index)
            ev = evaluate(vv, "tuketim", "_pred")
            row[fn] = float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])
        results[hl] = row
        w(f"| {hl if hl else 'yok'} | {row['F1']:.4f} | {row['F2']:.4f} | {row['F3']:.4f} |")
    w()
    # F1'e göre en iyi (referansı da dahil)
    best_hl = min(HALFLIVES, key=lambda h: results[h]["F1"])
    w(f"- En iyi halflife (F1'e göre): **{best_hl if best_hl else 'yok (ağırlıksız)'}** "
      f"→ F1 {results[best_hl]['F1']:.4f} (ağırlıksız {results[None]['F1']:.4f})")
    w()

    # full + submission (en iyi halflife)
    print(f"[FULL] halflife={best_hl} ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    Xtest = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xtest])
    comp_t = anchor_components(te, TRAIN_END, df)
    a_f, a_t = asm(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"]), \
        asm(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"])
    sw = rec_weight(metaf["tarih"], TRAIN_END, best_hl)
    preds = [np.log1p(train_pred(Xf, yf.to_numpy(), a_f, Xtest, a_t, sw, so))
             for so in SEEDS]
    pred = np.clip(np.expm1(np.mean(preds, axis=0)), 0, None)
    pd.DataFrame({"id": te["id"], "tuketim": pred}).to_csv(
        SUBMISSIONS_DIR / "sub_recency.csv", index=False)

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
    w("## 2. sub_recency.csv — kalibrasyon")
    w()
    w("| Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|")
    w("| " + " | ".join(f"{d:+.3f}" for d in devs) + f" | {max(abs(x) for x in devs):.3f} |")
    w()
    w(f"- **sub_recency.csv yazıldı** (halflife={best_hl}). Leak'siz. "
      f"sub_nowx_lo LB 1.06525 referans.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "recency.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'recency.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "recency.md").write_text(out.getvalue(), encoding="utf-8")
