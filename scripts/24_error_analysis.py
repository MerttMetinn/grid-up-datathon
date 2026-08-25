# -*- coding: utf-8 -*-
"""
24_error_analysis.py — Gap'in kaynağını bul. Kör model yerine teşhis.

En iyi modelimiz (hurdle+optuna) F1 valid'de nerede en çok kaybediyor?
Kareli-log hatasını kırılımlara göre topla: warm/cold, sıfır/nonzero, H_bucket,
ay, ilçe, guc_bucket, zero_streak. En yüksek hata payına sahip kesim = hedef.

Çıktı: reports/error_analysis.md
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
from src.config import REPORTS_DIR, SEED  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.features import (CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.train import align_categories, build_training_set  # noqa: E402
from src.validation import add_eval_columns, make_folds  # noqa: E402

ALPHA, W_COLD = 0.4, 0.70
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

out = io.StringIO()


def w(line=""):
    out.write(line + "\n"); print(line)


def main():
    df, profile = load_train(), load_profile()
    fold = make_folds(df, profile, seed=SEED)[0]     # F1 birincil
    X_tr, y_tr, meta = build_training_set(df, fold, profile, 0)
    vr = df.loc[fold["valid_idx"]]
    X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    align_categories([X_tr, X_va])
    valid = add_eval_columns(vr, fold, df)
    ic = valid["is_cold"].to_numpy()
    comp = anchor_components(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
    a_tr = (meta["anc_base"] + ALPHA * meta["anc_dev"] + meta["anc_zero"]).to_numpy()
    a_va = (comp["base"] + ALPHA * comp["season_dev"] + comp["zero_adj"]).to_numpy()
    cm = meta["is_cold_example"].to_numpy()
    b5c = b5_guc_lf(df.loc[fold["train_idx"]], vr[ic]).to_numpy()
    nz = meta["tuketim"].to_numpy() > 0
    cats = [c for c in CATEGORICAL_FEATURES if c in MAIN]

    print("egitim ...")
    bc = lgb.train(CLF, lgb.Dataset(X_tr[MAIN], label=(~nz).astype(int),
                   categorical_feature=cats), num_boost_round=300)
    p = np.clip(bc.predict(X_va[MAIN]), 0, 1)
    br = lgb.train(OPT, lgb.Dataset(X_tr[nz][MAIN], label=y_tr[nz], init_score=a_tr[nz],
                   categorical_feature=cats), num_boost_round=340)
    reg_l = br.predict(X_va[MAIN]) + a_va
    cnz = cm & nz
    cats_c = [c for c in CATEGORICAL_FEATURES if c in COLD]
    bcold = lgb.train(OPT, lgb.Dataset(X_tr[cnz][COLD], label=y_tr[cnz],
                      init_score=a_tr[cnz], categorical_feature=cats_c),
                      num_boost_round=200)
    cold_l = bcold.predict(X_va.loc[valid.index[ic]][COLD]) + a_va[ic]

    pred = np.clip(np.expm1((1 - p) * reg_l), 0, None)
    cp = np.clip(np.expm1((1 - p[ic]) * cold_l), 0, None)
    pred[ic] = np.expm1(W_COLD * np.log1p(cp) + (1 - W_COLD) * np.log1p(b5c))

    y = vr["tuketim"].to_numpy()
    e2 = (np.log1p(pred) - np.log1p(y)) ** 2
    tot = e2.sum()

    w("# Hata Analizi — gap nerede? (F1 valid, en iyi model)")
    w()
    w(f"Üretim: `scripts/24_error_analysis.py` · {datetime.now():%Y-%m-%d %H:%M}")
    w(f"- Toplam RMSLE: {np.sqrt(e2.mean()):.4f} · satır {len(y):,}")
    w()

    def breakdown(name, keys):
        g = pd.DataFrame({"e2": e2, "k": keys})
        agg = g.groupby("k", observed=True)["e2"].agg(["sum", "mean", "size"])
        agg["rmsle"] = np.sqrt(agg["mean"])
        agg["hata_payi_%"] = 100 * agg["sum"] / tot
        agg = agg.sort_values("sum", ascending=False)
        w(f"## {name}")
        w()
        w("| kesim | satır | pay% | RMSLE | **hata payı%** |")
        w("|---|---|---|---|---|")
        for k, r in agg.head(12).iterrows():
            w(f"| {k} | {int(r['size']):,} | {100*r['size']/len(y):.1f} | "
              f"{r['rmsle']:.3f} | **{r['hata_payi_%']:.1f}** |")
        w()

    breakdown("1. warm / cold", np.where(ic, "cold", "warm"))
    breakdown("2. sıfır / nonzero", np.where(y == 0, "gercek=0", "gercek>0"))
    seg = np.where(ic & (y == 0), "cold+0",
          np.where(ic & (y > 0), "cold+pozitif",
          np.where(~ic & (y == 0), "warm+0", "warm+pozitif")))
    breakdown("3. warm/cold × sıfır", seg)
    breakdown("4. ay", vr["tarih"].dt.month.astype(str).to_numpy())
    breakdown("5. guc_bucket", vr["guc_bucket"].astype(str).to_numpy())
    breakdown("6. ilçe (ilk 12 hata payı)", vr["ilce_key"].astype(str).to_numpy())
    breakdown("7. H_bucket", valid["H_bucket"].astype(str).to_numpy())
    breakdown("8. zero_streak durumu", valid["zero_streak_bucket"].astype(str).to_numpy())

    # en kötü 20 satır
    w("## 9. En yüksek hatalı 20 satır")
    w()
    idx = np.argsort(-e2)[:20]
    w("| gerçek | tahmin | log-hata² | cold | ay | ilçe |")
    w("|---|---|---|---|---|---|")
    for i in idx:
        w(f"| {y[i]:,.0f} | {pred[i]:,.0f} | {e2[i]:.1f} | "
          f"{'C' if ic[i] else 'W'} | {vr['tarih'].dt.month.iloc[i]} | "
          f"{str(vr['ilce_key'].iloc[i]).split('>')[-1]} |")
    w()
    w("> Yorum: en yüksek 'hata payı%' olan kesim, gap'i kapatmak için hedef bölge.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "error_analysis.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "error_analysis.md").write_text(out.getvalue(), encoding="utf-8")
