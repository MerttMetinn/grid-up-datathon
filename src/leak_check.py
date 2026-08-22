# -*- coding: utf-8 -*-
"""Feature sızıntı/sağlık kontrolü.

Her feature için:
  - train/valid dağılım kayması (ortalama farkı, train std birimiyle)
  - valid'de NaN oranı
  - tek-feature LightGBM F1 RMSLE — b6'yı 0.05+ geçen feature ŞÜPHELİ
"""
import numpy as np
import pandas as pd

from src.config import SEED

SINGLE_SAMPLE = 200_000   # tek-feature prob eğitimi için satır örneklemi


def run_leak_check(X_tr: pd.DataFrame, y_tr: pd.Series,
                   X_va: pd.DataFrame, y_va_true: pd.Series,
                   categorical: list[str], b6_score: float) -> pd.DataFrame:
    """y_tr: log1p hedef (eğitim), y_va_true: orijinal ölçek gerçek (valid)."""
    import lightgbm as lgb

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_tr), size=min(SINGLE_SAMPLE, len(X_tr)), replace=False)
    rows = []
    for col in X_tr.columns:
        tr_col, va_col = X_tr[col], X_va[col]
        nan_va = float(va_col.isna().mean())
        if col in categorical or str(tr_col.dtype) == "category":
            shift = np.nan
        else:
            ts = float(tr_col.std())
            shift = (float(va_col.mean()) - float(tr_col.mean())) / ts \
                if ts and ts > 0 else np.nan

        ds = lgb.Dataset(X_tr.iloc[idx][[col]], label=y_tr.iloc[idx],
                         categorical_feature=[col] if col in categorical else [])
        booster = lgb.train({"objective": "regression", "learning_rate": 0.1,
                             "num_leaves": 31, "verbose": -1, "seed": SEED},
                            ds, num_boost_round=200)
        pred = np.expm1(booster.predict(X_va[[col]])).clip(0)
        single = float(np.sqrt(np.mean(
            (np.log1p(pred) - np.log1p(y_va_true)) ** 2)))

        warns = []
        if not np.isnan(shift) and abs(shift) > 0.5:
            warns.append(f"dağılım kayması {shift:+.2f}σ")
        if single < b6_score - 0.05:
            warns.append(f"TEK BAŞINA b6'yı geçiyor ({single:.3f} < {b6_score:.3f}-0.05)")
        rows.append({"feature": col, "shift_sigma": shift, "valid_nan": nan_va,
                     "single_rmsle": single, "uyari": "; ".join(warns)})
    return pd.DataFrame(rows).sort_values("single_rmsle")
