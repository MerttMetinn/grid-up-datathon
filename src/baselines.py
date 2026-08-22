# -*- coding: utf-8 -*-
"""Baseline tahminciler. Hepsi orijinal ölçekte tahmin döndürür ve cold
trafolarda da (fallback zinciriyle) tahmin üretebilir.

Ortak imza: fonk(train, target) -> pd.Series (target ile aynı index)
train  = fold'un train satırları (df.loc[train_idx])
target = tahmin istenen satırlar (valid veya test)

Medyanlar log uzayında alınır: expm1(median(log1p(x))) — RMSLE ile uyumlu.
LF istatistiklerinde bozuk satırlar (is_bad_row) dışlanır.
"""
import numpy as np
import pandas as pd


def _logmed(s: pd.Series) -> float:
    return float(np.log1p(s).median())


def _group_logmed(train: pd.DataFrame, keys: list[str]) -> pd.Series:
    return (np.log1p(train["tuketim"])
            .groupby([train[k] for k in keys], observed=True).median())


def _lookup(target: pd.DataFrame, stat: pd.Series, keys: list[str]) -> pd.Series:
    idx = pd.MultiIndex.from_frame(target[keys]) if len(keys) > 1 \
        else pd.Index(target[keys[0]])
    return pd.Series(stat.reindex(idx).to_numpy(), index=target.index)


def b1_global_median(train, target):
    """Global medyan (log uzayında)."""
    return pd.Series(np.expm1(_logmed(train["tuketim"])), index=target.index)


def b2_trafo_median(train, target):
    """Trafo medyanı; görülmemiş trafo → global medyan."""
    stat = _group_logmed(train, ["tanim"])
    pred = _lookup(target, stat, ["tanim"])
    return np.expm1(pred.fillna(_logmed(train["tuketim"])))


def b3_trafo_ay_median(train, target):
    """Trafo × ay medyanı; fallback: trafo → global."""
    s_ta = _group_logmed(train, ["tanim", "ay_no"])
    s_t = _group_logmed(train, ["tanim"])
    pred = _lookup(target, s_ta, ["tanim", "ay_no"])
    pred = pred.fillna(_lookup(target, s_t, ["tanim"]))
    return np.expm1(pred.fillna(_logmed(train["tuketim"])))


def b4_trafo_ay_haftaici_median(train, target):
    """Trafo × ay × haftaiçi medyanı; fallback: b3 zinciri."""
    s_tah = _group_logmed(train, ["tanim", "ay_no", "haftaici"])
    s_ta = _group_logmed(train, ["tanim", "ay_no"])
    s_t = _group_logmed(train, ["tanim"])
    pred = _lookup(target, s_tah, ["tanim", "ay_no", "haftaici"])
    pred = pred.fillna(_lookup(target, s_ta, ["tanim", "ay_no"]))
    pred = pred.fillna(_lookup(target, s_t, ["tanim"]))
    return np.expm1(pred.fillna(_logmed(train["tuketim"])))


B5_CHAIN = [
    ["ilce_key", "ay_no", "haftaici"],
    ["ilce_key", "ay_no"],
    ["il", "guc_bucket", "ay_no"],
    ["guc_bucket", "ay_no"],
    ["ay_no"],
]


def b5_guc_lf(train, target, return_level=False):
    """guc × 24 × LF_medyan[ilçe, ay, haftaiçi] — cold baseline'ı.

    LF medyanı MEDYAN ile (raw LF), bozuk satırlar dışarıda.
    Fallback zinciri B5_CHAIN + global. return_level=True ise hangi seviyenin
    kullanıldığı da döner (teşhis).
    """
    tt = train.loc[~train["is_bad_row"]].copy()
    tt["lf"] = tt["tuketim"] / (tt["guc"] * 24.0)

    lf = pd.Series(np.nan, index=target.index)
    level = pd.Series("", index=target.index, dtype="object")
    for keys in B5_CHAIN:
        stat = tt.groupby(keys, observed=True)["lf"].median()
        cand = _lookup(target, stat, keys)
        take = lf.isna() & cand.notna()
        lf[take] = cand[take]
        level[take] = "+".join(keys)
    rest = lf.isna()
    lf[rest] = float(tt["lf"].median())
    level[rest] = "global"

    pred = target["guc"] * 24.0 * lf
    return (pred, level) if return_level else pred


def b6_hybrid(train, target, return_level=False):
    """Warm satırda b2 (trafo medyanı), cold satırda b5 (guc × LF).

    Sözleşme güncellemesi: warm bacağı b3 değil b2 — kırpılmış geçmiş rejiminde
    trafo×ay medyanı bayat kalıyor (YoY seviye kayması), b2 warm'da daha iyi.
    """
    warm_tx = set(train["tanim"].unique())
    is_warm = target["tanim"].isin(warm_tx)
    p2 = b2_trafo_median(train, target)
    p5, level5 = b5_guc_lf(train, target, return_level=True)
    pred = p5.copy()
    pred[is_warm] = p2[is_warm]
    if return_level:
        level = level5.where(~is_warm, "b2_warm")
        return pred, level
    return pred


BASELINES = {
    "b1_global": b1_global_median,
    "b2_trafo": b2_trafo_median,
    "b3_trafo_ay": b3_trafo_ay_median,
    "b4_trafo_ay_hi": b4_trafo_ay_haftaici_median,
    "b5_guc_lf": b5_guc_lf,
    "b6_hibrit": b6_hybrid,
}
