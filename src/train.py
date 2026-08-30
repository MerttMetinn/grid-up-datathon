# -*- coding: utf-8 -*-
"""Çok-origin LightGBM eğitimi.

Kurgu (onaylı sözleşme):
- Her fold için birden çok forecast_origin kesilir. Eğitim satırları HER ZAMAN
  origin'den SONRA (test geometrisiyle birebir); feature'lar origin'de, yalnızca
  origin ÖNCESİ geçmişle hesaplanır. Hedef penceresi ≤122 gün ve train_end'i aşmaz.
- Her origin'de test_history_profile.csv'den guc_bucket-stratified (H, giriş-offseti)
  çifti örneklenir; o origin'in geçmişi H ile kırpılır (H=0 → cold örneği) ve cold
  hedef satırlarının başı offset kadar kırpılır (cold satır payı test'e yaklaşır).
- Origin başına farklı seed → aynı trafo farklı origin'lerde farklı H alır.
  Bu, eski random dropout'un YERİNE geçer.

Varyantlar:
  m1: çok-origin + H-örneklemesi
  m2: m1 + init_score = log(guc*24)  (model yük faktörünü öğrenir) ← taban
  m3: m2 + ayrı cold modeli (static_/cal_/grp_ feature'ları; tahminde is_cold yönlendirir)
"""
import numpy as np
import pandas as pd

from src.config import SEED, TEST_N_DAYS
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, assemble_anchor, build_features)

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
}
NUM_ROUNDS = 5000
EARLY_STOP = 300

# fold başına forecast_origin listeleri (hedef: origin+1 .. min(origin+122, train_end))
# Origin'ler pencerenin TAMAMINA yayılır — hedef ayları tüm treni kapsasın
# (F3'ün sonbahar/kış boşluğu bu yüzden kapandı).
ORIGINS = {
    "F1": ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30",
           "2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"],
    "F2": ["2025-01-15", "2025-02-15", "2025-02-28", "2025-03-15"],
    "F3": ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
           "2025-06-30", "2025-07-31"],
}

COLD_MODEL_FEATURES = (FEATURE_GROUPS["static"] + FEATURE_GROUPS["cal"]
                       + FEATURE_GROUPS["grp"])


def _profile_pools(profile: pd.DataFrame):
    profile = profile.copy()
    profile["entry_offset"] = (
        profile["test_entry"] - profile["test_entry"].min()).dt.days
    pools = {b: g[["H", "entry_offset"]].to_numpy()
             for b, g in profile.groupby("guc_bucket", observed=True)}
    return pools, profile[["H", "entry_offset"]].to_numpy()


def build_origin_block(fold_train: pd.DataFrame, origin: pd.Timestamp,
                       train_end: pd.Timestamp, pools, all_pairs,
                       rng: np.random.Generator):
    """Tek origin için (features, y, meta) üretir."""
    horizon_end = min(origin + pd.Timedelta(days=TEST_N_DAYS), train_end)
    win_len = (horizon_end - origin).days

    hist_pool = fold_train[fold_train["tarih"] <= origin]
    targets = fold_train[(fold_train["tarih"] > origin) &
                         (fold_train["tarih"] <= horizon_end)]
    if targets.empty or hist_pool.empty:
        return None

    tx_bucket = targets.groupby("tanim", observed=True)["guc_bucket"].first()
    H_map, off_map = {}, {}
    for tx, b in tx_bucket.items():
        pool = pools.get(b, all_pairs)
        h, off = pool[rng.integers(len(pool))]
        H_map[tx] = int(h)
        off_map[tx] = int(round(off * win_len / TEST_N_DAYS))

    # geçmişi H ile kırp; hedef trafosu olmayanlar tam kalır (grp bağlamı)
    h_days = hist_pool["tanim"].map(H_map)
    min_keep = origin - pd.to_timedelta(h_days.fillna(10_000), unit="D")
    hist = hist_pool[h_days.isna() | (hist_pool["tarih"] > min_keep)]

    # cold örneklerinin hedef satırlarını giriş-offseti kadar baştan kırp
    is_cold_tx = targets["tanim"].map(lambda t: H_map.get(t, 1) == 0)
    entry = origin + pd.to_timedelta(targets["tanim"].map(off_map).fillna(0),
                                     unit="D")
    keep = ~is_cold_tx | (targets["tarih"] >= entry)
    targets = targets[keep]

    feats = build_features(targets, str(origin.date()), hist)
    comp = anchor_components(targets, str(origin.date()), hist)
    meta = pd.DataFrame({
        "tuketim": targets["tuketim"], "guc": targets["guc"],
        "is_bad_row": targets["is_bad_row"],
        "is_cold_example": targets["tanim"].map(
            lambda t: H_map.get(t, 1) == 0).astype(bool),
        "anc_base": comp["base"], "anc_dev": comp["season_dev"],
        "anc_zero": comp["zero_adj"], "anc_is_cold": comp["is_cold_anchor"],
        "tarih": targets["tarih"].to_numpy(),   # recency ağırlığı için hedef tarihi
    }, index=targets.index)
    return feats, meta


def build_training_set(df: pd.DataFrame, fold: dict, profile: pd.DataFrame,
                       fold_i: int):
    """Fold'un çok-origin eğitim matrisi. LF>1 satırları düşer, sıfırlar kalır."""
    pools, all_pairs = _profile_pools(profile)
    fold_train = df.loc[fold["train_idx"]]
    train_end = pd.Timestamp(fold["spec"]["train_end"])

    X_parts, meta_parts = [], []
    for j, o in enumerate(ORIGINS[fold["name"]]):
        rng = np.random.default_rng(SEED + 100 * fold_i + j)
        block = build_origin_block(fold_train, pd.Timestamp(o), train_end,
                                   pools, all_pairs, rng)
        if block is None:
            continue
        feats, meta = block
        X_parts.append(feats)
        meta_parts.append(meta)

    X = pd.concat(X_parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True)
    keep = ~meta["is_bad_row"]
    X, meta = X[keep.to_numpy()], meta[keep.to_numpy()]
    y = np.log1p(meta["tuketim"])
    return X.reset_index(drop=True), y.reset_index(drop=True), \
        meta.reset_index(drop=True)


def align_categories(frames: list[pd.DataFrame], cols=CATEGORICAL_FEATURES):
    """Kategori kodları train/valid arasında birebir aynı olsun (LGBM kod kullanır)."""
    for c in cols:
        cats = pd.api.types.union_categoricals(
            [f[c].astype("category") for f in frames]).categories
        for f in frames:
            f[c] = pd.Categorical(f[c], categories=cats)
    return frames


def fit_lgbm(X_tr, y_tr, X_va, y_va, features, init_tr=None, init_va=None,
             seed_offset: int = 0):
    import lightgbm as lgb
    params = dict(LGB_PARAMS)
    for k in ("seed", "feature_fraction_seed", "bagging_seed"):
        params[k] = LGB_PARAMS[k] + seed_offset
    ds_tr = lgb.Dataset(X_tr[features], label=y_tr, init_score=init_tr,
                        categorical_feature=[c for c in CATEGORICAL_FEATURES
                                             if c in features])
    ds_va = lgb.Dataset(X_va[features], label=y_va, init_score=init_va,
                        reference=ds_tr)
    booster = lgb.train(
        params, ds_tr, num_boost_round=NUM_ROUNDS,
        valid_sets=[ds_va], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    raw = booster.predict(X_va[features], num_iteration=booster.best_iteration)
    if init_va is not None:
        raw = raw + init_va
    pred = np.clip(np.expm1(raw), 0, None)
    return booster, pred, booster.best_iteration
