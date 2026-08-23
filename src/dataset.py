# -*- coding: utf-8 -*-
"""Materialize edilmiş feature datasetleri — optuna / feature-selection için hazır paket.

Neden tek fiziksel dosya DEĞİL: feature'ların yarısı (lvl_/grp_/seas_/anchor)
"bakış tarihi"ne (forecast_origin) bağlı. Aynı satır farklı fold'da farklı değer alır.
Hepsini tek statik tabloda dondurmak SIZINTI üretir (bkz. docs/DATASET.md).

Çözüm: her fold'un train+valid'i AYRI materialize edilir (sızıntısız), + full train
(final model) + test. Hepsi data/dataset/ altında, tek loader ile yüklenir.

Üretilen dosyalar (data/dataset/):
  f1_train.parquet / f1_valid.parquet   (F1 birincil fold)
  f2_train.parquet / f2_valid.parquet
  f3_train.parquet / f3_valid.parquet
  full_train.parquet                    (tüm veri, çok-origin — final model)
  test.parquet                          (final tahmin)

Her satır: [tanim, tarih, <75 feature>, y_log1p, tuketim, guc, is_cold,
            anc_base, anc_dev, anc_zero, init_score] (+ test'te id).
`init_score` = s2 çapası (α=0.4). Optuna alpha'yı denemek isterse anc_* kolonlarından
kendi init_score'unu kurar: base + alpha*anc_dev + anc_zero.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, SEED, TRAIN_END
from src.data import load_profile, load_test, load_train
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES,
                          anchor_components, build_features)
from src.train import ORIGINS, align_categories, build_training_set
from src.validation import add_eval_columns, make_folds

DATASET_DIR = PROCESSED_DIR.parent / "dataset"
ALPHA_S2 = 0.4                      # s2 çapası; init_score kolonu bununla kurulur
FEATURE_COLS = ALL_FEATURES
CAT_COLS = CATEGORICAL_FEATURES
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]


def _init_score(base, dev, zero):
    return (base + ALPHA_S2 * dev + zero).astype("float32")


def _train_frame(X, meta):
    """Çok-origin eğitim bloğunu tek tabloya çevirir (X + hedef + anchor)."""
    out = X.copy()
    out["y_log1p"] = np.log1p(meta["tuketim"]).astype("float32")
    out["tuketim"] = meta["tuketim"].astype("float32")
    out["guc"] = meta["guc"].astype("float32")
    out["is_cold"] = meta["is_cold_example"].to_numpy()
    out["anc_base"] = meta["anc_base"].astype("float32")
    out["anc_dev"] = meta["anc_dev"].astype("float32")
    out["anc_zero"] = meta["anc_zero"].astype("float32")
    out["init_score"] = _init_score(meta["anc_base"], meta["anc_dev"],
                                    meta["anc_zero"])
    return out


def _valid_frame(df, fold):
    """Fold valid satırları — sızıntısız (yalnızca fold train'inden feature)."""
    vr = df.loc[fold["valid_idx"]]
    train_end = fold["spec"]["train_end"]
    hist = df.loc[fold["train_idx"]]
    X = build_features(vr, train_end, hist)
    comp = anchor_components(vr, train_end, hist)
    ev = add_eval_columns(vr, fold, df)
    out = X.copy()
    out.insert(0, "tanim", vr["tanim"].to_numpy())
    out.insert(1, "tarih", vr["tarih"].to_numpy())
    out["y_log1p"] = np.log1p(vr["tuketim"]).astype("float32")
    out["tuketim"] = vr["tuketim"].astype("float32")
    out["guc"] = vr["guc"].astype("float32")
    out["is_cold"] = ev["is_cold"].to_numpy()
    out["anc_base"] = comp["base"].astype("float32")
    out["anc_dev"] = comp["season_dev"].astype("float32")
    out["anc_zero"] = comp["zero_adj"].astype("float32")
    out["init_score"] = _init_score(comp["base"], comp["season_dev"],
                                    comp["zero_adj"])
    return out


def build_datasets():
    """Tüm datasetleri üretip data/dataset/ altına yazar."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df = load_train()
    te = load_test()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    written = []
    for fi, fold in enumerate(folds):
        fn = fold["name"].lower()
        print(f"[{fold['name']}] train (çok-origin) ...")
        X, y, meta = build_training_set(df, fold, profile, fi)
        tr = _train_frame(X, meta)
        print(f"[{fold['name']}] valid ...")
        va = _valid_frame(df, fold)
        align_categories([tr, va])
        tr.to_parquet(DATASET_DIR / f"{fn}_train.parquet")
        va.to_parquet(DATASET_DIR / f"{fn}_valid.parquet")
        written += [f"{fn}_train ({len(tr):,})", f"{fn}_valid ({len(va):,})"]

    print("[FULL] train (çok-origin, tüm veri) ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index,
              "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    full = _train_frame(Xf, metaf)

    print("[TEST] ...")
    Xt = build_features(te, TRAIN_END, df)
    comp_t = anchor_components(te, TRAIN_END, df)
    test = Xt.copy()
    test.insert(0, "id", te["id"].to_numpy())
    test.insert(1, "tanim", te["tanim"].to_numpy())
    test.insert(2, "tarih", te["tarih"].to_numpy())
    test["guc"] = te["guc"].astype("float32")
    test["is_cold"] = (~te["tanim"].isin(set(df["tanim"].unique()))).to_numpy()
    test["anc_base"] = comp_t["base"].astype("float32")
    test["anc_dev"] = comp_t["season_dev"].astype("float32")
    test["anc_zero"] = comp_t["zero_adj"].astype("float32")
    test["init_score"] = _init_score(comp_t["base"], comp_t["season_dev"],
                                     comp_t["zero_adj"])
    align_categories([full, test])
    full.to_parquet(DATASET_DIR / "full_train.parquet")
    test.to_parquet(DATASET_DIR / "test.parquet")
    written += [f"full_train ({len(full):,})", f"test ({len(test):,})"]
    return written


def load_dataset(name: str) -> pd.DataFrame:
    """Materialize edilmiş bir dataseti yükler.

    name ∈ {f1_train, f1_valid, f2_train, f2_valid, f3_train, f3_valid,
            full_train, test}
    """
    path = DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} yok — önce: python scripts/18_build_dataset.py")
    return pd.read_parquet(path)
