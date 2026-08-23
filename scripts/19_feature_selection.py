# -*- coding: utf-8 -*-

import sys
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CAT_COLS, FEATURE_COLS, load_dataset  
from src.validation import rmsle 

MODEL_MIN_FOLDS = 2
MODEL_MIN_GAIN_SHARE = 0.005


def _to_float32_matrix(train: pd.DataFrame, valid: pd.DataFrame):
    """Tum satirlari kullanir; kategorikleri ortak kodlarla float32'e cevirir."""
    train_matrix = np.empty((len(train), len(FEATURE_COLS)), dtype=np.float32)
    valid_matrix = np.empty((len(valid), len(FEATURE_COLS)), dtype=np.float32)

    for i, col in enumerate(FEATURE_COLS):
        if col in CAT_COLS:
            categories = pd.api.types.union_categoricals(
                [train[col].astype("category"), valid[col].astype("category")]
            ).categories
            train_matrix[:, i] = pd.Categorical(
                train[col], categories=categories
            ).codes.astype(np.float32)
            valid_matrix[:, i] = pd.Categorical(
                valid[col], categories=categories
            ).codes.astype(np.float32)
        else:
            train_matrix[:, i] = train[col].to_numpy(
                dtype=np.float32, na_value=np.nan
            )
            valid_matrix[:, i] = valid[col].to_numpy(
                dtype=np.float32, na_value=np.nan
            )
    return train_matrix, valid_matrix



def train_fold_model(train: pd.DataFrame, valid: pd.DataFrame,
                     fold_name: str) -> lgb.Booster:
    """Bir fold'un tum train/valid satirlariyla LightGBM modeli egitir."""
    params = {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
    }
    train_matrix, valid_matrix = _to_float32_matrix(train, valid)
    cat_indices = [FEATURE_COLS.index(c) for c in CAT_COLS if c in FEATURE_COLS]
    train_set = lgb.Dataset(
        train_matrix,
        label=train["y_log1p"].to_numpy(dtype=np.float32),
        init_score=train["init_score"].to_numpy(dtype=np.float32),
        feature_name=FEATURE_COLS,
        categorical_feature=cat_indices,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        valid_matrix,
        label=valid["y_log1p"].to_numpy(dtype=np.float32),
        init_score=valid["init_score"].to_numpy(dtype=np.float32),
        reference=train_set,
        free_raw_data=True,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        valid_names=[f"{fold_name.lower()}_valid"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )



def main() -> None:
    results_dir = ROOT / "data" / "feature-selection-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_tables = []
    fold_summaries = []

    for fold_name in ("F1", "F2", "F3"):
        train = load_dataset(f"{fold_name.lower()}_train")
        valid = load_dataset(f"{fold_name.lower()}_valid")
        print(f"\n[{fold_name}] train: {len(train):,} | valid: {len(valid):,}")
        print(f"[{fold_name}] Tum satirlar kullanilacak; model egitiliyor...")

        model = train_fold_model(train, valid, fold_name)
        valid_matrix = _to_float32_matrix(train.iloc[:0], valid)[1]
        raw = model.predict(valid_matrix) + valid["init_score"].to_numpy()
        pred = np.clip(np.expm1(raw), 0, None)
        score = rmsle(valid["tuketim"], pred)
        print(f"[{fold_name}] Best iteration: {model.best_iteration}")
        print(f"[{fold_name}] valid RMSLE: {score:.4f}")

        table = pd.DataFrame({
            "feature": FEATURE_COLS,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        })
        total = table["gain"].sum()
        table["gain_share"] = table["gain"] / total if total else 0.0
        table["fold"] = fold_name
        table["group"] = table["feature"].str.split("_", n=1).str[0]
        table["status"] = np.select(
            [table["gain"] <= 0, table["gain_share"] < 0.005],
            ["kullanilmadi", "dusuk_katki"],
            default="katkili",
        )
        all_tables.append(table)
        fold_summaries.append({
            "fold": fold_name,
            "train_rows": len(train),
            "valid_rows": len(valid),
            "best_iteration": model.best_iteration,
            "valid_rmsle": score,
        })

        del model, train, valid, valid_matrix, raw, pred
        gc.collect()

    details = pd.concat(all_tables, ignore_index=True)
    details.to_csv(results_dir / "feature_importance_all_folds.csv", index=False)

    stable = (details.assign(used=details["gain"] > 0)
              .groupby(["feature", "group"], as_index=False)
              .agg(avg_gain_share=("gain_share", "mean"),
                   folds_used=("used", "sum"),
                   avg_split=("split", "mean")))
    stable["status"] = np.select(
        [stable["folds_used"] == 0, stable["folds_used"] == 1,
         stable["avg_gain_share"] < 0.005],
        ["hic_kullanilmadi", "tek_fold_katkili", "dusuk_istikrarli_katki"],
        default="istikrarli_katkili",
    )
    stable = stable.sort_values(
        ["folds_used", "avg_gain_share"], ascending=[False, False]
    ).reset_index(drop=True)

    model_features = stable.loc[
        (stable["folds_used"] >= MODEL_MIN_FOLDS)
        & (stable["avg_gain_share"] >= MODEL_MIN_GAIN_SHARE),
        "feature",
    ].tolist()
    model_features_path = results_dir / "model_features.json"
    model_features_path.write_text(
        json.dumps(model_features, indent=2),
        encoding="utf-8",
    )

    print("\nFOLD OZETI\n")
    print(pd.DataFrame(fold_summaries).to_string(index=False))
    print("\nISTIKRARLI FEATURE SIRASI\n")
    print(stable.to_string(index=False,
                           formatters={"avg_gain_share": "{:.2%}".format}))
    print(f"- Uc foldda kullanilan: {(stable['folds_used'] == 3).sum()} feature")
    print(f"- En az bir foldda kullanilan: {(stable['folds_used'] > 0).sum()} feature")
    print(f"- Hic kullanilmayan: {(stable['folds_used'] == 0).sum()} feature")
    print(f"- Model icin ayrilan: {len(model_features)} feature")
    print(f"- Kriter: en az {MODEL_MIN_FOLDS} fold, ortalama gain >= "
          f"%{MODEL_MIN_GAIN_SHARE * 100:.1f}")
    print("- Model feature adlari:")
    print("  " + ", ".join(model_features))
    print(f"\nDetay: {results_dir / 'feature_importance_all_folds.csv'}")
    print(f"Model listesi: {model_features_path}")


if __name__ == "__main__":
    main()
