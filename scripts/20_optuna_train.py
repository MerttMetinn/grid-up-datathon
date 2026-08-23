# -*- coding: utf-8 -*-

import json
import sys
import time
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print(f"ROOT: {ROOT}")

from src.dataset import CAT_COLS, load_dataset 
from src.validation import rmsle  

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"

FOLDS = ["f1", "f2", "f3"]
N_TRIALS = 25              # fold basina Optuna deneme sayisi
SEED = 42
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100

# Log-olcekli aranan parametreler icin ortalama alirken geometrik ortalama
# kullanilir (aritmetik ortalama log-uniform dagilimda yanlis egilir).
LOG_SCALE_PARAMS = {"learning_rate", "lambda_l1", "lambda_l2"}
INT_PARAMS = {"num_leaves", "min_data_in_leaf"}

# combined_params icinde raporda gosterilmeyecek sabit/teknik anahtarlar
_NON_TUNED_KEYS = (
    "objective", "metric", "verbosity", "feature_pre_filter",
    "seed", "feature_fraction_seed", "bagging_seed", "bagging_freq",
)


def load_model_features() -> list[str]:
    """19 scriptinin urettigi model feature isimlerini okur."""
    path = ROOT / "data" / "feature-selection-results" / "model_features.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok. Once python scripts/19_feature_selection.py calistir."
        )
    features = json.loads(path.read_text(encoding="utf-8"))
    if not features:
        raise ValueError("model_features.json bos.")
    return features


def to_float32_matrix(frame: pd.DataFrame, features: list[str]):
    """LightGBM icin tum satirlari float32 matrise cevirir."""
    matrix = np.empty((len(frame), len(features)), dtype=np.float32)
    for i, col in enumerate(features):
        if col in CAT_COLS:
            categories = frame[col].astype("category").cat.categories
            matrix[:, i] = pd.Categorical(
                frame[col], categories=categories
            ).codes.astype(np.float32)
        else:
            matrix[:, i] = frame[col].to_numpy(
                dtype=np.float32, na_value=np.nan
            )
    return matrix


def make_datasets(train, valid, features):
    """Train ve valid dataframe'lerini LightGBM Dataset nesnelerine cevirir.

    Kategorik kodlar train+valid BIRLIKTE hesaplanir ki iki tarafta
    ayni kategori ayni koda dussun.
    """
    combined = pd.concat([train[features], valid[features]], ignore_index=True)
    combined_matrix = to_float32_matrix(combined, features)
    train_matrix = combined_matrix[:len(train)]
    valid_matrix = combined_matrix[len(train):]
    del combined, combined_matrix

    cat_indices = [features.index(c) for c in CAT_COLS if c in features]
    train_set = lgb.Dataset(
        train_matrix,
        label=train["y_log1p"].to_numpy(dtype=np.float32),
        init_score=train["init_score"].to_numpy(dtype=np.float32),
        feature_name=features,
        categorical_feature=cat_indices,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_matrix,
        label=valid["y_log1p"].to_numpy(dtype=np.float32),
        init_score=valid["init_score"].to_numpy(dtype=np.float32),
        feature_name=features,
        categorical_feature=cat_indices,
        reference=train_set,
        free_raw_data=False,
    )
    return train_set, valid_set, valid_matrix


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "feature_pre_filter": False,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 20.0, log=True),
    }


def run_optuna_for_fold(fold: str, features: list[str]):
    """Bir fold'un TUM train/valid ciftinde bagimsiz Optuna calismasi yapar."""
    train = load_dataset(f"{fold}_train")
    valid = load_dataset(f"{fold}_valid")

    print(f"\n=== {fold.upper()} - Optuna arama ({N_TRIALS} trial) ===")
    print(f"Tum veri: train={len(train):,}, valid={len(valid):,}")

    train_set, valid_set, valid_matrix = make_datasets(train, valid, features)
    valid_y = valid["tuketim"].to_numpy()
    valid_init = valid["init_score"].to_numpy()
    trials = []

    def objective(trial):
        params = suggest_params(trial)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_set],
            valid_names=[f"{fold}_valid"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        raw = model.predict(valid_matrix, num_iteration=model.best_iteration)
        pred = np.clip(np.expm1(raw + valid_init), 0, None)
        score = rmsle(valid_y, pred)
        trials.append({
            "fold": fold, "trial": trial.number, "rmsle": score,
            "best_iteration": model.best_iteration, **params,
        })
        print(f"[{fold.upper()}] Trial {trial.number + 1:02d}/{N_TRIALS}: "
              f"RMSLE={score:.5f}, best_iteration={model.best_iteration}")
        del model, pred, raw
        gc.collect()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    del train_set, valid_set, valid_matrix, train, valid
    gc.collect()

    print(f"[{fold.upper()}] EN IYI RMSLE: {study.best_value:.5f} "
          f"(trial {study.best_trial.number})")
    return study, pd.DataFrame(trials)


def combine_params(fold_studies: list[dict]) -> dict:
    """Uc fold'un en iyi parametrelerini tek  sete birlestirir."""
    param_keys = list(fold_studies[0]["best_params"].keys())
    combined = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "feature_pre_filter": False,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "bagging_freq": 1,
    }
    for key in param_keys:
        values = np.array([fs["best_params"][key] for fs in fold_studies], dtype="float64")
        if key in LOG_SCALE_PARAMS:
            value = float(np.exp(np.mean(np.log(values))))
        else:
            value = float(np.mean(values))
        if key in INT_PARAMS:
            value = int(round(value))
        combined[key] = value
    return combined


def validate_combined_params(fold: str, features: list[str], params: dict):
    """Birlestirilmis parametreleri bir fold'un kendi TUM valid setinde test eder."""
    train = load_dataset(f"{fold}_train")
    valid = load_dataset(f"{fold}_valid")

    train_set, valid_set, valid_matrix = make_datasets(train, valid, features)
    model = lgb.train(
        params, train_set, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_set], valid_names=[f"{fold}_valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    raw = model.predict(valid_matrix, num_iteration=model.best_iteration)
    pred = np.clip(np.expm1(raw + valid["init_score"].to_numpy()), 0, None)
    score = rmsle(valid["tuketim"].to_numpy(), pred)
    best_iter = model.best_iteration
    del model, train_set, valid_set, valid_matrix, train, valid
    gc.collect()
    return score, best_iter


def train_final_model(features: list[str], params: dict, num_boost_round: int):
    """Birlestirilmis parametrelerle TUM full_train uzerinde final modeli egitir."""
    full = load_dataset("full_train")
    test = load_dataset("test")
    print(f"\nFull train: {len(full):,} satir | Test: {len(test):,} satir")

    combined = pd.concat([full[features], test[features]], ignore_index=True)
    combined_matrix = to_float32_matrix(combined, features)
    full_matrix = combined_matrix[:len(full)]
    test_matrix = combined_matrix[len(full):]
    del combined, combined_matrix
    gc.collect()

    cat_indices = [features.index(c) for c in CAT_COLS if c in features]
    full_set = lgb.Dataset(
        full_matrix,
        label=full["y_log1p"].to_numpy(dtype=np.float32),
        init_score=full["init_score"].to_numpy(dtype=np.float32),
        feature_name=features,
        categorical_feature=cat_indices,
        free_raw_data=False,
    )

    fixed_params = {k: v for k, v in params.items() if k != "metric"}
    t0 = time.time()
    model = lgb.train(fixed_params, full_set, num_boost_round=num_boost_round)
    print(f"Final model egitim suresi: {time.time() - t0:.1f}s")

    raw = model.predict(test_matrix)
    pred = np.clip(np.expm1(raw + test["init_score"].to_numpy()), 0, None)
    return model, test["id"].to_numpy(), pred


def build_markdown_summary(summary: dict) -> str:
    """optuna_summary sozlugunu okunabilir bir markdown rapora cevirir."""
    lines = []
    lines.append("# Optuna Ozet Raporu (F1 / F2 / F3 -> full_train)")
    lines.append("")
    lines.append("## Fold Bazli En Iyi Sonuclar (her fold kendi optimum parametreleriyle)")
    lines.append("")
    lines.append("| Fold | RMSLE | Best Iteration |")
    lines.append("|---|---|---|")
    for fs in summary["fold_own_best"]:
        lines.append(f"| {fs['fold'].upper()} | {fs['best_rmsle']:.5f} | {fs['best_iteration']} |")
    own_avg = float(np.mean([fs["best_rmsle"] for fs in summary["fold_own_best"]]))
    lines.append(f"| **Ortalama** | **{own_avg:.5f}** | - |")
    lines.append("")

    lines.append("## Fold Bazli En Iyi Hiperparametreler")
    lines.append("")
    param_names = list(summary["fold_own_best"][0]["best_params"].keys())
    header = "| Parametre | " + " | ".join(fs["fold"].upper() for fs in summary["fold_own_best"]) + " |"
    sep = "|---|" + "|".join(["---"] * len(summary["fold_own_best"])) + "|"
    lines.append(header)
    lines.append(sep)
    for p in param_names:
        row = [f"{fs['best_params'][p]:.6g}" for fs in summary["fold_own_best"]]
        lines.append(f"| {p} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Birlestirilmis Parametreler")
    lines.append("")
    lines.append("*Log-olcekli parametreler (learning_rate, lambda_l1, lambda_l2) icin "
                  "geometrik ortalama, digerleri icin aritmetik ortalama kullanildi.*")
    lines.append("")
    lines.append("| Parametre | Deger |")
    lines.append("|---|---|")
    for k, v in summary["combined_params"].items():
        if k in _NON_TUNED_KEYS:
            continue
        val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {val_str} |")
    lines.append("")

    lines.append("## Birlestirilmis Parametrelerin Fold Dogrulamasi")
    lines.append("")
    lines.append("| Fold | RMSLE | Best Iteration |")
    lines.append("|---|---|---|")
    for cv in summary["combined_validation"]:
        lines.append(f"| {cv['fold'].upper()} | {cv['rmsle']:.5f} | {cv['best_iteration']} |")
    lines.append(f"| **Ortalama** | **{summary['combined_avg_rmsle']:.5f}** | "
                  f"**{summary['final_num_rounds']}** |")
    lines.append("")

    lines.append("## Final Model")
    lines.append("")
    lines.append(f"- Egitim verisi: `full_train`")
    lines.append(f"- Tur sayisi: **{summary['final_num_rounds']}** "
                  f"(uc fold dogrulamasindaki ortalama best_iteration)")
    lines.append(f"- Birlestirilmis parametrelerin ortalama RMSLE'si: "
                  f"**{summary['combined_avg_rmsle']:.5f}**")
    lines.append(f"- Model dosyasi: `models/optuna_final_full_train.txt`")
    lines.append(f"- Submission dosyasi: `submissions/sub_s2_optuna.csv`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dataset_dir = ROOT / "data" / "dataset"
    features = load_model_features()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Feature sayisi: {len(features)}")
    print(f"Fold'lar: {', '.join(f.upper() for f in FOLDS)} | Trial/fold: {N_TRIALS}")

    # 1) Her fold icin ayri Optuna calismasi
    fold_studies = []
    for fold in FOLDS:
        study, trials_df = run_optuna_for_fold(fold, features)
        fold_studies.append({
            "fold": fold,
            "best_rmsle": study.best_value,
            "best_params": study.best_params,
            "best_iteration": int(trials_df.loc[trials_df["rmsle"].idxmin(), "best_iteration"]),
        })

    print("\n" + "=" * 60)
    print("FOLD BAZLI EN IYI SONUCLAR (kendi optimum parametreleriyle)")
    print("=" * 60)
    for fs in fold_studies:
        print(f"  {fs['fold'].upper()}: RMSLE={fs['best_rmsle']:.5f}, "
              f"best_iteration={fs['best_iteration']}")
    own_avg = np.mean([fs["best_rmsle"] for fs in fold_studies])
    print(f"  Ortalama (her fold kendi params'iyla): {own_avg:.5f}")

    # 2) Parametreleri birlestir
    combined_params = combine_params(fold_studies)
    print("\nBIRLESTIRILMIS PARAMETRELER:")
    for k, v in combined_params.items():
        if k not in _NON_TUNED_KEYS:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("BIRLESTIRILMIS PARAMETRELERIN FOLD DOGRULAMASI")
    print("=" * 60)
    val_scores, val_iters = [], []
    for fold in FOLDS:
        score, best_iter = validate_combined_params(fold, features, combined_params)
        val_scores.append(score)
        val_iters.append(best_iter)
        print(f"  {fold.upper()}: RMSLE={score:.5f}, best_iteration={best_iter}")

    combined_avg_rmsle = float(np.mean(val_scores))
    final_num_rounds = int(round(np.mean(val_iters)))
    print(f"\n  Birlestirilmis parametrelerin ortalama RMSLE'si: {combined_avg_rmsle:.5f}")
    print(f"  Final tur sayisi : {final_num_rounds}")

    summary = {
        "fold_own_best": fold_studies,
        "combined_params": combined_params,
        "combined_validation": [
            {"fold": f, "rmsle": s, "best_iteration": it}
            for f, s, it in zip(FOLDS, val_scores, val_iters)
        ],
        "combined_avg_rmsle": combined_avg_rmsle,
        "final_num_rounds": final_num_rounds,
    }

    # 4) Final modeli TUM full_train ile egit
    print("\n" + "=" * 60)
    print("FINAL MODEL - full_train ile egitim")
    print("=" * 60)
    model, ids, preds = train_final_model(features, combined_params, final_num_rounds)

    # CIKTI 1: models/optuna_final_full_train.txt
    model_path = MODELS_DIR / "optuna_final_full_train.txt"
    model.save_model(str(model_path))

    # CIKTI 2: reports/optuna_summary.md
    md_report = build_markdown_summary(summary)
    report_path = REPORTS_DIR / "optuna_summary.md"
    report_path.write_text(md_report, encoding="utf-8")

    # CIKTI 3: submissions/sub_s2_optuna.csv (id,tuketim)
    submission = pd.DataFrame({"id": ids, "tuketim": preds})
    sub_path = SUBMISSIONS_DIR / "sub_s2_optuna.csv"
    submission.to_csv(sub_path, index=False)

    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    print(f"Fold'lar kendi optimum params'iyla ortalama RMSLE : {own_avg:.5f}")
    print(f"Birlestirilmis (final) params ortalama RMSLE      : {combined_avg_rmsle:.5f}")
    print(f"Final full_train modeli tur sayisi                : {final_num_rounds}")
    print("\nOlusturulan dosyalar:")
    print(f"  {model_path}")
    print(f"  {report_path}")
    print(f"  {sub_path}")


if __name__ == "__main__":
    main()