# -*- coding: utf-8 -*-
"""
06_train.py — 3 fold × 3 varyant LightGBM + leak_check + raporlama.

Çıktılar: reports/model_v1.md · experiments/log.csv (append)
Kullanım: python scripts/06_train.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EXPERIMENTS_DIR, REPORTS_DIR, SEED, TARGET_BLEND  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.features import ALL_FEATURES, CATEGORICAL_FEATURES, build_features  # noqa: E402
from src.leak_check import run_leak_check  # noqa: E402
from src.train import make_xy, train_variant  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

B6_F1_BLEND = 1.2692   # referans (reports/baseline_results.md)
VARIANTS = ["v1", "v2", "v3"]

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def score_pack(valid_eval: pd.DataFrame, e2: pd.Series, valid: pd.DataFrame) -> dict:
    get = lambda k, s: valid_eval.loc[
        (valid_eval["kirilim"] == k) & (valid_eval["seviye"] == s), "rmsle"]
    nz = valid["tuketim"] > 0
    return {
        "all": float(valid_eval.loc[valid_eval["kirilim"] == "global", "rmsle"].iloc[0]),
        "warm": float(get("warm_cold", "warm").iloc[0]),
        "cold": float(get("warm_cold", "cold").iloc[0]),
        "blend": float(valid_eval.loc[valid_eval["kirilim"] == "blend", "rmsle"].iloc[0]),
        "nz_all": float(np.sqrt(e2[nz].mean())),
        "nz_warm": float(np.sqrt(e2[nz & ~valid["is_cold"]].mean())),
        "nz_cold": float(np.sqrt(e2[nz & valid["is_cold"]].mean())),
    }


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model v1 — LightGBM (wx_ hariç tüm feature'lar)")
    w()
    w(f"Üretim: `scripts/06_train.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    scores, importances, best_iters = {}, {}, {}
    leak_report = None

    for fold in folds:
        fn = fold["name"]
        origin = fold["spec"]["train_end"]
        train_rows = df.loc[fold["train_idx"]]
        valid_rows = df.loc[fold["valid_idx"]]
        both = pd.concat([train_rows, valid_rows])
        print(f"[{fn}] feature build (origin={origin}) ...")
        feats = build_features(both, origin, train_rows)

        X_tr, y_tr, tr_kept = make_xy(feats, train_rows, train_mode=True)
        X_va, y_va, _ = make_xy(feats, valid_rows, train_mode=False)
        valid = add_eval_columns(valid_rows, fold, df)

        for var in VARIANTS:
            print(f"[{fn}] {var} egitim ...")
            booster, pred, bi = train_variant(
                var, X_tr, y_tr, X_va, y_va,
                guc_tr=tr_kept["guc"], guc_va=valid_rows["guc"])
            valid["_pred"] = pred
            ev = evaluate(valid, "tuketim", "_pred")
            e2 = (np.log1p(pred.clip(0)) - np.log1p(valid["tuketim"])) ** 2
            scores[(fn, var)] = score_pack(ev, e2, valid)
            best_iters[(fn, var)] = bi
            if fn == "F1":
                importances[var] = pd.Series(
                    booster.feature_importance("gain"), index=ALL_FEATURES)

        if fn == "F1":
            print("[F1] leak_check ...")
            leak_report = run_leak_check(
                X_tr, y_tr, X_va, valid_rows["tuketim"],
                CATEGORICAL_FEATURES, B6_F1_BLEND)

    # ------------------------------------------------------------ skor tabloları
    w("## 1. Skorlar (RMSLE)")
    w()
    for fn in ["F1", "F2", "F3"]:
        w(f"### {fn}")
        w()
        w("| varyant | all | warm | cold | blend | nz_all | nz_warm | nz_cold | best_iter |")
        w("|---|---|---|---|---|---|---|---|---|")
        for var in VARIANTS:
            s = scores[(fn, var)]
            w(f"| {var} | {s['all']:.4f} | {s['warm']:.4f} | {s['cold']:.4f} | "
              f"**{s['blend']:.4f}** | {s['nz_all']:.4f} | {s['nz_warm']:.4f} | "
              f"{s['nz_cold']:.4f} | {best_iters[(fn, var)]} |")
        w()

    best_var = min(VARIANTS, key=lambda v: scores[("F1", v)]["blend"])
    best = scores[("F1", best_var)]
    w(f"- En iyi varyant (F1 blend): **{best_var} = {best['blend']:.4f}**")
    w(f"- b6 referans: {B6_F1_BLEND} → fark **{B6_F1_BLEND - best['blend']:+.4f}**")
    w(f"- Hedef çıta: {TARGET_BLEND} → kalan mesafe {best['blend'] - TARGET_BLEND:+.4f}")
    w()

    # ------------------------------------------------------------ importance
    w(f"## 2. F1 feature importance (gain) — {best_var}, ilk 25")
    w()
    imp = importances[best_var].sort_values(ascending=False)
    tot = imp.sum()
    w("| # | feature | gain payı |")
    w("|---|---|---|")
    for i, (f, v) in enumerate(imp.head(25).items(), 1):
        w(f"| {i} | {f} | %{100*v/tot:.2f} |")
    w()

    # ------------------------------------------------------------ leak check
    w("## 3. leak_check (F1)")
    w()
    warned = leak_report[leak_report["uyari"] != ""]
    w(f"- Kontrol edilen feature: {len(leak_report)} · uyarılı: {len(warned)}")
    w()
    if len(warned):
        w("| feature | shift σ | valid NaN | tek-feature RMSLE | uyarı |")
        w("|---|---|---|---|---|")
        for _, r in warned.iterrows():
            sh = f"{r['shift_sigma']:+.2f}" if not np.isnan(r["shift_sigma"]) else "·"
            w(f"| {r['feature']} | {sh} | %{100*r['valid_nan']:.1f} | "
              f"{r['single_rmsle']:.4f} | {r['uyari']} |")
    else:
        w("Uyarı yok.")
    w()
    w("En güçlü 10 tek-feature RMSLE (bilgi amaçlı):")
    w()
    w("| feature | tek-feature RMSLE | valid NaN |")
    w("|---|---|---|")
    for _, r in leak_report.head(10).iterrows():
        w(f"| {r['feature']} | {r['single_rmsle']:.4f} | %{100*r['valid_nan']:.1f} |")
    w()

    # ------------------------------------------------------------ kabul
    w("## 4. Kabul kriteri")
    w()
    gecti = best["blend"] <= B6_F1_BLEND - 0.15
    w(f"- Şart: F1 blend ≤ b6 − 0.15 = {B6_F1_BLEND - 0.15:.4f}")
    w(f"- Gerçekleşen: {best['blend']:.4f} → **{'GEÇTİ' if gecti else 'GEÇEMEDİ'}**")
    w()

    # ------------------------------------------------------------ log.csv
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for var in VARIANTS:
        rows.append({
            "timestamp": ts, "exp_id": f"lgbm_{var}",
            "feature_set": "static+cal+lvl+grp+seas", "model": f"lgbm_{var}",
            "f1_all": round(scores[("F1", var)]["all"], 4),
            "f1_warm": round(scores[("F1", var)]["warm"], 4),
            "f1_cold": round(scores[("F1", var)]["cold"], 4),
            "f1_blend": round(scores[("F1", var)]["blend"], 4),
            "f2_all": round(scores[("F2", var)]["all"], 4),
            "f3_all": round(scores[("F3", var)]["all"], 4),
            "lb": "", "note": "06_train ilk model",
        })
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not log_path.exists(), index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_v1.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_v1.md'}")


if __name__ == "__main__":
    main()
