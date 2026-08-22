# -*- coding: utf-8 -*-
"""
09_train_full.py — Mevsim-nötr lvl_*_full çıpaları + p1/p2/p3.

  p1: n3 kurgusu + lvl_*_full  (tek model + cold-only model + b5 harmanı, w F2'de)
  p2: p1, ama w ÜÇ fold ortalamasında optimize (F2 overfit'ine karşı)
  p3: p2 + seed averaging (3 seed, log uzayında ortalama)

Ek: lvl_full_over_90d ablation'ı (F3) · b düşerse F3 model−b6 kesim analizi.
Çıktılar: reports/model_v4.md · experiments/log.csv (append)
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import b5_guc_lf, b6_hybrid  # noqa: E402
from src.config import EXPERIMENTS_DIR, REPORTS_DIR, SEED  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.features import ALL_FEATURES, FEATURE_GROUPS, build_features  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, ORIGINS, align_categories,
                       build_training_set, fit_lgbm)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds, rmsle  # noqa: E402

B6_BLEND = {"F1": 1.2692, "F2": 1.2654, "F3": 1.3055}
VARIANTS = ["p1", "p2", "p3"]
SEEDS = [0, 1, 2]
LVL_FULL = ["lvl_mean_log_full", "lvl_std_log_full", "lvl_zero_ratio_full",
            "lvl_lf_median_full", "lvl_mean_log_364d", "lvl_full_over_90d",
            "lvl_full_over_28d"]

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def logmean(preds: list[np.ndarray]) -> np.ndarray:
    return np.expm1(np.mean([np.log1p(p) for p in preds], axis=0))


def logblend(pm, pb, wgt):
    return np.expm1(wgt * np.log1p(pm) + (1 - wgt) * np.log1p(pb))


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model v4 — mevsim-nötr çıpalar (lvl_*_full) + p1/p2/p3")
    w()
    w(f"Üretim: `scripts/09_train_full.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    store, imps, cold_imps = {}, {}, {}

    for fi, fold in enumerate(folds):
        fn = fold["name"]
        print(f"[{fn}] egitim seti ({len(ORIGINS[fn])} origin) ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)
        valid_rows = df.loc[fold["valid_idx"]]
        X_va = build_features(valid_rows, fold["spec"]["train_end"],
                              df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        y_va = np.log1p(valid_rows["tuketim"])
        valid = add_eval_columns(valid_rows, fold, df)
        is_cold_va = valid["is_cold"].to_numpy()

        init_tr = np.log(meta["guc"].to_numpy() * 24.0)
        init_va = np.log(valid_rows["guc"].to_numpy() * 24.0)
        cold_mask = meta["is_cold_example"].to_numpy()
        Xvc = X_va.loc[valid.index[is_cold_va]]
        yvc = y_va[is_cold_va]
        ivc = init_va[is_cold_va]

        mains, colds, iters = [], [], []
        for so in SEEDS:
            print(f"[{fn}] seed+{so}: ana model ...")
            bm, pm, itm = fit_lgbm(X_tr, y_tr, X_va, y_va, ALL_FEATURES,
                                   init_tr, init_va, seed_offset=so)
            print(f"[{fn}] seed+{so}: cold modeli ...")
            bc, pc, itc = fit_lgbm(X_tr[cold_mask], y_tr[cold_mask], Xvc, yvc,
                                   COLD_MODEL_FEATURES, init_tr[cold_mask], ivc,
                                   seed_offset=so)
            mains.append(np.asarray(pm))
            colds.append(np.asarray(pc))
            iters.append((itm, itc))
            if so == 0:
                imps[fn] = pd.Series(bm.feature_importance("gain"),
                                     index=ALL_FEATURES)
                cold_imps[fn] = pd.Series(bc.feature_importance("gain"),
                                          index=COLD_MODEL_FEATURES)

        pred_b5_cold = b5_guc_lf(df.loc[fold["train_idx"]],
                                 valid_rows[is_cold_va]).to_numpy()
        store[fn] = {
            "fold": fold, "valid": valid, "valid_rows": valid_rows,
            "is_cold": is_cold_va, "mains": mains, "colds": colds,
            "b5_cold": pred_b5_cold,
            "y_cold": valid_rows.loc[is_cold_va, "tuketim"].to_numpy(),
            "iters": iters, "X_tr_cols": (X_tr, y_tr, meta, X_va, y_va,
                                          init_tr, init_va),
        }

    # ---------------------------------------------------------- w optimizasyonu
    grid = np.arange(0, 1.0001, 0.05)
    w_per_fold = {}
    for fn in store:
        s = store[fn]
        w_per_fold[fn] = float(grid[int(np.argmin(
            [rmsle(s["y_cold"], logblend(s["colds"][0], s["b5_cold"], g))
             for g in grid]))])
    w_f2 = w_per_fold["F2"]
    w_avg = float(grid[int(np.argmin(
        [np.mean([rmsle(store[fn]["y_cold"],
                        logblend(store[fn]["colds"][0], store[fn]["b5_cold"], g))
                  for fn in store]) for g in grid]))])

    # ---------------------------------------------------------- varyant tahminleri
    def combine(fn, seeds, wgt):
        s = store[fn]
        pm = logmean([s["mains"][i] for i in seeds])
        pc = logmean([s["colds"][i] for i in seeds])
        p = np.array(pm, dtype="float64")
        p[s["is_cold"]] = logblend(pc, s["b5_cold"], wgt)
        return p

    scores = {}
    for fn in ["F1", "F2", "F3"]:
        preds = {"p1": combine(fn, [0], w_f2),
                 "p2": combine(fn, [0], w_avg),
                 "p3": combine(fn, [0, 1, 2], w_avg)}
        valid = store[fn]["valid"]
        for var, p in preds.items():
            valid["_pred"] = pd.Series(p, index=valid.index)
            ev = evaluate(valid, "tuketim", "_pred")
            e2 = (np.log1p(valid["_pred"].clip(0)) -
                  np.log1p(valid["tuketim"])) ** 2
            nz = valid["tuketim"] > 0
            get = lambda k, sv: float(ev.loc[
                (ev["kirilim"] == k) & (ev["seviye"] == sv), "rmsle"].iloc[0])
            scores[(fn, var)] = {
                "all": float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0]),
                "warm": get("warm_cold", "warm"), "cold": get("warm_cold", "cold"),
                "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
                "nz_warm": float(np.sqrt(e2[nz & ~valid["is_cold"]].mean())),
                "nz_cold": float(np.sqrt(e2[nz & valid["is_cold"]].mean())),
            }

    # ---------------------------------------------------------- 1. skorlar
    w("## 1. Skorlar")
    w()
    for fn in ["F1", "F2", "F3"]:
        it = store[fn]["iters"]
        w(f"### {fn}  (b6 blend = {B6_BLEND[fn]})")
        w()
        w("| varyant | all | warm | cold | blend | model−b6 | nz_warm | nz_cold | best_iter |")
        w("|---|---|---|---|---|---|---|---|---|")
        for var in VARIANTS:
            s = scores[(fn, var)]
            bi = f"{it[0][0]}/{it[0][1]}c" if var != "p3" else \
                ",".join(f"{a}/{b}c" for a, b in it)
            w(f"| {var} | {s['all']:.4f} | {s['warm']:.4f} | {s['cold']:.4f} | "
              f"**{s['blend']:.4f}** | {s['blend']-B6_BLEND[fn]:+.4f} | "
              f"{s['nz_warm']:.4f} | {s['nz_cold']:.4f} | {bi} |")
        w()

    # ---------------------------------------------------------- 2. F2 warm kontrolü
    f2_warm = scores[("F2", "p3")]["warm"]
    w("## 2. ÖZEL KONTROL — F2 warm")
    w()
    w(f"- Önceki tur: 0.7746 (sabitti) · bu tur: **{f2_warm:.4f}** "
      f"({f2_warm - 0.7746:+.4f})")
    if f2_warm >= 0.7746:
        w("- **DÜŞMEDİ** — lvl_*_full çıpaları F2 warm'a katkı vermedi; yaz "
          "rampası hâlâ fold içinden öğrenilemiyor.")
    w()

    # ---------------------------------------------------------- 3. importance
    w("## 3. Feature importance (ana model, gain)")
    w()
    for fn in ["F2", "F3"]:
        imp = imps[fn].sort_values(ascending=False)
        tot = imp.sum()
        w(f"### {fn} — ilk 20")
        w()
        w("| # | feature | gain payı |")
        w("|---|---|---|")
        for i, (f, v) in enumerate(imp.head(20).items(), 1):
            w(f"| {i} | {f} | %{100*v/tot:.2f} |")
        full_share = 100 * imps[fn].reindex(LVL_FULL).fillna(0).sum() / tot
        w(f"- **lvl_*_full ailesi gain payı: %{full_share:.1f}**")
        w()

    # ---------------------------------------------------------- 4. cold model gain
    w("## 4. Cold model gain (yeni kriter d)")
    w()
    grp_cold_share = {}
    for fn in ["F1", "F2", "F3"]:
        ci = cold_imps[fn].sort_values(ascending=False)
        tot = ci.sum()
        grp_cold_share[fn] = 100 * ci.reindex(
            FEATURE_GROUPS["grp"]).fillna(0).sum() / tot
        w(f"### {fn} cold modeli — ilk 10 · grp_ toplamı %{grp_cold_share[fn]:.1f}")
        w()
        w("| # | feature | gain payı |")
        w("|---|---|---|")
        for i, (f, v) in enumerate(ci.head(10).items(), 1):
            w(f"| {i} | {f} | %{100*v/tot:.2f} |")
        w()

    # ---------------------------------------------------------- 5. w raporu
    w("## 5. Harman ağırlığı w")
    w()
    w(f"- Fold başına optimum: " +
      " · ".join(f"{fn}={w_per_fold[fn]:.2f}" for fn in w_per_fold))
    w(f"- p1 kullanılan (F2): {w_f2:.2f} · p2/p3 kullanılan (3-fold ort.): "
      f"**{w_avg:.2f}**")
    w()

    # ---------------------------------------------------------- 6. ablation
    w("## 6. Ablation — lvl_full_over_90d (F3)")
    w()
    print("[F3] ablation: lvl_full_over_90d cikartilarak ...")
    X_tr, y_tr, meta, X_va, y_va, init_tr, init_va = store["F3"]["X_tr_cols"]
    feats_wo = [f for f in ALL_FEATURES if f != "lvl_full_over_90d"]
    _, pm_wo, _ = fit_lgbm(X_tr, y_tr, X_va, y_va, feats_wo, init_tr, init_va)
    s3 = store["F3"]
    p_wo = np.array(pm_wo, dtype="float64")
    p_wo[s3["is_cold"]] = logblend(s3["colds"][0], s3["b5_cold"], w_avg)
    valid3 = s3["valid"]
    valid3["_pred"] = pd.Series(p_wo, index=valid3.index)
    ev_wo = evaluate(valid3, "tuketim", "_pred")
    blend_wo = float(ev_wo.loc[ev_wo["kirilim"] == "blend", "rmsle"].iloc[0])
    w(f"- p2 (feature'la): {scores[('F3','p2')]['blend']:.4f} · "
      f"feature'sız: {blend_wo:.4f} → katkı **{blend_wo - scores[('F3','p2')]['blend']:+.4f}**")
    w()

    # ---------------------------------------------------------- kabul
    best_var = min(VARIANTS, key=lambda v: scores[("F2", v)]["blend"])
    w("## 7. Kabul kriterleri")
    w()
    ka = scores[("F2", best_var)]["blend"] <= 1.205
    kb = scores[("F3", best_var)]["blend"] < B6_BLEND["F3"]
    kc = scores[("F1", best_var)]["blend"] <= 1.13
    kd = grp_cold_share["F2"] >= 25.0
    ke = scores[("F2", best_var)]["warm"] < 0.7746
    w(f"- En iyi varyant (F2 blend): **{best_var}**")
    w(f"- a) F2 blend ≤ 1.205: {scores[('F2', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if ka else 'SAĞLANMADI'}")
    w(f"- b) F3 blend < 1.3055: {scores[('F3', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kb else 'SAĞLANMADI'}")
    w(f"- c) F1 blend ≤ 1.13: {scores[('F1', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kc else 'SAĞLANMADI'}")
    w(f"- d) cold model grp_ gain ≥ %25 (F2): %{grp_cold_share['F2']:.1f} → "
      f"{'SAĞLANDI' if kd else 'SAĞLANMADI'}")
    w(f"- e) F2 warm < 0.7746: {scores[('F2', best_var)]['warm']:.4f} → "
      f"{'SAĞLANDI' if ke else 'SAĞLANMADI'}")
    hepsi = all([ka, kb, kc, kd, ke])
    w(f"- **SONUÇ: {'KABUL' if hepsi else 'KRİTER DÜŞTÜ — DUR'}**")
    w()

    # ---------------------------------------------------------- b düşerse kesim analizi
    if not kb:
        w("## 8. F3 kesim analizi — model−b6 nerede kaybediyor")
        w()
        s3 = store["F3"]
        valid3, vr3 = s3["valid"], s3["valid_rows"]
        p_model = combine("F3", [0, 1, 2] if best_var == "p3" else [0],
                          w_f2 if best_var == "p1" else w_avg)
        p_b6 = b6_hybrid(df.loc[s3["fold"]["train_idx"]], vr3).to_numpy()
        e_m = (np.log1p(np.clip(p_model, 0, None)) - np.log1p(vr3["tuketim"])) ** 2
        e_b = (np.log1p(np.clip(p_b6, 0, None)) - np.log1p(vr3["tuketim"])) ** 2
        segs = {
            "warm · sıfır": (~s3["is_cold"]) & (vr3["tuketim"] == 0),
            "warm · sıfırdışı": (~s3["is_cold"]) & (vr3["tuketim"] > 0),
            "cold · sıfır": s3["is_cold"] & (vr3["tuketim"] == 0),
            "cold · sıfırdışı": s3["is_cold"] & (vr3["tuketim"] > 0),
        }
        w("| kesim | n | model | b6 | model−b6 |")
        w("|---|---|---|---|---|")
        for name, m in segs.items():
            m = np.asarray(m)
            rm, rb = np.sqrt(e_m[m].mean()), np.sqrt(e_b[m].mean())
            w(f"| {name} | {int(m.sum()):,} | {rm:.4f} | {rb:.4f} | {rm-rb:+.4f} |")
        w()
        w("| H_bucket | n | model | b6 | model−b6 |")
        w("|---|---|---|---|---|")
        for hb, g in valid3.groupby("H_bucket", observed=True):
            m = valid3.index.isin(g.index)
            rm, rb = np.sqrt(e_m[m].mean()), np.sqrt(e_b[m].mean())
            w(f"| {hb} | {int(m.sum()):,} | {rm:.4f} | {rb:.4f} | {rm-rb:+.4f} |")
        w()

    # ---------------------------------------------------------- log
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for var in VARIANTS:
        rows.append({
            "timestamp": ts, "exp_id": f"lgbm_full_{var}",
            "feature_set": "static+cal+lvl_full+grp2+seas+horizon",
            "model": f"lgbm_{var}",
            "f1_all": round(scores[("F1", var)]["all"], 4),
            "f1_warm": round(scores[("F1", var)]["warm"], 4),
            "f1_cold": round(scores[("F1", var)]["cold"], 4),
            "f1_blend": round(scores[("F1", var)]["blend"], 4),
            "f2_all": round(scores[("F2", var)]["all"], 4),
            "f3_all": round(scores[("F3", var)]["all"], 4),
            "lb": "", "note": f"09 lvl_full w_avg={w_avg:.2f}",
        })
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not log_path.exists(), index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_v4.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_v4.md'}")


if __name__ == "__main__":
    main()
