# -*- coding: utf-8 -*-
"""
12_train_anchor.py — Mevsim-farkındalıklı init_score (anchor) + r1/r2/r3.

  r2: tek ana model, yeni anchor, saf (cold yönlendirme yok)
  r1: r2 + cold-only model yönlendirmesi
  r3: r1 + cold harmanı (w=0.45, b5 ile)
  p3r: referans — AYNI feature seti, ESKİ çapa log(guc*24), r3 kurgusu

Rapor: skorlar, çapa etkisi (F1 yan yana), gain kayması, aylık kalibrasyon (d'),
anchor'ın kendi kalibrasyonu. Çıktı: reports/model_v6.md
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import b5_guc_lf  # noqa: E402
from src.config import (EXPERIMENTS_DIR, REPORTS_DIR, SEED, TRAIN_END,
                        YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (ALL_FEATURES, build_anchor, build_features)  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set, fit_lgbm)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

B6_BLEND = {"F1": 1.2692, "F2": 1.2654, "F3": 1.3055}
W_COLD = 0.45
SEEDS = [0, 1, 2]
VARIANTS = ["r1", "r2", "r3", "p3r"]
FINAL_ROUNDS_MAIN, FINAL_ROUNDS_COLD = 126, 73

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def logmean(preds):
    return np.expm1(np.mean([np.log1p(p) for p in preds], axis=0))


def logblend(pm, pb, wgt):
    return np.expm1(wgt * np.log1p(pm) + (1 - wgt) * np.log1p(pb))


def train_pack(X_tr, y_tr, meta, X_va, y_va, is_cold_va, valid_index,
               init_tr, init_va, tag):
    """3 seed ana + cold model; tahminleri döndürür."""
    cold_mask = meta["is_cold_example"].to_numpy()
    Xvc = X_va.loc[valid_index[is_cold_va]]
    mains, colds, imp = [], [], None
    for so in SEEDS:
        print(f"  [{tag}] seed+{so} ...")
        bm, pm, _ = fit_lgbm(X_tr, y_tr, X_va, y_va, ALL_FEATURES,
                             init_tr, init_va, seed_offset=so)
        bc, pc, _ = fit_lgbm(X_tr[cold_mask], y_tr[cold_mask],
                             Xvc, y_va[is_cold_va], COLD_MODEL_FEATURES,
                             init_tr[cold_mask], init_va[is_cold_va],
                             seed_offset=so)
        mains.append(np.asarray(pm))
        colds.append(np.asarray(pc))
        if so == 0:
            imp = pd.Series(bm.feature_importance("gain"), index=ALL_FEATURES)
    return logmean(mains), logmean(colds), imp


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model v6 — mevsim-farkındalıklı çapa (r1/r2/r3 + p3r referans)")
    w()
    w(f"Üretim: `scripts/12_train_anchor.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    scores, imps = {}, {}
    for fi, fold in enumerate(folds):
        fn = fold["name"]
        print(f"[{fn}] egitim seti ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)
        valid_rows = df.loc[fold["valid_idx"]]
        train_end = fold["spec"]["train_end"]
        X_va = build_features(valid_rows, train_end, df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        y_va = np.log1p(valid_rows["tuketim"])
        valid = add_eval_columns(valid_rows, fold, df)
        is_cold_va = valid["is_cold"].to_numpy()

        anchor_tr = meta["anchor"].to_numpy()
        anchor_va = build_anchor(valid_rows, train_end,
                                 df.loc[fold["train_idx"]]).to_numpy()
        old_tr = np.log(meta["guc"].to_numpy() * 24.0)
        old_va = np.log(valid_rows["guc"].to_numpy() * 24.0)

        pm_new, pc_new, imp_new = train_pack(
            X_tr, y_tr, meta, X_va, y_va, is_cold_va, valid.index,
            anchor_tr, anchor_va, f"{fn} yeni-çapa")
        pm_old, pc_old, imp_old = train_pack(
            X_tr, y_tr, meta, X_va, y_va, is_cold_va, valid.index,
            old_tr, old_va, f"{fn} eski-çapa")
        if fn == "F1":
            imps["yeni"] = imp_new
            imps["eski"] = imp_old

        b5c = b5_guc_lf(df.loc[fold["train_idx"]],
                        valid_rows[is_cold_va]).to_numpy()

        preds = {}
        preds["r2"] = pm_new
        p = np.array(pm_new, dtype="float64"); p[is_cold_va] = pc_new
        preds["r1"] = p
        p = np.array(pm_new, dtype="float64")
        p[is_cold_va] = logblend(pc_new, b5c, W_COLD)
        preds["r3"] = p
        p = np.array(pm_old, dtype="float64")
        p[is_cold_va] = logblend(pc_old, b5c, W_COLD)
        preds["p3r"] = p

        for var, pv in preds.items():
            valid["_pred"] = pd.Series(pv, index=valid.index)
            ev = evaluate(valid, "tuketim", "_pred")
            get = lambda k, sv: float(ev.loc[
                (ev["kirilim"] == k) & (ev["seviye"] == sv), "rmsle"].iloc[0])
            scores[(fn, var)] = {
                "all": float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0]),
                "warm": get("warm_cold", "warm"), "cold": get("warm_cold", "cold"),
                "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0]),
            }

    # ------------------------------------------------------------ 1. skorlar
    w("## 1. Skorlar")
    w()
    for fn in ["F1", "F2", "F3"]:
        w(f"### {fn}  (b6 = {B6_BLEND[fn]})")
        w()
        w("| varyant | all | warm | cold | blend | model−b6 |")
        w("|---|---|---|---|---|---|")
        for var in VARIANTS:
            s = scores[(fn, var)]
            w(f"| {var} | {s['all']:.4f} | {s['warm']:.4f} | {s['cold']:.4f} | "
              f"**{s['blend']:.4f}** | {s['blend']-B6_BLEND[fn]:+.4f} |")
        w()

    # ------------------------------------------------------------ 2. çapa etkisi
    w("## 2. Çapa etkisi (F1, aynı feature seti, r3 kurgusu)")
    w()
    w(f"- Eski çapa log(guc·24): blend {scores[('F1','p3r')]['blend']:.4f} · "
      f"yeni mevsim-farkındalıklı çapa: {scores[('F1','r3')]['blend']:.4f} → "
      f"fark {scores[('F1','r3')]['blend']-scores[('F1','p3r')]['blend']:+.4f}")
    w()

    # ------------------------------------------------------------ 3. gain kayması
    w("## 3. F1 gain — lvl_lf_median_90d payı")
    w()
    for name, imp in imps.items():
        tot = imp.sum()
        pay = 100 * imp.get("lvl_lf_median_90d", 0) / tot
        top5 = imp.sort_values(ascending=False).head(5)
        w(f"- **{name} çapa:** lvl_lf_median_90d %{pay:.1f} · ilk 5: "
          + " · ".join(f"{f} %{100*v/tot:.1f}" for f, v in top5.items()))
    w()

    # ------------------------------------------------------------ 4-6. tam eğitim sağlık
    w("## 4. Tam-eğitim aylık kalibrasyon (d') — 1 seed")
    w()
    print("[FULL] tam egitim ...")
    tr, te = df, load_test()
    ORIGINS["FULL"] = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                       "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                       "2025-10-31", "2025-11-30"]
    pseudo = {"name": "FULL", "train_idx": tr.index,
              "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(tr, pseudo, profile, fold_i=9)
    Xt = build_features(te, TRAIN_END, tr)
    align_categories([Xf, Xt])
    anchor_f = metaf["anchor"].to_numpy()
    anchor_t = build_anchor(te, TRAIN_END, tr).to_numpy()
    old_f = np.log(metaf["guc"].to_numpy() * 24.0)
    old_t = np.log(te["guc"].to_numpy() * 24.0)
    cold_f = metaf["is_cold_example"].to_numpy()
    is_cold_te = ~te["tanim"].isin(set(tr["tanim"].unique())).to_numpy()

    import lightgbm as lgb

    def fit_full(X, y, feats, init, rounds):
        ds = lgb.Dataset(X[feats], label=y, init_score=init,
                         categorical_feature=[c for c in feats
                                              if str(X[c].dtype) == "category"])
        return lgb.train(LGB_PARAMS, ds, num_boost_round=rounds)

    bm_new = fit_full(Xf, yf, ALL_FEATURES, anchor_f, FINAL_ROUNDS_MAIN)
    p_main_new = np.clip(np.expm1(
        bm_new.predict(Xt[ALL_FEATURES]) + anchor_t), 0, None)
    bc_new = fit_full(Xf[cold_f], yf[cold_f], COLD_MODEL_FEATURES,
                      anchor_f[cold_f], FINAL_ROUNDS_COLD)
    p_cold_new = np.clip(np.expm1(
        bc_new.predict(Xt.loc[is_cold_te, COLD_MODEL_FEATURES])
        + anchor_t[is_cold_te]), 0, None)
    bm_old = fit_full(Xf, yf, ALL_FEATURES, old_f, FINAL_ROUNDS_MAIN)
    p_main_old = np.clip(np.expm1(
        bm_old.predict(Xt[ALL_FEATURES]) + old_t), 0, None)
    bc_old = fit_full(Xf[cold_f], yf[cold_f], COLD_MODEL_FEATURES,
                      old_f[cold_f], FINAL_ROUNDS_COLD)
    p_cold_old = np.clip(np.expm1(
        bc_old.predict(Xt.loc[is_cold_te, COLD_MODEL_FEATURES])
        + old_t[is_cold_te]), 0, None)
    p_b5t = b5_guc_lf(tr, te[is_cold_te]).to_numpy()

    test_preds = {}
    test_preds["r2"] = p_main_new
    p = p_main_new.copy(); p[is_cold_te] = p_cold_new
    test_preds["r1"] = p
    p = p_main_new.copy()
    p[is_cold_te] = logblend(p_cold_new, p_b5t, W_COLD)
    test_preds["r3"] = p
    p = p_main_old.copy()
    p[is_cold_te] = logblend(p_cold_old, p_b5t, W_COLD)
    test_preds["p3r"] = p

    # 2025 tabanları — tüm test trafoları + tam-kapsamlı kohort
    te_ay = te["tarih"].dt.to_period("M")
    te_tx = set(te["tanim"].unique())
    tr_s = tr[tr["tanim"].isin(te_tx)].copy()
    tr_s["ay_p"] = tr_s["tarih"].dt.to_period("M")
    aylar = [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]
    cnt = tr_s[tr_s["ay_p"].isin(aylar)].groupby("tanim", observed=True)["tarih"].nunique()
    full_cov = set(cnt[cnt >= 110].index)   # 122 günün ≥110'u mevcut
    base_all, base_cov = {}, {}
    for m in (4, 5, 6, 7):
        rows = tr_s[tr_s["ay_p"] == pd.Period(f"2025-{m:02d}")]
        base_all[m] = float(np.log1p(rows["tuketim"]).mean())
        base_cov[m] = float(np.log1p(
            rows.loc[rows["tanim"].isin(full_cov), "tuketim"]).mean())
    w(f"- 2025 tabanı: tüm test trafoları (kompozisyon tam eş DEĞİL — bazılarının "
      f"2025 verisi yok) ve tam-kapsamlı kohort ({len(full_cov):,} trafo, ≥110/122 gün).")
    w()

    def calib_table(pred, mask=None, base=base_all, label=""):
        rows = []
        for m in (4, 5, 6, 7):
            sel = (te_ay == pd.Period(f"2026-{m:02d}")).to_numpy()
            if mask is not None:
                sel = sel & mask
            p26 = float(np.log1p(pred[sel]).mean())
            rows.append((m, p26, p26 - base[m] - YOY_DRIFT))
        return rows

    d_ok = {}
    w("| varyant | Nis | May | Haz | Tem | max |sapma| | d' |")
    w("|---|---|---|---|---|---|---|")
    for var in VARIANTS:
        devs = [r[2] for r in calib_table(test_preds[var])]
        mx = max(abs(d) for d in devs)
        d_ok[var] = mx <= 0.12
        w(f"| {var} | " + " | ".join(f"{d:+.3f}" for d in devs)
          + f" | {mx:.3f} | {'✓' if d_ok[var] else '✗'} |")
    w()
    w("Tam-kapsamlı kohort tabanıyla (kompozisyon-eş):")
    w()
    w("| varyant | Nis | May | Haz | Tem |")
    w("|---|---|---|---|---|")
    for var in VARIANTS:
        devs = [r[2] for r in calib_table(test_preds[var], base=base_cov)]
        w(f"| {var} | " + " | ".join(f"{d:+.3f}" for d in devs) + " |")
    w()

    # 5. warm/cold ayrı
    w("## 5. Aylık kalibrasyon — warm / cold ayrı (tüm-trafo tabanı)")
    w()
    for seg, mask in [("warm", ~is_cold_te), ("cold", is_cold_te)]:
        w(f"### {seg}")
        w()
        w("| varyant | Nis | May | Haz | Tem |")
        w("|---|---|---|---|---|")
        for var in VARIANTS:
            devs = [r[2] for r in calib_table(test_preds[var], mask=mask)]
            w(f"| {var} | " + " | ".join(f"{d:+.3f}" for d in devs) + " |")
        w()

    # 6. anchor'ın kendi kalibrasyonu
    w("## 6. Anchor kalibrasyonu (init_score aylık ortalaması vs 2025+drift)")
    w()
    w("| ay | anchor ort. | 2025+drift | fark |")
    w("|---|---|---|---|")
    for m in (4, 5, 6, 7):
        sel = (te_ay == pd.Period(f"2026-{m:02d}")).to_numpy()
        a = float(anchor_t[sel].mean())
        w(f"| {m:02d} | {a:.4f} | {base_all[m]+YOY_DRIFT:.4f} | "
          f"{a-base_all[m]-YOY_DRIFT:+.4f} |")
    w()

    # ------------------------------------------------------------ kabul
    rvars = ["r1", "r2", "r3"]
    best_var = min(rvars, key=lambda v: scores[("F1", v)]["blend"])
    w("## 7. Kabul kriterleri")
    w()
    ka = scores[("F1", best_var)]["blend"] <= 1.13
    kb = scores[("F3", best_var)]["blend"] <= 1.3155
    kc = d_ok[best_var]
    w(f"- En iyi varyant (F1 blend): **{best_var}**")
    w(f"- a) F1 blend ≤ 1.13: {scores[('F1', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if ka else 'SAĞLANMADI'}")
    w(f"- b) F3 blend ≤ 1.3155: {scores[('F3', best_var)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kb else 'SAĞLANMADI'}")
    w(f"- c) d' dört ayda sapma ≤ 0.12: {'✓' if kc else '✗'} → "
      f"{'SAĞLANDI' if kc else 'SAĞLANMADI'}")
    w(f"- **SONUÇ: {'KABUL' if all([ka, kb, kc]) else 'KRİTER DÜŞTÜ — DUR'}**")
    w()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for var in VARIANTS:
        rows.append({
            "timestamp": ts, "exp_id": f"lgbm_anchor_{var}",
            "feature_set": "clean+anchor" if var != "p3r" else "clean+oldanchor",
            "model": f"lgbm_{var}",
            "f1_all": round(scores[("F1", var)]["all"], 4),
            "f1_warm": round(scores[("F1", var)]["warm"], 4),
            "f1_cold": round(scores[("F1", var)]["cold"], 4),
            "f1_blend": round(scores[("F1", var)]["blend"], 4),
            "f2_all": round(scores[("F2", var)]["all"], 4),
            "f3_all": round(scores[("F3", var)]["all"], 4),
            "lb": "", "note": "12 mevsim-farkindalikli anchor",
        })
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not log_path.exists(), index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_v6.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_v6.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # çökme durumunda da o ana kadarki rapor diske yazılsın
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_v6.md").write_text(out.getvalue(),
                                                 encoding="utf-8")
