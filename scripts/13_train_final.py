# -*- coding: utf-8 -*-
"""
13_train_final.py — A×B çarpımı: s1/s2/s3 + p3r/r3 referans.

  p3r: eski çapa log(guc·24), ALL_FEATURES
  r3 : yeni çapa (alpha=1, cold_adj=False), ALL_FEATURES
  s1 : yeni çapa (alpha=1, cold_adj=True) + A (mevsim cal çıkarıldı)
  s2 : yeni çapa (alpha=α*, cold_adj=True) + B, takvim aynen
  s3 : yeni çapa (alpha=α*, cold_adj=True) + A + B

A = ALL_FEATURES − {cal_ay, cal_doy_sin, cal_doy_cos, cal_hafta} (ana ve cold model).
B = alpha yumuşatma; α* kalibrasyona göre grid'den (min max|aylık sapma|).
Cold anchor sıfır düzeltmesi: base += log(1 − zero_rate).

Anchor bileşenlerden runtime kurulur → feature build fold başına bir kez.
Çıktı: reports/model_v7.md · submissions/sub_s.csv · experiments/log.csv
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
from src.config import (EXPERIMENTS_DIR, REPORTS_DIR, SEED, SUBMISSIONS_DIR,
                        TRAIN_END, YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.predict import write_submission  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

B6_BLEND = {"F1": 1.2692, "F2": 1.2654, "F3": 1.3055}
W_COLD = 0.45
SEEDS = [0, 1, 2]
ALPHA_GRID = [0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
MEVSIM_CAL = ["cal_ay", "cal_doy_sin", "cal_doy_cos", "cal_hafta"]
FEATS_A = [f for f in ALL_FEATURES if f not in MEVSIM_CAL]
COLD_A = [f for f in COLD_MODEL_FEATURES if f not in MEVSIM_CAL]
FINAL_ROUNDS_MAIN, FINAL_ROUNDS_COLD = 126, 73
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def anc_from_meta(meta, alpha, cold_adj):
    a = meta["anc_base"] + alpha * meta["anc_dev"]
    if cold_adj:
        a = a + meta["anc_zero"]
    return a.to_numpy()


def anc_from_comp(comp, alpha, cold_adj):
    a = comp["base"] + alpha * comp["season_dev"]
    if cold_adj:
        a = a + comp["zero_adj"]
    return a.to_numpy()


def logmean(preds):
    return np.expm1(np.mean([np.log1p(p) for p in preds], axis=0))


def logblend(pm, pb, wgt):
    return np.expm1(wgt * np.log1p(pm) + (1 - wgt) * np.log1p(pb))


def fit(X, y, feats, init, rounds, seed_off, early=None, Xv=None, yv=None, iv=None):
    import lightgbm as lgb
    params = dict(LGB_PARAMS)
    for k in ("seed", "feature_fraction_seed", "bagging_seed"):
        params[k] = LGB_PARAMS[k] + seed_off
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    ds = lgb.Dataset(X[feats], label=y, init_score=init, categorical_feature=cats)
    cbs = []
    valid_sets = None
    if Xv is not None:
        valid_sets = [lgb.Dataset(Xv[feats], label=yv, init_score=iv, reference=ds)]
        cbs = [lgb.early_stopping(300, verbose=False)]
        rounds = 5000
    booster = lgb.train(params, ds, num_boost_round=rounds,
                        valid_sets=valid_sets, callbacks=cbs)
    return booster


def predict(booster, X, feats, init):
    it = booster.best_iteration or booster.current_iteration()
    return np.clip(np.expm1(booster.predict(X[feats], num_iteration=it) + init),
                   0, None)


def main():
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)
    te = load_test()

    w("# Model v7 — A×B çarpımı (s1/s2/s3) + p3r/r3 referans")
    w()
    w(f"Üretim: `scripts/13_train_final.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    # ===================== fold verilerini bir kez kur =====================
    fold_data = {}
    for fi, fold in enumerate(folds):
        fn = fold["name"]
        print(f"[{fn}] feature build ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile, fi)
        valid_rows = df.loc[fold["valid_idx"]]
        X_va = build_features(valid_rows, fold["spec"]["train_end"],
                              df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        y_va = np.log1p(valid_rows["tuketim"])
        valid = add_eval_columns(valid_rows, fold, df)
        is_cold = valid["is_cold"].to_numpy()
        comp_va = anchor_components(valid_rows, fold["spec"]["train_end"],
                                    df.loc[fold["train_idx"]])
        old_tr = np.log(meta["guc"].to_numpy() * 24.0)
        old_va = np.log(valid_rows["guc"].to_numpy() * 24.0)
        b5c = b5_guc_lf(df.loc[fold["train_idx"]],
                        valid_rows[is_cold]).to_numpy()
        fold_data[fn] = dict(
            X_tr=X_tr, y_tr=y_tr, meta=meta, X_va=X_va, y_va=y_va, valid=valid,
            is_cold=is_cold, comp_va=comp_va, old_tr=old_tr, old_va=old_va,
            b5c=b5c, cold_mask=meta["is_cold_example"].to_numpy(),
            valid_rows=valid_rows, y=valid_rows["tuketim"].to_numpy())

    # ===================== tam eğitim verisi bir kez =====================
    print("[FULL] feature build ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, fold_i=9)
    Xt = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xt])
    comp_t = anchor_components(te, TRAIN_END, df)
    old_f = np.log(metaf["guc"].to_numpy() * 24.0)
    old_t = np.log(te["guc"].to_numpy() * 24.0)
    cold_f = metaf["is_cold_example"].to_numpy()
    is_cold_te = ~te["tanim"].isin(set(df["tanim"].unique())).to_numpy()
    b5t = b5_guc_lf(df, te[is_cold_te]).to_numpy()
    te_ay = te["tarih"].dt.to_period("M")

    # kohort-eş 2025 tabanı (≥110/122 gün)
    te_tx = set(te["tanim"].unique())
    tr_s = df[df["tanim"].isin(te_tx)].copy()
    tr_s["ay_p"] = tr_s["tarih"].dt.to_period("M")
    aylar = [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]
    cnt = tr_s[tr_s["ay_p"].isin(aylar)].groupby("tanim", observed=True)["tarih"].nunique()
    cov = set(cnt[cnt >= 110].index)
    base_cov = {m: float(np.log1p(tr_s.loc[
        (tr_s["ay_p"] == pd.Period(f"2025-{m:02d}"))
        & (tr_s["tanim"].isin(cov)), "tuketim"]).mean()) for m in (4, 5, 6, 7)}

    def calib(pred, mask=None, base=base_cov):
        rows = []
        for m in (4, 5, 6, 7):
            sel = (te_ay == pd.Period(f"2026-{m:02d}")).to_numpy()
            if mask is not None:
                sel = sel & mask
            p26 = float(np.log1p(pred[sel]).mean())
            rows.append(p26 - base[m] - YOY_DRIFT)
        return rows

    # full eğitim tahmini üreten yardımcı (1 seed, alpha/feats/adj/anchor'a göre)
    def full_predict(anchor_kind, feats, cold_feats, alpha, cold_adj, seeds):
        if anchor_kind == "old":
            init_f, init_t = old_f, old_t
            init_fc, init_tc = old_f[cold_f], old_t[is_cold_te]
        else:
            init_f = anc_from_meta(metaf, alpha, cold_adj)
            init_t = anc_from_comp(comp_t, alpha, cold_adj)
            init_fc, init_tc = init_f[cold_f], init_t[is_cold_te]
        mains, colds = [], []
        for so in seeds:
            bm = fit(Xf, yf, feats, init_f, FINAL_ROUNDS_MAIN, so)
            mains.append(predict(bm, Xt, feats, init_t))
            bc = fit(Xf[cold_f], yf[cold_f], cold_feats, init_fc,
                     FINAL_ROUNDS_COLD, so)
            colds.append(predict(bc, Xt.loc[is_cold_te], cold_feats, init_tc))
        pm = logmean(mains)
        pc = logmean(colds)
        p = pm.copy()
        p[is_cold_te] = logblend(pc, b5t, W_COLD)
        return p, pm, pc

    # ===================== FAZ A: alpha grid (kalibrasyona göre) =====================
    print("== FAZ A: alpha grid ==")
    w("## 1. Alpha grid (B) — α seçimi kalibrasyona göre")
    w()
    w("| alpha | Nis | May | Haz | Tem | max|sapma| | F1 blend |")
    w("|---|---|---|---|---|---|---|")
    fd1 = fold_data["F1"]
    grid_rows = {}
    for a in ALPHA_GRID:
        # tam eğitim kalibrasyon (1 seed), cold_adj=True (s kurgusu)
        p_full, _, _ = full_predict("new", ALL_FEATURES, COLD_MODEL_FEATURES,
                                    a, True, [0])
        devs = calib(p_full)
        mx = max(abs(d) for d in devs)
        # F1 blend (1 seed)
        init_tr = anc_from_meta(fd1["meta"], a, True)
        init_va = anc_from_comp(fd1["comp_va"], a, True)
        bm = fit(fd1["X_tr"], fd1["y_tr"], ALL_FEATURES, init_tr, None, 0,
                 Xv=fd1["X_va"], yv=fd1["y_va"], iv=init_va)
        pm = predict(bm, fd1["X_va"], ALL_FEATURES, init_va)
        cm = fd1["cold_mask"]
        ivc = init_va[fd1["is_cold"]]
        bc = fit(fd1["X_tr"][cm], fd1["y_tr"][cm], COLD_MODEL_FEATURES,
                 init_tr[cm], None, 0,
                 Xv=fd1["X_va"].loc[fd1["valid"].index[fd1["is_cold"]]],
                 yv=fd1["y_va"][fd1["is_cold"]], iv=ivc)
        pc = predict(bc, fd1["X_va"].loc[fd1["valid"].index[fd1["is_cold"]]],
                     COLD_MODEL_FEATURES, ivc)
        p = pm.copy()
        p[fd1["is_cold"]] = logblend(pc, fd1["b5c"], W_COLD)
        vv = fd1["valid"].copy()
        vv["_pred"] = pd.Series(p, index=vv.index)
        ev = evaluate(vv, "tuketim", "_pred")
        f1b = float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])
        grid_rows[a] = (devs, mx, f1b)
        w(f"| {a} | " + " | ".join(f"{d:+.3f}" for d in devs)
          + f" | {mx:.3f} | {f1b:.4f} |")
    alpha_star = min(ALPHA_GRID, key=lambda a: grid_rows[a][1])
    w()
    w(f"- **Seçilen α* = {alpha_star}** (min max|aylık sapma| = "
      f"{grid_rows[alpha_star][1]:.3f})")
    w()

    # ===================== FAZ B: 5 varyant CV (1 seed) =====================
    print("== FAZ B: varyant CV ==")
    VARIANTS = {
        "p3r": dict(anchor="old", feats=ALL_FEATURES, cold_feats=COLD_MODEL_FEATURES,
                    alpha=1.0, cold_adj=False),
        "r3":  dict(anchor="new", feats=ALL_FEATURES, cold_feats=COLD_MODEL_FEATURES,
                    alpha=1.0, cold_adj=False),
        "s1":  dict(anchor="new", feats=FEATS_A, cold_feats=COLD_A,
                    alpha=1.0, cold_adj=True),
        "s2":  dict(anchor="new", feats=ALL_FEATURES, cold_feats=COLD_MODEL_FEATURES,
                    alpha=alpha_star, cold_adj=True),
        "s3":  dict(anchor="new", feats=FEATS_A, cold_feats=COLD_A,
                    alpha=alpha_star, cold_adj=True),
    }
    scores = {}
    for fn in ["F1", "F2", "F3"]:
        fdd = fold_data[fn]
        for var, cfg in VARIANTS.items():
            print(f"[{fn}] {var} ...")
            if cfg["anchor"] == "old":
                itr, iva = fdd["old_tr"], fdd["old_va"]
            else:
                itr = anc_from_meta(fdd["meta"], cfg["alpha"], cfg["cold_adj"])
                iva = anc_from_comp(fdd["comp_va"], cfg["alpha"], cfg["cold_adj"])
            bm = fit(fdd["X_tr"], fdd["y_tr"], cfg["feats"], itr, None, 0,
                     Xv=fdd["X_va"], yv=fdd["y_va"], iv=iva)
            pm = predict(bm, fdd["X_va"], cfg["feats"], iva)
            cm = fdd["cold_mask"]
            vc_idx = fdd["valid"].index[fdd["is_cold"]]
            ivc = iva[fdd["is_cold"]]
            bc = fit(fdd["X_tr"][cm], fdd["y_tr"][cm], cfg["cold_feats"], itr[cm],
                     None, 0, Xv=fdd["X_va"].loc[vc_idx],
                     yv=fdd["y_va"][fdd["is_cold"]], iv=ivc)
            pc = predict(bc, fdd["X_va"].loc[vc_idx], cfg["cold_feats"], ivc)
            p = pm.copy()
            p[fdd["is_cold"]] = logblend(pc, fdd["b5c"], W_COLD)
            vv = fdd["valid"].copy()
            vv["_pred"] = pd.Series(p, index=vv.index)
            ev = evaluate(vv, "tuketim", "_pred")
            get = lambda k, sv: float(ev.loc[
                (ev["kirilim"] == k) & (ev["seviye"] == sv), "rmsle"].iloc[0])
            scores[(fn, var)] = {
                "all": float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0]),
                "warm": get("warm_cold", "warm"), "cold": get("warm_cold", "cold"),
                "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])}

    w("## 2. Skorlar (5 varyant × 3 fold, 1 seed CV)")
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

    # ===================== FAZ C: tam eğitim kalibrasyon (5 varyant, 1 seed) =====
    print("== FAZ C: tam egitim kalibrasyon ==")
    full_preds = {}
    for var, cfg in VARIANTS.items():
        print(f"[FULL] {var} ...")
        p, pm, pc = full_predict(cfg["anchor"], cfg["feats"], cfg["cold_feats"],
                                 cfg["alpha"], cfg["cold_adj"], [0])
        full_preds[var] = (p, pm, pc)

    w("## 3. KOHORT-EŞ aylık kalibrasyon (birincil karar) — max|sapma| eşik 0.15")
    w()
    w("| varyant | Nis | May | Haz | Tem | max|sapma| | ✓ |")
    w("|---|---|---|---|---|---|---|")
    calib_ok = {}
    for var in VARIANTS:
        devs = calib(full_preds[var][0])
        mx = max(abs(d) for d in devs)
        calib_ok[var] = mx <= 0.15
        w(f"| {var} | " + " | ".join(f"{d:+.3f}" for d in devs)
          + f" | {mx:.3f} | {'✓' if calib_ok[var] else '✗'} |")
    w()

    w("## 4. Kalibrasyon — warm / cold ayrı (kohort-eş taban)")
    w()
    for seg, mask in [("warm", ~is_cold_te), ("cold", is_cold_te)]:
        w(f"### {seg}")
        w()
        w("| varyant | Nis | May | Haz | Tem |")
        w("|---|---|---|---|---|")
        for var in VARIANTS:
            devs = calib(full_preds[var][0], mask=mask)
            w(f"| {var} | " + " | ".join(f"{d:+.3f}" for d in devs) + " |")
        w()

    # ===================== 5. cold bias öncesi/sonrası =====================
    w("## 5. Cold seviye bias — sıfır düzeltmesi öncesi/sonrası")
    w()
    # bias = cold satır tahmin log ort − aynı (ilce,ay,bucket) warm tahmin log ort
    def cold_bias(pred):
        tmp = te.copy()
        tmp["lp"] = np.log1p(pred)
        tmp["cold"] = is_cold_te
        tmp["ay"] = tmp["tarih"].dt.month
        warm_grp = (tmp[~tmp["cold"]].groupby(["ilce_key", "ay", "guc_bucket"],
                    observed=True)["lp"].mean())
        cold_rows = tmp[tmp["cold"]]
        idx = pd.MultiIndex.from_frame(cold_rows[["ilce_key", "ay", "guc_bucket"]])
        warm_ref = pd.Series(pd.to_numeric(warm_grp.reindex(idx), errors="coerce")
                             .to_numpy(), index=cold_rows.index)
        diff = cold_rows["lp"] - warm_ref
        return float(diff.mean(skipna=True))

    # r3 = düzeltmesiz, s2 = düzeltmeli (aynı alpha değil ama cold_adj farkı temsili)
    # net öncesi/sonrası: r3 cold anchor'ı adj=False, s2 adj=True
    w("| varyant | cold_adj | cold bias (cold−warm eş grup) |")
    w("|---|---|---|")
    for var in ["r3", "s2", "s3"]:
        w(f"| {var} | {VARIANTS[var]['cold_adj']} | "
          f"{cold_bias(full_preds[var][0]):+.4f} |")
    w()

    # ===================== 6. F1 gain =====================
    w("## 6. F1 ana model gain — mevsim çapası çift-sayımı kırıldı mı")
    w()
    # s3 (A: mevsim cal çıkarılmış) F1 ana modelini gain için yeniden eğit
    fdd = fold_data["F1"]
    itr = anc_from_meta(fdd["meta"], alpha_star, True)
    iva = anc_from_comp(fdd["comp_va"], alpha_star, True)
    bm = fit(fdd["X_tr"], fdd["y_tr"], FEATS_A, itr, None, 0,
             Xv=fdd["X_va"], yv=fdd["y_va"], iv=iva)
    imp = pd.Series(bm.feature_importance("gain"), index=FEATS_A)
    tot = imp.sum()
    w(f"- s3 F1: lvl_lf_median_90d %{100*imp.get('lvl_lf_median_90d',0)/tot:.1f}")
    w(f"- kalan mevsim feature (FEATS_A'da doy/ay yok): "
      f"cal_horizon_days %{100*imp.get('cal_horizon_days',0)/tot:.1f} · "
      f"seas_ toplam %{100*imp.reindex(FEATURE_GROUPS['seas']).fillna(0).sum()/tot:.1f}")
    top = imp.sort_values(ascending=False).head(8)
    w("- ilk 8: " + " · ".join(f"{f} %{100*v/tot:.1f}" for f, v in top.items()))
    w()

    # ===================== 7. kabul + submission =====================
    best = min(VARIANTS, key=lambda v: max(abs(d) for d in calib(full_preds[v][0])))
    devs_b = calib(full_preds[best][0])
    mx_b = max(abs(d) for d in devs_b)
    w("## 7. Kabul kriterleri")
    w()
    ka = mx_b <= 0.15
    kb = scores[("F1", best)]["blend"] <= 1.14
    kc = scores[("F3", best)]["blend"] <= 1.3155
    kd = abs(cold_bias(full_preds[best][0])) <= 0.15
    w(f"- Kazanan (min max|sapma|): **{best}**")
    w(f"- a) max|aylık sapma| ≤ 0.15: {mx_b:.3f} → {'SAĞLANDI' if ka else 'SAĞLANMADI'}")
    w(f"- b) F1 blend ≤ 1.14: {scores[('F1',best)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kb else 'SAĞLANMADI'}")
    w(f"- c) F3 blend ≤ 1.3155: {scores[('F3',best)]['blend']:.4f} → "
      f"{'SAĞLANDI' if kc else 'SAĞLANMADI'}")
    w(f"- d) |cold bias| ≤ 0.15: {abs(cold_bias(full_preds[best][0])):.4f} → "
      f"{'SAĞLANDI' if kd else 'SAĞLANMADI'}")
    w(f"- **SONUÇ: {'KABUL' if all([ka,kb,kc,kd]) else 'KRİTER DÜŞTÜ'}**")
    w()

    # kazanan varyantla 3-seed submission
    print(f"== submission: {best} 3-seed ==")
    cfg = VARIANTS[best]
    p_sub, _, _ = full_predict(cfg["anchor"], cfg["feats"], cfg["cold_feats"],
                               cfg["alpha"], cfg["cold_adj"], SEEDS)
    sub = pd.DataFrame({"id": te["id"], "tuketim": p_sub})
    write_submission(sub, SUBMISSIONS_DIR / "sub_s.csv")
    w(f"- submissions/sub_s.csv yazıldı (kazanan={best}, 3-seed).")
    w()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "log.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for var in VARIANTS:
        rows.append({
            "timestamp": ts, "exp_id": f"lgbm_v7_{var}",
            "feature_set": "anchor-AB", "model": f"lgbm_{var}",
            "f1_all": round(scores[("F1", var)]["all"], 4),
            "f1_warm": round(scores[("F1", var)]["warm"], 4),
            "f1_cold": round(scores[("F1", var)]["cold"], 4),
            "f1_blend": round(scores[("F1", var)]["blend"], 4),
            "f2_all": round(scores[("F2", var)]["all"], 4),
            "f3_all": round(scores[("F3", var)]["all"], 4),
            "lb": "", "note": f"13 AxB alpha*={alpha_star} best={best}"}
            )
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not log_path.exists(), index=False)
    w(f"- experiments/log.csv güncellendi ({len(rows)} satır)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_v7.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_v7.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_v7.md").write_text(out.getvalue(), encoding="utf-8")
