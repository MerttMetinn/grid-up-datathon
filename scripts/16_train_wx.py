# -*- coding: utf-8 -*-
"""
16_train_wx.py — Hava (wx_) with/without: s2 kurgusu üzerinde A/B.

s2 sabit: mevsim-farkındalıklı anchor (α=0.4, cold_adj=True), cold-only model +
b5 harmanı (w=0.45), 3-seed (CV'de 1 seed).

  t0 = wx YOK  (ALL_FEATURES − wx; cold model static+cal+grp)   ← s2 referans
  t1 = wx VAR  (tüm 75 feature; cold model + wx)

F2 = wx_/seas_ karar fold'u (CLAUDE.md): mutlak skor değil, with/without DELTA okunur.
Çıktı: reports/model_wx.md · (t1 kazanırsa) submissions/sub_wx.csv
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

from src.baselines import b5_guc_lf  # noqa: E402
from src.config import (REPORTS_DIR, SEED, SUBMISSIONS_DIR, TRAIN_END,
                        YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.predict import write_submission  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set)  # noqa: E402
from src.validation import add_eval_columns, evaluate, make_folds  # noqa: E402

B6_BLEND = {"F1": 1.2692, "F2": 1.2654, "F3": 1.3055}
S2_BLEND = {"F1": 1.1244, "F2": 1.2432, "F3": 1.2479}   # reports/model_v7.md
ALPHA, W_COLD = 0.4, 0.45
SEEDS = [0, 1, 2]
FINAL_ROUNDS_MAIN, FINAL_ROUNDS_COLD = 126, 73
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]

WX = FEATURE_GROUPS["wx"]
MAIN_NOWX = [f for f in ALL_FEATURES if f not in WX]
MAIN_WX = ALL_FEATURES
COLD_NOWX = COLD_MODEL_FEATURES
COLD_WX = COLD_MODEL_FEATURES + WX
FEATS = {"t0": (MAIN_NOWX, COLD_NOWX), "t1": (MAIN_WX, COLD_WX)}

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def assemble(base, dev, zero):
    return (base + ALPHA * dev + zero).to_numpy()   # cold_adj=True (zero warm'da 0)


def fit(X, y, feats, init, rounds, so):
    params = dict(LGB_PARAMS)
    for k in ("seed", "feature_fraction_seed", "bagging_seed"):
        params[k] = LGB_PARAMS[k] + so
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    ds = lgb.Dataset(X[feats], label=y, init_score=init, categorical_feature=cats)
    return lgb.train(params, ds, num_boost_round=rounds)


def predict(b, X, feats, init):
    return np.clip(np.expm1(b.predict(X[feats]) + init), 0, None)


def logmean(ps):
    return np.expm1(np.mean([np.log1p(p) for p in ps], axis=0))


def logblend(pm, pb, wgt):
    return np.expm1(wgt * np.log1p(pm) + (1 - wgt) * np.log1p(pb))


def main():
    df = load_train()
    te = load_test()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Model wx — hava with/without (s2 üzerinde)")
    w()
    w(f"Üretim: `scripts/16_train_wx.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    scores, imp_f2 = {}, {}
    for fold in folds:
        fn = fold["name"]
        print(f"[{fn}] feature build ...")
        X_tr, y_tr, meta = build_training_set(df, fold, profile,
                                              {"F1": 0, "F2": 1, "F3": 2}[fn])
        vr = df.loc[fold["valid_idx"]]
        X_va = build_features(vr, fold["spec"]["train_end"], df.loc[fold["train_idx"]])
        align_categories([X_tr, X_va])
        y_va = np.log1p(vr["tuketim"])
        valid = add_eval_columns(vr, fold, df)
        is_cold = valid["is_cold"].to_numpy()
        comp_va = anchor_components(vr, fold["spec"]["train_end"],
                                   df.loc[fold["train_idx"]])
        itr = assemble(meta["anc_base"], meta["anc_dev"], meta["anc_zero"])
        iva = assemble(comp_va["base"], comp_va["season_dev"], comp_va["zero_adj"])
        cm = meta["is_cold_example"].to_numpy()
        vc_idx = valid.index[is_cold]
        b5c = b5_guc_lf(df.loc[fold["train_idx"]], vr[is_cold]).to_numpy()

        for var, (mfeats, cfeats) in FEATS.items():
            print(f"[{fn}] {var} ...")
            bm = fit(X_tr, y_tr, mfeats, itr, FINAL_ROUNDS_MAIN, 0)
            pm = predict(bm, X_va, mfeats, iva)
            bc = fit(X_tr[cm], y_tr[cm], cfeats, itr[cm], FINAL_ROUNDS_COLD, 0)
            pc = predict(bc, X_va.loc[vc_idx], cfeats, iva[is_cold])
            p = pm.copy()
            p[is_cold] = logblend(pc, b5c, W_COLD)
            vv = valid.copy()
            vv["_pred"] = pd.Series(p, index=vv.index)
            ev = evaluate(vv, "tuketim", "_pred")
            g = lambda k, s: float(ev.loc[(ev["kirilim"] == k) &
                                          (ev["seviye"] == s), "rmsle"].iloc[0])
            scores[(fn, var)] = {
                "all": float(ev.loc[ev["kirilim"] == "global", "rmsle"].iloc[0]),
                "warm": g("warm_cold", "warm"), "cold": g("warm_cold", "cold"),
                "blend": float(ev.loc[ev["kirilim"] == "blend", "rmsle"].iloc[0])}
            if fn == "F2" and var == "t1":
                imp_f2 = pd.Series(bm.feature_importance("gain"), index=mfeats)

    # ---- skorlar + delta ----
    w("## 1. Skorlar — t0 (wx yok) vs t1 (wx var)")
    w()
    for fn in ["F1", "F2", "F3"]:
        w(f"### {fn}  (b6={B6_BLEND[fn]} · s2={S2_BLEND[fn]})")
        w()
        w("| var | all | warm | cold | blend | wx Δ(blend) |")
        w("|---|---|---|---|---|---|")
        d = scores[(fn, "t1")]["blend"] - scores[(fn, "t0")]["blend"]
        for var in ["t0", "t1"]:
            s = scores[(fn, var)]
            dd = f"{d:+.4f}" if var == "t1" else ""
            w(f"| {var} | {s['all']:.4f} | {s['warm']:.4f} | {s['cold']:.4f} | "
              f"**{s['blend']:.4f}** | {dd} |")
        w()

    w("## 2. F2 wx_ önem payı (karar fold'u)")
    w()
    tot = imp_f2.sum()
    wx_gain = 100 * imp_f2.reindex(WX).fillna(0).sum() / tot
    w(f"- **wx_ ailesi toplam gain (F2): %{wx_gain:.1f}**")
    top = imp_f2.sort_values(ascending=False).head(12)
    w("- ilk 12: " + " · ".join(f"{f} %{100*v/tot:.1f}" for f, v in top.items()))
    w()

    # ---- tam eğitim kalibrasyon (her iki varyant, 1 seed) ----
    print("[FULL] tam egitim ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, 9)
    Xt = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xt])
    comp_t = anchor_components(te, TRAIN_END, df)
    itr = assemble(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"])
    itt = assemble(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"])
    cold_f = metaf["is_cold_example"].to_numpy()
    is_cold_te = ~te["tanim"].isin(set(df["tanim"].unique())).to_numpy()
    b5t = b5_guc_lf(df, te[is_cold_te]).to_numpy()
    te_ay = te["tarih"].dt.to_period("M")

    tr_s = df[df["tanim"].isin(set(te["tanim"].unique()))].copy()
    tr_s["ay_p"] = tr_s["tarih"].dt.to_period("M")
    aylar = [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]
    cnt = tr_s[tr_s["ay_p"].isin(aylar)].groupby("tanim", observed=True)["tarih"].nunique()
    cov = set(cnt[cnt >= 110].index)
    base_cov = {m: float(np.log1p(tr_s.loc[
        (tr_s["ay_p"] == pd.Period(f"2025-{m:02d}")) &
        (tr_s["tanim"].isin(cov)), "tuketim"]).mean()) for m in (4, 5, 6, 7)}

    def calib(pred):
        return [float(np.log1p(pred[(te_ay == pd.Period(f'2026-{m:02d}')).to_numpy()]).mean())
                - base_cov[m] - YOY_DRIFT for m in (4, 5, 6, 7)]

    full_pred = {}
    for var, (mfeats, cfeats) in FEATS.items():
        mains, colds = [], []
        for so in SEEDS:
            bm = fit(Xf, yf, mfeats, itr, FINAL_ROUNDS_MAIN, so)
            mains.append(predict(bm, Xt, mfeats, itt))
            bc = fit(Xf[cold_f], yf[cold_f], cfeats, itr[cold_f], FINAL_ROUNDS_COLD, so)
            colds.append(predict(bc, Xt.loc[is_cold_te], cfeats, itt[is_cold_te]))
        p = logmean(mains)
        p[is_cold_te] = logblend(logmean(colds), b5t, W_COLD)
        full_pred[var] = p

    w("## 3. Kohort-eş aylık kalibrasyon (tam eğitim, 3-seed)")
    w()
    w("| var | Nis | May | Haz | Tem | max|sapma| |")
    w("|---|---|---|---|---|---|")
    calib_mx = {}
    for var in ["t0", "t1"]:
        d = calib(full_pred[var])
        calib_mx[var] = max(abs(x) for x in d)
        w(f"| {var} | " + " | ".join(f"{x:+.3f}" for x in d)
          + f" | {calib_mx[var]:.3f} |")
    w()

    # ---- kabul + submission ----
    d_f2 = scores[("F2", "t1")]["blend"] - scores[("F2", "t0")]["blend"]
    d_f1 = scores[("F1", "t1")]["blend"] - scores[("F1", "t0")]["blend"]
    d_f3 = scores[("F3", "t1")]["blend"] - scores[("F3", "t0")]["blend"]
    w("## 4. Karar")
    w()
    w(f"- F2 wx Δ (karar fold'u): **{d_f2:+.4f}** ({'wx YARDIM ediyor' if d_f2 < 0 else 'wx yardım etmiyor'})")
    w(f"- F1 wx Δ: {d_f1:+.4f} · F3 wx Δ: {d_f3:+.4f}")
    w(f"- Kalibrasyon: t0 max {calib_mx['t0']:.3f} → t1 max {calib_mx['t1']:.3f}")
    w(f"- wx_ gain payı (F2): %{wx_gain:.1f}")
    w()
    # karar: F2 delta negatif VE F1 gerilemiyor VE kalibrasyon bozulmuyor
    wx_wins = (d_f2 < -0.002 and d_f1 <= 0.005 and calib_mx["t1"] <= 0.16)
    w(f"- **SONUÇ: {'wx KABUL (t1) — submission üretiliyor' if wx_wins else 'wx marjinal/olumsuz — karar kullanıcıya'}**")
    w()
    if wx_wins:
        sub = pd.DataFrame({"id": te["id"], "tuketim": full_pred["t1"]})
        write_submission(sub, SUBMISSIONS_DIR / "sub_wx.csv")
        w("- submissions/sub_wx.csv yazıldı (t1, wx'li, 3-seed).")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_wx.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'model_wx.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_wx.md").write_text(out.getvalue(), encoding="utf-8")
