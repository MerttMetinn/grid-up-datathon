# -*- coding: utf-8 -*-
"""
10_submit.py — Final eğitim + iki submission + tahmin-seviyesi sağlık kontrolü.

  sub_b6.csv — b6 baseline (warm b2 + cold b5)
  sub_p3.csv — p3 kurgusu: çok-origin ana model + cold-only model + b5 harmanı
               (w=0.45), 3-seed log ortalaması

Final eğitimde valid yok → iterasyon sayısı F1'in best_iter'i × 1.1
(ROADMAP kuralı). Sağlık kontrolü: reports/pred_sanity.md
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
from src.config import (REPORTS_DIR, SEED, SUBMISSIONS_DIR, TRAIN_END,
                        YOY_DRIFT)  # noqa: E402
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import ALL_FEATURES, build_features  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set)  # noqa: E402
from src.predict import write_submission  # noqa: E402

W_COLD = 0.45          # 09 raporundaki 3-fold ortalama optimumu
# F1 p3 best_iter'leri: ana [116,121,107] · cold [63,63,74] → ort × 1.1
FINAL_ROUNDS_MAIN = 126
FINAL_ROUNDS_COLD = 73
SEEDS = [0, 1, 2]

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")
    print(line)


def fit_final(X, y, features, init, rounds, seed_offset):
    import lightgbm as lgb
    params = dict(LGB_PARAMS)
    for k in ("seed", "feature_fraction_seed", "bagging_seed"):
        params[k] = LGB_PARAMS[k] + seed_offset
    ds = lgb.Dataset(X[features], label=y, init_score=init,
                     categorical_feature=[c for c in
                                          ["static_guc_bucket", "static_il",
                                           "static_bolge", "static_ilce_key",
                                           "cal_holiday_name"] if c in features])
    return lgb.train(params, ds, num_boost_round=rounds)


def main() -> None:
    tr = load_train()
    te = load_test()
    profile = load_profile()

    w("# Tahmin-seviyesi sağlık kontrolü (pred_sanity)")
    w()
    w(f"Üretim: `scripts/10_submit.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    # ------------------------------------------------------------- b6 submission
    print("b6 submission ...")
    pred_b6 = b6_hybrid(tr, te)
    sub_b6 = pd.DataFrame({"id": te["id"], "tuketim": pred_b6.to_numpy()})
    write_submission(sub_b6, SUBMISSIONS_DIR / "sub_b6.csv")

    # ------------------------------------------------------------- final p3 eğitim
    ORIGINS["FULL"] = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                       "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                       "2025-10-31", "2025-11-30"]
    pseudo_fold = {"name": "FULL", "train_idx": tr.index,
                   "spec": {"train_end": TRAIN_END}}
    print("cok-origin final egitim seti ...")
    X_tr, y_tr, meta = build_training_set(tr, pseudo_fold, profile, fold_i=9)

    print("test feature'lari ...")
    X_te = build_features(te, TRAIN_END, tr)
    align_categories([X_tr, X_te])
    init_tr = np.log(meta["guc"].to_numpy() * 24.0)
    init_te = np.log(te["guc"].to_numpy() * 24.0)
    cold_mask = meta["is_cold_example"].to_numpy()

    is_cold_te = ~te["tanim"].isin(set(tr["tanim"].unique())).to_numpy()

    mains, colds = [], []
    for so in SEEDS:
        print(f"seed+{so}: ana model ({FINAL_ROUNDS_MAIN} tur) ...")
        bm = fit_final(X_tr, y_tr, ALL_FEATURES, init_tr,
                       FINAL_ROUNDS_MAIN, so)
        pm = np.clip(np.expm1(bm.predict(X_te[ALL_FEATURES]) + init_te), 0, None)
        mains.append(pm)
        print(f"seed+{so}: cold modeli ({FINAL_ROUNDS_COLD} tur) ...")
        bc = fit_final(X_tr[cold_mask], y_tr[cold_mask], COLD_MODEL_FEATURES,
                       init_tr[cold_mask], FINAL_ROUNDS_COLD, so)
        pc = np.clip(np.expm1(
            bc.predict(X_te.loc[is_cold_te, COLD_MODEL_FEATURES])
            + init_te[is_cold_te]), 0, None)
        colds.append(pc)

    pred = np.expm1(np.mean([np.log1p(p) for p in mains], axis=0))
    pred_cold_model = np.expm1(np.mean([np.log1p(p) for p in colds], axis=0))
    pred_b5_cold = b5_guc_lf(tr, te[is_cold_te]).to_numpy()
    pred[is_cold_te] = np.expm1(
        W_COLD * np.log1p(pred_cold_model)
        + (1 - W_COLD) * np.log1p(pred_b5_cold))

    sub_p3 = pd.DataFrame({"id": te["id"], "tuketim": pred})
    write_submission(sub_p3, SUBMISSIONS_DIR / "sub_p3.csv")

    # ------------------------------------------------------------- sağlık kontrolü
    te_s = te.copy()
    te_s["pred"] = pred
    te_s["logp"] = np.log1p(pred)
    te_s["ay_p"] = te_s["tarih"].dt.to_period("M")

    tr_s = tr[tr["tanim"].isin(set(te["tanim"].unique()))].copy()
    tr_s["ay_p"] = tr_s["tarih"].dt.to_period("M")
    tr_2025 = tr_s[tr_s["ay_p"].isin(
        [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)])]
    tr_2025_log = np.log1p(tr_2025["tuketim"])

    w("## a–c. Aylık ortalama log1p — tahmin vs geçen yıl + drift")
    w()
    w("| ay | 2026 tahmin | 2025 gerçek (test trafoları) | beklenen (2025+0.102) | fark |")
    w("|---|---|---|---|---|")
    ratios = {}
    for m in (4, 5, 6, 7):
        p26 = float(te_s.loc[te_s["ay_p"] == pd.Period(f"2026-{m:02d}"), "logp"].mean())
        a25 = float(tr_2025_log[tr_2025["ay_p"] == pd.Period(f"2025-{m:02d}")].mean())
        ratios[m] = p26
        w(f"| {m:02d} | {p26:.4f} | {a25:.4f} | {a25 + YOY_DRIFT:.4f} | "
          f"{p26 - a25 - YOY_DRIFT:+.4f} |")
    w()

    # d. Temmuz/Mayıs geometrik oranı
    def geo_ratio(logs_jul, logs_may):
        return float(np.expm1(np.mean(logs_jul)) / np.expm1(np.mean(logs_may)))

    r_pred = geo_ratio(te_s.loc[te_s["ay_p"] == pd.Period("2026-07"), "logp"],
                       te_s.loc[te_s["ay_p"] == pd.Period("2026-05"), "logp"])
    r_2025 = geo_ratio(
        tr_2025_log[tr_2025["ay_p"] == pd.Period("2025-07")],
        tr_2025_log[tr_2025["ay_p"] == pd.Period("2025-05")])
    w("## d. Temmuz/Mayıs oranı (geometrik)")
    w()
    w(f"- Tahmin edilen (2026): **{r_pred:.2f}×** · 2025 gerçek (test trafoları): "
      f"{r_2025:.2f}× · beklenen ~1.86×")
    if r_pred < 1.3:
        w(f"- **KIRMIZI ALARM: {r_pred:.2f}× < 1.3 — model yaz rampasını KAÇIRIYOR.**")
    elif r_pred < 1.6:
        w(f"- **UYARI: {r_pred:.2f}× < 1.6 — ramp zayıf yakalanıyor.**")
    else:
        w(f"- Ramp makul yakalanıyor ({r_pred:.2f}× ≥ 1.6).")
    w()

    # e. ilçe bazında ilk 10
    w("## e. İlçe bazında Temmuz/Mayıs (test satırı en çok 10 ilçe)")
    w()
    top_ilce = te_s["ilce_key"].value_counts().head(10).index
    w("| ilçe | tahmin 2026 | gerçek 2025 |")
    w("|---|---|---|")
    for ic in top_ilce:
        rp = geo_ratio(
            te_s.loc[(te_s["ilce_key"] == ic) & (te_s["ay_p"] == pd.Period("2026-07")), "logp"],
            te_s.loc[(te_s["ilce_key"] == ic) & (te_s["ay_p"] == pd.Period("2026-05")), "logp"])
        m25 = tr_2025[tr_2025["ilce_key"] == ic]
        ra = geo_ratio(
            np.log1p(m25.loc[m25["ay_p"] == pd.Period("2025-07"), "tuketim"]),
            np.log1p(m25.loc[m25["ay_p"] == pd.Period("2025-05"), "tuketim"])) \
            if len(m25) else float("nan")
        w(f"| {ic} | {rp:.2f}× | {ra:.2f}× |")
    w()

    # f. cold trafolar
    w("## f. Cold trafolar (train'de hiç yok)")
    w()
    tc = te_s[is_cold_te]
    w("| ay | tahmin ort. log1p |")
    w("|---|---|")
    for m in (4, 5, 6, 7):
        v = tc.loc[tc["ay_p"] == pd.Period(f"2026-{m:02d}"), "logp"]
        w(f"| {m:02d} | {float(v.mean()):.4f} |" if len(v) else f"| {m:02d} | · |")
    rc = geo_ratio(tc.loc[tc["ay_p"] == pd.Period("2026-07"), "logp"],
                   tc.loc[tc["ay_p"] == pd.Period("2026-05"), "logp"])
    w()
    w(f"- Cold Temmuz/Mayıs: **{rc:.2f}×**")
    w()
    w(f"- sub_b6.csv ve sub_p3.csv `submissions/` altında, doğrulamadan geçti. "
      f"LB'ye kullanıcı yükleyecek.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "pred_sanity.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nRapor: {REPORTS_DIR / 'pred_sanity.md'}")


if __name__ == "__main__":
    main()
