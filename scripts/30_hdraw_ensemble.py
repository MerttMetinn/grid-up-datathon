# -*- coding: utf-8 -*-
"""
30_hdraw_ensemble.py — geçmiş-uzunluğu (H) çekilişi üzerinden topluluk.

GEREKÇE (LB'den ölçüldü):
  sub_nowx_lo (LB 1.06525) ve sub_notebook (LB 1.05764) aynı model, aynı feature,
  aynı parametrelerdir. Aralarındaki TEK yapısal fark eğitim matrisi kurulurken
  her trafoya atanan H (geçmiş uzunluğu) çekilişidir. Skor farkı 0.0076.

  Yani tek bir H çekilişinin skora etkisi ±0.008 mertebesinde ve bu varyans
  şu an HİÇ ortalanmıyor: notebook 3 tohum kullanıyor ama üçü de AYNI eğitim
  matrisini paylaşıyor (`fold_i=9` sabit). Tohum ortalaması yalnızca LightGBM'in
  kendi bagging/feature gürültüsünü söndürüyor, H gürültüsünü değil.

Bu script H çekilişini de çeşitlendirir: N_DRAWS farklı çekiliş × SEEDS tohum,
tahminler log uzayında ortalanır. Varyans azaltma — teorik olarak sağlam,
ek varsayım gerektirmez.

Çıktı: submissions/sub_hens.csv · reports/hdraw_ensemble.md
Kullanım: python scripts/30_hdraw_ensemble.py
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

# Notebook kaynağını tek doğruluk kaynağı olarak kullan: final eğitim
# bloğundan ÖNCESİNİ çalıştır, fonksiyonları oradan al.
_NB = ROOT / "notebooks" / "gridup_leakfree_submission.py"
_src = _NB.read_text(encoding="utf-8")
_head = _src[:_src.index("# 8a. FINAL")]
NB: dict = {"__name__": "nb_head"}
exec(compile(_head, str(_NB), "exec"), NB)

SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"

N_DRAWS = 4                     # farklı H çekilişi sayısı
DRAW_IDS = [9, 19, 29, 39]      # build_training_set(fold_i=...) -> rng tohumu
SEEDS = NB["SEEDS"]             # [0, 1, 2]
LEVEL_SHIFT = NB["LEVEL_SHIFT"]
FINAL_ROUNDS = NB["FINAL_ROUNDS"]
BEST_PARAMS = NB["BEST_PARAMS"]
CATEGORICAL = NB["CATEGORICAL"]
SEED = NB["SEED"]
TRAIN_END = NB["TRAIN_END"]

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def main():
    df, te, profile = NB["df"], NB["te"], NB["profile"]
    build_training_set = NB["build_training_set"]
    build_features = NB["build_features"]
    anchor_components = NB["anchor_components"]
    assemble_anchor = NB["assemble_anchor"]
    align_categories = NB["align_categories"]
    FULL_ORIGINS = NB["FULL_ORIGINS"]

    w("# H-çekilişi topluluğu — eğitim matrisi varyansının söndürülmesi")
    w()
    w(f"Üretim: `scripts/30_hdraw_ensemble.py` · {datetime.now():%Y-%m-%d %H:%M}")
    w(f"- {N_DRAWS} H çekilişi × {len(SEEDS)} tohum = {N_DRAWS*len(SEEDS)} model")
    w(f"- Taban: sub_notebook.csv (tek çekiliş, 3 tohum) LB **1.05764**")
    w()

    print("[test] feature + anchor (cekilisten bagimsiz, bir kez) ...")
    X_test = build_features(te, TRAIN_END, df)
    a_test = assemble_anchor(anchor_components(te, TRAIN_END, df))

    draw_logs = []       # her çekilişin kendi ortalama log tahmini
    for d, fold_i in enumerate(DRAW_IDS[:N_DRAWS]):
        print(f"\n[cekilis {d+1}/{N_DRAWS}] fold_i={fold_i} egitim matrisi ...")
        X_full, y_full, a_full = build_training_set(
            df, df.index, TRAIN_END, FULL_ORIGINS, profile, fold_i=fold_i)
        Xt = X_test.copy()
        align_categories([X_full, Xt])
        preds = []
        for so in SEEDS:
            params = dict(BEST_PARAMS, seed=SEED + so)
            ds = lgb.Dataset(X_full, label=y_full, init_score=a_full,
                             categorical_feature=CATEGORICAL)
            booster = lgb.train(params, ds, num_boost_round=FINAL_ROUNDS)
            preds.append(booster.predict(Xt) + a_test)
        dl = np.mean(preds, axis=0)
        draw_logs.append(dl)
        print(f"  cekilis {d+1} ortalama log1p (kaydirmasiz): {dl.mean():.4f}")

    A = np.vstack(draw_logs)
    w("## 1. Çekilişler arası yayılım (kaydırma öncesi, log1p ortalaması)")
    w()
    w("| çekiliş | fold_i | ortalama log1p |")
    w("|---|---|---|")
    for d, fold_i in enumerate(DRAW_IDS[:N_DRAWS]):
        w(f"| {d+1} | {fold_i} | {A[d].mean():.4f} |")
    w()
    w(f"- Çekilişler arası ortalama-seviye std: **{A.mean(axis=1).std():.4f}** log")
    w(f"- Satır bazında çekilişler arası std (medyan): "
      f"**{np.median(A.std(axis=0)):.4f}** log")
    w()
    w("Bu yayılım tek çekilişli modelde tamamen tahmine geçiyor; topluluk onu söndürür.")
    w()

    log_pred = A.mean(axis=0)
    pred = np.clip(np.expm1(log_pred + LEVEL_SHIFT), 0, None)

    sample = NB["sample"]
    sub = pd.DataFrame({"id": te["id"].astype(str), "tuketim": pred})
    sample_ids = sample["id"].astype(str)
    assert set(sub["id"]) == set(sample_ids)
    sub = sub.set_index("id").reindex(sample_ids).reset_index()
    assert sub["id"].tolist() == sample_ids.tolist()
    assert sub["tuketim"].notna().all() and (sub["tuketim"] >= 0).all()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSIONS / "sub_hens.csv", index=False)

    base = pd.read_csv(SUBMISSIONS / "sub_notebook.csv")
    lb_ = np.log1p(base["tuketim"].to_numpy())
    ln_ = np.log1p(sub["tuketim"].to_numpy())
    w("## 2. sub_hens.csv")
    w()
    w(f"- yazıldı · {len(sub):,} satır")
    w(f"- sub_notebook'a göre: ortalama fark **{(ln_-lb_).mean():+.4f}** · "
      f"MAE {np.abs(ln_-lb_).mean():.4f} · korelasyon {np.corrcoef(ln_, lb_)[0,1]:.5f}")
    w()
    te_ay = te["tarih"].dt.to_period("M").astype(str).to_numpy()
    w("| ay | sub_notebook | sub_hens |")
    w("|---|---|---|")
    for m in sorted(set(te_ay)):
        k = te_ay == m
        w(f"| {m} | {lb_[k].mean():.4f} | {ln_[k].mean():.4f} |")
    w()
    w("**Beklenti:** varyans azaltma; seviye neredeyse aynı kalmalı, "
      "satır bazında gürültü düşmeli. LB ile doğrulanmalı.")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "hdraw_ensemble.md").write_text(out.getvalue(), encoding="utf-8")
