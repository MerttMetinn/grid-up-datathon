# -*- coding: utf-8 -*-
"""
31_zero_anchor.py — anchor'ın sıfır düzeltmesini RMSLE-doğru forma çevirir.

SORUN (LB ile doğrulandı):
  Sıfır-şişirilmiş bir dağılımda y=0 olasılığı p, pozitifken log1p seviyesi L ise
  RMSLE'yi minimize eden tahmin log uzayının BEKLENEN DEĞERİDİR:

      E[log1p(y)] = (1-p)·L          <- çarpımsal

  Mevcut anchor ise toplamsal düzeltme kullanıyor:

      anchor = L + log(1-p)          <- ham ölçekte ORTALAMA için doğru, log için DEĞİL

  Fark p ile büyür:  p=0.06 -> +0.38 · p=0.20 -> +1.26 · p=0.35 -> +2.16 log.

KANIT:
  Segment-kaydırma deneyi (genel ortalama sabit tutularak) cold/warm sapmalarını
  ayırdı: cold **+0.184**, warm **-0.035**. Global ortalama +0.013 olduğu için
  bu sapma global kalibrasyonda GÖRÜNMÜYORDU. Ölçülen cold sapması, yukarıdaki
  matematiksel hatanın beklenen büyüklüğünün (+0.38) yaklaşık yarısı — yani GBM
  hatanın bir kısmını düzeltmiş, kalanı tahmine geçmiş.

  Üniform cold kaydırması (sub_sp15) yalnızca ORTALAMA sapmayı siler. Satır bazlı
  yapı — yüksek p'li satırların çok daha aşağı itilmesi — ancak anchor düzeltilerek
  giderilir. Hatanın %56'sı tam olarak o satırlarda (cold + gerçek sıfır, %1.6).

DEĞİŞİKLİK: yalnızca COLD satırlarda
      eski:  base + alpha*season_dev + log(1-p)
      yeni:  (1-p) * (base + alpha*season_dev)
  `base` zaten SIFIR OLMAYAN satırlardan kuruluyor (hz filtresi tuketim>0), yani
  tam olarak L'dir — çarpan doğrudan uygulanabilir.

  Warm DEĞİŞTİRİLMEZ: ölçülen warm sapması -0.035 (zaten kalibre) ve warm anchor'ı
  medyan tabanlı, farklı bir kurgu. Ölçüm desteklemediği yerde değişiklik yapılmaz.

Çıktı: submissions/sub_zanch.csv · reports/zero_anchor.md
Kullanım: python scripts/31_zero_anchor.py
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

_NB = ROOT / "notebooks" / "gridup_leakfree_submission.py"
_src = _NB.read_text(encoding="utf-8")
NB: dict = {"__name__": "nb_head"}
exec(compile(_src[:_src.index("# 8a. FINAL")], str(_NB), "exec"), NB)

SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"

SEED = NB["SEED"]
SEEDS = NB["SEEDS"]
DRAW_IDS = NB["DRAW_IDS"]
ALPHA = NB["ALPHA"]
FINAL_ROUNDS = NB["FINAL_ROUNDS"]
BEST_PARAMS = NB["BEST_PARAMS"]
CATEGORICAL = NB["CATEGORICAL"]
TRAIN_END = NB["TRAIN_END"]
LEVEL_SHIFT = NB["LEVEL_SHIFT"]

out = io.StringIO()


def w(line=""):
    out.write(line + "\n")
    print(line)


def assemble_anchor_zfix(comp: pd.DataFrame, alpha: float = ALPHA) -> np.ndarray:
    """RMSLE-doğru sıfır düzeltmesi (yalnız cold satırlarda çarpımsal).

    comp["zero_adj"] = log(1-p) olarak geliyor  ->  p = 1 - exp(zero_adj)
    warm satırlarda zero_adj = 0  ->  p = 0  ->  çarpan 1  (davranış değişmez)
    """
    L = comp["base"] + alpha * comp["season_dev"]
    p = 1.0 - np.exp(comp["zero_adj"].to_numpy())
    keep = np.where(comp["is_cold_anchor"].to_numpy(), 1.0 - p, 1.0)
    return (L.to_numpy() * keep)


def main():
    df, te, profile = NB["df"], NB["te"], NB["profile"]
    build_training_set = NB["build_training_set"]
    build_features = NB["build_features"]
    anchor_components = NB["anchor_components"]
    assemble_old = NB["assemble_anchor"]
    align_categories = NB["align_categories"]
    FULL_ORIGINS = NB["FULL_ORIGINS"]

    w("# Anchor sıfır düzeltmesi — RMSLE-doğru forma çevrildi")
    w()
    w(f"Üretim: `scripts/31_zero_anchor.py` · {datetime.now():%Y-%m-%d %H:%M}")
    w(f"- {len(DRAW_IDS)} H çekilişi × {len(SEEDS)} tohum (taban ile aynı kurgu)")
    w(f"- Taban: `sub_hens_lo` LB **1.05737** · üniform cold kaydırması "
      f"`sub_sp15` öngörü 1.0535")
    w()
    w("## 1. Değişikliğin büyüklüğü (test satırlarında)")
    w()

    comp_t = anchor_components(te, TRAIN_END, df)
    a_old = assemble_old(comp_t)
    a_new = assemble_anchor_zfix(comp_t)
    ic = comp_t["is_cold_anchor"].to_numpy()
    d = a_new - a_old
    w("| kesim | n | eski anchor | yeni anchor | fark |")
    w("|---|---|---|---|---|")
    w(f"| cold | {ic.sum():,} | {a_old[ic].mean():.4f} | {a_new[ic].mean():.4f} "
      f"| {d[ic].mean():+.4f} |")
    w(f"| warm | {(~ic).sum():,} | {a_old[~ic].mean():.4f} | {a_new[~ic].mean():.4f} "
      f"| {d[~ic].mean():+.4f} |")
    w()
    q = np.percentile(d[ic], [10, 50, 90])
    w(f"- cold farkının dağılımı: p10 {q[0]:+.3f} · medyan {q[1]:+.3f} · p90 {q[2]:+.3f}")
    w("- Üniform kaydırmadan farkı: düzeltme **satır bazlı**; sıfır olasılığı "
      "yüksek satırlar çok daha aşağı iniliyor.")
    w()

    print("[test] feature ...")
    X_test_base = build_features(te, TRAIN_END, df)

    draw_logs = []
    for i, fold_i in enumerate(DRAW_IDS):
        print(f"\n[cekilis {i+1}/{len(DRAW_IDS)}] fold_i={fold_i} ...")
        # Eğitim matrisini kur; anchor'ı YENİ formülle yeniden hesapla.
        # build_training_set eski anchor'ı meta'ya yazdığı için bileşenleri
        # burada tekrar üretmek yerine, aynı origin bloklarını yeniden kuruyoruz.
        X_full, y_full, a_full_old = build_training_set(
            df, df.index, TRAIN_END, FULL_ORIGINS, profile, fold_i=fold_i)
        X_test = X_test_base.copy()
        align_categories([X_full, X_test])
        seed_logs = []
        for so in SEEDS:
            params = dict(BEST_PARAMS, seed=SEED + so)
            ds = lgb.Dataset(X_full, label=y_full, init_score=a_full_old,
                             categorical_feature=CATEGORICAL)
            b = lgb.train(params, ds, num_boost_round=FINAL_ROUNDS)
            # Eğitim anchor'ı değişmedi; tahminde YENİ anchor kullanılır.
            # Model, eski anchor üzerindeki artığı öğrendi; yeni anchor ile
            # toplayınca cold satırlarda RMSLE-doğru seviyeye iniyoruz.
            seed_logs.append(b.predict(X_test) + a_new)
        draw_logs.append(np.mean(seed_logs, axis=0))
        print(f"  ortalama log1p (kaydirmasiz): {draw_logs[-1].mean():.4f}")

    log_pred = np.vstack(draw_logs).mean(axis=0)
    pred = np.clip(np.expm1(log_pred + LEVEL_SHIFT), 0, None)

    sample = NB["sample"]
    sub = pd.DataFrame({"id": te["id"].astype(str), "tuketim": pred})
    sids = sample["id"].astype(str)
    assert set(sub["id"]) == set(sids)
    sub = sub.set_index("id").reindex(sids).reset_index()
    assert sub["id"].tolist() == sids.tolist()
    assert sub["tuketim"].notna().all() and (sub["tuketim"] >= 0).all()
    sub.to_csv(SUBMISSIONS / "sub_zanch.csv", index=False)

    base = pd.read_csv(SUBMISSIONS / "sub_hens_lo.csv")
    lb_, ln_ = np.log1p(base["tuketim"].to_numpy()), np.log1p(sub["tuketim"].to_numpy())
    w("## 2. sub_zanch.csv")
    w()
    w(f"- yazıldı · {len(sub):,} satır")
    w(f"- `sub_hens_lo`'a göre: cold ortalama {ln_[ic].mean()-lb_[ic].mean():+.4f} · "
      f"warm {ln_[~ic].mean()-lb_[~ic].mean():+.4f} · genel {(ln_-lb_).mean():+.4f}")
    w(f"- genel ortalama log1p: {ln_.mean():.4f} (taban {lb_.mean():.4f})")
    w()
    w("**LB ile doğrulanmalı.** Beklenti: üniform kaydırmanın (sub_sp15) üstüne, "
      "satır bazlı düzeltmenin katkısı kadar iyileşme.")


if __name__ == "__main__":
    try:
        main()
    finally:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "zero_anchor.md").write_text(out.getvalue(), encoding="utf-8")
