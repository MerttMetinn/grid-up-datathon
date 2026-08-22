# -*- coding: utf-8 -*-
"""
14_save_model.py — Kazanan s2 modelini eğitip DİSKE kaydeder.

s2 kurgusu (reports/model_v7.md): mevsim-farkındalıklı anchor (α=0.4, cold_adj=True),
ALL_FEATURES, cold-only model + b5 harmanı (w=0.45), 3-seed log ortalaması.

Kaydedilen artifact'lar (models/):
  s2_main_seed{0,1,2}.txt   — ana LightGBM booster (native, okunabilir)
  s2_cold_seed{0,1,2}.txt   — cold-only booster
  MODEL_CARD.md             — model tanımı, hiperparametreler, yükleme talimatı

DETERMİNİZM: SEED sabit, tur sayısı sabit → sub_s.csv ile birebir aynı tahmin
beklenir. Bu script sub_s.csv'nin ÜZERİNE YAZMAZ; sub_s_check.csv üretir, sonra
byte-karşılaştırması yapılır (aşağıdaki komut).

Kullanım: python scripts/14_save_model.py
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

from src.baselines import b5_guc_lf  # noqa: E402
from src.config import SEED, SUBMISSIONS_DIR, TRAIN_END  # noqa: E402
from src.data import load_test, load_train, load_profile  # noqa: E402
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES,
                          anchor_components, build_features)  # noqa: E402
from src.predict import write_submission  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set)  # noqa: E402

MODELS_DIR = ROOT / "models"
ALPHA_STAR = 0.4          # reports/model_v7.md alpha grid seçimi (kalibrasyon)
W_COLD = 0.45
SEEDS = [0, 1, 2]
FINAL_ROUNDS_MAIN, FINAL_ROUNDS_COLD = 126, 73
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]


def assemble(base, dev, zero, alpha, adj):
    a = base + alpha * dev
    if adj:
        a = a + zero
    return a.to_numpy()


def fit(X, y, feats, init, rounds, seed_off):
    params = dict(LGB_PARAMS)
    for k in ("seed", "feature_fraction_seed", "bagging_seed"):
        params[k] = LGB_PARAMS[k] + seed_off
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    ds = lgb.Dataset(X[feats], label=y, init_score=init, categorical_feature=cats)
    return lgb.train(params, ds, num_boost_round=rounds)


def predict(booster, X, feats, init):
    return np.clip(np.expm1(booster.predict(X[feats]) + init), 0, None)


def logmean(preds):
    return np.expm1(np.mean([np.log1p(p) for p in preds], axis=0))


def logblend(pm, pb, wgt):
    return np.expm1(wgt * np.log1p(pm) + (1 - wgt) * np.log1p(pb))


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_train()
    te = load_test()
    profile = load_profile()

    print("Feature build (tam eğitim + test) ...")
    ORIGINS["FULL"] = FULL_ORIGINS
    pseudo = {"name": "FULL", "train_idx": df.index, "spec": {"train_end": TRAIN_END}}
    Xf, yf, metaf = build_training_set(df, pseudo, profile, fold_i=9)
    Xt = build_features(te, TRAIN_END, df)
    align_categories([Xf, Xt])
    comp_t = anchor_components(te, TRAIN_END, df)

    init_f = assemble(metaf["anc_base"], metaf["anc_dev"], metaf["anc_zero"],
                      ALPHA_STAR, True)
    init_t = assemble(comp_t["base"], comp_t["season_dev"], comp_t["zero_adj"],
                      ALPHA_STAR, True)
    cold_f = metaf["is_cold_example"].to_numpy()
    is_cold_te = ~te["tanim"].isin(set(df["tanim"].unique())).to_numpy()

    mains, colds = [], []
    for so in SEEDS:
        print(f"seed+{so}: ana model ({FINAL_ROUNDS_MAIN} tur) ...")
        bm = fit(Xf, yf, ALL_FEATURES, init_f, FINAL_ROUNDS_MAIN, so)
        bm.save_model(str(MODELS_DIR / f"s2_main_seed{so}.txt"))
        mains.append(predict(bm, Xt, ALL_FEATURES, init_t))

        print(f"seed+{so}: cold modeli ({FINAL_ROUNDS_COLD} tur) ...")
        bc = fit(Xf[cold_f], yf[cold_f], COLD_MODEL_FEATURES, init_f[cold_f],
                 FINAL_ROUNDS_COLD, so)
        bc.save_model(str(MODELS_DIR / f"s2_cold_seed{so}.txt"))
        colds.append(predict(bc, Xt.loc[is_cold_te], COLD_MODEL_FEATURES,
                             init_t[is_cold_te]))

    pm = logmean(mains)
    pc = logmean(colds)
    pred = pm.copy()
    b5c = b5_guc_lf(df, te[is_cold_te]).to_numpy()
    pred[is_cold_te] = logblend(pc, b5c, W_COLD)

    sub = pd.DataFrame({"id": te["id"], "tuketim": pred})
    write_submission(sub, SUBMISSIONS_DIR / "sub_s_check.csv")

    # ---- model kartı ----
    card = f"""# Model Kartı — s2 (final)

Üretim: `scripts/14_save_model.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}

## Ne bu model
Grid Up Datathon kazanan varyantı **s2** — trafo bazlı günlük tüketim tahmini.
Yerel CV: F1 blend **1.1244** (baseline b6: 1.2692), aylık kalibrasyon max sapma 0.099.

## Mimari
- **Hedef:** `log1p(tuketim)`, tahmin `expm1(model + init_score)`, `clip(0, None)`.
- **init_score (fiziksel çıpa):** mevsim-farkındalıklı anchor, `α={ALPHA_STAR}` yumuşatma,
  cold sıfır düzeltmeli. warm: `lvl_median_log_full + α·mevsim_sapması`.
  cold: `log(guc·24) + α·log(LF_nz_mevsim) + log(1−zero_rate)`.
- **Ana model:** LightGBM, {len(ALL_FEATURES)} feature (static_/cal_/lvl_/grp_/seas_),
  {FINAL_ROUNDS_MAIN} tur, 3 seed ({SEEDS}) log-uzayı ortalaması.
- **Cold model:** yalnızca cold örneklerle, {len(COLD_MODEL_FEATURES)} feature
  (static_+cal_+grp_), {FINAL_ROUNDS_COLD} tur. Cold satırlarda `w={W_COLD}` ile b5 baseline'ı
  harmanlanır: `w·model_cold + (1−w)·b5`.

## Hiperparametreler (LightGBM)
```
{chr(10).join(f'{k} = {v}' for k, v in LGB_PARAMS.items() if k not in ('verbose',))}
num_boost_round = {FINAL_ROUNDS_MAIN} (ana) / {FINAL_ROUNDS_COLD} (cold)
```

## Artifact'lar
| dosya | içerik |
|---|---|
| `s2_main_seed{{0,1,2}}.txt` | ana booster (LightGBM native metin formatı) |
| `s2_cold_seed{{0,1,2}}.txt` | cold-only booster |

## Nasıl yüklenir / tahmin üretilir
```python
import lightgbm as lgb
from src.features import build_features, anchor_components
# ... build_features + anchor kur (bkz. 14_save_model.py), sonra:
bm = lgb.Booster(model_file="models/s2_main_seed0.txt")
pred_log = bm.predict(X_test[ALL_FEATURES]) + init_score
pred = np.clip(np.expm1(pred_log), 0, None)
```
Not: init_score (anchor) tahmin sırasında `anchor_components` ile YENİDEN kurulmalıdır —
booster onu içermez. Tam akış `scripts/14_save_model.py`'de.

## Yeniden üretilebilirlik
SEED={SEED} sabit, tur sayısı sabit. Bu script `sub_s.csv` ile birebir aynı tahmini
üretir (doğrulama: `sub_s_check.csv` ile diff).
"""
    (MODELS_DIR / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print(f"\nModeller: {MODELS_DIR}")
    print("Model kartı: models/MODEL_CARD.md")


if __name__ == "__main__":
    main()
