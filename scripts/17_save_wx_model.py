# -*- coding: utf-8 -*-
"""
17_save_wx_model.py — wx'li final modeli (s2+wx) eğitip DİSKE kaydeder + submission.

s2+wx kurgusu: mevsim-farkındalıklı anchor (α=0.4, cold_adj=True), ana model TÜM
75 feature (wx dahil), cold-only model static+cal+grp+wx, b5 harmanı (w=0.45), 3-seed.

Artifact'lar (models/):
  wx_main_seed{0,1,2}.txt · wx_cold_seed{0,1,2}.txt · MODEL_CARD_wx.md
Submission: submissions/sub_wx.csv

Kullanım: python scripts/17_save_wx_model.py
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
from src.data import load_profile, load_test, load_train  # noqa: E402
from src.features import (ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS,
                          anchor_components, build_features)  # noqa: E402
from src.predict import write_submission  # noqa: E402
from src.train import (COLD_MODEL_FEATURES, LGB_PARAMS, ORIGINS,
                       align_categories, build_training_set)  # noqa: E402

MODELS_DIR = ROOT / "models"
ALPHA, W_COLD = 0.4, 0.45
SEEDS = [0, 1, 2]
FINAL_ROUNDS_MAIN, FINAL_ROUNDS_COLD = 126, 73
FULL_ORIGINS = ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31",
                "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30",
                "2025-10-31", "2025-11-30"]

WX = FEATURE_GROUPS["wx"]
MAIN_FEATS = ALL_FEATURES                    # wx dahil (75)
COLD_FEATS = COLD_MODEL_FEATURES + WX        # cold model + hava


def assemble(base, dev, zero):
    return (base + ALPHA * dev + zero).to_numpy()


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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df, te, profile = load_train(), load_test(), load_profile()

    print("Feature build (tam eğitim + test) ...")
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

    mains, colds = [], []
    for so in SEEDS:
        print(f"seed+{so}: ana model ({len(MAIN_FEATS)} feat, {FINAL_ROUNDS_MAIN} tur) ...")
        bm = fit(Xf, yf, MAIN_FEATS, itr, FINAL_ROUNDS_MAIN, so)
        bm.save_model(str(MODELS_DIR / f"wx_main_seed{so}.txt"))
        mains.append(predict(bm, Xt, MAIN_FEATS, itt))
        print(f"seed+{so}: cold modeli ({len(COLD_FEATS)} feat, {FINAL_ROUNDS_COLD} tur) ...")
        bc = fit(Xf[cold_f], yf[cold_f], COLD_FEATS, itr[cold_f], FINAL_ROUNDS_COLD, so)
        bc.save_model(str(MODELS_DIR / f"wx_cold_seed{so}.txt"))
        colds.append(predict(bc, Xt.loc[is_cold_te], COLD_FEATS, itt[is_cold_te]))

    pred = logmean(mains)
    b5t = b5_guc_lf(df, te[is_cold_te]).to_numpy()
    pred[is_cold_te] = logblend(logmean(colds), b5t, W_COLD)

    sub = pd.DataFrame({"id": te["id"], "tuketim": pred})
    write_submission(sub, SUBMISSIONS_DIR / "sub_wx.csv")

    card = f"""# Model Kartı — s2+wx (hava durumlu final)

Üretim: `scripts/17_save_wx_model.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}

## Ne bu model
s2 kurgusu + hava durumu (wx_) feature ailesi. Trafo bazlı günlük tüketim tahmini.

## Mimari
- Hedef `log1p(tuketim)`, tahmin `expm1(model + init_score)`, `clip(0)`.
- init_score: mevsim-farkındalıklı anchor, α={ALPHA}, cold sıfır düzeltmeli.
- Ana model: LightGBM, {len(MAIN_FEATS)} feature (static/cal/lvl/grp/seas/**wx**),
  {FINAL_ROUNDS_MAIN} tur, 3 seed log ortalaması.
- Cold model: cold örneklerle, {len(COLD_FEATS)} feature (static+cal+grp+wx),
  {FINAL_ROUNDS_COLD} tur. Cold satırlarda w={W_COLD} ile b5 harmanı.
- **wx_ (17 feature):** CDD/CDD²/CDD³/HDD, sıcaklık/apparent/nem, tarımsal
  (ET0 7g, yağış açığı 30g, toprak nemi), termal kütle (CDD 7g MA), ilk-sıcak-gün anomalisi.
  Kaynak: Open-Meteo arşivi (test dönemi gerçek gözlem). Cache: data/external/weather.parquet.

## Artifact'lar
| dosya | içerik |
|---|---|
| `wx_main_seed{{0,1,2}}.txt` | ana booster (wx dahil) |
| `wx_cold_seed{{0,1,2}}.txt` | cold-only booster (wx dahil) |

## Yükleme
`scripts/17_save_wx_model.py` akışıyla aynı: build_features (wx cache'ten) + anchor kur,
booster'ı `lgb.Booster(model_file=...)` ile yükle, `expm1(predict + init_score)`.
weather.parquet gerekli (yoksa `scripts/15_fetch_weather.py`).

## Yeniden üretilebilirlik
SEED={SEED}, sabit tur. sub_wx.csv ile birebir.
"""
    (MODELS_DIR / "MODEL_CARD_wx.md").write_text(card, encoding="utf-8")
    print(f"\nModeller: {MODELS_DIR} (wx_*)· Submission: sub_wx.csv")


if __name__ == "__main__":
    main()
