# Dataset Paketi — Optuna & Feature Selection Kılavuzu

Tüm feature'lar (75, hava dahil) önceden hesaplanmış, `data/dataset/` altında hazır.
Bu doküman **nasıl kullanılacağını ve hangi hataların yarışmayı bozacağını** anlatır.

Üretim / güncelleme:
```bash
python scripts/15_fetch_weather.py   # hava (bir kez; değişince)
python scripts/18_build_dataset.py   # dataset paketi (feature değişince tekrar)
```

---

## 1. Neden tek dosya değil? (önce bunu oku)

Feature'ların yarısı **"bakış tarihi"ne (forecast_origin) bağlı**:
- `static_`, `cal_`, `wx_` → tarihe/ilçeye bağlı, sabit. (origin-bağımsız)
- `lvl_`, `grp_`, `seas_`, `anchor` → "şu tarihe kadarki geçmişten" hesaplanır. (origin-bağımlı)

Aynı trafo-gün satırı, farklı bir bakış tarihinden hesaplanınca farklı değer alır.
Hepsini tek statik tabloda dondurmak, geçmiş feature'ını gelecek veriyle kirletir =
**veri sızıntısı**. Deneme skorun harika çıkar, Kaggle'da çökersin (projenin v1'i
tam bu yüzden elendi).

Çözüm: her fold'un train+valid'i **ayrı** materialize edildi (her biri sızıntısız),
artı full_train (final model) + test.

---

## 2. Dosyalar

| dosya | ne | satır |
|---|---|---|
| `f1_train` / `f1_valid` | **F1 — birincil fold** (train 12 ay, valid Oca–Mar 2026) | büyük |
| `f2_train` / `f2_valid` | F2 — yön kontrolü (yaz hedefi) | orta |
| `f3_train` / `f3_valid` | F3 — kırılganlık alarmı | büyük |
| `full_train` | tüm veri, çok-origin — **final model eğitimi** | ~2M |
| `test` | final tahmin (Kaggle submission) | 714,688 |

Yükleme:
```python
from src.dataset import load_dataset, FEATURE_COLS, CAT_COLS
tr = load_dataset("f1_train")
va = load_dataset("f1_valid")
```

## 3. Kolonlar

- **75 feature:** `FEATURE_COLS` listesinde. Kategorikler: `CAT_COLS`.
- `y_log1p` — model hedefi (log1p tüketim). **Bunu tahmin et.**
- `tuketim` — orijinal ölçek gerçek değer (skorlama için; train/valid'de var).
- `init_score` — s2 fiziksel çapası (α=0.4). LightGBM'e `init_score` olarak ver
  (aşağıda). Kendi α'nı denemek istersen: `init_score = anc_base + α*anc_dev + anc_zero`.
- `is_cold` — trafo eğitimde görülmemiş mi (cold-start ayrımı).
- `guc`, `tanim`, `tarih`, (`id` sadece test'te).

---

## 4. Optuna — DOĞRU kurulum

**Altın kural: `full_train`'i fold valid'lerle KARIŞTIRMA.** full_train tüm trafoları
içerir; F1 valid trafoları da orada → sızıntı. Optuna'da hep **eşleşen çift** kullan:
`f1_train` → eğit, `f1_valid` → skorla.

```python
import lightgbm as lgb, numpy as np, optuna
from src.dataset import load_dataset, FEATURE_COLS, CAT_COLS
from src.validation import rmsle

tr, va = load_dataset("f1_train"), load_dataset("f1_valid")

def objective(trial):
    params = {
        "objective": "regression", "verbose": -1,
        "learning_rate": trial.suggest_float("lr", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_data_in_leaf": trial.suggest_int("min_data", 50, 300),
        "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bf", 0.6, 1.0),
        "lambda_l2": trial.suggest_float("l2", 0.0, 5.0),
        "seed": 42,
    }
    ds = lgb.Dataset(tr[FEATURE_COLS], label=tr["y_log1p"],
                     init_score=tr["init_score"],
                     categorical_feature=CAT_COLS)
    dv = lgb.Dataset(va[FEATURE_COLS], label=va["y_log1p"],
                     init_score=va["init_score"], reference=ds)
    booster = lgb.train(params, ds, num_boost_round=3000,
                        valid_sets=[dv],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
    raw = booster.predict(va[FEATURE_COLS]) + va["init_score"].to_numpy()
    pred = np.clip(np.expm1(raw), 0, None)
    return rmsle(va["tuketim"], pred)      # düşük = iyi

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

**Sonra F2 ve F3'te DOĞRULA.** F1'de kazanan parametre F2/F3'te de iyiyse gerçek;
sadece F1'de iyiyse overfit (bkz. `CLAUDE.md` fold rolleri). F1 kış, test yaz —
F1'e körü körüne overfit tehlikeli.

**warm/cold ayrı bak:** `va["is_cold"]` ile böl, iki skoru ayrı raporla. Global
düşüp cold yükseliyorsa testte zarar verir.

## 5. Feature Selection

- Önem (importance) / permutation önemini **F1 train'de** hesapla, **F1 valid'de**
  doğrula. Attığın feature F2/F3'te de zarar vermiyorsa güvenli.
- `seas_` (kapsam %35) ve bazı `wx_` feature'ları zayıf olabilir — adaylar bunlar.
- Kategorik feature çıkarırken `CAT_COLS`'u da güncelle.

## 6. Final model + submission

En iyi parametreyle `full_train`'de eğit, `test`'te tahmin et:
```python
best = study.best_params | {"objective":"regression","verbose":-1,"seed":42}
full, test = load_dataset("full_train"), load_dataset("test")
ds = lgb.Dataset(full[FEATURE_COLS], label=full["y_log1p"],
                 init_score=full["init_score"], categorical_feature=CAT_COLS)
booster = lgb.train(best, ds, num_boost_round=BEST_ITER)   # F1 best_iteration
raw = booster.predict(test[FEATURE_COLS]) + test["init_score"].to_numpy()
pred = np.clip(np.expm1(raw), 0, None)
# cold satırlarda b5 harmanı için bkz. scripts/17_save_wx_model.py
```
Not: yukarısı yalnızca **ana model**. Bizim en iyi kurgu (s2/wx) cold satırlarda
ayrı bir cold model + b5 baseline harmanı kullanır (`scripts/17`). Optuna'yı önce
ana modelde kullan; cold harmanı sonra sabit eklenir.

## 7. Metrik

```python
from src.validation import rmsle          # sqrt(mean((log1p(p)-log1p(y))^2))
# birleşik (test kompozisyonu): warm %77.8, cold %22.2
blend = (0.778*warm_mse + 0.222*cold_mse) ** 0.5
```

---

## Yapma listesi (özet)

- ❌ `full_train` + fold valid karıştırma (sızıntı)
- ❌ Random KFold / `train_test_split` (verilen fold'ları kullan)
- ❌ `init_score`'u eğitimde verip tahminde unutma (ikisinde de ekle)
- ❌ Sadece F1'e optimize (F2/F3 doğrula)
- ❌ warm/cold'u birlikte gömme (ayrı raporla)
- ✅ Dataset'i hava değişince `scripts/18` ile tazele
