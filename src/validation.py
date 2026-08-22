# -*- coding: utf-8 -*-
"""Zaman bölmesi + geçmiş-uzunluğu (H) eşlemeli fold'lar, RMSLE, kırılımlı evaluate.

Fold mantığı (CLAUDE.md kural 4): cold-start ayrı vaka değil, H=0 halidir.
Her valid trafosuna test_history_profile.csv'den guc_bucket-stratified bir
(H, giriş-offseti) çifti örneklenir:
  - train tarafında yalnızca (train_end - H, train_end] aralığı bırakılır (H=0 → cold)
  - valid tarafında satırların başı, ölçekli giriş-offseti kadar kırpılır
H ve offset AYNI profil satırından birlikte örneklenir; böylece "cold trafo geç girer"
korelasyonu korunur ve cold satır payı test'teki %22.2'ye kendiliğinden oturur.

H ataması bucket içinde KANTİL EŞLEMELİDİR: örneklenen H'ler sıralanır ve derin
geçmişi olan trafoya uzun H, sığ olana kısa H düşer. Bağımsız rastgele atama,
kırpma tek yönlü olduğu için (veri eklenemez, sadece silinir) lag_364 kapsamını
sistematik olarak çökertir; test'te de cold/kısa-geçmiş trafolar zaten en yeni
trafolardır, yani eşleme gerçeğe daha sadıktır.
"""
import numpy as np
import pandas as pd

from src.config import (COLD_ROW_SHARE, H_MEDIAN, LAG364_COV_PM7, SEED,
                        TEST_N_DAYS, TEST_START, WARM_ROW_SHARE)

FOLD_SPECS = [
    {"name": "F1", "train_end": "2025-12-31",
     "valid_start": "2026-01-01", "valid_end": "2026-03-31"},
    {"name": "F2", "train_end": "2025-03-31",
     "valid_start": "2025-04-01", "valid_end": "2025-07-31"},
    {"name": "F3", "train_end": "2025-08-31",
     "valid_start": "2025-09-01", "valid_end": "2025-12-31"},
]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0, None)
    y_true = np.asarray(y_true, dtype="float64")
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def make_folds(df: pd.DataFrame, profile: pd.DataFrame, seed: int = SEED) -> list[dict]:
    """df: load_train() çıktısı. profile: load_profile() çıktısı.

    Dönen fold nesnesi: {"name", "train_idx", "valid_idx", "cold_tx", "H_map",
                         "spec"}. Filtreleme TEK noktadan: eğitim ve grp_
    istatistikleri yalnızca df.loc[train_idx] kullanmalıdır.
    """
    profile = profile.copy()
    profile["entry_offset"] = (
        profile["test_entry"] - pd.Timestamp(TEST_START)).dt.days
    by_bucket = {b: g[["H", "entry_offset"]].to_numpy()
                 for b, g in profile.groupby("guc_bucket", observed=True)}
    all_pairs = profile[["H", "entry_offset"]].to_numpy()

    tx_bucket = df.groupby("tanim", observed=True)["guc_bucket"].first()

    folds = []
    for i, spec in enumerate(FOLD_SPECS):
        rng = np.random.default_rng(seed + i)
        train_end = pd.Timestamp(spec["train_end"])
        v_start = pd.Timestamp(spec["valid_start"])
        v_end = pd.Timestamp(spec["valid_end"])
        valid_len = (v_end - v_start).days + 1

        in_train_win = df["tarih"] <= train_end
        in_valid_win = df["tarih"].between(v_start, v_end)
        valid_tx = df.loc[in_valid_win, "tanim"].unique()

        # her trafonun fold penceresi içindeki geçmiş derinliği (gün)
        first_seen = (df.loc[in_train_win].groupby("tanim", observed=True)["tarih"].min())
        depth = ((train_end - first_seen).dt.days + 1).clip(lower=0)

        # bucket içi kantil eşleme: örneklenen (H, offset) çiftleri H'ye göre,
        # trafolar derinliğe göre sıralanır ve hizalanarak atanır
        H_map, off_map = {}, {}
        vt = pd.Series(list(valid_tx))
        vt_bucket = vt.map(tx_bucket)
        for b, grp in vt.groupby(vt_bucket, observed=True):
            pool = by_bucket.get(b, all_pairs)
            pairs = pool[rng.integers(len(pool), size=len(grp))]
            pairs = pairs[np.argsort(pairs[:, 0])]              # H artan
            txs = sorted(grp, key=lambda t: depth.get(t, 0))    # derinlik artan
            for tx, (h, off) in zip(txs, pairs):
                H_map[tx] = int(h)
                # test 122 günlük; offseti valid pencere uzunluğuna orantıla
                off_map[tx] = int(round(off * valid_len / TEST_N_DAYS))

        tanim = df["tanim"]
        h_days = tanim.map(H_map)
        off_days = tanim.map(off_map)
        is_valid_tx = h_days.notna()

        # train tarafı: valid trafolarında sadece son H gün; diğer trafolar tam kalır
        min_keep = train_end - pd.to_timedelta(h_days.fillna(0), unit="D")
        keep_train = in_train_win & (~is_valid_tx | (df["tarih"] > min_keep))

        # valid tarafı: giriş-offseti kadar baştan kırp
        entry_date = v_start + pd.to_timedelta(off_days.fillna(0), unit="D")
        keep_valid = in_valid_win & (df["tarih"] >= entry_date)

        # cold = fold train'inde satırı KALMAYAN valid trafosu. H=0 atananlar +
        # zaten train penceresinde hiç görünmeyenler (H>0 atansa bile veri yok).
        surviving = set(df.loc[keep_train, "tanim"].unique())
        cold_tx = {tx for tx in valid_tx if tx not in surviving}

        # Kalibrasyon: doğal cold'lar valid'de az satırlı olduğundan cold SATIR
        # payı hedefin altında kalabilir. Hedefe ulaşana dek en sığ geçmişli warm
        # trafolar cold'a terfi eder (H'leri zaten küçük → lag kapsamı etkilenmez).
        vc = df.loc[keep_valid].groupby("tanim", observed=True).size()
        total_rows = int(vc.sum())
        cold_rows = int(vc.reindex(list(cold_tx)).fillna(0).sum())
        if cold_rows / total_rows < COLD_ROW_SHARE - 0.01:
            for tx in sorted((tx for tx in valid_tx if tx not in cold_tx),
                             key=lambda t: depth.get(t, 0)):
                if cold_rows / total_rows >= COLD_ROW_SHARE:
                    break
                cold_tx.add(tx)
                cold_rows += int(vc.get(tx, 0))
            keep_train = keep_train & ~df["tanim"].isin(cold_tx)

        H_map = {tx: (0 if tx in cold_tx else h) for tx, h in H_map.items()}
        folds.append({
            "name": spec["name"],
            "spec": spec,
            "train_idx": df.index[keep_train],
            "valid_idx": df.index[keep_valid],
            "cold_tx": cold_tx,
            "H_map": H_map,
        })
    return folds


def _lag364_cov_pm7(valid: pd.DataFrame, history: pd.DataFrame) -> float:
    """Valid satırlarının kaçı için tarih-364 ±7 gün penceresinde history kaydı var."""
    left = valid[["tanim", "tarih"]].copy()
    left["lag_date"] = left["tarih"] - pd.Timedelta(days=364)
    left["tanim"] = left["tanim"].astype("string")
    right = (history[["tanim", "tarih"]].drop_duplicates()
             .rename(columns={"tarih": "hist_date"}))
    right["tanim"] = right["tanim"].astype("string")
    m = pd.merge_asof(
        left.sort_values("lag_date"), right.sort_values("hist_date"),
        left_on="lag_date", right_on="hist_date", by="tanim",
        direction="nearest", tolerance=pd.Timedelta(days=7))
    return float(m["hist_date"].notna().mean())


def verify_fold(fold: dict, df: pd.DataFrame) -> dict:
    """Fold, test'in bilgi rejimini eşliyor mu? Hedeflerle kıyas.

    Sapma eşiği: oranlarda 0.05 mutlak, H medyanında %5 göreli.
    F2/F3'te lag364 kontrolü YAPISAL N/A: lag hedef penceresi (valid - 364 gün)
    veri başlangıcından (2025-01-01) önceye düşer, hiçbir fold kurgusu kapsayamaz.
    Bu fold'larda sadece cold_row_share ve h_median kontrol edilir.
    """
    valid = df.loc[fold["valid_idx"]]
    history = df.loc[fold["train_idx"]]

    cold_share = float(valid["tanim"].isin(fold["cold_tx"]).mean())
    lag_cov = _lag364_cov_pm7(valid, history)
    h_med = float(np.median(list(fold["H_map"].values())))
    lag_structural_na = fold["name"] in ("F2", "F3")

    res = {
        "fold": fold["name"],
        "cold_row_share": cold_share, "cold_row_target": COLD_ROW_SHARE,
        "lag364_cov_pm7": lag_cov, "lag364_target": LAG364_COV_PM7,
        "lag364_structural_na": lag_structural_na,
        "h_median": h_med, "h_median_target": H_MEDIAN,
        "warnings": [],
    }
    if abs(cold_share - COLD_ROW_SHARE) > 0.05:
        res["warnings"].append(
            f"cold_row_share {cold_share:.3f} hedeften ({COLD_ROW_SHARE}) sapıyor")
    if not lag_structural_na and abs(lag_cov - LAG364_COV_PM7) > 0.05:
        res["warnings"].append(
            f"lag364_cov {lag_cov:.3f} hedeften ({LAG364_COV_PM7}) sapıyor")
    if abs(h_med - H_MEDIAN) / H_MEDIAN > 0.05:
        res["warnings"].append(f"h_median {h_med:.0f} hedeften ({H_MEDIAN}) sapıyor")
    for wmsg in res["warnings"]:
        print(f"  UYARI [{fold['name']}]: {wmsg}")
    return res


def add_eval_columns(valid: pd.DataFrame, fold: dict, df: pd.DataFrame) -> pd.DataFrame:
    """evaluate() kırılımları için yardımcı kolonlar (fold bağlamından)."""
    valid = valid.copy()
    v_start = pd.Timestamp(fold["spec"]["valid_start"])
    train_end = pd.Timestamp(fold["spec"]["train_end"])

    valid["is_cold"] = valid["tanim"].isin(fold["cold_tx"])
    h = valid["tanim"].map(fold["H_map"])
    valid["H_bucket"] = pd.cut(h, bins=[-1, 0, 30, 90, 180, 300, 455],
                               labels=["0 (cold)", "1-30", "31-90", "91-180",
                                       "181-300", "301-455"])
    valid["ay"] = valid["tarih"].dt.to_period("M").astype(str)
    valid["ufuk_haftasi"] = ((valid["tarih"] - v_start).dt.days // 7 + 1)

    # forecast_origin itibarıyla sıfır serisi durumu (fold train verisinden)
    hist = df.loc[fold["train_idx"], ["tanim", "tarih", "tuketim"]]
    last_nonzero = (hist[hist["tuketim"] > 0]
                    .groupby("tanim", observed=True)["tarih"].max())
    streak = (train_end - valid["tanim"].map(last_nonzero)).dt.days
    valid["zero_streak_bucket"] = np.select(
        [valid["is_cold"], streak.isna(), streak < 30, streak < 120],
        ["cold", "hic_nonzero_yok", "aktif(<30)", "30-119"],
        default="120+")
    return valid


def evaluate(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """Kırılımlı RMSLE tablosu. df, add_eval_columns'tan geçmiş valid frame'idir."""
    err2 = (np.log1p(np.clip(df[y_pred_col], 0, None)) -
            np.log1p(df[y_true_col])) ** 2
    work = df.assign(_e2=err2)

    rows = [{"kirilim": "global", "seviye": "·", "n": len(work),
             "rmsle": float(np.sqrt(err2.mean()))}]

    # blend: test satır paylarıyla birleşik tahmin
    warm_mse = work.loc[~work["is_cold"], "_e2"].mean()
    cold_mse = work.loc[work["is_cold"], "_e2"].mean()
    if not np.isnan(cold_mse):
        blend = float(np.sqrt(WARM_ROW_SHARE * warm_mse + COLD_ROW_SHARE * cold_mse))
        rows.append({"kirilim": "blend", "seviye": "0.778w+0.222c",
                     "n": len(work), "rmsle": blend})

    dims = [("warm_cold", work["is_cold"].map({False: "warm", True: "cold"})),
            ("H_bucket", work.get("H_bucket")),
            ("ay", work.get("ay")),
            ("guc_bucket", work.get("guc_bucket")),
            ("il", work.get("il")),
            ("ilce", work.get("ilce_key")),
            ("ufuk_haftasi", work.get("ufuk_haftasi")),
            ("zero_streak", work.get("zero_streak_bucket"))]
    for name, col in dims:
        if col is None:
            continue
        g = work.groupby(col, observed=True)["_e2"].agg(["mean", "size"])
        for lvl, r in g.iterrows():
            rows.append({"kirilim": name, "seviye": str(lvl), "n": int(r["size"]),
                         "rmsle": float(np.sqrt(r["mean"]))})
    return pd.DataFrame(rows)
