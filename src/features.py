# -*- coding: utf-8 -*-
"""Feature engineering. Sözleşme: build_features(df, forecast_origin, history).

- df:       featurize edilecek satırlar (train+valid veya train+test birleşimi).
            Gerekli kolonlar: tanim, guc, tarih, il, bolge, ilce_key, guc_bucket,
            ay_no, dow, haftaici
- history:  hedef geçmişi — YALNIZCA forecast_origin ve öncesi satırlar
            (fold'un train_idx satırları; cold trafoların hiçbir satırı girmez)

Prefix kuralı: static_ / cal_ / lvl_ / grp_ / seas_   (wx_ ayrı adımda)
lvl_ ve seas_ cold trafolarda NaN kalır — doldurulmaz (CLAUDE.md).
grp_ istatistikleri MEDYAN/GEOMETRİK ile kurulur, aritmetik değil; trafo
sayısı <10 hücrelere üst seviyeye doğru shrinkage uygulanır.
"""
import numpy as np
import pandas as pd

from src.config import YOY_DRIFT

SHRINK_K = 10          # shrinkage kuvveti (trafo sayısı cinsinden)
RAMAZAN = [("2025-03-01", "2025-03-29"), ("2026-02-18", "2026-03-19")]
CDD_THRESHOLD = 22.0   # soğutma derece-gün eşiği (Ege; 21–24 arası optimize edilebilir)
HDD_THRESHOLD = 18.0

_WX_CACHE = None        # türetilmiş hava feature'ları (ilce_key, tarih) indeksli


def _weather_features() -> pd.DataFrame:
    """weather.parquet'ten wx_ feature'larını türetir (modül-cache).

    Hepsi (ilce_key, tarih) fonksiyonu — dış veri, hedeften bağımsız, fold/origin
    bağımsız (sızıntı yok). Hareketli pencereler hedef gününden geriye bakar; hava
    test döneminde de gözlemlendiği için test'te de doludur.
    """
    global _WX_CACHE
    if _WX_CACHE is not None:
        return _WX_CACHE
    from src.weather import _CACHE
    if not _CACHE.exists():
        raise FileNotFoundError("weather.parquet yok — önce scripts/15_fetch_weather.py")
    wx = pd.read_parquet(_CACHE).sort_values(["ilce_key", "tarih"]).copy()
    g = wx.groupby("ilce_key", observed=True)
    T = wx["wx_ham_temperature_2m_mean"]

    wx["wx_cdd"] = (T - CDD_THRESHOLD).clip(lower=0)
    wx["wx_cdd2"] = wx["wx_cdd"] ** 2                     # klima üstel artış
    wx["wx_cdd3"] = wx["wx_cdd"] ** 3
    wx["wx_hdd"] = (HDD_THRESHOLD - T).clip(lower=0)
    wx["wx_t_mean"] = T
    wx["wx_t_max"] = wx["wx_ham_temperature_2m_max"]
    wx["wx_t_range"] = (wx["wx_ham_temperature_2m_max"]
                        - wx["wx_ham_temperature_2m_min"])
    wx["wx_apparent_max"] = wx["wx_ham_apparent_temperature_max"]
    wx["wx_humidity"] = wx["wx_ham_relative_humidity_2m_mean"]
    wx["wx_precip"] = wx["wx_ham_precipitation_sum"]
    wx["wx_et0"] = wx["wx_ham_et0_fao_evapotranspiration"]
    wx["wx_soil_moist"] = wx["wx_ham_soil_moisture_0_to_7cm_mean"]
    wx["wx_radiation"] = wx["wx_ham_shortwave_radiation_sum"]

    # hareketli pencereler (termal kütle + tarımsal su dengesi)
    wx["wx_cdd_ma7"] = g["wx_cdd"].transform(
        lambda s: s.rolling(7, min_periods=1).mean())
    wx["wx_et0_sum7"] = g["wx_et0"].transform(
        lambda s: s.rolling(7, min_periods=1).sum())
    wx["wx_precip_30d"] = g["wx_precip"].transform(
        lambda s: s.rolling(30, min_periods=1).sum())     # düşük = kurak = sulama
    # "ilk sıcak gün" anomalisi: bugün vs önceki 7 gün ort (klimanın ilk açılışı)
    prev7 = g["wx_t_mean"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    wx["wx_t_anom7"] = (T - prev7)

    for c in WX_FEATURES:
        wx[c] = wx[c].astype("float32")
    wx["_ik"] = wx["ilce_key"].astype(str)
    _WX_CACHE = wx.set_index(["_ik", "tarih"])[WX_FEATURES]
    return _WX_CACHE


WX_FEATURES = ["wx_cdd", "wx_cdd2", "wx_cdd3", "wx_hdd", "wx_t_mean", "wx_t_max",
               "wx_t_range", "wx_apparent_max", "wx_humidity", "wx_precip",
               "wx_et0", "wx_soil_moist", "wx_radiation", "wx_cdd_ma7",
               "wx_et0_sum7", "wx_precip_30d", "wx_t_anom7"]


# ---------------------------------------------------------------- yardımcılar
def _tr_holidays(years=(2025, 2026)) -> dict:
    """holidays.TR — resmî + dinî tatiller. Anahtar kontrol: 23 Nisan, 1 Mayıs,
    19 Mayıs, 15 Temmuz, 30 Ağustos, 29 Ekim + Ramazan/Kurban bayramları."""
    import holidays as hol
    tr = hol.TR(years=list(years))
    for must in ["2026-04-23", "2026-05-01", "2026-05-19", "2026-07-15"]:
        assert pd.Timestamp(must) in tr, f"holidays.TR eksik: {must}"
    return {pd.Timestamp(d): name for d, name in tr.items()}


# ---------------------------------------------------------------- ana fonksiyon
def build_features(df: pd.DataFrame, forecast_origin: str,
                   history: pd.DataFrame) -> pd.DataFrame:
    origin = pd.Timestamp(forecast_origin)
    assert history["tarih"].max() <= origin, \
        "history, forecast_origin sonrası satır içeriyor — SIZINTI"

    out = pd.DataFrame(index=df.index)
    hist = history.copy()
    hist["log1p"] = np.log1p(hist["tuketim"])
    hist["lf"] = hist["tuketim"] / (hist["guc"] * 24.0)
    hist["is_zero"] = hist["tuketim"] == 0

    # ================================================== static_
    out["static_guc"] = df["guc"].astype("float32")
    out["static_log_guc"] = np.log1p(df["guc"]).astype("float32")
    out["static_guc_bucket"] = df["guc_bucket"]
    out["static_il"] = df["il"]
    out["static_bolge"] = df["bolge"]
    out["static_ilce_key"] = df["ilce_key"]

    first_seen = pd.concat([hist[["tanim", "tarih"]], df[["tanim", "tarih"]]]) \
        .groupby("tanim", observed=True)["tarih"].min()
    out["static_is_first_day"] = (
        df["tarih"] == df["tanim"].map(first_seen)).astype("int8")
    bad = hist.groupby("tanim", observed=True)["is_bad_row"].any()
    out["static_has_bad_rows"] = df["tanim"].map(bad).fillna(False).astype("int8")

    # ================================================== cal_
    out["cal_ay"] = df["ay_no"]
    out["cal_dow"] = df["dow"]
    out["cal_is_weekend"] = (~df["haftaici"]).astype("int8")
    doy = df["tarih"].dt.dayofyear
    out["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    out["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    out["cal_hafta"] = df["tarih"].dt.isocalendar().week.astype("int16")

    hmap = _tr_holidays()
    hdates = pd.DatetimeIndex(sorted(hmap))
    out["cal_is_holiday"] = df["tarih"].isin(hdates).astype("int8")
    out["cal_holiday_name"] = df["tarih"].map(
        lambda t: hmap.get(t, "yok")).astype("category")
    pos_next = hdates.searchsorted(df["tarih"], side="left")
    pos_prev = hdates.searchsorted(df["tarih"], side="right") - 1
    nxt = hdates[np.clip(pos_next, 0, len(hdates) - 1)]
    prv = hdates[np.clip(pos_prev, 0, len(hdates) - 1)]
    out["cal_days_to_holiday"] = np.clip(
        (nxt - df["tarih"].values).days, 0, 15).astype("int8")
    out["cal_days_since_holiday"] = np.clip(
        (df["tarih"].values - prv).days, 0, 15).astype("int8")
    # köprü günü: Pzt ve Salı tatil, ya da Cuma ve Perşembe tatil
    next_is_hol = (df["tarih"] + pd.Timedelta(days=1)).isin(hdates)
    prev_is_hol = (df["tarih"] - pd.Timedelta(days=1)).isin(hdates)
    out["cal_is_bridge_day"] = (((df["dow"] == 0) & next_is_hol) |
                                ((df["dow"] == 4) & prev_is_hol)).astype("int8")
    ram = pd.Series(False, index=df.index)
    for a, b in RAMAZAN:
        ram |= df["tarih"].between(a, b)
    out["cal_is_ramadan"] = ram.astype("int8")
    # tahmin başlangıcına uzaklık (test geometrisi: 1..122)
    out["cal_horizon_days"] = (df["tarih"] - origin).dt.days.astype("int16")

    # ================================================== lvl_
    g_all = hist.groupby("tanim", observed=True)
    out["lvl_history_days"] = df["tanim"].map(
        g_all["tarih"].nunique()).astype("float32")

    for wdays in (28, 56, 90):
        sub = hist[hist["tarih"] > origin - pd.Timedelta(days=wdays)]
        out[f"lvl_mean_log_{wdays}d"] = df["tanim"].map(
            sub.groupby("tanim", observed=True)["log1p"].mean()).astype("float32")
    sub90 = hist[hist["tarih"] > origin - pd.Timedelta(days=90)].copy()
    g90 = sub90.groupby("tanim", observed=True)
    out["lvl_std_log_90d"] = df["tanim"].map(g90["log1p"].std()).astype("float32")

    # 90 günlük lineer trend eğimi: cov(x,y)/var(x), x = güne uzaklık
    sub90["x"] = (sub90["tarih"] - origin).dt.days.astype("float32")
    stats = g90.agg(n=("x", "size"), sx=("x", "sum"), sy=("log1p", "sum"))
    sub90["xy"] = sub90["x"] * sub90["log1p"]
    sub90["xx"] = sub90["x"] * sub90["x"]
    stats["sxy"] = g90["xy"].sum()
    stats["sxx"] = g90["xx"].sum()
    var = stats["sxx"] - stats["sx"] ** 2 / stats["n"]
    cov = stats["sxy"] - stats["sx"] * stats["sy"] / stats["n"]
    slope = (cov / var.replace(0, np.nan))
    out["lvl_trend_slope_90d"] = df["tanim"].map(slope).astype("float32")

    out["lvl_lf_median_90d"] = df["tanim"].map(
        sub90.loc[~sub90["is_bad_row"]].groupby("tanim", observed=True)["lf"]
        .median()).astype("float32")

    for wdays in (30, 90):
        sub = hist[hist["tarih"] > origin - pd.Timedelta(days=wdays)]
        out[f"lvl_zero_ratio_{wdays}d"] = df["tanim"].map(
            sub.groupby("tanim", observed=True)["is_zero"].mean()).astype("float32")

    # --- mevsim-nötr tam-pencere seviye çıpaları (b2'nin gördüğü bilgi) -------
    out["lvl_mean_log_full"] = df["tanim"].map(g_all["log1p"].mean()).astype("float32")
    out["lvl_std_log_full"] = df["tanim"].map(g_all["log1p"].std()).astype("float32")
    out["lvl_zero_ratio_full"] = df["tanim"].map(
        g_all["is_zero"].mean()).astype("float32")
    out["lvl_lf_median_full"] = df["tanim"].map(
        hist.loc[~hist["is_bad_row"]].groupby("tanim", observed=True)["lf"]
        .median()).astype("float32")
    sub364 = hist[hist["tarih"] > origin - pd.Timedelta(days=364)]
    out["lvl_mean_log_364d"] = df["tanim"].map(
        sub364.groupby("tanim", observed=True)["log1p"].mean()).astype("float32")
    # "son dönem, trafonun normalinden ne kadar sapmış" — mevsimsel sapma ölçüsü
    out["lvl_full_over_90d"] = (out["lvl_mean_log_full"]
                                - out["lvl_mean_log_90d"]).astype("float32")
    out["lvl_full_over_28d"] = (out["lvl_mean_log_full"]
                                - out["lvl_mean_log_28d"]).astype("float32")

    last_nonzero = hist.loc[~hist["is_zero"]].groupby(
        "tanim", observed=True)["tarih"].max()
    dsl = (origin - df["tanim"].map(last_nonzero)).dt.days.astype("float32")
    out["lvl_days_since_last_nonzero"] = dsl
    # ardışık sıfır satır sayısı (son sıfır-dışı kayıttan sonra kalan satırlar)
    hist["_last_nz"] = hist["tanim"].map(last_nonzero)
    trailing = hist[hist["_last_nz"].isna() | (hist["tarih"] > hist["_last_nz"])] \
        .groupby("tanim", observed=True).size()
    streak = df["tanim"].map(trailing).astype("float32")
    in_hist = df["tanim"].isin(set(hist["tanim"].unique()))
    streak[in_hist & streak.isna()] = 0.0   # geçmişi var, sıfır serisi yok
    out["lvl_zero_streak_days"] = streak
    out["lvl_is_dead_flag"] = np.where(
        in_hist, (streak >= 30).astype("float32"), np.nan)

    # ================================================== grp_
    # yalnızca history'den (fold train penceresi); valid/cold trafo verisi girmez
    ntx_cell = hist.groupby(["ilce_key", "ay_no"], observed=True)["tanim"].nunique()

    def shrunk(cell_stat, parent_stat, parent_keys, cell_keys, n_cell):
        """cell'i, hücre trafo sayısıyla parent'a doğru çeker (n/(n+K))."""
        cell_df = cell_stat.rename("v").reset_index()
        cell_df = cell_df.merge(n_cell.rename("n").reset_index(), on=cell_keys,
                                how="left")
        cell_df = cell_df.merge(parent_stat.rename("p").reset_index(),
                                on=parent_keys, how="left")
        wgt = cell_df["n"] / (cell_df["n"] + SHRINK_K)
        cell_df["v"] = wgt * cell_df["v"] + (1 - wgt) * cell_df["p"]
        return cell_df.set_index(cell_keys)["v"]

    def look(stat, keys):
        idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 \
            else pd.Index(df[keys[0]])
        return pd.Series(stat.reindex(idx).to_numpy(), index=df.index,
                         dtype="float32")

    ok = ~hist["is_bad_row"]
    hist["log_lf"] = np.log1p(hist["lf"].clip(lower=0))

    # ilçe × ay medyan log-LF (il × ay parent'ına shrink)
    c1 = hist[ok].groupby(["ilce_key", "ay_no"], observed=True)["log_lf"].median()
    p1 = hist[ok].groupby(["il", "ay_no"], observed=True)["log_lf"].median()
    key_il = hist[["ilce_key", "il"]].drop_duplicates().set_index("ilce_key")["il"]
    c1s = c1.rename("v").reset_index()
    c1s["il"] = c1s["ilce_key"].map(key_il)
    c1s = c1s.merge(ntx_cell.rename("n").reset_index(), on=["ilce_key", "ay_no"])
    c1s = c1s.merge(p1.rename("p").reset_index(), on=["il", "ay_no"], how="left")
    wgt = c1s["n"] / (c1s["n"] + SHRINK_K)
    c1s["v"] = wgt * c1s["v"] + (1 - wgt) * c1s["p"].fillna(c1s["v"])
    out["grp_lf_med_ilce_ay"] = look(
        c1s.set_index(["ilce_key", "ay_no"])["v"], ["ilce_key", "ay_no"])

    # ilçe × ay geometrik mevsim indeksi (log-fark; il indeksine shrink)
    cell_m = hist.groupby(["ilce_key", "ay_no"], observed=True)["log1p"].median()
    base_m = hist.groupby("ilce_key", observed=True)["log1p"].median()
    idx_cell = (cell_m - base_m.reindex(cell_m.index.get_level_values(0)).values)
    cell_il = hist.groupby(["il", "ay_no"], observed=True)["log1p"].median()
    base_il = hist.groupby("il", observed=True)["log1p"].median()
    idx_il = (cell_il - base_il.reindex(cell_il.index.get_level_values(0)).values)
    ci = idx_cell.rename("v").reset_index()
    ci["il"] = ci["ilce_key"].map(key_il)
    ci = ci.merge(ntx_cell.rename("n").reset_index(), on=["ilce_key", "ay_no"])
    ci = ci.merge(idx_il.rename("p").reset_index(), on=["il", "ay_no"], how="left")
    wgt = ci["n"] / (ci["n"] + SHRINK_K)
    ci["v"] = wgt * ci["v"] + (1 - wgt) * ci["p"].fillna(ci["v"])
    out["grp_seasonal_ilce_ay"] = look(
        ci.set_index(["ilce_key", "ay_no"])["v"], ["ilce_key", "ay_no"])

    # il × guc_bucket × ay medyan log-LF
    c3 = hist[ok].groupby(["il", "guc_bucket", "ay_no"], observed=True)["log_lf"].median()
    out["grp_lf_med_il_bucket_ay"] = look(c3, ["il", "guc_bucket", "ay_no"])

    # ilçe × dow oranı (trafo-ay içi normalize, medyan)
    tm = hist.groupby(["tanim", "ay_no"], observed=True)["tuketim"].transform("mean")
    hist["_ratio"] = hist["tuketim"] / tm.replace(0, np.nan)
    c4 = hist.groupby(["ilce_key", "dow"], observed=True)["_ratio"].median()
    out["grp_dow_ratio_ilce"] = look(c4, ["ilce_key", "dow"])

    out["grp_n_transformers"] = look(
        hist.groupby("ilce_key", observed=True)["tanim"].nunique().astype("float32"),
        ["ilce_key"])

    # sıfır oranları (ilçe geneline shrink)
    z_cell = hist.groupby(["ilce_key", "ay_no"], observed=True)["is_zero"].mean()
    z_ilce = hist.groupby("ilce_key", observed=True)["is_zero"].mean()
    zc = z_cell.rename("v").reset_index()
    zc = zc.merge(ntx_cell.rename("n").reset_index(), on=["ilce_key", "ay_no"])
    zc = zc.merge(z_ilce.rename("p").reset_index(), on="ilce_key", how="left")
    wgt = zc["n"] / (zc["n"] + SHRINK_K)
    zc["v"] = wgt * zc["v"] + (1 - wgt) * zc["p"]
    out["grp_zero_rate_ilce_ay"] = look(
        zc.set_index(["ilce_key", "ay_no"])["v"], ["ilce_key", "ay_no"])
    out["grp_zero_rate_bucket"] = look(
        hist.groupby("guc_bucket", observed=True)["is_zero"].mean(), ["guc_bucket"])

    # --- grp_ ayrıştırması: b5'in gördüğü bilgi modele aynen verilir ---------
    # Sıfırlar HARİÇ seviye istatistikleri + sıfır oranı ayrı ayrı; hepsi
    # shrinkage'lı, yalnızca history'den.
    def shrunk_stat(frame, keys, parent_keys, col, aggfunc):
        cell = frame.groupby(keys, observed=True)[col].agg(aggfunc) \
            .rename("v").reset_index()
        n = frame.groupby(keys, observed=True)["tanim"].nunique() \
            .rename("n").reset_index()
        cell = cell.merge(n, on=keys)
        if "il" in parent_keys and "il" not in cell.columns:
            cell["il"] = cell["ilce_key"].map(key_il)
        par = frame.groupby(parent_keys, observed=True)[col].agg(aggfunc) \
            .rename("p").reset_index()
        cell = cell.merge(par, on=parent_keys, how="left")
        wgt = cell["n"] / (cell["n"] + SHRINK_K)
        cell["v"] = wgt * cell["v"] + (1 - wgt) * cell["p"].fillna(cell["v"])
        return cell.set_index(keys)["v"]

    hz = hist[ok & ~hist["is_zero"]]
    out["grp_lf_nz_med_ilce_ay_hi"] = look(
        shrunk_stat(hz, ["ilce_key", "ay_no", "haftaici"],
                    ["il", "ay_no", "haftaici"], "log_lf", "median"),
        ["ilce_key", "ay_no", "haftaici"])
    out["grp_lf_nz_med_il_bucket_ay"] = look(
        shrunk_stat(hz, ["il", "guc_bucket", "ay_no"],
                    ["guc_bucket", "ay_no"], "log_lf", "median"),
        ["il", "guc_bucket", "ay_no"])
    out["grp_zero_rate_ilce_ay_hi"] = look(
        shrunk_stat(hist, ["ilce_key", "ay_no", "haftaici"],
                    ["il", "ay_no", "haftaici"], "is_zero", "mean"),
        ["ilce_key", "ay_no", "haftaici"])
    out["grp_lf_nz_p25_ilce_ay"] = look(
        shrunk_stat(hz, ["ilce_key", "ay_no"], ["il", "ay_no"], "log_lf",
                    lambda s: s.quantile(0.25)),
        ["ilce_key", "ay_no"])
    out["grp_lf_nz_p75_ilce_ay"] = look(
        shrunk_stat(hz, ["ilce_key", "ay_no"], ["il", "ay_no"], "log_lf",
                    lambda s: s.quantile(0.75)),
        ["ilce_key", "ay_no"])

    # NOT: lvl_season_adjusted_90d/28d ve lvl_season_gap kaldırıldı — F1'de
    # %0.00 gain (model_v5). Mevsim bilgisi artık build_anchor'da taşınıyor.
    # b2'nin birebir çıpası: tam pencere MEDYANI (ortalama değil)
    out["lvl_median_log_full"] = df["tanim"].map(
        g_all["log1p"].median()).astype("float32")
    out["lvl_lf_median_364d"] = df["tanim"].map(
        sub364.loc[~sub364["is_bad_row"]].groupby("tanim", observed=True)["lf"]
        .median()).astype("float32")

    # ================================================== seas_
    # trafo-günlük seriye ±7 gün merkezli 15 günlük medyan uygula, lag364'te oku
    daily = (hist.groupby(["tanim", "tarih"], observed=True)["log1p"].mean()
             .reset_index())

    def _roll(g):
        s = g.set_index("tarih")["log1p"].asfreq("D")
        return s.rolling(15, center=True, min_periods=1).median()

    rolled = daily.groupby("tanim", observed=True)[["tarih", "log1p"]] \
        .apply(_roll).rename("seas_val").reset_index()
    lag_key = pd.DataFrame({
        "tanim": df["tanim"],
        "tarih": df["tarih"] - pd.Timedelta(days=364)})
    m = lag_key.merge(rolled, on=["tanim", "tarih"], how="left")
    seas = pd.Series(m["seas_val"].to_numpy(), index=df.index, dtype="float32")
    out["seas_lag364_log1p"] = seas
    out["seas_lag364_available"] = seas.notna().astype("int8")
    own_mean = df["tanim"].map(g_all["log1p"].mean())
    out["seas_lag364_ratio_own"] = (seas - own_mean).astype("float32")
    out["seas_lag364_drift_adj"] = (seas + YOY_DRIFT).astype("float32")

    # ================================================== wx_ (dış veri, hedef günün havası)
    wxf = _weather_features()
    ik = df["ilce_key"].astype(str).to_numpy()
    ridx = pd.MultiIndex.from_arrays([ik, df["tarih"].to_numpy()])
    for col in WX_FEATURES:
        out[col] = pd.Series(wxf[col].reindex(ridx).to_numpy(),
                             index=df.index, dtype="float32")

    return out


def anchor_components(df: pd.DataFrame, forecast_origin: str,
                      history: pd.DataFrame) -> pd.DataFrame:
    """Anchor'ı bileşenlere ayırır: base (mevsim-nötr seviye) + season_dev
    (mevsim sapması) + zero_adj (cold sıfır düzeltmesi). assemble_anchor bunları
    alpha ve cold_zero_adj ile birleştirir — böylece feature build bir kez yapılıp
    tüm alpha/varyantlar ucuza türetilir.

    warm:  base = lvl_median_log_full · dev = seas_idx[ilce,ay] − yıllık_ort · zadj=0
    cold:  base = log(guc*24) + log(LF_nz_year[ilce]) · dev = log(LF_nz[ilce,ay,hi])
           − log(LF_nz_year) · zadj = log(1 − zero_rate[ilce,ay])
    Yalnızca origin öncesi geçmişten; eğitim ve tahminde AYNI kurulur.
    """
    origin = pd.Timestamp(forecast_origin)
    assert history["tarih"].max() <= origin, "anchor: history origin'i aşıyor"
    hist = history.copy()
    hist["log1p"] = np.log1p(hist["tuketim"])
    hist["lf"] = hist["tuketim"] / (hist["guc"] * 24.0)
    hist["is_zero"] = hist["tuketim"] == 0

    def map_num(series, mapping):
        m = series.astype(object).map(mapping.to_dict())
        return pd.Series(pd.to_numeric(m, errors="coerce").to_numpy(),
                         index=series.index, dtype="float64")

    def chain(frame, col, chains, aggfunc):
        """Fallback zinciri ile satır bazında değer (log ölçekte)."""
        val = pd.Series(np.nan, index=df.index)
        for keys in chains:
            stat = frame.groupby(keys, observed=True)[col].agg(aggfunc)
            ridx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 \
                else pd.Index(df[keys[0]])
            cand = pd.Series(pd.to_numeric(stat.reindex(ridx), errors="coerce")
                             .to_numpy(), index=df.index)
            val = val.fillna(cand)
        return val.fillna(float(frame[col].agg(aggfunc)))

    # ---- warm: tam pencere medyanı + shrunk mevsim sapması ----
    med_full = map_num(df["tanim"],
                       hist.groupby("tanim", observed=True)["log1p"].median())
    key_il = hist[["ilce_key", "il"]].drop_duplicates().set_index("ilce_key")["il"]
    ntx = hist.groupby(["ilce_key", "ay_no"], observed=True)["tanim"].nunique()
    cell = hist.groupby(["ilce_key", "ay_no"], observed=True)["log1p"].median()
    base_i = hist.groupby("ilce_key", observed=True)["log1p"].median()
    idx = (cell - base_i.reindex(cell.index.get_level_values(0)).values)
    cell_il = hist.groupby(["il", "ay_no"], observed=True)["log1p"].median()
    base_il = hist.groupby("il", observed=True)["log1p"].median()
    idx_il = (cell_il - base_il.reindex(cell_il.index.get_level_values(0)).values)
    ci = idx.rename("v").reset_index()
    ci["il"] = ci["ilce_key"].map(key_il)
    ci = ci.merge(ntx.rename("n").reset_index(), on=["ilce_key", "ay_no"])
    ci = ci.merge(idx_il.rename("p").reset_index(), on=["il", "ay_no"], how="left")
    wgt = ci["n"] / (ci["n"] + SHRINK_K)
    ci["v"] = wgt * ci["v"] + (1 - wgt) * ci["p"].fillna(ci["v"])
    seas = ci.set_index(["ilce_key", "ay_no"])["v"]
    seas_year = seas.groupby(level=0).mean()
    row_idx = pd.MultiIndex.from_frame(df[["ilce_key", "ay_no"]])
    dev_warm = (pd.Series(pd.to_numeric(seas.reindex(row_idx), errors="coerce")
                          .to_numpy(), index=df.index)
                - map_num(df["ilce_key"], seas_year)).fillna(0.0)

    # ---- cold: guc çapası + mevsimli LF sapması + sıfır düzeltmesi ----
    hz = hist[(~hist["is_bad_row"]) & (hist["tuketim"] > 0)].copy()
    hz["log_lf"] = np.log(hz["lf"].clip(lower=1e-3))
    lf_cell = chain(hz, "log_lf",
                    [["ilce_key", "ay_no", "haftaici"], ["ilce_key", "ay_no"],
                     ["il", "guc_bucket", "ay_no"], ["guc_bucket", "ay_no"]],
                    "median")
    lf_year = chain(hz, "log_lf",
                    [["ilce_key"], ["il", "guc_bucket"], ["guc_bucket"]], "median")
    cold_base = np.log(df["guc"] * 24.0) + lf_year
    cold_dev = lf_cell - lf_year
    # sıfır düzeltmesi: log(1 − zero_rate), shrunk (ilce genelıne)
    zr = chain(hist, "is_zero", [["ilce_key", "ay_no"], ["ilce_key"],
                                 ["guc_bucket"]], "mean").clip(0, 0.95)
    zero_adj = np.log(1.0 - zr)

    is_cold = med_full.isna()
    return pd.DataFrame({
        "base": cold_base.where(is_cold, med_full),
        "season_dev": cold_dev.where(is_cold, dev_warm),
        "zero_adj": zero_adj.where(is_cold, 0.0),
        "is_cold_anchor": is_cold,
    }, index=df.index)


def assemble_anchor(comp: pd.DataFrame, alpha: float = 1.0,
                    cold_zero_adj: bool = False) -> pd.Series:
    """Bileşenlerden init_score kur: base + alpha·season_dev (+ cold zero_adj)."""
    a = comp["base"] + alpha * comp["season_dev"]
    if cold_zero_adj:
        a = a + comp["zero_adj"]        # zero_adj warm satırlarda zaten 0
    return a.astype("float64")


def build_anchor(df: pd.DataFrame, forecast_origin: str, history: pd.DataFrame,
                 alpha: float = 1.0, cold_zero_adj: bool = False) -> pd.Series:
    """Geriye uyumlu sarmalayıcı (varsayılan = eski davranış: alpha=1, adj yok)."""
    return assemble_anchor(anchor_components(df, forecast_origin, history),
                           alpha, cold_zero_adj)


FEATURE_GROUPS: dict[str, list[str]] = {
    "static": ["static_guc", "static_log_guc", "static_guc_bucket", "static_il",
               "static_bolge", "static_ilce_key", "static_is_first_day",
               "static_has_bad_rows"],
    "cal": ["cal_ay", "cal_dow", "cal_is_weekend", "cal_doy_sin", "cal_doy_cos",
            "cal_hafta", "cal_is_holiday", "cal_holiday_name",
            "cal_days_to_holiday", "cal_days_since_holiday", "cal_is_bridge_day",
            "cal_is_ramadan", "cal_horizon_days"],
    "lvl": ["lvl_history_days", "lvl_mean_log_28d", "lvl_mean_log_56d",
            "lvl_mean_log_90d", "lvl_std_log_90d", "lvl_trend_slope_90d",
            "lvl_lf_median_90d", "lvl_zero_streak_days", "lvl_zero_ratio_30d",
            "lvl_zero_ratio_90d", "lvl_days_since_last_nonzero", "lvl_is_dead_flag",
            "lvl_mean_log_full", "lvl_std_log_full", "lvl_zero_ratio_full",
            "lvl_lf_median_full", "lvl_mean_log_364d", "lvl_full_over_90d",
            "lvl_full_over_28d", "lvl_median_log_full", "lvl_lf_median_364d"],
    "grp": ["grp_lf_med_ilce_ay", "grp_seasonal_ilce_ay", "grp_lf_med_il_bucket_ay",
            "grp_dow_ratio_ilce", "grp_n_transformers", "grp_zero_rate_ilce_ay",
            "grp_zero_rate_bucket", "grp_lf_nz_med_ilce_ay_hi",
            "grp_lf_nz_med_il_bucket_ay", "grp_zero_rate_ilce_ay_hi",
            "grp_lf_nz_p25_ilce_ay", "grp_lf_nz_p75_ilce_ay"],
    "seas": ["seas_lag364_log1p", "seas_lag364_available", "seas_lag364_ratio_own",
             "seas_lag364_drift_adj"],
    "wx": ["wx_cdd", "wx_cdd2", "wx_cdd3", "wx_hdd", "wx_t_mean", "wx_t_max",
           "wx_t_range", "wx_apparent_max", "wx_humidity", "wx_precip",
           "wx_et0", "wx_soil_moist", "wx_radiation", "wx_cdd_ma7",
           "wx_et0_sum7", "wx_precip_30d", "wx_t_anom7"],
}

ALL_FEATURES = [f for grp in FEATURE_GROUPS.values() for f in grp]
CATEGORICAL_FEATURES = ["static_guc_bucket", "static_il", "static_bolge",
                        "static_ilce_key", "cal_holiday_name"]
