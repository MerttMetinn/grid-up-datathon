# -*- coding: utf-8 -*-
"""Veri yükleme + temizlik. İlk çalıştırmada parquet cache üretir, sonra hep oradan okur."""
import pandas as pd

from src.config import (GUC_BUCKET_BINS, GUC_BUCKET_LABELS, PROCESSED_DIR,
                        RAW_DIR)

_TRAIN_PQ = PROCESSED_DIR / "train.parquet"
_TEST_PQ = PROCESSED_DIR / "test.parquet"


def _parse_lokasyon(df: pd.DataFrame) -> pd.DataFrame:
    """İki format: 'İZMİR>BÖLGE>İLÇE' (3 parça) ve 'MANİSA>İLÇE' (2 parça).

    il = ilk parça, ilce = son parça, bolge = sadece 3 parçalıda orta parça.
    ilce_key = 'İL>İLÇE' — ilçe adı çakışmalarına karşı birleşik anahtar.
    """
    parts = df["lokasyon"].astype("string").str.split(">")
    n = parts.str.len()
    il = parts.str[0].str.strip()
    ilce = parts.str[-1].str.strip()
    bolge = parts.str[1].str.strip().where(n == 3)
    df["il"] = il.astype("category")
    df["bolge"] = bolge.astype("category")
    df["ilce"] = ilce.astype("category")
    df["ilce_key"] = (il + ">" + ilce).astype("category")
    return df


def _add_common(df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_lokasyon(df)
    df["guc_bucket"] = pd.cut(df["guc"], bins=GUC_BUCKET_BINS,
                              labels=GUC_BUCKET_LABELS)
    df["ay_no"] = df["tarih"].dt.month.astype("int8")
    df["dow"] = df["tarih"].dt.dayofweek.astype("int8")
    df["haftaici"] = (df["dow"] < 5)
    return df


def load_train(refresh: bool = False) -> pd.DataFrame:
    """Train + türetilmiş kolonlar. `is_bad_row` işaretlenir, satır SİLİNMEZ."""
    if _TRAIN_PQ.exists() and not refresh:
        return pd.read_parquet(_TRAIN_PQ)
    df = pd.read_csv(
        RAW_DIR / "train.csv",
        dtype={"tanim": "category", "guc": "float32", "tuketim": "float32",
               "lokasyon": "category"},
        parse_dates=["tarih"],
    )
    df = _add_common(df)
    df["is_bad_row"] = (df["tuketim"] / (df["guc"] * 24.0) > 1.0)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_TRAIN_PQ)
    return df


def load_test(refresh: bool = False) -> pd.DataFrame:
    if _TEST_PQ.exists() and not refresh:
        return pd.read_parquet(_TEST_PQ)
    df = pd.read_csv(
        RAW_DIR / "test.csv",
        dtype={"id": "string", "tanim": "category", "guc": "float32",
               "lokasyon": "category"},
        parse_dates=["tarih"],
    )
    df = _add_common(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_TEST_PQ)
    return df


def load_profile() -> pd.DataFrame:
    """recon-3'ün ürettiği test geçmiş profili (tanim, guc_bucket, H, test_entry...)."""
    from src.config import PROFILE_CSV
    prof = pd.read_csv(PROFILE_CSV, parse_dates=["test_entry"])
    return prof
