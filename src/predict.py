# -*- coding: utf-8 -*-
"""Submission yazımı + zorunlu doğrulamalar."""
import numpy as np
import pandas as pd

from src.config import RAW_DIR


class SubmissionError(ValueError):
    pass


def write_submission(pred_df: pd.DataFrame, path) -> None:
    """pred_df: kolonlar ['id', 'tuketim']. sample_submission ile id kümesi VE
    sırası birebir doğrulanır; NaN/negatif/satır sayısı kontrol edilir.
    Hata varsa SubmissionError — sessizce düzeltme YAPILMAZ.
    """
    ss = pd.read_csv(RAW_DIR / "sample_submission.csv", usecols=["id"],
                     dtype={"id": "string"})

    if list(pred_df.columns) != ["id", "tuketim"]:
        raise SubmissionError(f"Kolonlar ['id','tuketim'] olmalı: {list(pred_df.columns)}")
    if len(pred_df) != len(ss):
        raise SubmissionError(f"Satır sayısı {len(pred_df)} != {len(ss)}")
    if pred_df["tuketim"].isna().any():
        raise SubmissionError(f"NaN tahmin: {int(pred_df['tuketim'].isna().sum())} satır")
    if (pred_df["tuketim"] < 0).any():
        raise SubmissionError(f"Negatif tahmin: {int((pred_df['tuketim'] < 0).sum())} satır")
    ids = pred_df["id"].astype("string")
    if not ids.equals(ss["id"]):
        n_diff = int((ids.to_numpy() != ss["id"].to_numpy()).sum()) \
            if len(ids) == len(ss) else -1
        raise SubmissionError(f"id sırası/kümesi sample_submission ile uyuşmuyor "
                              f"(farklı konum: {n_diff})")

    path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(path, index=False)
    print(f"Submission yazıldı: {path} ({len(pred_df):,} satır)")
