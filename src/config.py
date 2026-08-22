# -*- coding: utf-8 -*-
"""Sabitler, yollar, tarih sınırları. Her modül buradan okur."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_DIR = ROOT / "data" / "external"
REPORTS_DIR = ROOT / "reports"
EXPERIMENTS_DIR = ROOT / "experiments"
SUBMISSIONS_DIR = ROOT / "submissions"

SEED = 42

TRAIN_START, TRAIN_END = "2025-01-01", "2026-03-31"
TEST_START, TEST_END = "2026-04-01", "2026-07-31"
TEST_N_DAYS = 122

# Test'ten ölçülen hedefler (reports/recon2.md, recon3.md)
COLD_ROW_SHARE = 0.2216       # cold satır payı
LAG364_COV_PM7 = 0.350        # lag_364 ±7 gün kapsamı
H_MEDIAN = 105                # test trafolarının geçmiş-gün medyanı (cold=0 dahil)

# guc_bucket sınırları — recon3 / test_history_profile.csv ile aynı
GUC_BUCKETS = [(0, 160), (161, 400), (401, 1000), (1001, 1600), (1601, None)]
GUC_BUCKET_BINS = [0, 160, 400, 1000, 1600, float("inf")]
GUC_BUCKET_LABELS = ["<=160", "250-400", "630-1000", "1250-1600", ">1600"]

# Skor birleştirme ağırlıkları (satır payları)
WARM_ROW_SHARE = 1 - COLD_ROW_SHARE

PROFILE_CSV = PROCESSED_DIR / "test_history_profile.csv"

# Sabit kohorttan ölçülen YoY seviye kayması (reports/diagnosis.md bölüm 7).
# lag_364 feature'larının drift-düzeltilmiş kolonunda kullanılır.
YOY_DRIFT = 0.102

# Hedef çıta: sıfır problemi mükemmel çözülse ulaşılacak F1 blend seviyesi.
TARGET_BLEND = 1.07

# NOT: HISTORY_DROPOUT kaldırıldı — yerine çok-origin eğitimde origin başına
# H-örneklemesi kullanılır (src/train.py).
