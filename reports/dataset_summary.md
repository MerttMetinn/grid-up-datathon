# Dataset Paketi Özeti

Üretim: `scripts/18_build_dataset.py` · 2026-08-23 13:43

Konum: `data/dataset/` · toplam 75 feature (hava dahil)

### Üretilen dosyalar

- f1_train (2,001,625)
- f1_valid (299,929)
- f2_train (255,713)
- f2_valid (251,757)
- f3_train (1,202,176)
- f3_valid (320,622)
- full_train (3,032,692)
- test (714,688)

### Feature grupları

- `static_` : 8 feature
- `cal_` : 13 feature
- `lvl_` : 21 feature
- `grp_` : 12 feature
- `seas_` : 4 feature
- `wx_` : 17 feature

- Kategorik feature'lar: ['static_guc_bucket', 'static_il', 'static_bolge', 'static_ilce_key', 'cal_holiday_name']

### Feature dışı kolonlar (her dosyada)

['id', 'tanim', 'tarih', 'guc', 'is_cold', 'anc_base', 'anc_dev', 'anc_zero', 'init_score']

- `y_log1p` : model hedefi (log1p tüketim) — train/valid'de
- `tuketim` : orijinal ölçek gerçek değer — train/valid'de (skorlama için)
- `init_score` : s2 fiziksel çapası (α=0.4). LightGBM'e init_score olarak ver.
- `anc_base/anc_dev/anc_zero` : çapa bileşenleri — kendi α'nı denemek için
  `init_score = anc_base + α*anc_dev + anc_zero`
- `is_cold` : trafo train'de görülmemiş mi (cold-start)

Kullanım: `from src.dataset import load_dataset; df = load_dataset('f1_train')`

Detaylı kullanım + UYARILAR: `docs/DATASET.md`
