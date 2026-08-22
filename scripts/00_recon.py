# -*- coding: utf-8 -*-
"""
00_recon.py — Tek seferlik veri keşif scripti.

CSV'ler chunk'lanarak, dtype-optimize edilerek okunur (bellek dostu).
Çıktı: reports/recon.md

Kullanım:  python scripts/00_recon.py
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "recon.md"

CHUNK = 500_000

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")


def fmt(x, nd=2):
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


# ---------------------------------------------------------------- train okuma
# Chunk'la oku, dtype küçült, kompakt frame'de topla.
train_dtypes = {
    "tanim": "category",  # saf sayı değil: '202917T' gibi değerler var
    "guc": "float32",
    "tuketim": "float32",
    "lokasyon": "category",
}
chunks = []
raw_mem = 0.0  # optimizasyonsuz (int64/float64/object) tahmini bellek
for ch in pd.read_csv(RAW / "train.csv", dtype=train_dtypes, chunksize=CHUNK):
    # optimizasyonsuz bellek: aynı chunk'ı default dtype'a çevirip ölç
    raw_mem += (
        ch["tanim"].astype("object").memory_usage(deep=True, index=False)
        + ch["guc"].astype("float64").memory_usage(deep=True, index=False)
        + ch["tarih"].memory_usage(deep=True, index=False)  # object string
        + ch["tuketim"].astype("float64").memory_usage(deep=True, index=False)
        + ch["lokasyon"].astype("object").memory_usage(deep=True, index=False)
    )
    ch["tarih"] = pd.to_datetime(ch["tarih"], format="%Y-%m-%d")
    chunks.append(ch)
tr = pd.concat(chunks, ignore_index=True)
del chunks
# concat sonrası kategorileri tekrar birleştir (chunk'lar farklı kategori setleriyle gelir)
tr["lokasyon"] = tr["lokasyon"].astype("category")
tr["tanim"] = tr["tanim"].astype("category")
opt_mem = tr.memory_usage(deep=True, index=False).sum()

# ---------------------------------------------------------------- test okuma
test_dtypes = {
    "id": "string",
    "tanim": "category",
    "guc": "float32",
    "lokasyon": "category",
}
te_chunks = []
for ch in pd.read_csv(RAW / "test.csv", dtype=test_dtypes, chunksize=CHUNK):
    ch["tarih"] = pd.to_datetime(ch["tarih"], format="%Y-%m-%d")
    te_chunks.append(ch)
te = pd.concat(te_chunks, ignore_index=True)
del te_chunks
te["lokasyon"] = te["lokasyon"].astype("category")
te["tanim"] = te["tanim"].astype("category")

ss_ids = pd.read_csv(RAW / "sample_submission.csv", usecols=["id"], dtype={"id": "string"})["id"]

# ================================================================ YAPI
w("# Veri Keşif Raporu (recon)")
w()
w(f"Üretim: `scripts/00_recon.py` · Tarih: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
w()
w("## YAPI")
w()

# 1. satır / tekil tanim
tr_tanim = set(tr["tanim"].unique())
te_tanim = set(te["tanim"].unique())
w("### 1. Satır ve tekil trafo sayıları")
w()
w(f"| | satır | tekil tanim |")
w(f"|---|---|---|")
w(f"| train | {fmt(len(tr))} | {fmt(len(tr_tanim))} |")
w(f"| test  | {fmt(len(te))} | {fmt(len(te_tanim))} |")
w()

# 2. kesişim
only_te = te_tanim - tr_tanim
only_tr = tr_tanim - te_tanim
inter = tr_tanim & te_tanim
w("### 2. KRİTİK — tanim kümeleri kesişimi")
w()
w(f"- Kesişim: **{fmt(len(inter))}** trafo")
w(f"- Test'te olup train'de OLMAYAN: **{fmt(len(only_te))}** "
  f"(test trafolarının %{100*len(only_te)/max(len(te_tanim),1):.2f}'i)")
w(f"- Train'de olup test'te olmayan: **{fmt(len(only_tr))}** "
  f"(train trafolarının %{100*len(only_tr)/max(len(tr_tanim),1):.2f}'i)")
if len(only_te) / max(len(te_tanim), 1) > 0.01:
    w()
    w("> **UYARI:** Test'te train'de hiç görülmemiş trafo oranı %1'in üzerinde →"
      " **cold-start stratejisi gerekecek** (lokasyon × güç grubu medyanına düşüş vb.).")
w()

# 3. tarih aralıkları
w("### 3. Tarih aralıkları")
w()
for name, df in [("train", tr), ("test", te)]:
    dmin, dmax = df["tarih"].min(), df["tarih"].max()
    ndays = df["tarih"].nunique()
    span = (dmax - dmin).days + 1
    w(f"- **{name}**: {dmin:%Y-%m-%d} → {dmax:%Y-%m-%d} · takvim {span} gün · "
      f"veride {ndays} tekil gün" + (" · **takvimde eksik gün var**" if ndays != span else ""))
w()

# 4. trafo başına gün dağılımı
w("### 4. Trafo başına gün sayısı (train)")
w()
g = tr.groupby("tanim")["tarih"].agg(["count", "min", "max", "nunique"])
g["span"] = (g["max"] - g["min"]).dt.days + 1
g["gap"] = g["span"] - g["nunique"]
dup_rows = int((g["count"] - g["nunique"]).sum())
desc = g["nunique"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
w("Tekil gün sayısı dağılımı:")
w()
w("| istatistik | değer |")
w("|---|---|")
for k in ["min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"]:
    w(f"| {k} | {desc[k]:.0f} |")
w()
full_days = tr["tarih"].nunique()
n_full = int((g["nunique"] == full_days).sum())
w(f"- Tam panel ({full_days} gün) olan trafo: {fmt(n_full)} / {fmt(len(g))} "
  f"(%{100*n_full/len(g):.1f})")
w(f"- Kendi aralığında boşluğu (gap) olan trafo: {fmt(int((g['gap']>0).sum()))} "
  f"(toplam eksik gün: {fmt(int(g['gap'].sum()))})")
w(f"- Aynı trafo+gün mükerrer satır: {fmt(dup_rows)}")
late_start = int((g["min"] > tr["tarih"].min()).sum())
early_end = int((g["max"] < tr["tarih"].max()).sum())
w(f"- Sonradan başlayan trafo (ilk gün > {tr['tarih'].min():%Y-%m-%d}): {fmt(late_start)}")
w(f"- Erken susan trafo (son gün < {tr['tarih'].max():%Y-%m-%d}): {fmt(early_end)}")
w()

# test paneli dengeli mi
gt = te.groupby("tanim")["tarih"].nunique()
w(f"- Test panel: trafo başına gün min={gt.min()}, max={gt.max()} "
  + ("(dengeli)" if gt.min() == gt.max() else "(**dengesiz**)"))
w()

# 5. bellek
w("### 5. Bellek (train)")
w()
w(f"- Optimizasyonsuz (int64/float64/object string): ~{raw_mem/1e6:,.0f} MB")
w(f"- Optimize (category tanim/lokasyon, float32, datetime64): "
  f"**{opt_mem/1e6:,.0f} MB**")
w()

# ================================================================ HEDEF
w("## HEDEF")
w()
t = tr["tuketim"]
w("### 6. tuketim istatistikleri")
w()
w("| istatistik | değer |")
w("|---|---|")
w(f"| min | {t.min():,.2f} |")
w(f"| %25 | {t.quantile(.25):,.2f} |")
w(f"| medyan | {t.median():,.2f} |")
w(f"| %75 | {t.quantile(.75):,.2f} |")
w(f"| max | {t.max():,.2f} |")
w(f"| ortalama | {t.mean():,.2f} |")
w(f"| NaN | {fmt(int(t.isna().sum()))} |")
w(f"| negatif | {fmt(int((t < 0).sum()))} |")
w(f"| sıfır | {fmt(int((t == 0).sum()))} (%{100*(t==0).mean():.2f}) |")
w()

# 7. log1p histogram
w("### 7. log1p(tuketim) histogramı")
w()
lt = np.log1p(t.clip(lower=0).dropna())
hist, edges = np.histogram(lt, bins=15)
maxbar = hist.max()
w("```")
for i, h in enumerate(hist):
    bar = "#" * max(1, round(40 * h / maxbar)) if h else ""
    w(f"[{edges[i]:5.2f}, {edges[i+1]:5.2f})  {h:>9,}  {bar}")
w("```")
w()

# 8. yük faktörü
w("### 8. Yük faktörü = tuketim / (guc*24)")
w()
lf = tr["tuketim"] / (tr["guc"] * 24.0)
valid_lf = lf.replace([np.inf, -np.inf], np.nan).dropna()
w(f"- Hesaplanabilen satır: {fmt(len(valid_lf))} (guc=0 veya NaN nedeniyle düşen: "
  f"{fmt(len(tr)-len(valid_lf))})")
w(f"- Medyan: {valid_lf.median():.4f} · %95: {valid_lf.quantile(.95):.4f} · "
  f"max: {valid_lf.max():.2f}")
n_gt1 = int((valid_lf > 1).sum())
w(f"- **1'i aşan satır: {fmt(n_gt1)} (%{100*n_gt1/len(valid_lf):.3f})** — veri hatası sinyali")
n_gt1_trafo = tr.loc[valid_lf.index[valid_lf > 1], "tanim"].nunique() if n_gt1 else 0
w(f"- 1'i aşan satırların dokunduğu trafo: {fmt(n_gt1_trafo)}")
w()

# 9. ardışık sıfır blokları
w("### 9. Ardışık sıfır blokları (30+ gün)")
w()
z = tr[["tanim", "tarih", "tuketim"]].sort_values(["tanim", "tarih"])
z["is_zero"] = (z["tuketim"] == 0).astype("int8")
# blok id: trafo değişince veya zero->nonzero geçişte artar
blk = ((z["tanim"] != z["tanim"].shift()) | (z["is_zero"] != z["is_zero"].shift())).cumsum()
runs = z[z["is_zero"] == 1].groupby(blk).agg(tanim=("tanim", "first"), n=("is_zero", "size"),
                                            son=("tarih", "max"))
long_runs = runs[runs["n"] >= 30]
w(f"- 30+ gün ardışık sıfır bloğu sayısı: {fmt(len(long_runs))}")
w(f"- Bu bloklara sahip tekil trafo: **{fmt(long_runs['tanim'].nunique())}**")
if len(long_runs):
    w(f"- En uzun blok: {fmt(int(runs['n'].max()))} gün")
    # train sonunda hâlâ sıfırda olan (kapanmış aday)
    tr_end = tr["tarih"].max()
    dead = long_runs[long_runs["son"] == tr_end]
    w(f"- Train'in son gününde hâlâ sıfır bloğunda olan trafo (kapanmış aday): "
      f"**{fmt(dead['tanim'].nunique())}**")
w()

# ================================================================ KOLONLAR
w("## KOLONLAR")
w()

# 10. guc
w("### 10. guc")
w()
gu = tr.groupby("tanim")["guc"].first()  # trafo bazında
w(f"- Tekil guc değeri (train, satır bazında): {tr['guc'].nunique()}")
w(f"- Trafo bazında guc: min={gu.min():,.0f} · medyan={gu.median():,.0f} · "
  f"max={gu.max():,.0f}")
w(f"- guc=0 satır: {fmt(int((tr['guc']==0).sum()))} · guc NaN satır: "
  f"{fmt(int(tr['guc'].isna().sum()))}")
vc = gu.value_counts().head(10)
w()
w("En yaygın 10 guc değeri (trafo sayısı):")
w()
w("| guc (kVA) | trafo |")
w("|---|---|")
for k, v in vc.items():
    w(f"| {k:,.0f} | {fmt(int(v))} |")
# guc trafo içinde sabit mi
gvar = tr.groupby("tanim")["guc"].nunique()
w()
w(f"- guc'u zaman içinde değişen trafo: {fmt(int((gvar>1).sum()))}")
w()

# 11. lokasyon
w("### 11. lokasyon")
w()
lok = tr["lokasyon"].astype("string")
pat = re.compile(r"^[^>]+>[^>]+>[^>]+$")
uniq_lok = lok.dropna().unique()
match_map = {u: bool(pat.match(u)) for u in uniq_lok}
n_match = int(lok.map(match_map).fillna(False).sum())
gediz_mask = lok.str.contains("GEDİZ EDAŞ", na=False)
w(f"- Tekil lokasyon (train): {len(uniq_lok)}")
w(f"- `İL>BÖLGE>İLÇE` (2 adet `>`) formatına uyan satır: %{100*n_match/len(tr):.2f}")
w(f"- Jenerik `GEDİZ EDAŞ` içeren satır: {fmt(int(gediz_mask.sum()))} "
  f"(%{100*gediz_mask.mean():.2f}) · trafo: {fmt(tr.loc[gediz_mask,'tanim'].nunique())}")
parts = pd.Series([u for u in uniq_lok if match_map[u]]).str.split(">", expand=True)
if len(parts):
    w(f"- Ayrıştırma (formata uyanlar): il={parts[0].str.strip().nunique()} · "
      f"bölge={parts[1].str.strip().nunique()} · ilçe={parts[2].str.strip().nunique()}")
    w(f"- İller: {sorted(parts[0].str.strip().unique())}")
w(f"- NaN lokasyon satırı: {fmt(int(lok.isna().sum()))}")
# formata uymayan örnekler
nonmatch = [u for u in uniq_lok if not match_map[u]][:8]
if nonmatch:
    w(f"- Formata uymayan tekil değer örnekleri: {nonmatch}")
# lokasyon trafo içinde sabit mi
lvar = tr.groupby("tanim")["lokasyon"].nunique()
w(f"- Lokasyonu zaman içinde değişen trafo: {fmt(int((lvar>1).sum()))}")
w()

# 12. id formatı
w("### 12. test id ↔ sample_submission")
w()
expected = te["tanim"].astype("string") + "_" + te["tarih"].dt.strftime("%Y-%m-%d")
id_ok = bool((te["id"] == expected).all())
w(f"- test.csv id formatı `tanim_YYYY-MM-DD` mi: **{'EVET' if id_ok else 'HAYIR'}**")
w(f"- sample_submission satır sayısı: {fmt(len(ss_ids))} · test satır: {fmt(len(te))}")
same_order = len(ss_ids) == len(te) and bool((ss_ids.values == te["id"].values).all())
same_set = set(ss_ids) == set(te["id"])
w(f"- Küme olarak birebir eşleşme: **{'EVET' if same_set else 'HAYIR'}**")
w(f"- Sıra da aynı mı: **{'EVET' if same_order else 'HAYIR'}**")
w(f"- test id mükerrer: {fmt(int(te['id'].duplicated().sum()))}")
w()

# ================================================================ ZAMAN
w("## ZAMAN")
w()
tr["ay"] = tr["tarih"].dt.to_period("M")

# 13. aylık toplam
w("### 13. Aylık toplam tüketim (train)")
w()
mon = tr.groupby("ay")["tuketim"].agg(["sum", "mean", "count"])
w("| ay | toplam | ortalama/satır | satır |")
w("|---|---|---|---|")
for ay, r in mon.iterrows():
    w(f"| {ay} | {r['sum']:,.0f} | {r['mean']:,.1f} | {int(r['count']):,} |")
w()

# 14. haftanın günü
w("### 14. Haftanın gününe göre ortalama tüketim")
w()
gunler = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
dw = tr.groupby(tr["tarih"].dt.dayofweek)["tuketim"].mean()
w("| gün | ortalama |")
w("|---|---|")
for i, v in dw.items():
    w(f"| {gunler[i]} | {v:,.1f} |")
w()

# 15. 2025 Nisan–Temmuz
w("### 15. 2025 Nisan–Temmuz aylık toplamları (test dönemi kıyası)")
w()
m45 = mon.loc[[pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]]
w("| ay | toplam | ortalama/satır |")
w("|---|---|---|")
for ay, r in m45.iterrows():
    w(f"| {ay} | {r['sum']:,.0f} | {r['mean']:,.1f} |")
w()

# ================================================================ ANOMALİLER
w("## DİKKAT ÇEKEN ANOMALİLER")
w()
anom = []
if len(only_te):
    anom.append(f"Test'te train'de görülmemiş {fmt(len(only_te))} trafo "
                f"(%{100*len(only_te)/len(te_tanim):.2f}) → cold-start gereksinimi.")
if int((t < 0).sum()):
    anom.append(f"Negatif tuketim: {fmt(int((t<0).sum()))} satır (min {t.min():,.2f}).")
if int(t.isna().sum()):
    anom.append(f"NaN tuketim: {fmt(int(t.isna().sum()))} satır.")
if n_gt1:
    anom.append(f"Yük faktörü > 1 olan {fmt(n_gt1)} satır "
                f"(%{100*n_gt1/len(valid_lf):.3f}, {fmt(n_gt1_trafo)} trafo); "
                f"max yük faktörü {valid_lf.max():.1f}.")
if int((tr['guc']==0).sum()) or int(tr['guc'].isna().sum()):
    anom.append(f"guc=0 satır: {fmt(int((tr['guc']==0).sum()))}, "
                f"guc NaN: {fmt(int(tr['guc'].isna().sum()))}.")
if len(long_runs):
    anom.append(f"30+ gün ardışık sıfır bloğu olan {fmt(long_runs['tanim'].nunique())} trafo; "
                f"{fmt(dead['tanim'].nunique())} tanesi train sonunda hâlâ sıfırda (kapanmış aday).")
if dup_rows:
    anom.append(f"Aynı trafo+gün için mükerrer {fmt(dup_rows)} satır.")
if int((g['gap']>0).sum()):
    anom.append(f"Kendi tarih aralığında boşluğu olan {fmt(int((g['gap']>0).sum()))} trafo "
                f"(toplam {fmt(int(g['gap'].sum()))} eksik gün) → panel dengesiz.")
if int((gvar>1).sum()):
    anom.append(f"guc değeri zaman içinde değişen {fmt(int((gvar>1).sum()))} trafo.")
if int((lvar>1).sum()):
    anom.append(f"lokasyon değeri zaman içinde değişen {fmt(int((lvar>1).sum()))} trafo.")
if not same_order:
    anom.append("sample_submission id sırası test.csv ile birebir aynı DEĞİL.")
if gt.min() != gt.max():
    anom.append(f"Test paneli dengesiz: trafo başına gün {gt.min()}–{gt.max()} arası.")
if not anom:
    anom.append("Beklenmedik yapısal anomali gözlenmedi.")
for a in anom:
    w(f"- {a}")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(out.getvalue(), encoding="utf-8")
print(f"Rapor yazıldı: {REPORT}")
