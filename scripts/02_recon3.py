# -*- coding: utf-8 -*-
"""
02_recon3.py — Üçüncü keşif turu (STRATEGY_v3 madde 9.1).

Sorular:
  1. Toplu giriş teyidi (cold ve train, günlük histogram + tarih örüntüsü)
  2. Ramp-up testi — yeni giren trafo rampalanıyor mu? (RAMP VAR / RAMP YOK)
  3. Test trafolarının geçmiş uzunluğu (H) dağılımı → validation.py girdisi
  4. Sıfır bloğu devam oranı q ve dönüş seviyesi L → x* = q·L
  5. İlçe Temmuz/Mayıs oranının robustluğu (ortalama / medyan / geometrik)

Çıktı: reports/recon3.md + data/processed/test_history_profile.csv
Kullanım: python scripts/02_recon3.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "recon3.md"
PROFILE_CSV = ROOT / "data" / "processed" / "test_history_profile.csv"

BULK_THRESHOLD = 30  # bir günde bu kadar+ trafo giriyorsa "toplu giriş" günü

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")


def fmt(x):
    return f"{x:,}"


def pct(a, b):
    return f"%{100 * a / max(b, 1):.2f}"


GUNLER = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]


def guc_bucket(g: pd.Series) -> pd.Series:
    return pd.cut(g, bins=[0, 160, 400, 1000, 1600, np.inf],
                  labels=["<=160", "250-400", "630-1000", "1250-1600", ">1600"])


# ---------------------------------------------------------------- yükleme
tr = pd.read_csv(
    RAW / "train.csv",
    dtype={"tanim": "string", "guc": "float32", "tuketim": "float32",
           "lokasyon": "string"},
    parse_dates=["tarih"],
)
te = pd.read_csv(
    RAW / "test.csv",
    dtype={"id": "string", "tanim": "string", "guc": "float32",
           "lokasyon": "string"},
    parse_dates=["tarih"],
)
for df in (tr, te):
    parts = df["lokasyon"].str.split(">")
    df["il"] = parts.str[0].str.strip()
    df["ilce"] = df["il"] + ">" + parts.str[-1].str.strip()

tr_tanim = set(tr["tanim"].unique())
cold_set = set(te["tanim"].unique()) - tr_tanim

w("# Recon-3 Raporu")
w()
w(f"Üretim: `scripts/02_recon3.py` · Tarih: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
w()

# ================================================================ 1. TOPLU GİRİŞ
w("## 1. Toplu giriş teyidi")
w()

# --- 1a cold günlük histogram
te_first = te.groupby("tanim", observed=True)["tarih"].min()
cold_first = te_first[te_first.index.isin(cold_set)]
cold_daily = cold_first.value_counts().sort_index()
w("### 1a. Cold trafoların test'e giriş günü — en yoğun 10 tarih")
w()
w("| tarih | gün | ayın günü | trafo | cold'ların payı |")
w("|---|---|---|---|---|")
for t, n in cold_daily.sort_values(ascending=False).head(10).items():
    w(f"| {t:%Y-%m-%d} | {GUNLER[t.dayofweek]} | {t.day} | {fmt(int(n))} | "
      f"{pct(int(n), len(cold_first))} |")
w()
top1_share = cold_daily.max() / len(cold_first)
w(f"- Tek tepe: {cold_daily.idxmax():%Y-%m-%d} günü {fmt(int(cold_daily.max()))} trafo "
  f"({pct(int(cold_daily.max()), len(cold_first))}) — "
  + ("**belirgin toplu giriş**" if top1_share > 0.2 else "belirgin tepe yok"))
w(f"- Giriş görülen tekil gün sayısı: {cold_daily.size} / 122")
w()

# --- 1b train günlük histogram
tr_first = tr.groupby("tanim", observed=True)["tarih"].min()
late_first = tr_first[tr_first > tr["tarih"].min()]
late_daily = late_first.value_counts().sort_index()
w("### 1b. Train'de sonradan başlayan trafolar — en yoğun 10 tarih")
w()
w("| tarih | gün | ayın günü | trafo | payı |")
w("|---|---|---|---|---|")
top_train = late_daily.sort_values(ascending=False).head(10)
for t, n in top_train.items():
    w(f"| {t:%Y-%m-%d} | {GUNLER[t.dayofweek]} | {t.day} | {fmt(int(n))} | "
      f"{pct(int(n), len(late_first))} |")
w()
bulk_dates = set(late_daily[late_daily >= BULK_THRESHOLD].index)
n_bulk_trafo = int(late_daily[late_daily >= BULK_THRESHOLD].sum())
w(f"- Toplu giriş günü tanımı: ≥{BULK_THRESHOLD} trafo/gün → {len(bulk_dates)} gün, "
  f"{fmt(n_bulk_trafo)} trafo ({pct(n_bulk_trafo, len(late_first))})")
w()

# --- 1c tarih örüntüsü
w("### 1c. Tarih örüntüsü — haftanın günü / ayın günü")
w()
bulk_days = pd.Series(sorted(bulk_dates))
if len(bulk_days):
    dow_dist = bulk_days.dt.dayofweek.value_counts().sort_index()
    dom_first = int((bulk_days.dt.day == 1).sum())
    w(f"- Train toplu giriş günlerinin haftanın günü dağılımı: "
      + " · ".join(f"{GUNLER[i]}={int(v)}" for i, v in dow_dist.items()))
    w(f"- Ayın 1'ine denk gelen toplu gün: {dom_first} / {len(bulk_days)}")
    w(f"- Ayın günü dağılımı (toplu günler): "
      + " · ".join(str(int(d)) for d in sorted(bulk_days.dt.day.unique())))
cold_peak = cold_daily.idxmax()
w(f"- Test cold tepesi: {cold_peak:%Y-%m-%d} ({GUNLER[cold_peak.dayofweek]}, ayın {cold_peak.day}'i)")
# ağırlıklı örüntü: toplu giren trafoların dow/dom dağılımı
w()
w(f"> **Sonuç (1):** Cold girişlerinin {pct(int(cold_daily.max()), len(cold_first))}'i "
  f"tek günde ({cold_daily.idxmax():%Y-%m-%d}) — train'de de girişlerin "
  f"{pct(n_bulk_trafo, len(late_first))}'i {len(bulk_dates)} toplu güne yığılmış; "
  f"bu tek tek saha kurulumu değil, dönemsel toplu sisteme alım imzasıdır.")
w()

# ================================================================ 2. RAMP-UP
w("## 2. Ramp-up testi")
w()
late_set = set(late_first.index)
sub = tr[tr["tanim"].isin(late_set)].copy()
sub["ilk_gun"] = sub["tanim"].map(late_first)
sub["dse"] = (sub["tarih"] - sub["ilk_gun"]).dt.days  # days_since_entry
sub["log1p"] = np.log1p(sub["tuketim"])

# en az 90 günlük geçmiş
max_dse = sub.groupby("tanim", observed=True)["dse"].max()
eligible = set(max_dse[max_dse >= 90].index)
sub = sub[sub["tanim"].isin(eligible)]

# taban: 60-90 gün penceresi ortalama log1p
base = (sub[(sub["dse"] >= 60) & (sub["dse"] <= 90)]
        .groupby("tanim", observed=True)["log1p"].mean())
base = base[base > 0.1]  # taban sıfırsa oran anlamsız
sub = sub[sub["tanim"].isin(set(base.index))]
sub["norm"] = sub["log1p"] / sub["tanim"].map(base)
sub["is_bulk"] = sub["ilk_gun"].isin(bulk_dates)

n_bulk_coh = sub.loc[sub["is_bulk"], "tanim"].nunique()
n_drip_coh = sub.loc[~sub["is_bulk"], "tanim"].nunique()
w(f"- Uygun trafo (≥90 gün geçmiş, taban>0): toplu-giriş kohortu {fmt(n_bulk_coh)} · "
  f"tek-tük kohortu {fmt(n_drip_coh)}")
w()

sub90 = sub[(sub["dse"] >= 0) & (sub["dse"] <= 90)]
bins = list(range(0, 7))
labels_daily = [str(i) for i in range(7)]
week_edges = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]


def ramp_table(df):
    med = {}
    for i in range(7):  # ilk hafta gün gün
        v = df.loc[df["dse"] == i, "norm"]
        med[str(i)] = (v.median(), len(v))
    lo = 7
    for hi in week_edges[1:]:
        v = df.loc[(df["dse"] >= lo) & (df["dse"] < hi), "norm"]
        med[f"{lo}-{hi-1}"] = (v.median(), len(v))
        lo = hi
    return med


tab_bulk = ramp_table(sub90[sub90["is_bulk"]])
tab_drip = ramp_table(sub90[~sub90["is_bulk"]])
w("Medyan norm = log1p(tuketim) / trafonun 60–90. gün ortalama log1p'i")
w()
w("| days_since_entry | toplu giriş kohortu | n | tek tük kohortu | n |")
w("|---|---|---|---|---|")
for k in tab_bulk:
    mb, nb = tab_bulk[k]
    md, nd = tab_drip[k]
    w(f"| {k} | {mb:.3f} | {nb:,} | {md:.3f} | {nd:,} |")
w()

first_week_bulk = sub90[(sub90["is_bulk"]) & (sub90["dse"] <= 6)]["norm"].median()
first_week_drip = sub90[(~sub90["is_bulk"]) & (sub90["dse"] <= 6)]["norm"].median()
ramp_var = (first_week_bulk < 0.9) or (first_week_drip < 0.9)
w(f"- İlk hafta (0–6 gün) medyan norm: toplu={first_week_bulk:.3f} · "
  f"tek-tük={first_week_drip:.3f} (1.0 = olgun seviye)")
w()
if ramp_var:
    w(f"> **Sonuç (2): RAMP VAR** — yeni giren trafo ilk haftada olgun seviyenin "
      f"belirgin altında başlıyor (toplu {first_week_bulk:.2f}, tek-tük "
      f"{first_week_drip:.2f}); `days_since_entry` bilgisi cold tahmininde zorunlu.")
else:
    w(f"> **Sonuç (2): RAMP YOK** — yeni giren trafo ilk günden olgun seviyede "
      f"(ilk hafta medyan norm toplu {first_week_bulk:.2f}, tek-tük "
      f"{first_week_drip:.2f}); `guc × LF` doğrudan çalışır, ramp feature'ı gereksiz.")
w()

# ================================================================ 3. GEÇMİŞ UZUNLUĞU
w("## 3. Test trafolarının geçmiş uzunluğu (H) dağılımı")
w()
tr_days = tr.groupby("tanim", observed=True)["tarih"].nunique()
te_trafo = te.groupby("tanim", observed=True).agg(
    guc=("guc", "first"), ilce=("ilce", "first"), il=("il", "first"),
    test_n_days=("tarih", "nunique"), test_entry=("tarih", "min"),
)
te_trafo["H"] = te_trafo.index.map(tr_days).fillna(0).astype(int)
te_trafo["guc_bucket"] = guc_bucket(te_trafo["guc"])

h_bins = [-1, 0, 30, 90, 180, 300, 455]
h_labels = ["0 (cold)", "1-30", "31-90", "91-180", "181-300", "301-455"]
te_trafo["H_bin"] = pd.cut(te_trafo["H"], bins=h_bins, labels=h_labels)

w("### 3a. H histogramı")
w()
hb = te_trafo["H_bin"].value_counts().reindex(h_labels)
w("| H aralığı | trafo | pay |")
w("|---|---|---|")
for k, v in hb.items():
    w(f"| {k} | {fmt(int(v))} | {pct(int(v), len(te_trafo))} |")
w(f"| **toplam** | {fmt(len(te_trafo))} | · |")
w()
w(f"- H medyanı: {te_trafo['H'].median():.0f} gün · warm'larda medyan: "
  f"{te_trafo.loc[te_trafo['H']>0, 'H'].median():.0f} gün")
w()

w("### 3b. H dağılımı × guc_bucket (satırlar guc_bucket, pay %)")
w()
ct = pd.crosstab(te_trafo["guc_bucket"], te_trafo["H_bin"], normalize="index") * 100
cnt = te_trafo["guc_bucket"].value_counts()
w("| guc_bucket | trafo | " + " | ".join(h_labels) + " |")
w("|---|---|" + "---|" * len(h_labels))
for b in ct.index:
    row = " | ".join(f"%{ct.loc[b, k]:.1f}" for k in h_labels)
    w(f"| {b} | {fmt(int(cnt[b]))} | {row} |")
w()

w("### 3c. Test'e giriş tarihi (tüm test trafoları) — en yoğun 10 gün")
w()
te_entry_daily = te_trafo["test_entry"].value_counts().sort_index()
w("| tarih | gün | trafo | pay |")
w("|---|---|---|---|")
for t, n in te_entry_daily.sort_values(ascending=False).head(10).items():
    w(f"| {t:%Y-%m-%d} | {GUNLER[t.dayofweek]} | {fmt(int(n))} | "
      f"{pct(int(n), len(te_trafo))} |")
w()
w(f"- Giriş görülen tekil gün: {te_entry_daily.size} / 122 · "
  f"ilk gün ({te['tarih'].min():%Y-%m-%d}) girenler: "
  f"{fmt(int(te_entry_daily.iloc[0]))} ({pct(int(te_entry_daily.iloc[0]), len(te_trafo))})")
w()

PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
te_trafo.reset_index()[["tanim", "guc", "guc_bucket", "il", "ilce", "H",
                        "test_entry", "test_n_days"]].to_csv(
    PROFILE_CSV, index=False, encoding="utf-8")
w(f"- Profil CSV yazıldı: `data/processed/test_history_profile.csv` "
  f"({fmt(len(te_trafo))} satır)")
w()
w(f"> **Sonuç (3):** Test trafolarının {pct(int(hb.iloc[0]), len(te_trafo))}'i cold, "
  f"warm'ların H medyanı {te_trafo.loc[te_trafo['H']>0, 'H'].median():.0f} gün ve "
  f"dağılım guc_bucket'a göre kayda değer değişiyor — `make_folds` H örneklemesini "
  f"bu CSV'deki bucket-bazlı dağılımdan yapmalı.")
w()

# ================================================================ 4. SIFIR BLOĞU
w("## 4. Sıfır bloğu devam oranı")
w()
z = tr[["tanim", "tarih", "tuketim"]].sort_values(["tanim", "tarih"]).reset_index(drop=True)
z["is_zero"] = (z["tuketim"] == 0).astype("int8")
blk = ((z["tanim"] != z["tanim"].shift()) |
       (z["is_zero"] != z["is_zero"].shift())).cumsum()
z["blk"] = blk
runs = (z[z["is_zero"] == 1].groupby("blk")
        .agg(tanim=("tanim", "first"), n=("is_zero", "size"),
             bas=("tarih", "min"), son=("tarih", "max")))
runs = runs[runs["n"] >= 30]
trafo_last = z.groupby("tanim", observed=True)["tarih"].max()
runs["trafo_son"] = runs["tanim"].map(trafo_last)
runs["bitti"] = runs["son"] < runs["trafo_son"]  # sonrasında kayıt var → o kayıt sıfır-dışı

n_bitti = int(runs["bitti"].sum())
n_censored = len(runs) - n_bitti
q = n_bitti / len(runs)
w("### 4a-b. Blok sayıları ve devam oranı")
w()
w(f"- 30+ gün sıfır bloğu (toplam): {fmt(len(runs))}")
w(f"- Biten (tüketim yeniden başladı): {fmt(n_bitti)}")
w(f"- Veri sonuna kadar süren (sansürlü): {fmt(n_censored)}")
w(f"- **q = biten / toplam = {q:.3f}**")
w()

# c: bittikten sonraki ilk 30 günün ortalama log1p'i
w("### 4c. Dönüş seviyesi L")
w()
ended = runs[runs["bitti"]]
post_means = []
for _, r in ended.iterrows():
    m = ((z["tanim"] == r["tanim"]) & (z["tarih"] > r["son"]) &
         (z["tarih"] <= r["son"] + pd.Timedelta(days=30)))
    vals = np.log1p(z.loc[m, "tuketim"])
    if len(vals):
        post_means.append(vals.mean())
post_means = pd.Series(post_means)
L = post_means.mean()
w(f"- Biten blok sonrası ilk 30 günün trafo-bazlı ortalama log1p'i: "
  f"**L ortalama = {L:.3f}** · medyan = {post_means.median():.3f} · "
  f"%25–%75 = {post_means.quantile(.25):.2f}–{post_means.quantile(.75):.2f}")
w(f"- x* = q·L = {q:.3f} × {L:.3f} = **{q*L:.3f}** (log1p ölçeği) → "
  f"tahmin ≈ **{np.expm1(q*L):,.1f} kWh**")
w()

# d: blok uzunluğuna göre q
w("### 4d. Blok uzunluğuna göre q")
w()
runs["uz_bin"] = pd.cut(runs["n"], bins=[29, 60, 120, 240, np.inf],
                        labels=["30-60", "61-120", "121-240", "240+"])
qt = runs.groupby("uz_bin", observed=True)["bitti"].agg(["sum", "count"])
qt["q"] = qt["sum"] / qt["count"]
w("| blok uzunluğu | toplam | biten | q |")
w("|---|---|---|---|")
for b, r in qt.iterrows():
    w(f"| {b} | {int(r['count']):,} | {int(r['sum']):,} | {r['q']:.3f} |")
w()
w(f"> **Sonuç (4):** 30+ günlük sıfır bloklarının {pct(n_bitti, len(runs))}'i "
  f"yeniden tüketime dönüyor (q={q:.2f}) ve dönüş seviyesi L≈{L:.1f} log1p; "
  f"blok uzadıkça q {'düşüyor' if qt['q'].iloc[-1] < qt['q'].iloc[0] else 'düşmüyor'} "
  f"— kapanmış-aday trafo tahmini x*=q·L≈{q*L:.2f} (≈{np.expm1(q*L):,.0f} kWh) "
  f"civarında olmalı, sert 0 override yanlış.")
w()

# ================================================================ 5. İLÇE ORANI
w("## 5. İlçe Temmuz/Mayıs oranı — üç yöntem")
w()
gun_say = tr.groupby("tanim", observed=True)["tarih"].nunique()
full_set = set(gun_say[gun_say == tr["tarih"].nunique()].index)
coh = tr[tr["tanim"].isin(full_set)].copy()
coh["ay_p"] = coh["tarih"].dt.to_period("M")
coh["log1p"] = np.log1p(coh["tuketim"])
sub5 = coh[coh["ay_p"].isin([pd.Period("2025-05"), pd.Period("2025-07")])]

agg = sub5.groupby(["ilce", "ay_p"], observed=True).agg(
    mean_t=("tuketim", "mean"), med_t=("tuketim", "median"),
    mean_l=("log1p", "mean"), n_trafo=("tanim", "nunique")).reset_index()
piv = agg.pivot(index="ilce", columns="ay_p")
may, jul = pd.Period("2025-05"), pd.Period("2025-07")
tab = pd.DataFrame({
    "trafo": piv[("n_trafo", may)],
    "aritmetik": piv[("mean_t", jul)] / piv[("mean_t", may)],
    "medyan": piv[("med_t", jul)] / piv[("med_t", may)],
    "geometrik": np.expm1(piv[("mean_l", jul)]) / np.expm1(piv[("mean_l", may)]),
}).sort_values("aritmetik", ascending=False)

w("| ilçe | trafo | aritmetik | medyan | geometrik |")
w("|---|---|---|---|---|")
for ilce, r in tab.iterrows():
    az = " ⚠️az-örnek" if r["trafo"] < 10 else ""
    w(f"| {ilce}{az} | {int(r['trafo'])} | {r['aritmetik']:.2f}× | "
      f"{r['medyan']:.2f}× | {r['geometrik']:.2f}× |")
w()
konak = tab.loc["İZMİR>KONAK"]
w(f"- **Konak kontrolü:** aritmetik {konak['aritmetik']:.2f}× · medyan "
  f"{konak['medyan']:.2f}× · geometrik {konak['geometrik']:.2f}×")
w()
corr = tab[["aritmetik", "medyan", "geometrik"]].corr(method="spearman")
w(f"- Yöntemler arası Spearman sıra korelasyonu: aritmetik–medyan "
  f"{corr.loc['aritmetik','medyan']:.2f} · aritmetik–geometrik "
  f"{corr.loc['aritmetik','geometrik']:.2f} · medyan–geometrik "
  f"{corr.loc['medyan','geometrik']:.2f}")
w()
konak_robust = konak[["medyan", "geometrik"]].min() > 3
w(f"> **Sonuç (5):** Konak'ın 5.0× aritmetik oranı medyanda {konak['medyan']:.2f}×, "
  f"geometrikte {konak['geometrik']:.2f}× — "
  + ("üç yöntemde de ayakta, ilçe etkisi gerçek" if konak_robust else
     "robust yöntemlerde eriyor, aritmetik oran birkaç büyük trafonun eseri; "
     "grp_ mevsimsel indeks robust (medyan/geometrik) istatistikle kurulmalı")
  + f"; az-örnekli ilçeler (⚠️, trafo<10) her yöntemde güvenilmez.")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(out.getvalue(), encoding="utf-8")
print(f"Rapor yazıldı: {REPORT}")
