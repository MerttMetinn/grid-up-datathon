# -*- coding: utf-8 -*-
"""
01_recon2.py — İkinci keşif turu (STRATEGY_v2 madde 6.1).

Sorular:
  1. Cold-start profili (satır oranı, guc/ilçe dağılımı, devreye giriş deseni)
  2. lag_364 kapsamı (test ve F1 fold'u için, ±3/±7 pencereli)
  3. Haftanın günü anomalisi — trafo-içi normalize kontrol
  4. LF>1 ve sıfır-blok trafolarının test'teki varlığı
  5. Tam panel kohortu üzerinden mevsimsellik + ilçe Temmuz/Mayıs oranı

Çıktı: reports/recon2.md
Kullanım: python scripts/01_recon2.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "recon2.md"

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")


def fmt(x):
    return f"{x:,}"


def pct(a, b):
    return f"%{100 * a / max(b, 1):.2f}"


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


def parse_ilce(lok: pd.Series) -> pd.DataFrame:
    """İZMİR>BÖLGE>İLÇE (3 parça) ve MANİSA>İLÇE (2 parça) formatlarını ayırır."""
    parts = lok.str.split(">")
    il = parts.str[0].str.strip()
    ilce = parts.str[-1].str.strip()
    return pd.DataFrame({"il": il, "ilce": il + ">" + ilce})


tr[["il", "ilce"]] = parse_ilce(tr["lokasyon"])
te[["il", "ilce"]] = parse_ilce(te["lokasyon"])

tr_tanim = set(tr["tanim"].unique())
te_tanim = set(te["tanim"].unique())
cold_set = te_tanim - tr_tanim
warm_set = te_tanim & tr_tanim
te["is_cold"] = te["tanim"].isin(cold_set)

w("# Recon-2 Raporu")
w()
w(f"Üretim: `scripts/01_recon2.py` · Tarih: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
w()

# ================================================================ 1. COLD-START
w("## 1. Cold-start profili")
w()

# --- 1a satır oranı
n_cold_rows = int(te["is_cold"].sum())
w("### 1a. Test satırlarının cold payı")
w()
w(f"- Cold trafo: {fmt(len(cold_set))} / {fmt(len(te_tanim))} ({pct(len(cold_set), len(te_tanim))})")
w(f"- **Cold SATIR: {fmt(n_cold_rows)} / {fmt(len(te))} ({pct(n_cold_rows, len(te))})**")
w()

# --- 1b guc karşılaştırması (trafo bazında)
w("### 1b. guc dağılımı — cold vs warm (trafo bazında)")
w()
te_trafo = te.groupby("tanim", observed=True).agg(
    guc=("guc", "first"), ilce=("ilce", "first"), il=("il", "first"),
    is_cold=("is_cold", "first"), n_gun=("tarih", "nunique"),
    ilk_gun=("tarih", "min"),
)
gc = te_trafo.loc[te_trafo["is_cold"], "guc"]
gw = te_trafo.loc[~te_trafo["is_cold"], "guc"]
w("| istatistik | cold | warm |")
w("|---|---|---|")
for name, q in [("min", 0), ("%25", .25), ("medyan", .5), ("%75", .75), ("max", 1)]:
    w(f"| {name} | {gc.quantile(q):,.0f} | {gw.quantile(q):,.0f} |")
w()
w("guc frekans tablosu (trafo sayısı ve kolon içi pay):")
w()
freq = pd.DataFrame({
    "cold": gc.value_counts(), "warm": gw.value_counts()
}).fillna(0).astype(int)
freq["cold_%"] = 100 * freq["cold"] / max(len(gc), 1)
freq["warm_%"] = 100 * freq["warm"] / max(len(gw), 1)
freq = freq.sort_values("cold", ascending=False)
w("| guc | cold | cold % | warm | warm % |")
w("|---|---|---|---|---|")
for guc_val, r in freq.head(15).iterrows():
    w(f"| {guc_val:,.0f} | {int(r['cold']):,} | {r['cold_%']:.1f} | "
      f"{int(r['warm']):,} | {r['warm_%']:.1f} |")
kalan = freq.iloc[15:]
if len(kalan):
    w(f"| (diğer {len(kalan)} değer) | {kalan['cold'].sum():,} | "
      f"{kalan['cold_%'].sum():.1f} | {kalan['warm'].sum():,} | {kalan['warm_%'].sum():.1f} |")
w()

# --- 1c ilçe dağılımı
w("### 1c. İlçe dağılımı — cold yoğunlaşması")
w()
ilce_tab = te_trafo.groupby("ilce", observed=True)["is_cold"].agg(["sum", "count"])
ilce_tab.columns = ["cold", "toplam"]
ilce_tab["cold_orani_%"] = 100 * ilce_tab["cold"] / ilce_tab["toplam"]
ilce_tab["cold_pay_%"] = 100 * ilce_tab["cold"] / max(len(cold_set), 1)
ilce_tab = ilce_tab.sort_values("cold", ascending=False)
w("Cold trafo sayısına göre ilk 15 ilçe:")
w()
w("| ilçe | cold | ilçedeki trafo | ilçe içi cold oranı | tüm cold'lar içindeki pay |")
w("|---|---|---|---|---|")
for ilce, r in ilce_tab.head(15).iterrows():
    w(f"| {ilce} | {int(r['cold']):,} | {int(r['toplam']):,} | "
      f"%{r['cold_orani_%']:.1f} | %{r['cold_pay_%']:.1f} |")
w()
il_tab = te_trafo.groupby("il", observed=True)["is_cold"].agg(["sum", "count"])
il_tab["oran_%"] = 100 * il_tab["sum"] / il_tab["count"]
w("İl bazında:")
w()
w("| il | cold | toplam | cold oranı |")
w("|---|---|---|---|")
for il, r in il_tab.iterrows():
    w(f"| {il} | {int(r['sum']):,} | {int(r['count']):,} | %{r['oran_%']:.1f} |")
w()
hh = ilce_tab["cold_pay_%"].head(5).sum()
w(f"- İlk 5 ilçe tüm cold trafoların %{hh:.1f}'ini içeriyor; "
  f"ilçe içi cold oranı %{ilce_tab['cold_orani_%'].min():.0f}–%{ilce_tab['cold_orani_%'].max():.0f} arası değişiyor.")
w()

# --- 1d cold gün sayısı + ilk görülme
w("### 1d. Cold trafoların test'teki gün sayısı ve devreye giriş tarihi")
w()
cold_tr = te_trafo[te_trafo["is_cold"]]
d = cold_tr["n_gun"].describe(percentiles=[.05, .25, .5, .75, .95])
w(f"- Gün sayısı: min={d['min']:.0f} · %25={d['25%']:.0f} · medyan={d['50%']:.0f} · "
  f"%75={d['75%']:.0f} · max={d['max']:.0f}")
n_full = int((cold_tr["n_gun"] == 122).sum())
w(f"- 122 günün tamamında görünen cold trafo: {fmt(n_full)} ({pct(n_full, len(cold_tr))})")
first_day_start = int((cold_tr["ilk_gun"] == te["tarih"].min()).sum())
w(f"- İlk gün {te['tarih'].min():%Y-%m-%d} olan cold trafo: {fmt(first_day_start)} "
  f"({pct(first_day_start, len(cold_tr))})")
w()
w("İlk görüldükleri tarih (aylık histogram):")
w()
hist_cold = cold_tr["ilk_gun"].dt.to_period("M").value_counts().sort_index()
w("| ay | trafo | pay |")
w("|---|---|---|")
for ay, v in hist_cold.items():
    w(f"| {ay} | {fmt(int(v))} | {pct(int(v), len(cold_tr))} |")
w()

# --- 1e train'de sonradan başlayanlarla kıyas
w("### 1e. Train'de 2025-01-01 sonrası başlayan warm trafolar vs cold devreye giriş")
w()
tr_first = tr.groupby("tanim", observed=True)["tarih"].min()
late = tr_first[tr_first > tr["tarih"].min()]
w(f"- Train'de sonradan başlayan trafo: {fmt(len(late))} / {fmt(len(tr_tanim))} "
  f"({pct(len(late), len(tr_tanim))})")
w()
hist_late = late.dt.to_period("M").value_counts().sort_index()
w("Train sonradan-başlama aylık histogram (455 günlük pencere):")
w()
w("| ay | trafo | aylık ort. yeni trafo/gün |")
w("|---|---|---|")
for ay, v in hist_late.items():
    w(f"| {ay} | {fmt(int(v))} | {int(v)/ay.days_in_month:.1f} |")
w()
late_rate = len(late) / 454  # ilk gün hariç günlük ortalama giriş
cold_after_first = len(cold_tr) - first_day_start
cold_rate = cold_after_first / 121 if len(cold_tr) else 0
w(f"- Train'de günlük ort. yeni trafo girişi: {late_rate:.1f}/gün · "
  f"Test'te ilk gün sonrası cold girişi: {cold_rate:.1f}/gün")
w()
peak_ay, peak_n = hist_cold.idxmax(), int(hist_cold.max())
w(f"> **Özet (1):** Test satırlarının {pct(n_cold_rows, len(te))}'i cold; cold trafolar "
  f"warm'dan daha büyük güçlü (medyan {gc.median():.0f} vs {gw.median():.0f} kVA) ve "
  f"belirli ilçelerde yoğun (ilçe içi cold oranı %{ilce_tab['cold_orani_%'].min():.0f}–"
  f"%{ilce_tab['cold_orani_%'].max():.0f}); neredeyse hiçbiri test başında yok — "
  f"%{100*peak_n/len(cold_tr):.0f}'i {peak_ay}'te devreye giriyor, yani cold'lar "
  f"train'deki filo büyümesinin (günde {late_rate:.1f} yeni trafo) devamı ama daha "
  f"hızlı ({cold_rate:.1f}/gün).")
w()

# ================================================================ 2. LAG_364
w("## 2. lag_364 kapsamı")
w()


def lag_coverage(target: pd.DataFrame, history: pd.DataFrame, lag_days=364):
    """target satırları için (tanim, tarih-lag) history'de var mı — exact/±3/±7."""
    left = target[["tanim", "tarih"]].copy()
    left["lag_date"] = left["tarih"] - pd.Timedelta(days=lag_days)
    right = history[["tanim", "tarih"]].drop_duplicates().rename(
        columns={"tarih": "hist_date"})
    # exact: merge
    exact = left.merge(right, left_on=["tanim", "lag_date"],
                       right_on=["tanim", "hist_date"], how="left")
    res = {"exact": exact["hist_date"].notna().to_numpy()}
    # pencere: merge_asof nearest + tolerance
    ls = left.sort_values("lag_date").reset_index()
    rs = right.sort_values("hist_date")
    for k in (3, 7):
        m = pd.merge_asof(ls, rs, left_on="lag_date", right_on="hist_date",
                          by="tanim", direction="nearest",
                          tolerance=pd.Timedelta(days=k))
        hit = pd.Series(m["hist_date"].notna().to_numpy(), index=m["index"]).sort_index()
        res[f"win{k}"] = hit.to_numpy()
    return res


def cov_row(label, cov, n):
    return (f"| {label} | {pct(int(cov['exact'].sum()), n)} | "
            f"{pct(int(cov['win3'].sum()), n)} | {pct(int(cov['win7'].sum()), n)} | {fmt(n)} |")


w("| kapsam | exact | ±3 gün | ±7 gün | satır |")
w("|---|---|---|---|---|")

# a-c: test tüm satırlar, history = tüm train
cov_all = lag_coverage(te, tr)
w(cov_row("test — tüm satırlar", cov_all, len(te)))

# d: sadece warm satırlar
te_warm = te[~te["is_cold"]]
cov_warm = lag_coverage(te_warm, tr)
w(cov_row("test — sadece warm satırlar", cov_warm, len(te_warm)))

# e: F1 fold (valid 2026-01→03, history ≤ 2025-12-31)
f1_valid = tr[tr["tarih"] >= "2026-01-01"]
f1_hist = tr[tr["tarih"] <= "2025-12-31"]
cov_f1 = lag_coverage(f1_valid, f1_hist)
w(cov_row("F1 fold — valid tüm satırlar", cov_f1, len(f1_valid)))
f1_warm_set = set(f1_hist["tanim"].unique())
f1_valid_warm = f1_valid[f1_valid["tanim"].isin(f1_warm_set)]
cov_f1w = lag_coverage(f1_valid_warm, f1_hist)
w(cov_row("F1 fold — history'de görülen (warm) satırlar", cov_f1w, len(f1_valid_warm)))
w()
w(f"- Test lag hedef aralığı: {te['tarih'].min() - pd.Timedelta(days=364):%Y-%m-%d} → "
  f"{te['tarih'].max() - pd.Timedelta(days=364):%Y-%m-%d} (train içinde)")
w(f"- F1 lag hedef aralığı: {f1_valid['tarih'].min() - pd.Timedelta(days=364):%Y-%m-%d} → "
  f"{f1_valid['tarih'].max() - pd.Timedelta(days=364):%Y-%m-%d}")
w()
cov7 = 100 * cov_all["win7"].sum() / len(te)
karar = ">%50 → seas_* planı uygulanır" if cov7 > 50 else \
        ("%20–50 bandı → lag_364 eklenir ama grp_ mevsimsel indeks öncelikli" if cov7 >= 20
         else "<%20 → lag_364 bırakılır")
cov0 = 100 * cov_all["exact"].sum() / len(te)
cov7w = 100 * cov_warm["win7"].sum() / len(te_warm)
w(f"> **Özet (2):** Test genelinde ±7 gün kapsam %{cov7:.1f} (warm'da %{cov7w:.1f}) — "
  f"STRATEGY_v2 eşiklerine göre: {karar}; pencere genişletmenin katkısı marjinal "
  f"(exact %{cov0:.1f} → ±7 %{cov7:.1f}, +{cov7-cov0:.1f} puan), yani boşluklar "
  f"birkaç günlük kaymalardan değil geçmişin hiç olmamasından kaynaklanıyor.")
w()

# ================================================================ 3. DOW
w("## 3. Haftanın günü anomalisi — normalize kontrol")
w()
gunler = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
tr["ay_p"] = tr["tarih"].dt.to_period("M")
tr["dow"] = tr["tarih"].dt.dayofweek
grp_mean = tr.groupby(["tanim", "ay_p"], observed=True)["tuketim"].transform("mean")
ratio = tr["tuketim"] / grp_mean.replace(0, np.nan)  # tamamen sıfır ay → dışarıda
raw_dow = tr.groupby("dow")["tuketim"].mean()
norm_dow = ratio.groupby(tr["dow"]).mean()
zero_dow = tr.loc[tr["tuketim"] == 0, "dow"].value_counts().sort_index()
n_zero = int((tr["tuketim"] == 0).sum())
w("| gün | ham ortalama | normalize oran (trafo-ay içi) | sıfır satır payı |")
w("|---|---|---|---|")
for i in range(7):
    w(f"| {gunler[i]} | {raw_dow[i]:,.1f} | {norm_dow[i]:.4f} | "
      f"{pct(int(zero_dow.get(i, 0)), n_zero)} |")
w()
spread_raw = (raw_dow.max() - raw_dow.min()) / raw_dow.mean()
spread_norm = norm_dow.max() - norm_dow.min()
w(f"- Ham tabloda göreli açıklık: %{100*spread_raw:.1f} · "
  f"normalize tabloda açıklık: {spread_norm:.4f} ({100*spread_norm:.1f} puan)")
w(f"- Normalize hesapta dışlanan satır (trafo-ay ortalaması 0): "
  f"{fmt(int(ratio.isna().sum()))}")
w()
if spread_norm < 0.5 * spread_raw:
    verdict = (f"Salı/Cumartesi tepesi kompozisyon artefaktı — normalize açıklık "
               f"%{100*spread_norm:.1f}'e düşüyor (ham %{100*spread_raw:.1f}); gerçek "
               f"trafo-içi dow etkisi küçük ({gunler[int(norm_dow.idxmin())]} en düşük "
               f"{norm_dow.min():.3f}, {gunler[int(norm_dow.idxmax())]} en yüksek "
               f"{norm_dow.max():.3f}) ve sıfır satırlar güne eşit dağılmış")
else:
    verdict = (f"dow etkisi normalize edince de sürüyor (açıklık %{100*spread_norm:.1f}) "
               f"— artefakt değil, gerçek")
w(f"> **Özet (3):** {verdict}.")
w()

# ================================================================ 4. LF>1 & SIFIR
w("## 4. LF>1 ve sıfır-blok trafoları test'te")
w()
lf = tr["tuketim"] / (tr["guc"] * 24.0)
lf_trafolar = set(tr.loc[lf > 1, "tanim"].unique())
lf_in_test = lf_trafolar & te_tanim
w("### 4a. LF>1 trafoları")
w()
w(f"- LF>1 satırı olan trafo: {fmt(len(lf_trafolar))}")
w(f"- Bunlardan test'te olan: **{fmt(len(lf_in_test))}** ({pct(len(lf_in_test), len(lf_trafolar))})")
w()

# sıfır blokları (recon-1 ile aynı yöntem)
z = tr[["tanim", "tarih", "tuketim"]].sort_values(["tanim", "tarih"])
z["is_zero"] = (z["tuketim"] == 0).astype("int8")
blk = ((z["tanim"] != z["tanim"].shift()) |
       (z["is_zero"] != z["is_zero"].shift())).cumsum()
runs = z[z["is_zero"] == 1].groupby(blk).agg(
    tanim=("tanim", "first"), n=("is_zero", "size"), son=("tarih", "max"))
tr_end = tr["tarih"].max()
dead_set = set(runs.loc[(runs["n"] >= 30) & (runs["son"] == tr_end), "tanim"])
dead_in_test = dead_set & te_tanim
w("### 4b. Train sonunda sıfır bloğunda olan trafolar")
w()
w(f"- Kapanmış aday (30+ gün sıfır, train sonunda hâlâ sıfır): {fmt(len(dead_set))}")
w(f"- Bunlardan test'te olan: **{fmt(len(dead_in_test))}** ({pct(len(dead_in_test), len(dead_set))})")
if dead_in_test:
    dd = te_trafo.loc[te_trafo.index.isin(dead_in_test), "n_gun"]
    w(f"- Test'teki gün sayıları: min={dd.min()} · medyan={dd.median():.0f} · "
      f"max={dd.max()} · toplam satır={fmt(int(dd.sum()))} "
      f"(test satırlarının {pct(int(dd.sum()), len(te))}'i)")
    n122 = int((dd == 122).sum())
    w(f"- 122 günün tamamında istenen: {fmt(n122)} trafo")
w()
w(f"> **Özet (4):** Bozuk-LF trafolarının {fmt(len(lf_in_test))}/{fmt(len(lf_trafolar))} "
  f"tanesi ve kapanmış-aday trafoların {fmt(len(dead_in_test))}/{fmt(len(dead_set))} "
  f"tanesi test'te tahmin bekliyor; kapanmış adaylar test satırlarının "
  f"{pct(int(dd.sum()), len(te))}'ini kaplıyor ve {fmt(n122)} tanesi 122 günün "
  f"tamamında istendiği için sıfır-override kuralının etki alanı küçük ama cezası büyük.")
w()

# ================================================================ 5. MEVSİMSELLİK
w("## 5. Mevsimsellik tabanı — tam panel kohortu")
w()
gun_say = tr.groupby("tanim", observed=True)["tarih"].nunique()
full_set = set(gun_say[gun_say == tr["tarih"].nunique()].index)
coh = tr[tr["tanim"].isin(full_set)].copy()
w(f"- Kohort: {fmt(len(full_set))} tam panelli trafo · {fmt(len(coh))} satır")
w()
coh["log1p"] = np.log1p(coh["tuketim"])
mon = coh.groupby("ay_p")["log1p"].mean()
mmax, mmin = mon.max(), mon.min()
w("Aylık ortalama log1p(tuketim) — sabit kohort:")
w()
w("| ay | ort. log1p | bar |")
w("|---|---|---|")
for ay, v in mon.items():
    bar = "#" * max(1, round(30 * (v - mmin) / (mmax - mmin)))
    w(f"| {ay} | {v:.4f} | `{bar}` |")
w()
temmuz = np.expm1(mon[pd.Period("2025-07")])
mayis = np.expm1(mon[pd.Period("2025-05")])
w(f"- Geometrik-ortalama ölçekte Temmuz/Mayıs oranı (kohort geneli): "
  f"{temmuz/mayis:.2f}×")
w()

# ilçe bazında Temmuz/Mayıs
w("İlçe bazında Temmuz/Mayıs oranı (kohort, ortalama tuketim):")
w()
sub = coh[coh["ay_p"].isin([pd.Period("2025-05"), pd.Period("2025-07")])]
piv = sub.pivot_table(index="ilce", columns="ay_p", values="tuketim",
                      aggfunc="mean", observed=True)
piv.columns = ["mayis", "temmuz"]
piv["oran"] = piv["temmuz"] / piv["mayis"]
piv["trafo"] = sub.groupby("ilce", observed=True)["tanim"].nunique()
piv = piv.sort_values("oran", ascending=False)
w("| ilçe | trafo | Mayıs ort. | Temmuz ort. | Temmuz/Mayıs |")
w("|---|---|---|---|---|")
for ilce, r in piv.iterrows():
    w(f"| {ilce} | {int(r['trafo'])} | {r['mayis']:,.0f} | {r['temmuz']:,.0f} | "
      f"**{r['oran']:.2f}×** |")
w()
top3 = " / ".join(piv.head(3).index)
bot = piv.index[-1]
w(f"> **Özet (5):** Yaz rampası sabit kohortta da gerçek (Temmuz/Mayıs geometrik "
  f"ortalamada {temmuz/mayis:.2f}×, dip ay Mayıs) ama ilçeler arası fark çok büyük "
  f"({piv['oran'].min():.2f}×–{piv['oran'].max():.2f}×): en sert patlama {top3}'ta, "
  f"en zayıf {bot}'de — mevsimsel düzeltme ilçe bazında yapılmadan tek küresel "
  f"eğriyle açıklanamaz.")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(out.getvalue(), encoding="utf-8")
print(f"Rapor yazıldı: {REPORT}")
