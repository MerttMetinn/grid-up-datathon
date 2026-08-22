# -*- coding: utf-8 -*-
"""
04_diagnose.py — Metriğin nereden geldiğini ayrıştırır. Model yok.

  1. RMSLE ayrıştırması: sıfır vs sıfır-dışı satırların MSE payı (fold × baseline)
  2. F1 valid sıfır profili (global/warm/cold + trafo-bazlı histogram)
  3. Sıfır tahmin edilebilirliği (kırılımlar + hücre-oranı AUC)
  4. Sıfırsız skorlar (yalnız gerçek>0 satırlar)
  5. Cold'da sabit tahmin vs b5 fiziksel çıpa
  6. Kaçınılmaz MSE tavanı (p(1-p)L²)
  7. Sabit kohortta YoY drift → lag_364 düzeltme katsayısı

Çıktı: reports/diagnosis.md
Kullanım: python scripts/04_diagnose.py
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import BASELINES, b5_guc_lf  # noqa: E402
from src.config import REPORTS_DIR, SEED  # noqa: E402
from src.data import load_profile, load_train  # noqa: E402
from src.validation import make_folds  # noqa: E402

out = io.StringIO()


def w(line: str = "") -> None:
    out.write(line + "\n")


def pct(x):
    return f"%{100 * x:.2f}"


def sq_log_err(y, p):
    return (np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2


def main() -> None:
    df = load_train()
    profile = load_profile()
    folds = make_folds(df, profile, seed=SEED)

    w("# Teşhis Raporu — RMSLE ayrıştırması")
    w()
    w(f"Üretim: `scripts/04_diagnose.py` · {datetime.now():%Y-%m-%d %H:%M} · SEED={SEED}")
    w()

    # tahminleri bir kez üret, 1 ve 4'te kullan
    preds = {}   # (fold, baseline) -> (valid_df, pred, e2)
    for fold in folds:
        train = df.loc[fold["train_idx"]]
        valid = df.loc[fold["valid_idx"]].copy()
        valid["is_cold"] = valid["tanim"].isin(fold["cold_tx"])
        for bname, bfunc in BASELINES.items():
            p = bfunc(train, valid)
            preds[(fold["name"], bname)] = (valid, p, sq_log_err(valid["tuketim"], p))

    # ============================================================ 1. AYRIŞTIRMA
    w("## 1. RMSLE ayrıştırması — sıfırların MSE payı")
    w()
    w("| fold | baseline | kesim | n | y=0 satır | y=0 pay | y=0 SSE | "
      "**y=0 MSE payı** | y>0 SSE | rmsle |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for fold in folds:
        fn = fold["name"]
        for bname in BASELINES:
            valid, p, e2 = preds[(fn, bname)]
            for seg, mask in [("warm", ~valid["is_cold"]), ("cold", valid["is_cold"])]:
                y = valid.loc[mask, "tuketim"]
                e = e2[mask]
                zm = y == 0
                sse_z, sse_nz = float(e[zm].sum()), float(e[~zm].sum())
                tot = sse_z + sse_nz
                w(f"| {fn} | {bname} | {seg} | {mask.sum():,} | {int(zm.sum()):,} | "
                  f"{pct(zm.mean())} | {sse_z:,.0f} | **{pct(sse_z / tot)}** | "
                  f"{sse_nz:,.0f} | {np.sqrt(e.mean()):.4f} |")
    w()
    # özet: b6 F1
    _, _, e2 = preds[("F1", "b6_hibrit")]
    v1 = preds[("F1", "b6_hibrit")][0]
    zshare_row = float((v1["tuketim"] == 0).mean())
    zshare_mse = float(e2[v1["tuketim"] == 0].sum() / e2.sum())
    w(f"> **Sonuç (1):** F1'de b6 hibritin toplam kareli hatasının {pct(zshare_mse)}'i "
      f"gerçek-sıfır satırlardan geliyor — bu satırların payı yalnızca {pct(zshare_row)} "
      f"iken; metriğin ana kaldıracı seviye tahmini değil, **sıfırları bilmek**.")
    w()

    # ============================================================ 2. SIFIR PROFİLİ
    w("## 2. F1 valid sıfır profili")
    w()
    f1 = folds[0]
    v = df.loc[f1["valid_idx"]].copy()
    v["is_cold"] = v["tanim"].isin(f1["cold_tx"])
    w("### 2a. Sıfır satır payı")
    w()
    w(f"- global: {pct((v['tuketim'] == 0).mean())} · "
      f"warm: {pct((v.loc[~v['is_cold'], 'tuketim'] == 0).mean())} · "
      f"cold: {pct((v.loc[v['is_cold'], 'tuketim'] == 0).mean())}")
    w()

    def zero_hist(sub: pd.DataFrame, title: str):
        zr = sub.groupby("tanim", observed=True)["tuketim"].agg(
            n="size", z=lambda s: (s == 0).mean())
        bins = [-0.001, 0, 0.05, 0.25, 0.75, 0.9999, 1.0]
        labels = ["%0", "%0-5", "%5-25", "%25-75", "%75-99", "%100"]
        zr["kova"] = pd.cut(zr["z"], bins=bins, labels=labels)
        g = zr.groupby("kova", observed=False).agg(trafo=("z", "size"), satir=("n", "sum"))
        zrows = sub.groupby("tanim", observed=True).apply(
            lambda s: (s["tuketim"] == 0).sum(), include_groups=False)
        g["sifir_satir"] = zr.assign(zn=zrows).groupby("kova", observed=False)["zn"].sum()
        w(f"### {title}")
        w()
        w("| trafo sıfır oranı | trafo | satır | sıfır satır |")
        w("|---|---|---|---|")
        for k, r in g.iterrows():
            w(f"| {k} | {int(r['trafo']):,} | {int(r['satir']):,} | "
              f"{int(r['sifir_satir']):,} |")
        w()
        return g

    gc = zero_hist(v[v["is_cold"]], "2b. Cold trafolar")
    gw = zero_hist(v[~v["is_cold"]], "2c. Warm trafolar")
    dead_share_c = gc.loc[["%75-99", "%100"], "sifir_satir"].sum() / max(gc["sifir_satir"].sum(), 1)
    dead_share_w = gw.loc[["%75-99", "%100"], "sifir_satir"].sum() / max(gw["sifir_satir"].sum(), 1)
    w(f"> **Sonuç (2):** Sıfırlar dağınık değil — cold'da sıfır satırların "
      f"{pct(dead_share_c)}'i, warm'da {pct(dead_share_w)}'i sıfır oranı %75+ olan "
      f"'ölü/yarı ölü' trafolarda toplanmış; problem 'hangi gün sıfır' değil, "
      f"büyük ölçüde '**hangi trafo ölü**' problemi.")
    w()

    # ============================================================ 3. TAHMİN EDİLEBİLİRLİK
    w("## 3. Sıfır tahmin edilebilir mi (train geneli)")
    w()
    tr_all = df.copy()
    tr_all["is_zero"] = tr_all["tuketim"] == 0

    w("### 3a. ilce_key bazında sıfır oranı (uç 10'ar)")
    w()
    zi = tr_all.groupby("ilce_key", observed=True)["is_zero"].agg(["mean", "size"])
    zi = zi.sort_values("mean", ascending=False)
    w("| ilce_key | sıfır oranı | satır |")
    w("|---|---|---|")
    for k, r in pd.concat([zi.head(10), zi.tail(10)]).iterrows():
        w(f"| {k} | {pct(r['mean'])} | {int(r['size']):,} |")
    w()

    w("### 3b. guc_bucket bazında")
    w()
    zb = tr_all.groupby("guc_bucket", observed=True)["is_zero"].mean()
    w("| guc_bucket | sıfır oranı |")
    w("|---|---|")
    for k, r in zb.items():
        w(f"| {k} | {pct(r)} |")
    w()

    w("### 3c. ay bazında")
    w()
    za = tr_all.groupby(tr_all["tarih"].dt.to_period("M"))["is_zero"].mean()
    w("| ay | sıfır oranı |")
    w("|---|---|")
    for k, r in za.items():
        w(f"| {k} | {pct(r)} |")
    w()

    w("### 3d. days_since_entry bazında")
    w()
    first = tr_all.groupby("tanim", observed=True)["tarih"].min()
    dse = (tr_all["tarih"] - tr_all["tanim"].map(first)).dt.days
    dse_bin = pd.cut(dse, bins=[-1, 0, 7, 30, 90, np.inf],
                     labels=["0", "1-7", "8-30", "31-90", "90+"])
    zd = tr_all.groupby(dse_bin, observed=True)["is_zero"].agg(["mean", "size"])
    w("| days_since_entry | sıfır oranı | satır |")
    w("|---|---|---|")
    for k, r in zd.iterrows():
        w(f"| {k} | {pct(r['mean'])} | {int(r['size']):,} |")
    w()

    w("### 3e. Hücre-oranı AUC (F1, hücre = ilce_key × guc_bucket × ay_no)")
    w()
    from sklearn.metrics import roc_auc_score
    tr1 = df.loc[f1["train_idx"]]
    cell = tr1.groupby(["ilce_key", "guc_bucket", "ay_no"], observed=True)["tuketim"] \
        .apply(lambda s: (s == 0).mean())
    idx = pd.MultiIndex.from_frame(v[["ilce_key", "guc_bucket", "ay_no"]])
    p_zero = pd.Series(cell.reindex(idx).to_numpy(), index=v.index) \
        .fillna(float((tr1["tuketim"] == 0).mean()))
    y_zero = (v["tuketim"] == 0).astype(int)
    auc_all = roc_auc_score(y_zero, p_zero)
    auc_cold = roc_auc_score(y_zero[v["is_cold"]], p_zero[v["is_cold"]])
    w(f"- AUC (tüm valid): **{auc_all:.4f}** · sadece cold satırlar: **{auc_cold:.4f}**")
    w()
    w(f"> **Sonuç (3):** Sıfır oranı ilçeye göre {pct(zi['mean'].min())}–"
      f"{pct(zi['mean'].max())} bandında, yeni giriş gününde {pct(zd.loc['0','mean'])} "
      f"ve statik hücre bilgisiyle bile AUC {auc_all:.2f} (cold'da {auc_cold:.2f}) — "
      f"sıfır olasılığı kısmen tahmin edilebilir, model bu sinyali kullanabilmeli.")
    w()

    # ============================================================ 4. SIFIRSIZ SKOR
    w("## 4. Sıfırsız skorlar (yalnız gerçek > 0)")
    w()
    for fold in folds:
        fn = fold["name"]
        w(f"### {fn}")
        w()
        w("| baseline | all | warm | cold |")
        w("|---|---|---|---|")
        for bname in BASELINES:
            valid, p, e2 = preds[(fn, bname)]
            nz = valid["tuketim"] > 0
            a = np.sqrt(e2[nz].mean())
            wm = np.sqrt(e2[nz & ~valid["is_cold"]].mean())
            cd = np.sqrt(e2[nz & valid["is_cold"]].mean())
            w(f"| {bname} | {a:.4f} | {wm:.4f} | {cd:.4f} |")
        w()
    _, _, e2h = preds[("F1", "b6_hibrit")]
    vh = preds[("F1", "b6_hibrit")][0]
    nz = vh["tuketim"] > 0
    w(f"> **Sonuç (4):** Sıfırlar atılınca b6 F1 skoru "
      f"{np.sqrt(e2h.mean()):.3f} → {np.sqrt(e2h[nz].mean()):.3f} — "
      f"seviye tahmini kalitesi göründüğünden çok daha iyi; skorun büyük kısmı "
      f"sıfır problemine gömülü.")
    w()

    # ============================================================ 5. COLD SABİT vs b5
    w("## 5. Cold'da sabit tahmin vs fiziksel çıpa (F1)")
    w()
    vc = v[v["is_cold"]]
    train1 = df.loc[f1["train_idx"]]
    const_log = float(np.log1p(vc["tuketim"]).mean())   # oracle sabit (log-ortalama)
    const_pred = np.expm1(const_log)
    e_const = sq_log_err(vc["tuketim"], pd.Series(const_pred, index=vc.index))
    p5 = b5_guc_lf(train1, vc)
    e_b5 = sq_log_err(vc["tuketim"], p5)
    w(f"### 5a-b. Tüm cold satırlar ({len(vc):,})")
    w()
    w(f"- Oracle sabit (valid'den log-ortalama = {const_log:.3f} → "
      f"{const_pred:,.0f} kWh): RMSLE **{np.sqrt(e_const.mean()):.4f}**")
    w(f"- b5 (guc×24×LF): RMSLE **{np.sqrt(e_b5.mean()):.4f}** "
      f"(fark {np.sqrt(e_const.mean()) - np.sqrt(e_b5.mean()):+.4f})")
    w()
    nzc = vc["tuketim"] > 0
    const_log_nz = float(np.log1p(vc.loc[nzc, "tuketim"]).mean())
    e_const_nz = sq_log_err(vc.loc[nzc, "tuketim"],
                            pd.Series(np.expm1(const_log_nz), index=vc.index[nzc]))
    w(f"### 5c. Sadece gerçek>0 cold satırlar ({int(nzc.sum()):,})")
    w()
    w(f"- Oracle sabit (log-ortalama {const_log_nz:.3f} → {np.expm1(const_log_nz):,.0f} kWh): "
      f"RMSLE **{np.sqrt(e_const_nz.mean()):.4f}**")
    w(f"- b5: RMSLE **{np.sqrt(e_b5[nzc].mean()):.4f}** "
      f"(fark {np.sqrt(e_const_nz.mean()) - np.sqrt(e_b5[nzc].mean()):+.4f})")
    w()
    gain_all = np.sqrt(e_const.mean()) - np.sqrt(e_b5.mean())
    gain_nz = np.sqrt(e_const_nz.mean()) - np.sqrt(e_b5[nzc].mean())
    w(f"> **Sonuç (5):** guc ölçeklemesi sıfır-dışı satırlarda sabitten "
      f"{gain_nz:+.3f} RMSLE {'kazandırıyor' if gain_nz > 0 else 'kaybettiriyor'} "
      f"(tüm cold'da {gain_all:+.3f}) — yani `guc` bilgisi "
      f"{'gerçek seviye sinyali taşıyor' if gain_nz > 0.02 else 'seviye tahmininde beklenenden az katkı veriyor'}; "
      f"oracle sabitin bile {np.sqrt(e_const.mean()):.2f}'de kalması cold probleminin "
      f"seviyeden çok sıfır/heterojenlik problemi olduğunu doğruluyor.")
    w()

    # ============================================================ 6. TAVAN
    w("## 6. Kaçınılmaz MSE tavanı (cold, F1)")
    w()
    p_z = float((vc["tuketim"] == 0).mean())
    L_nz = float(np.log1p(vc.loc[nzc, "tuketim"]).mean())
    floor_mse = p_z * (1 - p_z) * L_nz ** 2
    floor_rmsle = float(np.sqrt(floor_mse))
    b5_cold = float(np.sqrt(e_b5.mean()))
    w(f"- p (sıfır oranı) = {p_z:.4f} · L (sıfır-dışı ort. log1p) = {L_nz:.3f}")
    w(f"- Kaçınılmaz MSE = p(1-p)L² = {floor_mse:.4f} → **RMSLE tabanı = {floor_rmsle:.4f}**")
    w(f"- b5 mevcut cold RMSLE = {b5_cold:.4f} → taban ile ara: "
      f"**{b5_cold - floor_rmsle:.4f}**")
    w()
    w(f"> **Sonuç (6):** Sıfırlar hiç ayırt edilemese bile taban {floor_rmsle:.2f}; "
      f"b5'in {b5_cold:.2f}'lik cold skoru ile taban arasındaki {b5_cold - floor_rmsle:.2f}'lik "
      f"aralık, cold tarafında modellemeyle kazanılabilir alandır (sıfır olasılığı + "
      f"seviye heterojenliği).")
    w()

    # ============================================================ 7. YoY DRIFT
    w("## 7. YoY drift (sabit kohort) → lag_364 düzeltmesi")
    w()
    gun = df.groupby("tanim", observed=True)["tarih"].nunique()
    full_set = set(gun[gun == df["tarih"].nunique()].index)
    coh = df[df["tanim"].isin(full_set)].copy()
    coh["ay_p"] = coh["tarih"].dt.to_period("M")
    mon = np.log1p(coh["tuketim"]).groupby(coh["ay_p"]).mean()
    w("| ay | 2025 | 2026 | fark (log1p) |")
    w("|---|---|---|---|")
    diffs = []
    for m in (1, 2, 3):
        a, b = mon[pd.Period(f"2025-{m:02d}")], mon[pd.Period(f"2026-{m:02d}")]
        diffs.append(b - a)
        w(f"| {m:02d} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    drift = float(np.mean(diffs))
    w(f"| **ort** | · | · | **{drift:+.4f}** |")
    w()
    w(f"> **Sonuç (7):** Sabit kohortta YoY drift Oca–Mar ortalaması "
      f"**{drift:+.3f} log1p** (çarpan olarak ×{np.exp(drift):.3f}) — lag_364 "
      f"feature'larına önerilen düzeltme: `seas_lag364_log1p + {drift:.3f}` "
      f"(ya da modele ham ver, `cal_year` benzeri sinyalle öğrenmesine izin ver; "
      f"baseline kullanımında katsayı buradan).")
    w()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "diagnosis.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"Rapor: {REPORTS_DIR / 'diagnosis.md'}")


if __name__ == "__main__":
    main()
