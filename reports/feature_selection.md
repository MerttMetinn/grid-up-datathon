# feature importance ve model feature listesi

## amac

`scripts/19_feature_selection.py`, modelde kullanilan 75 feature'in onemini f1, f2 ve f3 fold'larinda olcer. Sonra importance sonuclarina gore modele girecek feature adlarini `model_features.json` dosyasina yazar.

## kullanilan veriler

Her fold kendi train ve valid datasetini kullanir:

```text
f1_train.parquet -> f1_valid.parquet
f2_train.parquet -> f2_valid.parquet
f3_train.parquet -> f3_valid.parquet
```

`full_train.parquet` feature selection sirasinda kullanilmaz. Bu dosya final model egitimi icindir.

Her fold'da tum satirlar ve 75 feature kullanilir.


## importance ve model listesi

### importance

Bir feature'in egitilmis model icindeki katkisini olcer. Kodda iki olcut kullanilir:

- `gain`: Feature'in model hatasini azaltmaya toplam katkisi.
- `split`: Feature'in agaclarda kac kez bolme icin kullanildigi.

Feature onemini yorumlarken `gain` ana olcut olarak kullanilir.

### model listesi

Script, feature'i model listesine almak icin iki kosul kullanir:

- En az 2 fold'da kullanilmis olmasi (`folds_used >= 2`).
- Uc fold'daki ortalama gain payinin en az `%0.5` olmasi.

Bu kosullari gecen feature adlari modelde kullanilmak uzere JSON'a kaydedilir. JSON sadece feature isimlerinden olusan bir listedir.

## feature durumlari

Tek fold detayinda:

- `katkili`: Gain payi en az `%0.5`.
- `dusuk_katki`: Modelde kullanilmis ancak gain payi `%0.5` altinda.
- `kullanilmadi`: Feature hicbir agac bolmesinde kullanilmamis.

Uc fold ozetinde:

- `istikrarli_katkili`: Uc fold'da da kullanilmis.
- `tek_fold_katkili`: Yalnizca bir fold'da kullanilmis.
- `dusuk_istikrarli_katki`: Kullanilmis ancak ortalama katkisi dusuk.
- `hic_kullanilmadi`: Hicbir fold'da kullanilmamis.

## csv ciktilari

### `feature_importance_all_folds.csv`

Her feature'in her fold'daki ayrintili sonucudur. 75 feature ve 3 fold oldugu icin 225 satir icerir.

Kolonlar:

- `feature`: Feature adi.
- `gain`: Toplam katki.
- `split`: Kullanim sayisi.
- `gain_share`: Toplam gain icindeki pay.
- `fold`: F1, F2 veya F3.
- `group`: static, cal, lvl, grp, seas veya wx.
- `status`: O fold'daki durum.

## json ciktilari

### `model_features.json`

Modelde kullanilacak feature adlarini icerir. 

## calistirma

Once dataset dosyalari olusturulur:

```powershell
python scripts\18_build_dataset.py
```

Sonra feature importance ve model feature listesi olusturulur:

```powershell
python scripts\19_feature_selection.py
```

Olusan dosyalar:

```text
data/feature-selection-results/feature_importance_all_folds.csv
data/feature-selection-results/model_features.json
```


