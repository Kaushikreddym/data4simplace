import sys, glob, pandas as pd
for tag, root in zip(("cmip6", "mswx"), sys.argv[1:3]):
    files = sorted(glob.glob(f"{root}/*/*.csv.gz"))
    d = pd.concat((pd.read_csv(f, sep="\t") for f in files), ignore_index=True)
    v = d[d.columns[1:8].drop("RefET")].join(d.RelHumCalc)
    s = v.replace(-99.9, float("nan"))
    print(f"== {tag}: {len(files)} files, {len(d)} rows")
    print(" -99.9 per col:", (v == -99.9).sum().to_dict())
    print(" NaN text:", d.isna().sum().sum())
    print(" min:", s.min().round(2).to_dict()); print(" max:", s.max().round(2).to_dict())
    print(" Tmin>Tmax:", int((s.TempMin > s.TempMax).sum()), " Tmin>Tmean:", int((s.TempMin > s.TempMean).sum()), " Tmean>Tmax:", int((s.TempMean > s.TempMax).sum()))
    print(" P<0:", int((s.Precipitation < 0).sum()), " Rad<=0:", int((s.Radiation <= 0).sum()), " RH>100:", int((s.RelHumCalc > 100).sum()), " RH<=0:", int((s.RelHumCalc <= 0).sum()), " Wind<=0:", int((s.Windspeed <= 0).sum()))
    dt = pd.to_datetime(d.Date, errors="coerce"); print(" bad dates:", int(dt.isna().sum()))
