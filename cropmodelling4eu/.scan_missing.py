import sys, glob, pandas as pd
files = sorted(glob.glob(f"{sys.argv[1]}/*/*.csv.gz"))
frac = {}
for f in files[:200]:
    d = pd.read_csv(f, sep="\t", usecols=["Date", "TempMax"], parse_dates=["Date"])
    frac[f.rsplit("/", 1)[-1]] = (d.TempMax == -99.9).mean()
s = pd.Series(frac); print("per-file missing frac:", s.describe().round(3).to_dict()); print("all-missing files:", int((s == 1).sum()), " none-missing:", int((s == 0).sum()))
f = s[(s > 0) & (s < 1)].index[:1].tolist() or s.index[:1].tolist()
d = pd.read_csv(next(x for x in files if x.endswith(f[0])), sep="\t", parse_dates=["Date"])
m = d.TempMax == -99.9
print(f[0], "missing by year:", m.groupby(d.Date.dt.year).mean().round(2).loc[lambda x: x > 0].to_dict())
print("missing by month:", m.groupby(d.Date.dt.month).mean().round(2).to_dict())
