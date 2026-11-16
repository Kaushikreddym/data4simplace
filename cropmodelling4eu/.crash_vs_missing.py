import sys, glob, gzip, re, pandas as pd
rows = []
for reg in (r.rstrip("/") + "/" for r in sorted(glob.glob(f"{sys.argv[1]}/region_*/"))):
    p = pd.read_csv(f"{reg}workspace/project/project_wheat.csv")
    def missing(r):
        f = glob.glob(f"{reg}weather/{r.vRow}/daily_mean_RES1_C{r.vColumn}R{r.vRow}.csv.gz")
        if not f: return None
        with gzip.open(f[0], "rt") as h:
            next(h); return all(l.split("\t")[4] == "-99.9" for _, l in zip(range(400), h))
    p["miss"] = [missing(r) for r in p.itertuples()]
    for log in glob.glob(f"{reg}log/lines_*.out"):
        a, b = map(int, re.search(r"lines_(\d+)-(\d+)", log).group(1, 2))
        seg = p.iloc[a - 1:b]
        txt = open(log, errors="ignore").read()
        rows.append(dict(region=reg[-10:-1], a=a, b=b, n_miss=int(seg.miss.fillna(False).sum()), no_file=int(seg.miss.isna().sum()), crash="SIGSEGV" in txt))
d = pd.DataFrame(rows); d["has_miss"] = d.n_miss > 0
print(pd.crosstab(d.has_miss, d.crash, margins=True))
print("crash rate by n_miss:\n", d.groupby(d.n_miss.clip(upper=5)).crash.agg(["mean", "size"]).round(3))
print("missing weather file:", d.no_file.sum())
