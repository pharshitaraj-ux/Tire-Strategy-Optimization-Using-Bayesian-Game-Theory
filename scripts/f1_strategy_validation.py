
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 0

# --------------------------------------------------------------------------- #
# 1. Data acquisition (FastF1)
# --------------------------------------------------------------------------- #
def fetch(seasons, out, cache="cache"):
    import fastf1
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    lap_cols = ["Driver", "Team", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
                "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate"]
    wx_cols = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]
    parts = []
    for yr in seasons:
        sched = fastf1.get_event_schedule(yr, include_testing=False)
        for _, ev in sched.iterrows():
            try:
                ses = fastf1.get_session(yr, int(ev["RoundNumber"]), "R")
                ses.load(laps=True, telemetry=False, weather=True, messages=False)
                laps = ses.laps
                wx = laps.get_weather_data().reset_index(drop=True)
                d = pd.concat([laps[lap_cols].reset_index(drop=True), wx[wx_cols]], axis=1)
                d["Season"], d["Round"] = yr, int(ev["RoundNumber"])
                d["Race"], d["Circuit"] = ev["EventName"], ev["Location"]
                parts.append(d)
                print("ok  ", yr, ev["EventName"], len(d))
            except Exception as e:  # keep going; log the gap for the paper
                print("SKIP", yr, ev["EventName"], repr(e))
    df = pd.concat(parts, ignore_index=True)
    df["LapTime_s"] = df["LapTime"].dt.total_seconds()
    df["PitIn"], df["PitOut"] = df["PitInTime"].notna(), df["PitOutTime"].notna()
    df.drop(columns=["LapTime", "PitInTime", "PitOutTime"]).to_csv(out, index=False)
    print("saved", out, df.shape)


def make_demo(seed=0):
    """Synthetic laps with the same schema as fetch(); only for smoke-testing the pipeline."""
    rng = np.random.default_rng(seed)
    teams = ["Red Bull Racing", "McLaren", "Ferrari", "Mercedes", "Aston Martin", "Alpine"]
    rows = []
    for yr in (2023, 2024, 2025):
        for rd in range(1, 11):
            base = 80 + 3 * (rd % 6) + rng.normal(0, .5)
            wet = rng.random() < 0.15
            for ti, team in enumerate(teams):
                for k in range(2):
                    drv = f"{team[:3].upper()}{k}"
                    comp = rng.choice(["SOFT", "MEDIUM", "HARD"], size=2)
                    stop = int(rng.integers(14, 36))
                    for lap in range(1, 51):
                        stint = 1 if lap <= stop else 2
                        c = comp[stint - 1]
                        life = lap if stint == 1 else lap - stop
                        deg = {"SOFT": .09, "MEDIUM": .05, "HARD": .03}[c]
                        t = base + .15 * ti + deg * life + rng.normal(0, .4)
                        pit_in, pit_out = lap == stop, lap == stop + 1
                        status = "4" if (rd % 4 == 0 and 20 <= lap <= 23) else "1"
                        if pit_in: t += 3
                        if pit_out: t += 18
                        if status == "4": t += 25
                        is_wet = wet and 25 <= lap <= 40
                        if is_wet:
                            t += 8
                            c = "INTERMEDIATE" if rng.random() < .7 else c
                        rows.append(dict(Driver=drv, Team=team, LapNumber=lap, Stint=stint, Compound=c,
                                         TyreLife=life, TrackStatus=status, IsAccurate=not (pit_in or pit_out),
                                         AirTemp=25 + rng.normal(0, 2), TrackTemp=38 + rng.normal(0, 4),
                                         Humidity=50 + rng.normal(0, 8), Pressure=1010 + rng.normal(0, 2),
                                         Rainfall=bool(is_wet), WindSpeed=2 + abs(rng.normal(0, 1)),
                                         Season=yr, Round=rd, Race=f"GP{rd}", Circuit=f"C{rd}",
                                         LapTime_s=t, PitIn=pit_in, PitOut=pit_out))
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# 2. Preparation, filtering, features (with an explicit input registry)
# --------------------------------------------------------------------------- #
# Every engineered variable lists the raw columns it is computed from.
# None may use LapTime_s (the regression target). PaceCategory is the classification
# LABEL (a function of LapTime_s) and is never allowed in the predictor set.
FEATURE_INPUTS = {
    "CompoundCode": ("Compound",),
    "WeatherState": ("Rainfall", "Compound"),
    "TemperatureLevel": ("TrackTemp",),
    "PitStop": ("PitIn", "PitOut"),
    "TyreWear": ("TyreLife", "Compound"),            # TyreLife / reference stint length of the compound
    "StintLength": ("RaceID", "Driver", "Stint", "LapNumber"),   # look-ahead: total laps of the stint
    "Strategy": ("Compound", "StintLength"),
}
LABEL_ONLY = {"PaceCategory": ("LapTime_s",)}
REF_LIFE = {"SOFT": 18, "MEDIUM": 28, "HARD": 40, "INTERMEDIATE": 25, "WET": 25, "UNKNOWN": 25}
NUM = ["TyreLife", "LapNumber", "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed",
       "TyreWear", "PitStop", "CompoundCode"]
CAT = ["Compound", "Team", "Circuit"]


def prepare(df, short=12, medium=25):
    df = df.copy()
    df["RaceID"] = df["Season"].astype(str) + "_" + df["Round"].astype(int).astype(str).str.zfill(2)
    df["Compound"] = df["Compound"].astype(str).str.upper().replace({"NAN": "UNKNOWN"})
    df = df[df["LapTime_s"].notna()].reset_index(drop=True)
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    for c in ("PitIn", "PitOut", "IsAccurate", "Rainfall"):
        df[c] = df[c].astype(bool)
    # laps filtered out in the "filtered" experiments: pit-in, pit-out, SC/VSC/yellow/red, inaccurate, lap 1
    df["Clean"] = (~df.PitIn) & (~df.PitOut) & (df.TrackStatus == "1") & df.IsAccurate & (df.LapNumber > 1)
    df["Wet"] = df.Rainfall | df.Compound.isin(["INTERMEDIATE", "WET"])
    # engineered features (inputs listed in FEATURE_INPUTS)
    df["CompoundCode"] = df.Compound.astype("category").cat.codes
    df["WeatherState"] = np.where(df.Wet, "Wet", "Dry")
    df["TemperatureLevel"] = pd.qcut(df.TrackTemp, 3, labels=["Low", "Medium", "High"]).astype(str)
    df["PitStop"] = (df.PitIn | df.PitOut).astype(int)
    df["TyreWear"] = df.TyreLife / df.Compound.map(REF_LIFE).fillna(25)
    df["StintLength"] = df.groupby(["RaceID", "Driver", "Stint"])["LapNumber"].transform("count")
    cat = np.where(df.StintLength <= short, "Short", np.where(df.StintLength <= medium, "Medium", "Long"))
    df["Strategy"] = df.Compound + "_" + cat
    # circuit-normalised delta (seconds) used by the payoff: lap time minus the race's clean-lap median
    med = df[df.Clean].groupby("RaceID")["LapTime_s"].median()
    df["Delta"] = df.LapTime_s - df.RaceID.map(med)
    # classification LABEL (tertiles of clean laps within each race); never a predictor
    q = df[df.Clean].groupby("RaceID")["LapTime_s"].quantile([1 / 3, 2 / 3]).unstack()
    lo, hi = df.RaceID.map(q.iloc[:, 0]), df.RaceID.map(q.iloc[:, 1])
    df["PaceCategory"] = np.where(df.LapTime_s <= lo, "Fast", np.where(df.LapTime_s <= hi, "Normal", "Slow"))
    assert not ({"LapTime_s", "PaceCategory", "Delta"} & set(NUM + CAT)), "target leakage in predictors"
    assert all("LapTime_s" not in v for v in FEATURE_INPUTS.values()), "engineered feature uses the target"
    return df

# --------------------------------------------------------------------------- #
# 3. Models and baselines
# --------------------------------------------------------------------------- #
class CircuitMean(BaseEstimator, RegressorMixin):
    """Baseline 1: predict the mean lap time of the circuit seen in training."""
    def fit(self, X, y):
        self.m_ = pd.Series(np.asarray(y), index=X["Circuit"].values).groupby(level=0).mean()
        self.g_ = float(np.mean(y)); return self
    def predict(self, X):
        return X["Circuit"].map(self.m_).fillna(self.g_).values


class AgeCompound(BaseEstimator, TransformerMixin):
    """Design matrix: tyre age, compound dummies and age x compound (optionally circuit dummies)."""
    def __init__(self, circuit=False): self.circuit = circuit
    def fit(self, X, y=None):
        self.comps_, self.circs_ = sorted(X["Compound"].unique()), sorted(X["Circuit"].unique()); return self
    def transform(self, X):
        age = X["TyreLife"].fillna(0).values.astype(float)
        cols = [age]
        for c in self.comps_:
            ind = (X["Compound"] == c).values.astype(float); cols += [ind, ind * age]
        if self.circuit:
            cols += [(X["Circuit"] == c).values.astype(float) for c in self.circs_]
        return np.column_stack(cols)


def pre():
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])


def reg_models(fast):
    gb = HistGradientBoostingRegressor(max_iter=300, random_state=SEED) if fast else \
        GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=SEED)
    m = {"CircuitMean": CircuitMean(),
         "LinReg(age x compound)": Pipeline([("d", AgeCompound()), ("m", LinearRegression())]),
         "LinReg(age x compound + circuit)": Pipeline([("d", AgeCompound(True)), ("m", LinearRegression())]),
         "RandomForest": Pipeline([("p", pre()), ("m", RandomForestRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=SEED))]),
         "GradientBoosting": Pipeline([("p", pre()), ("m", gb)])}
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = Pipeline([("p", pre()), ("m", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=.05,
                                                                subsample=.8, n_jobs=-1, random_state=SEED))])
    except ImportError:
        if not getattr(reg_models, "warned", False):
            print("xgboost not installed: XGBoost baseline skipped"); reg_models.warned = True
    return m


def cls_models(cap):
    return {"Majority": DummyClassifier(strategy="most_frequent"),
            "RandomForest": Pipeline([("p", pre()), ("m", RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=SEED))]),
            "SVM(RBF)": Pipeline([("p", pre()), ("s", StandardScaler()), ("m", SVC(C=1.0, gamma="scale"))])}

# --------------------------------------------------------------------------- #
# 4. Splitters: lap-random (original, optimistic), driver-race, race, leave-one-season-out
# --------------------------------------------------------------------------- #
def make_splits(d, scheme, reps):
    idx = np.arange(len(d))
    if scheme == "lap_random (original)":
        return list(ShuffleSplit(reps, test_size=.2, random_state=SEED).split(idx))
    if scheme == "driver_race groups":
        g = (d.RaceID + "|" + d.Driver).values
        return list(GroupShuffleSplit(reps, test_size=.2, random_state=SEED).split(idx, groups=g))
    if scheme == "race groups":
        return list(GroupShuffleSplit(reps, test_size=.2, random_state=SEED).split(idx, groups=d.RaceID.values))
    if scheme == "leave-one-season-out":
        s = d.Season.values
        return [(idx[s != y], idx[s == y]) for y in sorted(set(s))]
    raise ValueError(scheme)

SCHEMES = ["lap_random (original)", "driver_race groups", "race groups", "leave-one-season-out"]


def run_regression(df, reps, fast, out):
    rows = []
    for filt in (False, True):
        d = (df[df.Clean] if filt else df).reset_index(drop=True)
        for scheme in SCHEMES:
            for k, (tr, te) in enumerate(make_splits(d, scheme, reps)):
                Xtr, Xte, ytr, yte = d.iloc[tr], d.iloc[te], d.LapTime_s.values[tr], d.LapTime_s.values[te]
                for name, mdl in reg_models(fast).items():
                    p = clone(mdl).fit(Xtr, ytr).predict(Xte)
                    rows.append(dict(filtered=filt, scheme=scheme, split=k, model=name,
                                     MAE=mean_absolute_error(yte, p), RMSE=mean_squared_error(yte, p) ** .5, R2=r2_score(yte, p)))
            print("regression done:", "filtered" if filt else "unfiltered", scheme)
    res = pd.DataFrame(rows); res.to_csv(f"{out}/regression_results.csv", index=False)
    return res


def run_classification(df, reps, cap, out):
    rows = []
    for filt in (False, True):
        d = (df[df.Clean] if filt else df).reset_index(drop=True)
        for scheme in ("lap_random (original)", "race groups"):
            for k, (tr, te) in enumerate(make_splits(d, scheme, reps)):
                if len(tr) > cap:
                    tr = np.random.default_rng(k).choice(tr, cap, replace=False)  # SVC is O(n^2): cap train size
                Xtr, Xte, ytr, yte = d.iloc[tr], d.iloc[te], d.PaceCategory.values[tr], d.PaceCategory.values[te]
                for name, mdl in cls_models(cap).items():
                    p = clone(mdl).fit(Xtr, ytr).predict(Xte)
                    rows.append(dict(filtered=filt, scheme=scheme, split=k, model=name,
                                     Accuracy=accuracy_score(yte, p), WeightedF1=f1_score(yte, p, average="weighted")))
            print("classification done:", "filtered" if filt else "unfiltered", scheme)
    res = pd.DataFrame(rows); res.to_csv(f"{out}/classification_results.csv", index=False)
    return res


def leakage_audit(df, fast, out, reps=5):
    """(a) static: input registry; (b) empirical: |Spearman| with LapTime and MAE change when the feature is dropped."""
    d = df[df.Clean].reset_index(drop=True)
    rows = [dict(feature=f, inputs=", ".join(v), uses_target="LapTime_s" in v, role="predictor candidate") for f, v in FEATURE_INPUTS.items()]
    rows += [dict(feature=f, inputs=", ".join(v), uses_target=True, role="classification label only") for f, v in LABEL_ONLY.items()]
    audit = pd.DataFrame(rows)
    audit["abs_spearman_with_LapTime"] = audit.feature.map(
        lambda f: abs(d[f].rank().corr(d.LapTime_s.rank())) if f in d and pd.api.types.is_numeric_dtype(d[f]) else np.nan)
    splits = make_splits(d, "race groups", reps)
    def mae_with(num, cat):
        global NUM, CAT
        keep = (NUM, CAT); NUM, CAT = num, cat
        try:
            return np.mean([mean_absolute_error(d.LapTime_s.values[te], clone(reg_models(fast)["GradientBoosting"]).fit(d.iloc[tr], d.LapTime_s.values[tr]).predict(d.iloc[te])) for tr, te in splits])
        finally:
            NUM, CAT = keep
    full = mae_with(NUM, CAT)
    audit["delta_MAE_when_dropped"] = np.nan
    for f in ("TyreWear", "PitStop", "CompoundCode"):
        audit.loc[audit.feature == f, "delta_MAE_when_dropped"] = mae_with([x for x in NUM if x != f], CAT) - full
    audit.to_csv(f"{out}/leakage_audit.csv", index=False)
    return audit

# --------------------------------------------------------------------------- #
# 5. Payoff, switch points with cluster-bootstrap CIs, back-test
# --------------------------------------------------------------------------- #
P0 = 0.016486                         # baseline P(Wet) used in the original report
GRID = np.round(np.arange(0, 1.0001, .005), 3)


def payoff(d, strategies, min_laps):
    """U(s,w) = - mean(Delta) over laps of strategy s under weather w; NaN if fewer than min_laps laps."""
    g = d.groupby(["Strategy", "Wet"])["Delta"].agg(["sum", "count"])
    S = np.full((len(strategies), 2), np.nan); N = np.zeros((len(strategies), 2))
    for i, s in enumerate(strategies):
        for j, w in enumerate((False, True)):
            if (s, w) in g.index:
                sm, ct = g.loc[(s, w)]; N[i, j] = ct
                if ct >= min_laps: S[i, j] = -sm / ct
    return S, N


def first_switch(U, strategies, p0=P0):
    ok = ~np.isnan(U).any(1)
    if ok.sum() < 2: return None
    names, Uo = np.array(strategies)[ok], U[ok]
    E = (1 - GRID)[:, None] * Uo[:, 0] + GRID[:, None] * Uo[:, 1]
    b0 = int(np.argmax((1 - p0) * Uo[:, 0] + p0 * Uo[:, 1]))
    chg = np.where((GRID > p0) & (E.argmax(1) != b0))[0]
    return (GRID[chg[0]], names[b0], names[E[chg[0]].argmax()]) if len(chg) else (np.nan, names[b0], None)


def switch_points_ci(df, teams, min_laps, B, out):
    d = df[df.Clean]; strategies = sorted(d.Strategy.unique()); rng = np.random.default_rng(SEED)
    rows, nrows = [], []
    for team in teams:
        t = d[d.Team == team]
        if t.RaceID.nunique() < 5: continue
        races = sorted(t.RaceID.unique())
        SUM = np.zeros((len(races), len(strategies), 2)); CNT = np.zeros_like(SUM)
        for a, r in enumerate(races):
            g = t[t.RaceID == r].groupby(["Strategy", "Wet"])["Delta"].agg(["sum", "count"])
            for (s, w), (sm, ct) in g.iterrows():
                SUM[a, strategies.index(s), int(w)], CNT[a, strategies.index(s), int(w)] = sm, ct
        def U_from(w):
            s, c = np.tensordot(w, SUM, 1), np.tensordot(w, CNT, 1)
            return np.where(c >= min_laps, -s / np.maximum(c, 1), np.nan), c
        U, c = U_from(np.ones(len(races)))
        for i, s in enumerate(strategies):
            nrows.append(dict(team=team, strategy=s, n_dry=int(c[i, 0]), n_wet=int(c[i, 1])))
        point = first_switch(U, strategies)
        draws = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(races), len(races)), minlength=len(races)).astype(float)
            r = first_switch(U_from(w)[0], strategies)
            if r is not None: draws.append(r)
        ps = np.array([x[0] for x in draws], float)
        fin = ps[~np.isnan(ps)]
        same = np.mean([(x[1], x[2]) == (point[1], point[2]) for x in draws]) if point and draws else np.nan
        rows.append(dict(team=team, n_races=len(races), prev=point[1] if point else None,
                         switch_p_point=point[0] if point else np.nan, new=point[2] if point else None,
                         ci_low=np.percentile(fin, 2.5) if len(fin) else np.nan,
                         ci_high=np.percentile(fin, 97.5) if len(fin) else np.nan,
                         share_boot_with_any_switch=len(fin) / max(len(draws), 1), share_boot_same_pair=same, B=B))
    pd.DataFrame(nrows).to_csv(f"{out}/payoff_n_laps.csv", index=False)
    res = pd.DataFrame(rows); res.to_csv(f"{out}/switch_points_ci.csv", index=False)
    return res


def backtest(df, min_laps, min_test_laps, min_train_races, out):
    """Expanding-window back-test: recommend with races before r, evaluate on realised laps of race r."""
    d = df[df.Clean]; strategies = sorted(d.Strategy.unique())
    order = sorted(d.RaceID.unique()); rows = []
    for i, r in enumerate(order):
        if i < min_train_races: continue
        tr, te = d[d.RaceID.isin(order[:i])], d[d.RaceID == r]
        p_clim, p_orac = tr.Wet.mean(), te.Wet.mean()
        for team in te.Team.unique():
            U, _ = payoff(tr[tr.Team == team], strategies, min_laps)
            Ut, _ = payoff(te[te.Team == team], strategies, 1)
            real_mean = -np.array([te[(te.Team == team) & (te.Strategy == s)].Delta.mean() for s in strategies])
            cnt = np.array([((te.Team == team) & (te.Strategy == s)).sum() for s in strategies])
            run = (cnt >= min_test_laps) & ~np.isnan(real_mean)
            if run.sum() < 2: continue
            fam = tr[tr.Team == team].Strategy.value_counts()
            pol = {}
            for lab, p in (("payoff@climatology", p_clim), ("payoff@oracle-weather", p_orac)):
                ok = ~np.isnan(U).any(1)
                if ok.any(): pol[lab] = int(np.where(ok)[0][np.argmax(((1 - p) * U[:, 0] + p * U[:, 1])[ok])])
            okd = ~np.isnan(U[:, 0])
            if okd.any(): pol["payoff dry-only (p=0)"] = int(np.where(okd)[0][np.argmax(U[okd, 0])])
            if len(fam): pol["most-used (baseline)"] = strategies.index(fam.index[0])
            for lab, k in pol.items():
                covered = bool(run[k])
                best = np.nanmax(np.where(run, real_mean, np.nan))
                rows.append(dict(race=r, team=team, policy=lab, covered=covered,
                                 regret=best - real_mean[k] if covered else np.nan,
                                 hit=bool(real_mean[k] == best) if covered else np.nan))
    det = pd.DataFrame(rows); det.to_csv(f"{out}/backtest_detail.csv", index=False)
    summ = det.groupby("policy").agg(n=("covered", "size"), coverage=("covered", "mean"),
                                     top1_hit=("hit", lambda x: np.nanmean(x.astype(float))),
                                     mean_regret_s=("regret", "mean"), sd_regret_s=("regret", "std")).reset_index()
    summ.to_csv(f"{out}/backtest_summary.csv", index=False)
    return summ

# --------------------------------------------------------------------------- #
# 6. Reporting
# --------------------------------------------------------------------------- #
def ms(x): return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else float('nan'):.3f}"


def write_tables(reg, cls, audit, sw, bt, out):
    L = []
    for filt in (False, True):
        L.append(f"\n### Regression, {'FILTERED' if filt else 'UNFILTERED'} laps (mean ± SD over splits)\n")
        t = reg[reg.filtered == filt].groupby(["scheme", "model"]).agg(MAE=("MAE", ms), RMSE=("RMSE", ms), R2=("R2", ms))
        L.append(t.to_markdown())
    for filt in (False, True):
        L.append(f"\n### Classification, {'FILTERED' if filt else 'UNFILTERED'} laps\n")
        t = cls[cls.filtered == filt].groupby(["scheme", "model"]).agg(Accuracy=("Accuracy", ms), WeightedF1=("WeightedF1", ms))
        L.append(t.to_markdown())
    L += ["\n### Leakage audit\n", audit.to_markdown(index=False),
          "\n### Switch points (cluster bootstrap over races)\n", sw.to_markdown(index=False),
          "\n### Back-test\n", bt.to_markdown(index=False)]
    open(f"{out}/results_tables.md", "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--seasons", nargs="+", type=int, required=True); f.add_argument("--out", default="data/laps.csv")
    r = sub.add_parser("run")
    r.add_argument("--data"); r.add_argument("--demo", action="store_true"); r.add_argument("--out", default="results")
    r.add_argument("--reps", type=int, default=20); r.add_argument("--boot", type=int, default=1000)
    r.add_argument("--fast", action="store_true", help="HistGradientBoosting instead of classic GradientBoosting")
    r.add_argument("--svm-cap", type=int, default=15000); r.add_argument("--min-laps", type=int, default=30)
    r.add_argument("--teams", nargs="*", default=["Red Bull Racing", "McLaren", "Ferrari"])
    a = ap.parse_args()
    if a.cmd == "fetch": return fetch(a.seasons, a.out)
    os.makedirs(a.out, exist_ok=True)
    raw = make_demo() if a.demo else pd.read_csv(a.data, dtype={"TrackStatus": str})
    df = prepare(raw)
    print(f"{len(df)} laps, {df.RaceID.nunique()} races, seasons {sorted(df.Season.unique())}, clean laps {df.Clean.sum()}")
    reg = run_regression(df, a.reps, a.fast, a.out)
    cls = run_classification(df, a.reps, a.svm_cap, a.out)
    audit = leakage_audit(df, a.fast, a.out)
    sw = switch_points_ci(df, a.teams, a.min_laps, a.boot, a.out)
    bt = backtest(df, a.min_laps, 8, 6, a.out)
    write_tables(reg, cls, audit, sw, bt, a.out)
    print("done ->", a.out)


if __name__ == "__main__":
    main()