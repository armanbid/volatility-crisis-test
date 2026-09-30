# =============================================================================
# Session 2: build signals, train a linear volatility model on 2005-2019,
# and compare it with the naive benchmark in calm periods vs. shocks.
#
# Setup (terminal, with volenv active):   pip install statsmodels
# Run:                                    python session2_model.py
# (Ignore FutureWarnings from pandas, they don't affect results.)
# =============================================================================

# %% 1. Load the prices saved in Session 1 (no re-download needed)
import numpy as np
import pandas as pd
import statsmodels.api as sm

prices = pd.read_csv("prices_daily.csv", index_col=0, parse_dates=True)
rets = np.log(prices).diff()

# %% 2. Build monthly signals (rows = month-end, columns = ticker)
monthly = rets.groupby(pd.Grouper(freq="ME"))
rv = (monthly.std() * np.sqrt(252)).where(monthly.count() >= 15)   # same as Session 1

month_end_price = prices.groupby(pd.Grouper(freq="ME")).last()
high_52w = prices.rolling(252, min_periods=200).max().groupby(pd.Grouper(freq="ME")).last()

# Everything is known at the end of month t. The target is month t+1.
# Volatility goes in as logs: vol is skewed (occasional huge spikes) and logs tame that.
signals = {
    "log_rv_1m":  np.log(rv),                       # this month's vol
    "log_rv_3m":  np.log(rv.rolling(3).mean()),     # average vol over the last 3 months
    "log_rv_12m": np.log(rv.rolling(12).mean()),    # average vol over the last year
    "ret_1m":     np.log(month_end_price).diff(1),  # momentum: last month's return
    "ret_12m":    np.log(month_end_price).diff(12), # momentum: last year's return
    "drawdown":   month_end_price / high_52w - 1,   # how far below the 52-week high (0 = at the high)
    "log_rv_next": np.log(rv).shift(-1),            # TARGET: next month's vol
}
FEATURES = ["log_rv_1m", "log_rv_3m", "log_rv_12m", "ret_1m", "ret_12m", "drawdown"]

# %% 3. Reshape to one row per (month, ticker) and merge all signals into one table
def to_long(wide, name):
    w = wide.copy()
    w.index.name = "month"
    return w.reset_index().melt(id_vars="month", var_name="ticker", value_name=name)

df = None
for name, wide in signals.items():
    long = to_long(wide, name)
    df = long if df is None else df.merge(long, on=["month", "ticker"])
df = df.dropna()   # drops each ticker's first ~12 months (not enough history yet)

df["target_month"] = df["month"] + pd.offsets.MonthEnd(1)
df = df[df["target_month"] >= "2005-01-01"]

# %% 4. Regimes (same definitions as Session 1)
def regime(m):
    if m.year <= 2019:
        return "1_train 2005-19"
    if pd.Timestamp("2020-02-01") <= m <= pd.Timestamp("2020-04-30"):
        return "3_covid crash (Feb-Apr 2020)"
    if m.year == 2022:
        return "4_2022 selloff"
    return "2_calm test (2020-25 ex-shocks)"

df["regime"] = df["target_month"].apply(regime)

# %% 5. Train: one linear regression, pooled across all 8 tickers, 2005-2019 only
train = df[df["regime"].str.startswith("1_")]
model = sm.OLS(train["log_rv_next"], sm.add_constant(train[FEATURES])).fit()
print(model.summary())
# Read the "coef" column: positive = that signal pushes forecast vol up.
# The t-stats are optimistic because the 8 stocks move together, so treat them as indicative.

# %% 6. Forecast every month (in-sample for 2005-19, out-of-sample from 2020)
# Converting a log forecast back with exp() slightly under-forecasts on average.
# The "smearing" factor exp(residual variance / 2) corrects that.
smear = np.exp(model.resid.var() / 2)
df["model"] = np.exp(model.predict(sm.add_constant(df[FEATURES]))) * smear
df["naive"] = np.exp(df["log_rv_1m"])      # next month = this month
df["actual"] = np.exp(df["log_rv_next"])

# %% 7. Compare model vs naive in each regime
def summarise(g):
    out = {"n": len(g), "mean_actual_vol": g["actual"].mean()}
    for f in ["naive", "model"]:
        err = g[f] - g["actual"]
        out[f"{f}_relMAE"] = err.abs().mean() / g["actual"].mean()   # error as % of the vol level
        out[f"{f}_bias"] = err.mean()                               # negative = under-forecast
    out["model_wins_%"] = ((g["model"] - g["actual"]).abs()
                           < (g["naive"] - g["actual"]).abs()).mean() * 100
    return pd.Series(out)

print("\nModel vs naive, all tickers pooled:")
print(df.groupby("regime").apply(summarise).round(3))

print("\nModel vs naive, S&P 500 only:")
print(df[df["ticker"] == "SPX"].groupby("regime").apply(summarise).round(3))

# %% 8. The onset test: what did each forecast say going into March 2020?
spx = df[df["ticker"] == "SPX"].set_index("target_month")
print("\nS&P 500 forecasts around COVID (vol, annualized):")
print(spx.loc["2019-12":"2020-06", ["actual", "naive", "model"]].round(3))

df.to_csv("model_results.csv", index=False)   # input for Session 3's chart
