# =============================================================================
# Session 1: download prices, build monthly realized volatility, test the naive
# benchmark ("next month's volatility = this month's volatility").
#
# Run in Google Colab (one cell per "# %%" block) or locally.
# Setup (Colab or terminal):   pip install yfinance pandas numpy
# =============================================================================

# %% 1. Imports and settings
import numpy as np
import pandas as pd
import yfinance as yf

# ^GSPC = S&P 500 index. The other seven are the Magnificent 7.
TICKERS = ["^GSPC", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
START, END = "2004-01-01", "2026-01-01"   # 2004 gives a year of history before training starts in 2005

# %% 2. Download daily prices
# auto_adjust=True adjusts prices for splits and dividends, so returns are correct
# across events like NVDA's and TSLA's stock splits.
# threads=False downloads one ticker at a time. Parallel downloads can clash on
# yfinance's local cache ("database is locked") and silently lose tickers.
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False, threads=False)
prices = raw["Close"].rename(columns={"^GSPC": "SPX"})

# Stop if any ticker came back empty, rather than carrying on with missing data.
missing = list(prices.columns[prices.isna().all()])
if missing:
    raise SystemExit(f"Download failed for {missing}. Run the script again.")
prices.to_csv("prices_daily.csv")          # saved so Session 2 doesn't need to re-download
print(prices.tail())
print("First valid date per ticker:\n", prices.apply(lambda s: s.first_valid_index()))
# Expect TSLA to start mid-2010 and META mid-2012: they have shorter histories.

# %% 3. Daily log returns
# log(P_t) - log(P_{t-1}). Log returns add up over time, which is standard for volatility work.
rets = np.log(prices).diff()

# %% 4. Monthly realized volatility (annualized)
# Realized vol for a month = standard deviation of that month's daily returns,
# scaled by sqrt(252) (trading days per year) so it reads like "30% a year".
# "ME" = month-end. On pandas older than 2.2, use freq="M" instead.
monthly = rets.groupby(pd.Grouper(freq="ME"))
rv = monthly.std() * np.sqrt(252)
n_days = monthly.count()
rv = rv.where(n_days >= 15)                # drop part-months (e.g. IPO months) with too few days
rv.index.name = "month"
rv.to_csv("rv_monthly.csv")

# Sanity check: S&P 500 volatility should spike in March 2020 (roughly 80-90% annualized).
print(rv.loc["2020-01":"2020-06", "SPX"].round(3))

# %% 5. Reshape to one row per (month, ticker) and add the target
# rv.shift(-1) moves next month's value up one row: that is what we are trying to forecast.
def to_long(wide, name):
    return wide.reset_index().melt(id_vars="month", var_name="ticker", value_name=name)

df = (to_long(rv, "rv")
      .merge(to_long(rv.shift(-1), "rv_next"), on=["month", "ticker"])
      .dropna())

# "month" is when the forecast is made; the target month is the one after it.
df["target_month"] = df["month"] + pd.offsets.MonthEnd(1)
df = df[df["target_month"] >= "2005-01-01"]

# %% 6. Label each forecast by the regime of the month being forecast
def regime(m):
    if m.year <= 2019:
        return "1_train 2005-19"
    if pd.Timestamp("2020-02-01") <= m <= pd.Timestamp("2020-04-30"):
        return "3_covid crash (Feb-Apr 2020)"
    if m.year == 2022:
        return "4_2022 selloff"
    return "2_calm test (2020-25 ex-shocks)"

df["regime"] = df["target_month"].apply(regime)

# %% 7. Naive benchmark and its errors
df["naive"] = df["rv"]                     # forecast: next month looks like this month
df["err"] = df["naive"] - df["rv_next"]    # positive = forecast too high

def summarise(g):
    return pd.Series({
        "n": len(g),
        "mean_actual_vol": g["rv_next"].mean(),
        "MAE": g["err"].abs().mean(),
        "RMSE": np.sqrt((g["err"] ** 2).mean()),
        # MAE relative to the vol level: crises have higher vol, so raw errors
        # are naturally bigger. This makes calm and crisis periods comparable.
        "rel_MAE": g["err"].abs().mean() / g["rv_next"].mean(),
        "mean_bias": g["err"].mean(),      # negative = under-forecasting on average
    })

print("\nNaive benchmark, all tickers pooled:")
print(df.groupby("regime").apply(summarise).round(3))

print("\nNaive benchmark, S&P 500 only:")
print(df[df["ticker"] == "SPX"].groupby("regime").apply(summarise).round(3))

df.to_csv("naive_results.csv", index=False)
