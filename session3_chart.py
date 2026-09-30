# =============================================================================
# Session 3: the headline chart and one headline number.
#
# Setup (terminal, with volenv active):   pip install matplotlib
# Run:                                    python session3_chart.py
# Output: volatility_chart.png in the same folder.
# =============================================================================

# %% 1. Load the forecasts saved at the end of Session 2
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model_results.csv", parse_dates=["month", "target_month"])

# %% 2. Headline number: how much of the total error comes from one month?
# Squared errors punish big misses heavily, so they show whether the error is
# spread evenly or concentrated in a few extreme months (Taleb's point).
test = df[df["target_month"] >= "2020-01-01"].copy()     # out-of-sample only
is_march = test["target_month"] == "2020-03-31"

print("Out-of-sample 2020-2025, all 8 tickers:")
for name in ["naive", "model"]:
    sq_err = (test[name] - test["actual"]) ** 2
    share = sq_err[is_march].sum() / sq_err.sum()
    print(f"  {name}: March 2020 is {is_march.mean():.1%} of forecasts "
          f"but {share:.1%} of total squared error")

# %% 3. The chart: S&P 500, actual vs. both forecasts, 2018-2025
spx = df[df["ticker"] == "SPX"].set_index("target_month").sort_index().loc["2018":]

# Wide and short, with large text, so it stays readable when scaled to fit a page.
plt.rcParams["font.size"] = 12
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(spx.index, spx["actual"] * 100, color="black", lw=2, label="Actual volatility")
ax.plot(spx.index, spx["naive"] * 100, color="tab:orange", lw=1.3, ls="--",
        label='Naive: "next month = this month"')
ax.plot(spx.index, spx["model"] * 100, color="tab:blue", lw=1.6,
        label="Model (trained on 2005-2019)")

# Shade the two stress periods and mark where out-of-sample testing begins
ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-15"), color="red", alpha=0.12,
           label="COVID crash")
ax.axvspan(pd.Timestamp("2022-01-15"), pd.Timestamp("2022-12-15"), color="grey", alpha=0.15,
           label="2022 selloff")
ax.axvline(pd.Timestamp("2020-01-15"), color="grey", lw=1, ls=":")
ax.text(pd.Timestamp("2020-01-01"), 97, "out-of-sample →", ha="right", fontsize=11, color="grey")

# Point at the month that matters most
m = pd.Timestamp("2020-03-31")
ax.annotate(f"March 2020: actual {spx.loc[m, 'actual']:.0%}\n"
            f"naive forecast {spx.loc[m, 'naive']:.0%}, model {spx.loc[m, 'model']:.0%}",
            xy=(m, spx.loc[m, "actual"] * 100), xytext=(pd.Timestamp("2020-09-30"), 55),
            fontsize=11, arrowprops=dict(arrowstyle="->", color="black"))

ax.set_title("S&P 500 one-month volatility forecasts: fine in calm markets, blind to the shock",
             fontsize=14, loc="left")
ax.set_ylabel("Annualized volatility (%)")
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=11, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Data: Yahoo Finance daily closes. Realized volatility = std. dev. of daily "
         "log returns in the month, annualized.", fontsize=10, color="grey")

fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("volatility_chart.png", dpi=200)
print("\nSaved volatility_chart.png")
