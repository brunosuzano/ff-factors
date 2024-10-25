import pandas as pd
import numpy as np
import sqlite3
import statsmodels.formula.api as smf

from regtabletotext import prettify_result

# =============================================================================
# Data Preparation
# =============================================================================

# Connect to the SQLite database containing the financial data
tidy_finance = sqlite3.connect(
  database="data/tidy_finance_python.sqlite"
)

# Load the CRSP monthly data, including permno, gvkey, date, returns, and market cap
crsp_monthly = (pd.read_sql_query(
    sql=("SELECT permno, gvkey, date, ret_excess, mktcap, "
         "mktcap_lag, exchange FROM crsp_monthly"),
    con=tidy_finance,
    parse_dates={"date"})
  .dropna()
)

# Load the Compustat data, including book equity, operating profitability, and investment
compustat = (pd.read_sql_query(
    sql="SELECT gvkey, datadate, be, op, inv FROM compustat",
    con=tidy_finance,
    parse_dates={"datadate"})
  .dropna()
)

# Load the Fama-French 3-factor data (monthly)
factors_ff3_monthly = pd.read_sql_query(
  sql="SELECT date, smb, hml FROM factors_ff3_monthly",
  con=tidy_finance,
  parse_dates={"date"}
)

# Load the Fama-French 5-factor data (monthly)
factors_ff5_monthly = pd.read_sql_query(
  sql="SELECT date, smb, hml, rmw, cma FROM factors_ff5_monthly",
  con=tidy_finance,
  parse_dates={"date"}
)

# =============================================================================

# Prepare the firm size (me) variable by taking market cap data from June of each year 
# and lagging it by one month to use for sorting portfolios
size = (crsp_monthly
  .query("date.dt.month == 6")  # Select only the June data for each year
  .assign(sorting_date=lambda x: (x["date"]+pd.DateOffset(months=1)))  # Increase the date by 1 month to create sorting_date
  .get(["permno", "exchange", "sorting_date", "mktcap"])  # Keep relevant columns: permno, exchange, sorting_date, and mktcap
  .rename(columns={"mktcap": "size"})  # Rename 'mktcap' to 'size'
)

# Prepare the market equity (me) variable by taking market cap data from December and increasing its date by 7 months
market_equity = (crsp_monthly
  .query("date.dt.month == 12")  # Select only the December data
  .assign(sorting_date=lambda x: (x["date"]+pd.DateOffset(months=7)))  # Increase the date by 7 months to create sorting_date
  .get(["permno", "gvkey", "sorting_date", "mktcap"])  # Keep relevant columns: permno, gvkey, sorting_date, and mktcap
  .rename(columns={"mktcap": "me"})  # Rename 'mktcap' to 'me' (market equity)
)

# Compute the book-to-market (bm) ratio using the book equity from Compustat and market equity from CRSP
book_to_market = (compustat
  .assign(
    # Create a sorting_date by lagging book equity by one year, ensuring it's aligned with market equity
    sorting_date=lambda x: (pd.to_datetime(
      (x["datadate"].dt.year+1).astype(str)+"0701", format="%Y%m%d")
    )
  )
  # Merge the book equity (Compustat) with the market equity (CRSP) based on gvkey and sorting_date
  .merge(market_equity, how="inner", on=["gvkey", "sorting_date"])
  .assign(bm=lambda x: x["be"]/x["me"])  # Calculate the book-to-market ratio (bm = be / me)
  .get(["permno", "sorting_date", "me", "bm"])  # Keep relevant columns: permno, sorting_date, me, and bm
)

# Combine size and book-to-market (bm) variables for portfolio sorting, ensuring no missing values and no duplicate entries
sorting_variables = (size
  .merge(book_to_market, how="inner", on=["permno", "sorting_date"])
  .dropna()
  .drop_duplicates(subset=["permno", "sorting_date"])
 )

# =============================================================================
# Portfolio Sorts
# =============================================================================

# The function assigns portfolios based on NYSE-specific breakpoints
def assign_portfolio(data, sorting_variable, percentiles):
    """Assign portfolios to a bin according to a sorting variable."""
    
    # Compute breakpoints for NYSE stocks based on the specified percentiles
    breakpoints = (data
      .query("exchange == 'NYSE'")  # Use only NYSE stocks to calculate breakpoints
      .get(sorting_variable)  # Get the values of the sorting variable (either 'size' or 'bm')
      .quantile(percentiles, interpolation="linear")  # Calculate breakpoints using the given percentiles
    )
    # Set the first breakpoint to -Inf and the last breakpoint to +Inf for complete coverage
    breakpoints.iloc[0] = -np.Inf
    breakpoints.iloc[breakpoints.size-1] = np.Inf
    
    # Assign each firm to a portfolio based on the sorting variable and calculated breakpoints
    assigned_portfolios = pd.cut(
      data[sorting_variable],  # The sorting variable (either 'size' or 'bm')
      bins=breakpoints,  # Use the calculated breakpoints to define portfolio bins
      labels=pd.Series(range(1, breakpoints.size)),  # Label the portfolios from 1 to the number of breakpoints
      include_lowest=True,  # Include the lowest value in the first portfolio
      right=False  # The intervals are left-closed (i.e., [a, b))
    )
    
    return assigned_portfolios  # Return the assigned portfolios

# Sort firms into size and book-to-market portfolios based on the computed breakpoints
portfolios = (sorting_variables
  .groupby("sorting_date")  # Group data by sorting_date (which ensures portfolios are formed each year)
  .apply(lambda x: x
    .assign(
      # Assign firms to 2 size portfolios (based on the median NYSE size)
      portfolio_size=assign_portfolio(x, "size", [0, 0.5, 1]),  # Split at the median for size
      # Assign firms to 3 book-to-market portfolios (low, neutral, high)
      portfolio_bm=assign_portfolio(x, "bm", [0, 0.3, 0.7, 1])  # Use 30th and 70th percentiles for bm
    )
  )
  .reset_index(drop=True)  # Reset the index after applying the portfolio assignment
  .get(["permno", "sorting_date", "portfolio_size", "portfolio_bm"])  # Keep only relevant columns for further analysis
)

# Adjust the 'sorting_date' in the CRSP data based on the portfolio formation rule
# The sorting_date is set to July 1st of the previous year if the month is June or earlier,
# and July 1st of the current year if the month is July or later.
portfolios = (crsp_monthly
  .assign(
    # Create the 'sorting_date' column, which is July 1st of the prior year if the date is in June or earlier, 
    # and July 1st of the current year if the date is July or later.
    sorting_date=lambda x: (pd.to_datetime(
      x["date"].apply(lambda x: str(x.year-1)+
        "0701" if x.month <= 6 else str(x.year)+"0701")))
  )
  # Merge the computed portfolios with CRSP monthly data using 'permno' and 'sorting_date'
  .merge(portfolios, how="inner", on=["permno", "sorting_date"])
)

# =============================================================================
# Fama-French Three-Factor Model
# =============================================================================

# Calculate the value-weighted average return for each of the six portfolios
# Portfolio groups are defined by size (portfolio_size) and book-to-market (portfolio_bm)
factors_replicated = (portfolios
  .groupby(["portfolio_size", "portfolio_bm", "date"])
  .apply(lambda x: pd.Series({
    # Calculate the value-weighted average return for each portfolio using lagged market capitalization as weights
    "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])
    })
   )
  .reset_index()  # Reset the index after applying the group operation
  .groupby("date")  # Group data by date to compute the factors for each month
  .apply(lambda x: pd.Series({
    # SMB: Long the small portfolios (portfolio_size == 1) and short the large portfolios (portfolio_size == 2)
    "smb_replicated": (
      x["ret"][x["portfolio_size"] == 1].mean() -  # Average return of the small portfolios
        x["ret"][x["portfolio_size"] == 2].mean()),  # Subtract the average return of the large portfolios
    
    # HML: Long the high book-to-market portfolios (portfolio_bm == 3) and short the low book-to-market portfolios (portfolio_bm == 1)
    "hml_replicated": (
      x["ret"][x["portfolio_bm"] == 3].mean() -  # Average return of the high book-to-market portfolios
        x["ret"][x["portfolio_bm"] == 1].mean())  # Subtract the average return of the low book-to-market portfolios
    }))
  .reset_index()  # Reset the index after applying the group operation
)

# Merge the replicated factors with the original Fama-French three-factor data (smb and hml)
factors_replicated = (factors_replicated
  .merge(factors_ff3_monthly, how="inner", on="date")
  .round(4)  # Round the results to 4 decimal places for presentation
)

# =============================================================================
# Replication Evaluation
# =============================================================================

# We run an OLS regression to assess how well we replicated the SMB factor.
# The dependent variable is the original SMB factor ('smb'), and the independent variable 
# is the replicated SMB factor ('smb_replicated').
model_smb = (smf.ols(
    formula="smb ~ smb_replicated",  # OLS regression with smb as the dependent variable and smb_replicated as the independent variable
    data=factors_replicated
  )
  .fit()
)
prettify_result(model_smb)

# We run an OLS regression to assess how well we replicated the HML factor.
# The dependent variable is the original HML factor ('hml'), and the independent variable 
# is the replicated HML factor ('hml_replicated').
model_hml = (smf.ols(
    formula="hml ~ hml_replicated",  # OLS regression with hml as the dependent variable and hml_replicated as the independent variable
    data=factors_replicated
  )
  .fit()
)
prettify_result(model_hml)

# =============================================================================

# Define the start and end dates
start_date = "1970-07-01"
end_date = "2017-12-31"

# Subset the factors_replicated DataFrame for the specified date range
factors_subset = factors_replicated.query("date >= @start_date and date <= @end_date")

# Calculate the correlation between the original and replicated SMB factors
smb_correlation = factors_replicated["smb"].corr(factors_replicated["smb_replicated"])
print(f"Correlation between original SMB and replicated SMB: {smb_correlation:.4f}")

# Calculate the correlation between the original and replicated HML factors
hml_correlation = factors_replicated["hml"].corr(factors_replicated["hml_replicated"])
print(f"Correlation between original HML and replicated HML: {hml_correlation:.4f}")

# =============================================================================

import matplotlib.pyplot as plt

# Plotting the SMB factor
plt.figure(figsize=(10, 6))
plt.plot(factors_replicated['date'], factors_replicated['smb'], label='Original SMB', color='blue', linestyle='-')
plt.plot(factors_replicated['date'], factors_replicated['smb_replicated'], label='Replicated SMB', color='red', linestyle='--')
plt.title('Comparison of Original and Replicated SMB Factor')
plt.xlabel('Date')
plt.ylabel('SMB')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the HML factor
plt.figure(figsize=(10, 6))
plt.plot(factors_replicated['date'], factors_replicated['hml'], label='Original HML', color='blue', linestyle='-')
plt.plot(factors_replicated['date'], factors_replicated['hml_replicated'], label='Replicated HML', color='red', linestyle='--')
plt.title('Comparison of Original and Replicated HML Factor')
plt.xlabel('Date')
plt.ylabel('HML')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# Fama-French Five-Factor Model
# =============================================================================

# Extend the table with additional variables: operating profitability (op) and investment (inv)
# Sorting date is set to July of the next year as a typical Fama-French convention to avoid look-ahead bias
other_sorting_variables = (compustat
  .assign(
    sorting_date=lambda x: (pd.to_datetime(
      (x["datadate"].dt.year+1).astype(str)+"0701", format="%Y%m%d")  # Set sorting date to July 1 of the following year
    )
  )
  .merge(market_equity, how="inner", on=["gvkey", "sorting_date"])  # Merge with market equity (me)
  .assign(bm=lambda x: x["be"]/x["me"])  # Calculate book-to-market ratio (bm = be / me)
  .get(["permno", "sorting_date", "me", "bm", "op", "inv"])  # Select necessary columns
)

# Merge size and other sorting variables into a single table and remove duplicates and missing values
sorting_variables = (size
  .merge(other_sorting_variables, how="inner", on=["permno", "sorting_date"])
  .dropna()
  .drop_duplicates(subset=["permno", "sorting_date"])
 )

# Assign size portfolios independently
portfolios = (sorting_variables
  .groupby("sorting_date")
  .apply(lambda x: x
    .assign(
      portfolio_size=assign_portfolio(x, "size", [0, 0.5, 1])  # Assign size portfolios based on 50th percentile breakpoints
    )
  )
  .reset_index(drop=True)
  .groupby(["sorting_date", "portfolio_size"])  # Group by sorting date and size portfolio
  .apply(lambda x: x
    .assign(
      portfolio_bm=assign_portfolio(x, "bm", [0, 0.3, 0.7, 1]),  # Assign book-to-market portfolios (value)
      portfolio_op=assign_portfolio(x, "op", [0, 0.3, 0.7, 1]),  # Assign profitability portfolios
      portfolio_inv=assign_portfolio(x, "inv", [0, 0.3, 0.7, 1])  # Assign investment portfolios
    )
  )
  .reset_index(drop=True)
  .get(["permno", "sorting_date", 
        "portfolio_size", "portfolio_bm",
        "portfolio_op", "portfolio_inv"])
)

# Merge the portfolios back with the CRSP monthly return data
portfolios = (crsp_monthly
  .assign(
    sorting_date=lambda x: (pd.to_datetime(
      x["date"].apply(lambda x: str(x.year-1)+
        "0701" if x.month <= 6 else str(x.year)+"0701")))  # Adjust the sorting date to July of current year
  )
  .merge(portfolios, how="inner", on=["permno", "sorting_date"])
)

# =============================================================================

# Compute value-weighted returns for the portfolios based on book-to-market (HML)
portfolios_value = (portfolios
  .groupby(["portfolio_size", "portfolio_bm", "date"])  # Group by size, book-to-market portfolio, and date
  .apply(lambda x: pd.Series({
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])  # Calculate value-weighted returns for each portfolio
    })
  )
  .reset_index()
)

# Construct the value factor (HML): high minus low book-to-market
factors_value = (portfolios_value
  .groupby("date")
  .apply(lambda x: pd.Series({
    "hml_replicated": (
      x["ret"][x["portfolio_bm"] == 3].mean() -  # Long high book-to-market portfolios
        x["ret"][x["portfolio_bm"] == 1].mean())})  # Short low book-to-market portfolios
  )
  .reset_index()
)

# =============================================================================

# Compute value-weighted returns for profitability portfolios (RMW)
portfolios_profitability = (portfolios
  .groupby(["portfolio_size", "portfolio_op", "date"])  # Group by size, profitability, and date
  .apply(lambda x: pd.Series({
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])  # Calculate value-weighted returns
    })
  )
  .reset_index()
)

# Construct the profitability factor (RMW): robust minus weak profitability
factors_profitability = (portfolios_profitability
  .groupby("date")
  .apply(lambda x: pd.Series({
    "rmw_replicated": (
      x["ret"][x["portfolio_op"] == 3].mean() -  # Long high profitability portfolios
        x["ret"][x["portfolio_op"] == 1].mean())})  # Short low profitability portfolios
  )
  .reset_index()
)

# =============================================================================

# Compute value-weighted returns for investment portfolios (CMA)
portfolios_investment = (portfolios
  .groupby(["portfolio_size", "portfolio_inv", "date"])  # Group by size, investment, and date
  .apply(lambda x: pd.Series({
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])  # Calculate value-weighted returns
    })
  )
  .reset_index()
)

# Construct the investment factor (CMA): conservative minus aggressive
factors_investment = (portfolios_investment
  .groupby("date")
  .apply(lambda x: pd.Series({
    "cma_replicated": (
      x["ret"][x["portfolio_inv"] == 1].mean() -  # Long low investment portfolios
        x["ret"][x["portfolio_inv"] == 3].mean())})  # Short high investment portfolios
  )
  .reset_index()
)

# =============================================================================

# Combine all portfolios and compute the size factor (SMB)
factors_size = (
  pd.concat(  # Concatenate the value, profitability, and investment portfolios
    [portfolios_value, portfolios_profitability, portfolios_investment], 
    ignore_index=True
  )
  .groupby("date")
  .apply(lambda x: pd.Series({
    "smb_replicated": (
      x["ret"][x["portfolio_size"] == 1].mean() -  # Long small portfolios
        x["ret"][x["portfolio_size"] == 2].mean())})  # Short large portfolios
  )
  .reset_index()
)

# =============================================================================

# Merge the size, value, profitability, and investment factors into a single DataFrame
factors_replicated = (factors_size
  .merge(factors_value, how="outer", on="date")
  .merge(factors_profitability, how="outer", on="date")
  .merge(factors_investment, how="outer", on="date")
)

# Merge with the original Fama-French five-factor data for comparison
factors_replicated = (factors_replicated
  .merge(factors_ff5_monthly, how="inner", on="date")
  .round(4)  # Round values to 4 decimal places
)

# Run regression to compare the replicated SMB with the original SMB
model_smb = (smf.ols(
    formula="smb ~ smb_replicated",  # OLS regression of original SMB on replicated SMB
    data=factors_replicated
  )
  .fit()
)
prettify_result(model_smb)

# Run regression to compare the replicated HML with the original HML
model_hml = (smf.ols(
    formula="hml ~ hml_replicated",  # OLS regression of original HML on replicated HML
    data=factors_replicated
  )
  .fit()
)
prettify_result(model_hml)

# Run regression to compare the replicated RMW with the original RMW
model_rmw = (smf.ols(
    formula="rmw ~ rmw_replicated",  # OLS regression of original RMW on replicated RMW
    data=factors_replicated
  )
  .fit()
)
prettify_result(model_rmw)

# Run regression to compare the replicated CMA with the original CMA
model_cma = (smf.ols(
    formula="cma ~ cma_replicated",  # OLS regression of original CMA on replicated CMA
    data=factors_replicated
  )
  .fit()
)
prettify_result(model_cma)

# =============================================================================

# Define the start and end dates
start_date = "1966-07-01"
end_date = "2021-12-01"

# Subset the factors_replicated DataFrame for the specified date range
factors_subset = factors_replicated.query("date >= @start_date and date <= @end_date")

# Calculate the correlation between the original and replicated SMB factors
smb_correlation = factors_replicated["smb"].corr(factors_replicated["smb_replicated"])
print(f"Correlation between original SMB and replicated SMB: {smb_correlation:.4f}")

# Calculate the correlation between the original and replicated HML factors
hml_correlation = factors_replicated["hml"].corr(factors_replicated["hml_replicated"])
print(f"Correlation between original HML and replicated HML: {hml_correlation:.4f}")

# Calculate the correlation between the original and replicated RMW factors
rmw_correlation = factors_replicated["rmw"].corr(factors_replicated["rmw_replicated"])
print(f"Correlation between original RMW and replicated RMW: {rmw_correlation:.4f}")

# Calculate the correlation between the original and replicated CMA factors
cma_correlation = factors_replicated["cma"].corr(factors_replicated["cma_replicated"])
print(f"Correlation between original CMA and replicated CMA: {cma_correlation:.4f}")

# =============================================================================

import matplotlib.pyplot as plt

# Plotting the SMB factor
plt.figure(figsize=(10, 6))
plt.plot(factors_replicated['date'], factors_replicated['smb'], label='Original SMB', color='blue', linestyle='-')
plt.plot(factors_replicated['date'], factors_replicated['smb_replicated'], label='Replicated SMB', color='red', linestyle='--')
plt.title('Comparison of Original and Replicated SMB Factor')
plt.xlabel('Date')
plt.ylabel('SMB')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the HML factor
plt.figure(figsize=(10, 6))
plt.plot(factors_replicated['date'], factors_replicated['hml'], label='Original HML', color='blue', linestyle='-')
plt.plot(factors_replicated['date'], factors_replicated['hml_replicated'], label='Replicated HML', color='red', linestyle='--')
plt.title('Comparison of Original and Replicated HML Factor')
plt.xlabel('Date')
plt.ylabel('HML')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the RMW factor
plt.figure(figsize=(10, 6))
plt.plot(factors_replicated['date'], factors_replicated['rmw'], label='Original RMW', color='blue', linestyle='-')
plt.plot(factors_replicated['date'], factors_replicated['rmw_replicated'], label='Replicated RMW', color='red', linestyle='--')
plt.title('Comparison of Original and Replicated RMW Factor')
plt.xlabel('Date')
plt.ylabel('RMW')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the CMA factor
plt.figure(figsize=(10, 6))
plt.plot(factors_replicated['date'], factors_replicated['cma'], label='Original CMA', color='blue', linestyle='-')
plt.plot(factors_replicated['date'], factors_replicated['cma_replicated'], label='Replicated CMA', color='red', linestyle='--')
plt.title('Comparison of Original and Replicated CMA Factor')
plt.xlabel('Date')
plt.ylabel('CMA')
plt.legend()
plt.tight_layout()
plt.show()
