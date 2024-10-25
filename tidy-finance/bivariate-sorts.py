import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

# =============================================================================
# Data Preparation
# =============================================================================

# Establish a connection to the SQLite database
tidy_finance = sqlite3.connect(database="data/tidy_finance_python.sqlite")

# Load the CRSP monthly data
crsp_monthly = (pd.read_sql_query(
    sql=("SELECT permno, gvkey, date, ret_excess, mktcap, " 
         "mktcap_lag, exchange FROM crsp_monthly"),
    con=tidy_finance,
    parse_dates={"date"})
  .dropna()
)

# Load book equity data from Compustat
book_equity = (pd.read_sql_query(
    sql="SELECT gvkey, datadate, be FROM compustat",
    con=tidy_finance, 
    parse_dates={"datadate"})
  .dropna()
  .assign(
    # Convert 'datadate' to its monthly value by first converting to period ('M') and then back to timestamp
    date=lambda x: (
      pd.to_datetime(x["datadate"]).dt.to_period("M").dt.to_timestamp() 
    )
  )
)

# =============================================================================
# Book-to-Market Ratio
# =============================================================================

# Create the market equity (me) variable by lagging the CRSP market cap (mktcap) by one month
me = (crsp_monthly
  .assign(sorting_date=lambda x: x["date"] + pd.DateOffset(months=1))  # Increase date by 1 month to create sorting_date
  .rename(columns={"mktcap": "me"})  # Rename 'mktcap' to 'me' (market equity)
  .get(["permno", "sorting_date", "me"])  # Keep only permno, sorting_date, and market equity (me)
)

# Merge the book equity data from Compustat with CRSP market equity to calculate the book-to-market ratio (bm)
bm = (book_equity
  .merge(crsp_monthly, how="inner", on=["gvkey", "date"])  # Merge Compustat book equity (be) with CRSP monthly data on gvkey and date
  .assign(bm=lambda x: x["be"]/x["mktcap"],  # Calculate the book-to-market ratio (bm = be / me)
          sorting_date=lambda x: x["date"]+pd.DateOffset(months=6))  # Lag the book-to-market ratio by 6 months to prevent look-ahead bias
  .assign(comp_date=lambda x: x["sorting_date"])  # Store the 'sorting_date' as 'comp_date' for comparison later
  .get(["permno", "gvkey", "sorting_date", "comp_date", "bm"])  # Keep relevant columns for further processing
)

# Merge the CRSP monthly data with the book-to-market (bm) and market equity (me) data
data_for_sorts = (crsp_monthly
  .merge(bm, 
         how="left", 
         left_on=["permno", "gvkey", "date"], 
         right_on=["permno", "gvkey", "sorting_date"])
  .merge(me, 
         how="left", 
         left_on=["permno", "date"], 
         right_on=["permno", "sorting_date"])
  .get(["permno", "gvkey", "date", "ret_excess", 
        "mktcap_lag", "me", "bm", "exchange", "comp_date"])
)

# Sort data by firm and date, then carry forward the latest book-to-market ratio (bm) and accounting date (comp_date)
data_for_sorts = (data_for_sorts
  .sort_values(by=["permno", "gvkey", "date"])  # Sort data by permno, gvkey, and date
  .groupby(["permno", "gvkey"])  # Group by permno and gvkey to handle each firm separately
  .apply(lambda x: x.assign(
      bm=x["bm"].fillna(method="ffill"),  # Forward-fill missing book-to-market (bm) values with the most recent available value
      comp_date=x["comp_date"].fillna(method="ffill")  # Forward-fill missing comp_date values
    )
  )
  .reset_index(drop=True)  # Reset the index after applying the group operation
  .assign(threshold_date = lambda x: (x["date"]-pd.DateOffset(months=12)))  # Create a threshold date (1 year before current date)
  .query("comp_date > threshold_date")  # Filter out rows where the book-to-market ratio is more than 12 months old
  .drop(columns=["comp_date", "threshold_date"])  # Drop temporary columns used for filtering
  .dropna()  # Remove rows with any remaining missing data
)

# Define a function to assign portfolios based on breakpoints for a given sorting variable (e.g., size, book-to-market)
def assign_portfolio(data, exchanges, sorting_variable, n_portfolios):
    """Assign portfolio for a given sorting variable."""
    
    # Calculate the breakpoints for the sorting variable within the specified exchanges
    breakpoints = (data
      .query(f"exchange in {exchanges}")  # Filter data for the specified exchanges (e.g., NYSE, AMEX)
      .get(sorting_variable)  # Get the values for the sorting variable (e.g., size or book-to-market)
      .quantile(np.linspace(0, 1, num=n_portfolios+1),  # Calculate breakpoints for the sorting variable
                interpolation="linear")
      .drop_duplicates()  # Remove any duplicate breakpoints to ensure distinct portfolio ranges
    )
    # Set the first breakpoint to -Inf and the last breakpoint to +Inf for complete coverage
    breakpoints.iloc[0] = -np.Inf
    breakpoints.iloc[breakpoints.size-1] = np.Inf
    
    # Assign each firm to a portfolio based on the sorting variable and calculated breakpoints
    assigned_portfolios = pd.cut(
      data[sorting_variable],  # Use the sorting variable to assign firms to portfolios
      bins=breakpoints,  # Use the calculated breakpoints to define portfolio ranges
      labels=range(1, breakpoints.size),  # Label the portfolios from 1 to n_portfolios
      include_lowest=True,  # Include the lowest value in the first portfolio
      right=False  # The intervals are left-closed (i.e., [a, b))
    )
    
    return assigned_portfolios  # Return the assigned portfolios

# =============================================================================
# Independent Sorts
# =============================================================================

# Assign portfolios for book-to-market (bm) and market equity (me) independently, then combine them
value_portfolios = (data_for_sorts
  .groupby("date")  # Group data by 'date' to create portfolios for each month
  .apply(lambda x: x.assign(  # Assign portfolios for each month
      portfolio_bm=assign_portfolio(  # Assign portfolios based on book-to-market ratio (bm)
        data=x, sorting_variable="bm", n_portfolios=5, exchanges=["NYSE"]  # Sort firms into 5 portfolios based on bm, using NYSE as the exchange
      ),
      portfolio_me=assign_portfolio(  # Assign portfolios based on market equity (me)
        data=x, sorting_variable="me", n_portfolios=5, exchanges=["NYSE"]  # Sort firms into 5 portfolios based on me, using NYSE as the exchange
      )
    )
  )
  .reset_index(drop=True)  # Reset the index after applying the group operation
  .groupby(["date", "portfolio_bm", "portfolio_me"])  # Group data by date, portfolio_bm (book-to-market), and portfolio_me (size)
  .apply(lambda x: pd.Series({  # Compute the weighted average return within each portfolio
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])  # Value-weighted average return using lagged market cap (mktcap_lag)
    })
  )
  .reset_index()
)

# Compute the value premium, which is the return differential between the highest and lowest book-to-market portfolios
value_premium = (value_portfolios
  .groupby(["date", "portfolio_bm"])  # Group by date and book-to-market portfolios (bm)
  .aggregate({"ret": "mean"})  # Calculate the mean return for each book-to-market portfolio across different size portfolios
  .reset_index()  # Reset the index to prepare for further grouping
  .groupby("date")  # Group by date to compute the value premium for each month
  .apply(lambda x: pd.Series({
    "value_premium": (
        x.loc[x["portfolio_bm"] == x["portfolio_bm"].max(), "ret"].mean() -  # Find the mean return of the highest bm portfolio
          x.loc[x["portfolio_bm"] == x["portfolio_bm"].min(), "ret"].mean()  # Subtract the mean return of the lowest bm portfolio
      )
    })
  )
  .aggregate({"value_premium": "mean"})  # Calculate the mean value premium across all months
)

# =============================================================================
# Dependent Sorts
# =============================================================================

# First, assign firms to size (me) portfolios using the assign_portfolio function
value_portfolios = (data_for_sorts
  .groupby("date")  # Group data by 'date' to assign portfolios separately for each month
  .apply(lambda x: x.assign(  # Apply portfolio assignment for each month
      portfolio_me=assign_portfolio(  # Assign portfolios based on market equity (me)
        data=x, sorting_variable="me", n_portfolios=5, exchanges=["NYSE"]  # Create 5 size portfolios, using NYSE stocks as the sample
      )
    )
  )
  .reset_index(drop=True)  # Reset index after applying the assignment
  
  # Now, within each size portfolio, assign book-to-market (bm) portfolios
  .groupby(["date", "portfolio_me"])  # Group by 'date' and 'portfolio_me' to assign bm portfolios within each size bucket
  .apply(lambda x: x.assign(
      portfolio_bm=assign_portfolio(  # Assign portfolios based on book-to-market (bm) within each size portfolio
        data=x, sorting_variable="bm", n_portfolios=5, exchanges=["NYSE"]  # Create 5 book-to-market portfolios within each size bucket
      )
    )
  )
  .reset_index(drop=True)  # Reset index after applying the assignment
  
  # Group by date, book-to-market portfolio (bm), and size portfolio (me), then compute value-weighted portfolio returns
  .groupby(["date", "portfolio_bm", "portfolio_me"])  # Group by date, bm portfolio, and me portfolio
  .apply(lambda x: pd.Series({
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])  # Calculate the value-weighted return using lagged market cap
    })
  )
  .reset_index()  # Reset index after the calculation
)

# Compute the value premium by comparing the returns of the highest and lowest book-to-market portfolios
value_premium = (value_portfolios
  .groupby(["date", "portfolio_bm"])  # Group by date and bm portfolios
  .aggregate({"ret": "mean"})  # Calculate the mean return for each book-to-market portfolio
  .reset_index()  # Reset the index to prepare for the next group operation
  .groupby("date")  # Group by date to compute the value premium for each month
  .apply(lambda x: pd.Series({
    "value_premium": (
        x.loc[x["portfolio_bm"] == x["portfolio_bm"].max(), "ret"].mean() -  # Calculate the return of the highest book-to-market portfolio
          x.loc[x["portfolio_bm"] == x["portfolio_bm"].min(), "ret"].mean()  # Subtract the return of the lowest book-to-market portfolio
      )
    })
  )
  .aggregate({"value_premium": "mean"})  # Calculate the mean value premium across all months
)
