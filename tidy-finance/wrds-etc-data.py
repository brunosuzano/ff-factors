import pandas as pd
import numpy as np
import sqlite3

from plotnine import *
from mizani.formatters import comma_format, percent_format
from datetime import datetime

# Sample period
start_date = "01/01/1960"
end_date = "06/30/2024"

# =============================================================================
# Accessing WRDS
# =============================================================================

from sqlalchemy import create_engine

import os
from dotenv import load_dotenv
load_dotenv()

# Construct the connection string using credentials stored in environment variables
connection_string = (
  "postgresql+psycopg2://"                 # Specify the database dialect (PostgreSQL) and driver (psycopg2)
 f"{os.getenv('WRDS_USER')}:{os.getenv('WRDS_PASSWORD')}"  # Retrieve WRDS username and password from environment variables
  "@wrds-pgdata.wharton.upenn.edu:9737/wrds"  # Define the WRDS server address, port, and database
)

# Create a SQLAlchemy engine to connect to the WRDS PostgreSQL database
wrds = create_engine(connection_string, pool_pre_ping=True)  # pool_pre_ping ensures connections are valid before being used

# =============================================================================
# Downloading and Preparing CRSP
# =============================================================================

# Construct the SQL query to retrieve CRSP monthly stock data
crsp_monthly_query = (
  "SELECT msf.permno, date_trunc('month', msf.mthcaldt)::date AS date, "  # Select stock identifier (permno) and truncating the date to month level
         "msf.mthret AS ret, msf.shrout, msf.mthprc AS altprc, "  # Select monthly return (ret), shares outstanding (shrout), and monthly price (altprc)
         "ssih.primaryexch, ssih.siccd "  # Select the primary exchange and SIC code (industry classification)
    "FROM crsp.msf_v2 AS msf "  # From the CRSP monthly stock file (msf_v2)
    "INNER JOIN crsp.stksecurityinfohist AS ssih "  # Inner join with the CRSP security information history table
    "ON msf.permno = ssih.permno AND "  # Join on permno (unique stock identifier)
       "ssih.secinfostartdt <= msf.mthcaldt AND "  # Ensure the security info start date is before or on the monthly date
       "msf.mthcaldt <= ssih.secinfoenddt "  # Ensure the security info end date is after or on the monthly date
   f"WHERE msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}' "  # Filter by the date range between start_date and end_date
          "AND ssih.sharetype = 'NS' "  # Filter by share type: non-split-adjusted shares
          "AND ssih.securitytype = 'EQTY' "  # Filter by security type: equity
          "AND ssih.securitysubtype = 'COM' "  # Filter by security subtype: common stocks
          "AND ssih.usincflg = 'Y' "  # Filter for U.S.-incorporated companies
          "AND ssih.issuertype in ('ACOR', 'CORP') "  # Filter for corporate issuers (associations and corporations)
          "AND ssih.primaryexch in ('N', 'A', 'Q') "  # Filter for exchanges: NYSE (N), AMEX (A), NASDAQ (Q)
          "AND ssih.conditionaltype in ('RW', 'NW') "  # Filter for real-world (RW) or non-worldwide (NW) types
          "AND ssih.tradingstatusflg = 'A'"  # Filter for active trading status
)

# Execute the SQL query and load the results into a pandas DataFrame
crsp_monthly = (pd.read_sql_query(
    sql=crsp_monthly_query,  # SQL query constructed above
    con=wrds,  # Database connection (WRDS PostgreSQL)
    dtype={"permno": int, "siccd": int},  # Specify data types for 'permno' and 'siccd' as integers
    parse_dates={"date"})  # Parse the 'date' column as datetime for easier date manipulation
  .assign(shrout=lambda x: x["shrout"]*1000)  # Convert 'shrout' from thousands of shares to actual shares (multiplying by 1000)
)

# Calculate market capitalization in millions and replace zero market cap with NaN
crsp_monthly = (crsp_monthly
  .assign(mktcap=lambda x: x["shrout"] * x["altprc"] / 1000000)  # Market capitalization: shares outstanding * price, converted to millions
  .assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))  # Replace zero market cap values with NaN to handle missing or invalid data
)

# Create a lagged market capitalization column (mktcap_lag) with a 1-month offset
# (useful for calculating value-weighted returns later on)
mktcap_lag = (crsp_monthly
  .assign(
    date=lambda x: x["date"] + pd.DateOffset(months=1),  # Shift the date by 1 month to create a lag effect
    mktcap_lag=lambda x: x["mktcap"]  # Create the lagged market capitalization column
  )
  .get(["permno", "date", "mktcap_lag"])  # Keep only 'permno', 'date', and 'mktcap_lag' columns
)

# Merge the original dataset with the lagged market capitalization values
crsp_monthly = (crsp_monthly
  .merge(mktcap_lag, how="left", on=["permno", "date"])
)

# Function to assign exchange names based on primary exchange codes
def assign_exchange(primaryexch):
    if primaryexch == "N":
        return "NYSE"
    elif primaryexch == "A":
        return "AMEX"
    elif primaryexch == "Q":
        return "NASDAQ"
    else:
        return "Other"

# Apply the assign_exchange function to the 'primaryexch' column
crsp_monthly["exchange"] = (crsp_monthly["primaryexch"]
  .apply(assign_exchange)
)

# Function to assign industries based on SIC codes
def assign_industry(siccd):
    if 1 <= siccd <= 999:
        return "Agriculture"
    elif 1000 <= siccd <= 1499:
        return "Mining"
    elif 1500 <= siccd <= 1799:
        return "Construction"
    elif 2000 <= siccd <= 3999:
        return "Manufacturing"
    elif 4000 <= siccd <= 4899:
        return "Transportation"
    elif 4900 <= siccd <= 4999:
        return "Utilities"
    elif 5000 <= siccd <= 5199:
        return "Wholesale"
    elif 5200 <= siccd <= 5999:
        return "Retail"
    elif 6000 <= siccd <= 6799:
        return "Finance"
    elif 7000 <= siccd <= 8999:
        return "Services"
    elif 9000 <= siccd <= 9999:
        return "Public"
    else:
        return "Missing"

# Apply the assign_industry function to the 'siccd' column to categorize industries
crsp_monthly["industry"] = (crsp_monthly["siccd"]
  .apply(assign_industry)
)

# Connect to the SQLite database containing financial data
tidy_finance = sqlite3.connect(database="data/tidy_finance_python.sqlite")

# Read 'date' and 'rf' (risk-free rate) columns from the 'factors_ff3_monthly' table in the database
factors_ff3_monthly = pd.read_sql_query(
  sql="SELECT date, rf FROM factors_ff3_monthly",  # SQL query to select 'date' and 'rf' columns
  con=tidy_finance,                                # Specify the database connection
  parse_dates={"date"}                             # Parse the 'date' column as a datetime object for time-based operations
)

# Merge CRSP monthly data with the Fama-French risk-free rate
crsp_monthly = (crsp_monthly
  .merge(factors_ff3_monthly, how="left", on="date")  # Perform a left join on 'date' to align the risk-free rate (rf) with CRSP data
  .assign(ret_excess=lambda x: x["ret"] - x["rf"])    # Calculate excess return (ret_excess) as stock return (ret) minus risk-free rate (rf)
  .assign(ret_excess=lambda x: x["ret_excess"].clip(lower=-1))  # Clip excess return values at a minimum of -1 to handle extreme negative returns
  .drop(columns=["rf"])                               # Drop the 'rf' column as it's no longer needed after calculating excess return
)

# Drop rows with missing values in the 'ret_excess', 'mktcap', or 'mktcap_lag' columns
crsp_monthly = (crsp_monthly
  .dropna(subset=["ret_excess", "mktcap", "mktcap_lag"])
)

# Save the cleaned 'crsp_monthly' DataFrame into the SQLite database
(crsp_monthly
  .to_sql(name="crsp_monthly",            # Name the table 'crsp_monthly' in the SQLite database
          con=tidy_finance,               # Specify the SQLite database connection
          if_exists="replace",            # Replace the table if it already exists
          index=False)                    # Do not write the DataFrame's index to the database
)

# =============================================================================
# First Glimpse of the CRSP Sample
# =============================================================================

# Group the CRSP data by 'exchange' and 'date', and count the number of securities for each combination
securities_per_exchange = (crsp_monthly
  .groupby(["exchange", "date"])  # Group the data by exchange and date
  .size()                          # Count the number of securities in each group (exchange and date)
  .reset_index(name="n")           # Reset the index and create a column 'n' for the count of securities
)

# Create a line plot showing the number of securities listed on each exchange over time
securities_per_exchange_figure = (
  ggplot(securities_per_exchange, 
         aes(x="date", y="n", color="exchange", linetype="exchange")) +  # Set aesthetics: date on x-axis, count (n) on y-axis, color and line type by exchange
  geom_line() +                      # Add a line plot for each exchange
  labs(x="", y="", color="", linetype="",  # Set labels for axes and legends (empty labels for simplicity)
       title="Monthly number of securities by listing exchange") +  # Add a title to the plot
  scale_x_datetime(date_breaks="10 years", date_labels="%Y") +  # Format the x-axis to show breaks every 10 years, labeled by year
  scale_y_continuous(labels=comma_format())  # Format the y-axis with commas for thousands separators (e.g., 1,000)
)

# Render and display the plot
securities_per_exchange_figure.draw()

# =============================================================================

# Read the 'cpi_monthly' table from the SQLite database to get CPI data
cpi_monthly = pd.read_sql_query(
  sql="SELECT * FROM cpi_monthly",  # SQL query to select all columns from the 'cpi_monthly' table
  con=tidy_finance,                 # SQLite database connection
  parse_dates={"date"}              # Parse the 'date' column as a datetime object for time-based operations
)

# Calculate market capitalization per exchange, adjusted for inflation (in Dec 2023 USD)
market_cap_per_exchange = (crsp_monthly
  .merge(cpi_monthly, how="left", on="date")  # Merge CRSP monthly data with CPI data on the 'date' column
  .groupby(["date", "exchange"])  # Group by 'date' and 'exchange'
  .apply(  # Apply a custom aggregation to each group
    lambda group: pd.Series({
      "mktcap": group["mktcap"].sum() / group["cpi"].mean()  # Calculate inflation-adjusted market cap (sum of market cap / mean CPI)
    })
  )
  .reset_index()  # Reset the index to make 'date' and 'exchange' regular columns again
)

# Create a line plot to visualize the market capitalization by exchange, adjusted for inflation
market_cap_per_exchange_figure = (
  ggplot(market_cap_per_exchange,  # Use the aggregated DataFrame for plotting
         aes(x="date", y="mktcap/1000",  # Plot 'date' on the x-axis and market cap (in billions) on the y-axis
             color="exchange", linetype="exchange")) +  # Color and line type are based on 'exchange'
  geom_line() +  # Add a line plot to visualize the data
  labs(x="", y="", color="", linetype="",  # Remove labels from axes and legend (for simplicity)
       title=("Monthly market cap by listing exchange "  # Add a title to the plot
              "in billions of Dec 2023 USD")) + 
  scale_x_datetime(date_breaks="10 years", date_labels="%Y") +  # Format the x-axis to show date breaks every 10 years, labeled by year
  scale_y_continuous(labels=comma_format())  # Format the y-axis with commas for thousands (e.g., 1,000)
)

# Render and display the plot
market_cap_per_exchange_figure.draw()

# =============================================================================

# Group the CRSP data by 'industry' and 'date', and count the number of securities for each combination
securities_per_industry = (crsp_monthly
  .groupby(["industry", "date"])  # Group the data by industry and date
  .size()                         # Count the number of securities in each group
  .reset_index(name="n")          # Reset the index and create a column 'n' for the count of securities
)

# Define a list of custom line types for the plot
linetypes = ["-", "--", "-.", ":"]  # Four line types for use in the plot
n_industries = securities_per_industry["industry"].nunique()  # Number of unique industries in the data

# Create a line plot showing the number of securities in each industry over time
securities_per_industry_figure = (
  ggplot(securities_per_industry, 
         aes(x="date", y="n", color="industry", linetype="industry")) +  # Set aesthetics: date on x-axis, count (n) on y-axis, color and linetype by industry
  geom_line() +                      # Add line plots for each industry
  labs(x="", y="", color="", linetype="",  # Clear axis and legend labels (for simplicity)
       title="Monthly number of securities by industry") +  # Add a title to the plot
  scale_x_datetime(date_breaks="10 years", date_labels="%Y") +  # Format the x-axis to show breaks every 10 years, labeled by year
  scale_y_continuous(labels=comma_format()) +  # Format the y-axis with commas for thousands separators (e.g., 1,000)
  
  # Manually specify the linetypes for each industry, cycling through the 'linetypes' list
  scale_linetype_manual(
    values=[linetypes[l % len(linetypes)] for l in range(n_industries)]  # Assign line styles cyclically based on the number of industries
  )
)

# Render and display the plot
securities_per_industry_figure.draw()

# =============================================================================

# Calculate market capitalization per industry, adjusted for inflation (in Dec 2023 USD)
market_cap_per_industry = (crsp_monthly
  .merge(cpi_monthly, how="left", on="date")  # Merge CRSP monthly data with CPI data on 'date'
  .groupby(["date", "industry"])  # Group by 'date' and 'industry'
  .apply(  # Apply a custom aggregation to each group
    lambda group: pd.Series({
      "mktcap": (group["mktcap"].sum() / group["cpi"].mean())  # Calculate inflation-adjusted market cap (sum of market cap / mean CPI)
    })
  )
  .reset_index()  # Reset the index to make 'date' and 'industry' regular columns again
)

# Create a line plot to visualize the market capitalization by industry, adjusted for inflation
market_cap_per_industry_figure = (
  ggplot(market_cap_per_industry,  # Use the aggregated DataFrame for plotting
         aes(x="date", y="mktcap/1000",  # Plot 'date' on the x-axis and market cap (in billions) on the y-axis
             color="industry", linetype="industry")) +  # Color and line type are based on 'industry'
  geom_line() +  # Add a line plot to visualize the data
  labs(x="", y="", color="", linetype="",  # Remove labels from axes and legend (for simplicity)
       title="Monthly market cap by industry in billions of Dec 2023 USD") +  # Add a title to the plot
  scale_x_datetime(date_breaks="10 years", date_labels="%Y") +  # Format the x-axis to show date breaks every 10 years, labeled by year
  scale_y_continuous(labels=comma_format()) +  # Format the y-axis with commas for thousands separators (e.g., 1,000)
  
  # Manually specify the linetypes for each industry, cycling through the 'linetypes' list
  scale_linetype_manual(
    values=[linetypes[l % len(linetypes)] for l in range(n_industries)]  # Assign line styles cyclically based on the number of industries
  ) 
)

# Render and display the plot
market_cap_per_industry_figure.draw()

# =============================================================================
# Daily CRSP Data
# =============================================================================

# Load daily Fama-French 3-factor data from the SQLite database
factors_ff3_daily = pd.read_sql(
  sql="SELECT * FROM factors_ff3_daily",  # Query to select all columns from 'factors_ff3_daily' table
  con=tidy_finance,                       # SQLite connection
  parse_dates={"date"}                    # Parse the 'date' column as a datetime object
)

# Retrieve the distinct 'permno' values from WRDS database
permnos = pd.read_sql(
  sql="SELECT DISTINCT permno FROM crsp.stksecurityinfohist",  # Query to get distinct permno (stock identifiers) from security info
  con=wrds,                                                    # WRDS database connection
  dtype={"permno": int}                                        # Specify data type for 'permno' as integer
)

# Convert 'permno' column to a list of strings for formatting in SQL queries
permnos = list(permnos["permno"].astype(str))

# Set batch size for processing permnos in chunks
batch_size = 500  # Define the number of permnos to process per batch
batches = np.ceil(len(permnos)/batch_size).astype(int)  # Calculate the total number of batches required

# Loop through each batch of permnos
for j in range(1, batches+1):  
   
  # Get the current batch of permnos based on the batch number
  permno_batch = permnos[((j-1)*batch_size):(min(j*batch_size, len(permnos)))]
  
  # Format the batch of permnos for use in SQL query (turn them into a comma-separated string)
  permno_batch_formatted = ", ".join(f"'{permno}'" for permno in permno_batch)
  permno_string = f"({permno_batch_formatted})"  # Wrap the formatted string in parentheses for the SQL query
  
  # Construct the SQL query to retrieve daily return data for the batch of permnos
  crsp_daily_sub_query = (
    "SELECT dsf.permno, dlycaldt AS date, dlyret AS ret "  # Select permno, date (daily), and return
    "FROM crsp.dsf_v2 AS dsf "  # From CRSP daily stock file (dsf_v2)
    "INNER JOIN crsp.stksecurityinfohist AS ssih "  # Join with security info history to filter based on certain criteria
    "ON dsf.permno = ssih.permno AND "  # Join on 'permno'
       "ssih.secinfostartdt <= dsf.dlycaldt AND "  # Ensure security info start date is before or on daily date
       "dsf.dlycaldt <= ssih.secinfoenddt "  # Ensure security info end date is after or on daily date
    f"WHERE dsf.permno IN {permno_string} "  # Filter the data for the current batch of permnos
         f"AND dlycaldt BETWEEN '{start_date}' AND '{end_date}' "  # Filter by date range
          "AND ssih.sharetype = 'NS' "  # Filter for non-split-adjusted shares
          "AND ssih.securitytype = 'EQTY' "  # Filter for equities
          "AND ssih.securitysubtype = 'COM' "  # Filter for common stocks
          "AND ssih.usincflg = 'Y' "  # Filter for U.S.-incorporated companies
          "AND ssih.issuertype in ('ACOR', 'CORP') "  # Filter for certain issuer types (corporations)
          "AND ssih.primaryexch in ('N', 'A', 'Q') "  # Filter for NYSE, AMEX, and NASDAQ exchanges
          "AND ssih.conditionaltype in ('RW', 'NW') "  # Filter for certain conditional types (real-world and non-worldwide)
          "AND ssih.tradingstatusflg = 'A'"  # Filter for active trading status
  )
  
  # Execute the query and store the result in a DataFrame
  crsp_daily_sub = (pd.read_sql_query(
      sql=crsp_daily_sub_query,  # Execute the query for the current batch of permnos
      con=wrds,                  # WRDS database connection
      dtype={"permno": int},     # Specify data type for 'permno' as integer
      parse_dates={"date"}       # Parse the 'date' column as a datetime object
    )
    .dropna()  # Drop any rows with missing values
   )

  # If the result is not empty, process the data
  if not crsp_daily_sub.empty:
    
      # Merge the daily CRSP data with Fama-French risk-free rate data (ff3_daily) on the 'date' column
      crsp_daily_sub = (crsp_daily_sub
        .merge(factors_ff3_daily[["date", "rf"]], on="date", how="left")  # Left join to bring in 'rf' (risk-free rate)
        .assign(
          ret_excess = lambda x:  # Calculate excess return (ret - rf) and clip extreme values at -1
            ((x["ret"] - x["rf"]).clip(lower=-1))
        )
        .get(["permno", "date", "ret_excess"])  # Keep only the relevant columns (permno, date, ret_excess)
      )
      
      # Define the 'if_exists' argument for writing to the database (replace for the first batch, append for others)
      if j == 1:
        if_exists_string = "replace"  # Replace the table if it's the first batch
      else:
        if_exists_string = "append"   # Append to the table for subsequent batches

      # Write the processed batch data to the SQLite database (as 'crsp_daily' table)
      crsp_daily_sub.to_sql(
        name="crsp_daily", 
        con=tidy_finance, 
        if_exists=if_exists_string, 
        index=False  # Do not write the DataFrame index to the database
      )
            
  # Print progress for each batch
  print(f"Batch {j} out of {batches} done ({(j/batches)*100:.2f}%)\n")  # Display progress update

# =============================================================================
# Preparing Compustat Data
# =============================================================================

# Define the SQL query to retrieve annual financial data from Compustat for North American industrial firms
compustat_query = (
  "SELECT gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb, pstkrv, "
         "pstkl, pstk, capx, oancf, sale, cogs, xint, xsga "  # Select key financial variables for the analysis
    "FROM comp.funda "  # From the Compustat fundamentals annual dataset ('funda')
    "WHERE indfmt = 'INDL' "  # Filter for industrial companies (INDL format)
          "AND datafmt = 'STD' "  # Use standard format (STD)
          "AND consol = 'C' "  # Use consolidated reports (C)
          "AND curcd = 'USD' "  # Filter for companies reporting in USD
         f"AND datadate BETWEEN '{start_date}' AND '{end_date}'"  # Filter by the specified date range
)

# Execute the query and store the result in a DataFrame
compustat = pd.read_sql_query(
  sql=compustat_query,  # Execute the defined SQL query
  con=wrds,  # WRDS database connection
  dtype={"gvkey": str},  # Ensure 'gvkey' (unique company identifier) is treated as a string
  parse_dates={"datadate"}  # Parse the 'datadate' column as a datetime object to facilitate time-based operations
)

# Calculate the book equity (be) and operating profitability (op)
compustat = (compustat
  .assign(
    # Book equity (be) calculation based on Fama and French's approach
    be=lambda x: (
      # Use stockholders' equity (seq) if available, otherwise fallback to common equity (ceq) + preferred stock (pstk)
      x["seq"].combine_first(x["ceq"] + x["pstk"])
      # If seq and ceq+pstk are missing, fallback to total assets (at) minus total liabilities (lt)
      .combine_first(x["at"] - x["lt"]) +
      # Add deferred taxes (txditc), fallback to deferred tax balances (txdb + itcb) if missing
      x["txditc"].combine_first(x["txdb"] + x["itcb"]).fillna(0) -
      # Subtract preferred stock (pstkrv or pstkl), fallback to pstk if others are missing
      x["pstkrv"].combine_first(x["pstkl"]).combine_first(x["pstk"]).fillna(0)
    )
  )
  .assign(
    # Set book equity to NaN if it's less than or equal to 0 (common practice in book-to-market ratio calculations)
    be=lambda x: x["be"].apply(lambda y: np.nan if y <= 0 else y)
  )
  .assign(
    # Calculate operating profitability (op) as: (Sales - COGS - SG&A - Interest Expense) / Book Equity
    op=lambda x: (
      (x["sale"] - x["cogs"].fillna(0) -  # Subtract COGS (cost of goods sold)
       x["xsga"].fillna(0) -              # Subtract SG&A expenses (selling, general & administrative)
       x["xint"].fillna(0)) /             # Subtract interest expenses
       x["be"]                            # Divide by book equity (be)
    )
  )
)

# Extract the year from the 'datadate' and keep only the latest report for each firm-year group
compustat = (compustat
  .assign(year=lambda x: pd.DatetimeIndex(x["datadate"]).year)  # Extract the year from the 'datadate'
  .sort_values("datadate")  # Sort by 'datadate' to ensure we select the most recent data
  .groupby(["gvkey", "year"])  # Group by company (gvkey) and year
  .tail(1)  # Select the last (most recent) report for each group
  .reset_index()  # Reset the index after the group-by operation
)

# Create a lagged version of 'at' (total assets) for the previous year to calculate the investment ratio
compustat_lag = (compustat
  .get(["gvkey", "year", "at"])  # Select gvkey, year, and total assets (at)
  .assign(year=lambda x: x["year"]+1)  # Shift the year by 1 to represent lagged values
  .rename(columns={"at": "at_lag"})  # Rename 'at' to 'at_lag' for clarity
)

# Merge the lagged total assets (at_lag) into the main Compustat DataFrame
compustat = (compustat
  .merge(compustat_lag, how="left", on=["gvkey", "year"])  # Merge on company (gvkey) and year
  .assign(inv=lambda x: x["at"]/x["at_lag"]-1) # Calculate the investment ratio (inv) as the percentage change in total assets: (at / at_lag) - 1
  .assign(inv=lambda x: np.where(x["at_lag"] <= 0, np.nan, x["inv"]))  # Set investment (inv) to NaN where lagged assets (at_lag) are less than or equal to 0 (invalid data)
)

# Store the prepared Compustat data in the SQLite database under the 'compustat' table
(compustat
  .to_sql(name="compustat",
          con=tidy_finance,  # SQLite database connection
          if_exists="replace",  # Replace the table if it already exists
          index=False)  # Do not write the DataFrame's index to the table
)

# =============================================================================
# Merging CRSP with Compustat
# =============================================================================

# Define SQL query to retrieve the CRSP-Compustat linking table
ccm_linking_table_query = (
  "SELECT lpermno AS permno, gvkey, linkdt, "  # Select stock identifier (permno), firm identifier (gvkey), and link dates
         "COALESCE(linkenddt, CURRENT_DATE) AS linkenddt "  # Use CURRENT_DATE for links with no end date
    "FROM crsp.ccmxpf_linktable "  # From the CRSP-Compustat linking table
    "WHERE linktype IN ('LU', 'LC') "  # Filter for links based on link type ('LU': Link using historical permnos, 'LC': Link current permno)
          "AND linkprim IN ('P', 'C')"  # Keep only primary ('P') and consolidated ('C') links
)

# Execute the query and store the linking table in a DataFrame
ccm_linking_table = pd.read_sql_query(
  sql=ccm_linking_table_query,  # Execute the SQL query defined above
  con=wrds,  # WRDS database connection
  dtype={"permno": int, "gvkey": str},  # Specify the data types for 'permno' (int) and 'gvkey' (str)
  parse_dates={"linkdt", "linkenddt"}  # Parse 'linkdt' and 'linkenddt' as datetime objects
)

# Merge the CRSP monthly data with the linking table on 'permno' to map permno to gvkey
ccm_links = (crsp_monthly
  .merge(ccm_linking_table, how="inner", on="permno")  # Inner join keeps only rows where permno exists in both DataFrames
  .query("~gvkey.isnull() & (date >= linkdt) & (date <= linkenddt)")  # Filter valid rows where gvkey exists and dates fall within link period
  .get(["permno", "gvkey", "date"])  # Keep only permno, gvkey, and date columns for further use
)

# Merge the CRSP monthly data with the links (permno to gvkey)
crsp_monthly = (crsp_monthly
  .merge(ccm_links, how="left", on=["permno", "date"])  # Left join ensures all CRSP monthly data is retained, even if not linked to Compustat
)

# Store the updated CRSP monthly data with gvkey links into the SQLite database
(crsp_monthly
  .to_sql(name="crsp_monthly",  # Save the DataFrame to the 'crsp_monthly' table in SQLite
          con=tidy_finance,  # SQLite database connection
          if_exists="replace",  # Replace the existing table if it exists
          index=False)  # Do not write the DataFrame index to the table
)

# =============================================================================

# Calculate the share of securities with available book equity values (be) by exchange and year
share_with_be = (crsp_monthly
  .assign(year=lambda x: pd.DatetimeIndex(x["date"]).year)  # Extract the year from the 'date' column
  .sort_values("date")  # Sort the DataFrame by date to ensure proper time ordering
  .groupby(["permno", "year"])  # Group by permno (stock identifier) and year
  .tail(1)  # Keep only the last available record for each stock-year combination
  .reset_index()  # Reset the index after group-by operation
  .merge(compustat, how="left", on=["gvkey", "year"])  # Merge the Compustat data (which includes 'be') by gvkey and year
  .groupby(["exchange", "year"])  # Group by exchange and year
  .apply(  # Apply a custom aggregation function
    lambda x: pd.Series({
      "share": x["permno"][~x["be"].isnull()].nunique() / x["permno"].nunique()  # Calculate the share of permnos with non-null 'be'
    })
  )
  .reset_index()  # Reset the index after group-by operation
)

# Create a line plot showing the share of securities with book equity values (be) by exchange and year
share_with_be_figure = (
  ggplot(share_with_be,  # Use the DataFrame 'share_with_be' for plotting
         aes(x="year", y="share", color="exchange", linetype="exchange")) +  # Plot year on x-axis, share on y-axis, color and linetype by exchange
  geom_line() +  # Add line plot to show the trend of book equity availability over time
  labs(x="", y="", color="", linetype="",  # Remove axis and legend labels for simplicity
       title="Share of securities with book equity values by exchange") +  # Add title to the plot
  scale_y_continuous(labels=percent_format()) +  # Format y-axis labels as percentages
  coord_cartesian(ylim=(0, 1))  # Limit y-axis range to [0, 1] to represent share as a percentage
)

# Render and display the plot
share_with_be_figure.draw()
