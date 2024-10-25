import pandas as pd
import numpy as np

# Sample period
start_date = "1960-01-01"
end_date = "2023-12-31"

# =============================================================================
# Fama-French Data
# =============================================================================

import pandas_datareader as pdr

# Import 3-factor monthly data
factors_ff3_monthly_raw = pdr.DataReader(
  name="F-F_Research_Data_Factors",
  data_source="famafrench", 
  start=start_date, 
  end=end_date)[0]

# Process the 3-factor monthly data
factors_ff3_monthly = (factors_ff3_monthly_raw
  .divide(100)                            # Divide factor values by 100 to scale appropriately
  .reset_index(names="date")              # Reset the index to make 'date' a regular column
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))  # Convert the 'date' column to datetime format
  .rename(str.lower, axis="columns")      # Convert all column names to lowercase for consistency
  .rename(columns={"mkt-rf": "mkt_excess"})  # Rename 'mkt-rf' to 'mkt_excess' for clarity
)

# Import 5-factor monthly data
factors_ff5_monthly_raw = pdr.DataReader(
  name="F-F_Research_Data_5_Factors_2x3",
  data_source="famafrench", 
  start=start_date, 
  end=end_date)[0]

# Process the 5-factor monthly data
factors_ff5_monthly = (factors_ff5_monthly_raw
  .divide(100)                            # Divide factor values by 100 to scale appropriately
  .reset_index(names="date")              # Reset the index to make 'date' a regular column
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))  # Convert the 'date' column to datetime format
  .rename(str.lower, axis="columns")      # Convert all column names to lowercase for consistency
  .rename(columns={"mkt-rf": "mkt_excess"})  # Rename 'mkt-rf' to 'mkt_excess' for clarity
)

# Import 3-factor daily data
factors_ff3_daily_raw = pdr.DataReader(
  name="F-F_Research_Data_Factors_daily",
  data_source="famafrench", 
  start=start_date, 
  end=end_date)[0]


# Process the 3-factor daily data
factors_ff3_daily = (factors_ff3_daily_raw
  .divide(100)                            # Divide factor values by 100 to scale appropriately
  .reset_index(names="date")              # Reset the index to make 'date' a regular column
  .rename(str.lower, axis="columns")      # Convert all column names to lowercase for consistency
  .rename(columns={"mkt-rf": "mkt_excess"})  # Rename 'mkt-rf' to 'mkt_excess' for clarity
)

# Import 10-industry portfolio (monthly) data
industries_ff_monthly_raw = pdr.DataReader(
  name="10_Industry_Portfolios",
  data_source="famafrench", 
  start=start_date, 
  end=end_date)[0]

# Process the 10-industry portfolio (monthly) data
industries_ff_monthly = (industries_ff_monthly_raw
  .divide(100)                            # Divide the portfolio returns by 100 to scale appropriately
  .reset_index(names="date")              # Reset the index to make 'date' a regular column
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))  # Convert the 'date' column to datetime format
  .rename(str.lower, axis="columns")      # Convert all column names to lowercase for consistency
)

# =============================================================================
# q-Factors
# =============================================================================

# Link to the CSV file containing the Q5 factors data (monthly)
factors_q_monthly_link = (
  "https://global-q.org/uploads/1/2/2/6/122679606/"
  "q5_factors_monthly_2023.csv"
)

# Load the q-factors CSV, process the data, and filter by date range
factors_q_monthly = (pd.read_csv(factors_q_monthly_link)   # Read the CSV data into a DataFrame
  .assign(
    date=lambda x: ( # Combine 'year' and 'month' columns into a 'date' column and convert it to datetime format
      pd.to_datetime(x["year"].astype(str) + "-" +   # Convert 'year' to string and concatenate with 'month'
        x["month"].astype(str) + "-01"))            # Assume the first day of the month for simplicity
  )
  .drop(columns=["R_F", "R_MKT", "year"])           # Drop unnecessary columns that won't be used in the analysis
  .rename(columns=lambda x: x.replace("R_", "").lower())  # Rename columns, removing 'R_' prefix and converting to lowercase
  .query(f"date >= '{start_date}' and date <= '{end_date}'")  # Keep only rows within the specified date range
  .assign(
    **{col: lambda x: x[col]/100 for col in ["me", "ia", "roe", "eg"]}  # Divide by 100 to convert from % to decimal
  )
)

# =============================================================================
# Macroeconomic Predictors
# =============================================================================

# Google Sheets Download URL
sheet_id = "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG"
sheet_name = "macro_predictors.xlsx"
macro_predictors_link = (
  f"https://docs.google.com/spreadsheets/d/{sheet_id}" 
  f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
)

macro_predictors = (
  # Read the CSV data from the Google Sheets link
  pd.read_csv(macro_predictors_link, thousands=",")
  # Perform several transformations using .assign() to calculate new columns
  .assign(
    date=lambda x: pd.to_datetime(x["yyyymm"], format="%Y%m"),  # Convert 'yyyymm' column to datetime format
    dp=lambda x: np.log(x["D12"])-np.log(x["Index"]),           # Calculate the dividend-price ratio (dp)
    dy=lambda x: np.log(x["D12"])-np.log(x["Index"].shift(1)),  # Calculate the dividend yield (dy)
    ep=lambda x: np.log(x["E12"])-np.log(x["Index"]),           # Calculate the earnings-price ratio (ep)
    de=lambda x: np.log(x["D12"])-np.log(x["E12"]),             # Calculate the dividend-earnings ratio (de)
    tms=lambda x: x["lty"]-x["tbl"],                            # Calculate the term spread (tms)
    dfy=lambda x: x["BAA"]-x["AAA"]                             # Calculate the default yield spread (dfy)
  )
  .rename(columns={"b/m": "bm"}) # Rename 'b/m' column to 'bm' for consistency in naming conventions
  .get(["date", "dp", "dy", "ep", "de", "svar", "bm", 
        "ntis", "tbl", "lty", "ltr", "tms", "dfy", "infl"]) # Select only the relevant columns for further analysis
  .query("date >= @start_date and date <= @end_date") # Filter the data to only rows within the specified date range
  .dropna() # Drop any rows that contain missing values
)

# =============================================================================
# Other Macroeconomic Data
# =============================================================================

# Import monthly CPI (Consumer Price Index) data from the FRED data source
cpi_monthly = (pdr.DataReader(
    name="CPIAUCNS",                     # Name of the CPI dataset in FRED
    data_source="fred",                   # Data source: FRED (Federal Reserve Economic Data)
    start=start_date,                     # Start date for the data retrieval
    end=end_date                          # End date for the data retrieval
  )
  .reset_index(names="date")              # Reset the index and turn 'date' into a regular column
  .rename(columns={"CPIAUCNS": "cpi"})    # Rename the CPI column to 'cpi' for simplicity
  .assign(cpi=lambda x: x["cpi"]/x["cpi"].iloc[-1])  # Normalize CPI values by dividing each value by the most recent CPI
)

# =============================================================================
# Setting Up a Database
# =============================================================================

import os
import sqlite3

# Establish a connection to the SQLite database
os.makedirs("data", exist_ok=True)  # Create the 'data' directory if it doesn't exist
tidy_finance = sqlite3.connect("data/tidy_finance_python.sqlite")

# Save the factors_ff3_monthly DataFrame into the SQLite database as a new table
(factors_ff3_monthly
  .to_sql(name="factors_ff3_monthly",     # Name the table 'factors_ff3_monthly'
          con=tidy_finance,               # Specify the database connection
          if_exists="replace",            # Replace the table if it already exists
          index=False)                    # Do not write the DataFrame index to the database
)

# Query the SQLite database for 'date' and 'rf' columns from the 'factors_ff3_monthly' table
pd.read_sql_query(
  sql="SELECT date, rf FROM factors_ff3_monthly",  # SQL query to select 'date' and 'rf' columns
  con=tidy_finance,                                # Specify the database connection
  parse_dates={"date"}                             # Parse the 'date' column as a datetime object
)

# Dictionary to hold multiple DataFrames for batch insertion into the database
data_dict = {
  "factors_ff5_monthly": factors_ff5_monthly,       # Key: table name; Value: DataFrame for FF5 monthly factors
  "factors_ff3_daily": factors_ff3_daily,           # Key: table name; Value: DataFrame for FF3 daily factors
  "industries_ff_monthly": industries_ff_monthly,   # Key: table name; Value: DataFrame for FF 10-industry portfolios
  "factors_q_monthly": factors_q_monthly,           # Key: table name; Value: DataFrame for Q5 factors monthly data
  "macro_predictors": macro_predictors,             # Key: table name; Value: DataFrame for macro predictors data
  "cpi_monthly": cpi_monthly                       # Key: table name; Value: DataFrame for monthly CPI data
}

# Loop through the dictionary and save each DataFrame to the SQLite database
for key, value in data_dict.items():
    value.to_sql(name=key,                         # Use the dictionary key as the table name
                 con=tidy_finance,                 # Specify the database connection
                 if_exists="replace",              # Replace the table if it already exists
                 index=False)                      # Do not write the DataFrame index to the database

# =============================================================================

### From now on, to import data from the SQLite database use the following code:

import pandas as pd
import sqlite3

# Establish a connection to the SQLite database
tidy_finance = sqlite3.connect(database="data/tidy_finance_python.sqlite")

# Read the 'factors_q_monthly' table from the SQLite database into a DataFrame
factors_q_monthly_test = pd.read_sql_query(
  sql="SELECT * FROM factors_q_monthly",  # SQL query to select all columns from the 'factors_q_monthly' table
  con=tidy_finance,                       # Specify the database connection
  parse_dates={"date"}                    # Parse the 'date' column as a datetime object
)

# =============================================================================
# Managing SQLite Databases
# =============================================================================

# Perform a VACUUM cleanup on the SQLite database
tidy_finance.execute("VACUUM")
