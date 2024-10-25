import pandas as pd
import numpy as np
import datetime as dt
import wrds

# Fama-French stock filters:
# - NYSE, AMEX, NASDAQ only (EXCHGCD 1, 2, & 3)
# - Common stocks only (SHRCD 10 & 11)
# - Exclude multiple share classes by keeping only the largest ME (but the company ME will be the sum for all share classes)
# - Only data available in both CRSP and Compustat
# - Only firms that are on Compustat at least 2 years
# - No negative BE

# This code currently employs only the first 2 filters

# =============================================================================
# CRSP Download
# =============================================================================

# Establish a connection to WRDS
conn=wrds.Connection()

# CRSP sample period
start_date = "01/01/2014"
end_date = "06/30/2024"

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

# Download CRSP data
crsp_m = conn.raw_sql(crsp_monthly_query, date_cols=['date']) 

# =============================================================================
# PERMNO Sample
# =============================================================================

# Retrieve unique PERMNOs and count them
unique_permnos = crsp_m['permno'].unique()
unique_permnos_count = len(unique_permnos)

# Output the results
print(f"Number of unique PERMNOs in CRSP: {unique_permnos_count}")

# =============================================================================
# PERMNO-Date Sample
# =============================================================================

# Retrieve unique PERMNO-date combinations and count them
unique_permno_date_combinations = crsp_m[['permno', 'date']].drop_duplicates()
unique_permno_date_count = len(unique_permno_date_combinations)

# Output the results
print(f"Number of unique PERMNO-date combinations in CRSP: {unique_permno_date_count}")

# Save to csv
unique_permno_date_combinations.to_csv('unique_permno_date_combinations.csv', index=False)
