import pandas as pd
import numpy as np
import datetime as dt
import wrds
from datetime import datetime, timedelta

# Fama-French stock filters:
# - NYSE, AMEX, NASDAQ only (EXCHGCD 1, 2, & 3)
# - Common stocks only (SHRCD 10 & 11)
# - Exclude multiple share classes by keeping only the largest ME (but the company ME will be the sum for all share classes)
# - Only data available in both CRSP and Compustat
# - Only firms that are on Compustat at least 2 years
# - No negative BE

# This code currently employs only the first 2 filters

# Set data periodicity ('daily', 'monthly')
period = 'daily'

# Save samples to .csv?
savecsv = True

# =============================================================================
# CRSP Download
# =============================================================================

# Establish a connection to WRDS
conn = wrds.Connection(wrds_username='brumor')

# CRSP sample period
start_date = "01/01/2014"
end_date = "06/30/2024"

# Parse the original start date and shift back one year
original_start_date = datetime.strptime(start_date, "%d/%m/%Y")
shifted_start_date = original_start_date.replace(year=original_start_date.year - 1)
shifted_start_date_str = shifted_start_date.strftime("%d/%m/%Y")

if period == 'monthly':
    
    # Construct the SQL query to retrieve CRSP monthly stock data
    crsp_monthly_query = (
      "SELECT msf.permno, (date_trunc('month', msf.mthcaldt) + interval '1 month' - interval '1 day')::date AS date "  # Select stock identifier (permno) and set date to the end of the month
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
    permno_dates = conn.raw_sql(crsp_monthly_query, date_cols=['date'])
    permno_dates = permno_dates.sort_values(by=['permno', 'date']).reset_index(drop=True)
    permno_dates['date'] = permno_dates['date'].dt.date

elif period == 'daily':
    
    # Construct the SQL query to retrieve CRSP daily stock data
    crsp_daily_query = (
      "SELECT dsf.permno, date_trunc('day', dsf.dlycaldt)::date AS date "  # Select stock identifier (permno) and truncating the date to day level
        "FROM crsp.dsf_v2 AS dsf "  # From the CRSP daily stock file (dsf_v2)
        "INNER JOIN crsp.stksecurityinfohist AS ssih "  # Inner join with the CRSP security information history table
        "ON dsf.permno = ssih.permno AND "  # Join on permno (unique stock identifier)
           "ssih.secinfostartdt <= dsf.dlycaldt AND "  # Ensure the security info start date is before or on the daily date
           "dsf.dlycaldt <= ssih.secinfoenddt "  # Ensure the security info end date is after or on the daily date
       f"WHERE dsf.dlycaldt BETWEEN '{start_date}' AND '{end_date}' "  # Filter by the date range between start_date and end_date
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
    permno_dates = conn.raw_sql(crsp_daily_query, date_cols=['date'])
    permno_dates = permno_dates.sort_values(by=['permno', 'date']).reset_index(drop=True)

    # Extract year and month, and create a 'year_month' column
    permno_dates['year_month'] = permno_dates['date'].dt.to_period('M')

    # Select permno and year_month, then drop duplicates to get unique combinations
    permno_dates = permno_dates[['permno', 'year_month']].drop_duplicates().reset_index(drop=True)

    # Convert year_month to a full date at the end of the month
    permno_dates['date'] = permno_dates['year_month'].dt.to_timestamp(how='end').dt.date

    # Drop the original year_month column
    permno_dates = permno_dates.drop(columns=['year_month'])

# =============================================================================
# PERMNO Sample
# =============================================================================

# Retrieve unique PERMNOs and count them
permnos = permno_dates['permno'].unique()
permnos_count = len(permnos)

# Output the results
print(f"Number of unique PERMNOs in CRSP: {permnos_count}")

# Save to csv
if savecsv:
    unique_permnos_df = pd.DataFrame(permnos, columns=['permno'])
    unique_permnos_df.to_csv('unique_permnos.csv', index=False)

# =============================================================================
# PERMNO-Date Sample
# =============================================================================

# Count PERMNO-date combinations
permno_dates_count = len(permno_dates)

# Output the results
print(f"Number of unique PERMNO-date combinations in CRSP: {permno_dates_count}")

# Save to csv
if savecsv:
    permno_dates.to_csv('unique_permno_date_combinations.csv', index=False)
