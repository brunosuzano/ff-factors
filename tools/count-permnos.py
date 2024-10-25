import pandas as pd

# Ensure the 'date' column is in datetime format
crsp_monthly['date'] = pd.to_datetime(crsp_monthly['date'])

# Subset the data to only include dates between 2014 and 2023
crsp_subset = crsp_monthly[(crsp_monthly['date'] >= '2014-01-01') & (crsp_monthly['date'] <= '2024-06-30')]

# Count the number of unique permnos in the subset
unique_permnos_count = crsp_subset['permno'].nunique()

# Print the result
print(f"Number of unique permnos in CRSP from 2014 to 2023: {unique_permnos_count}")

# =============================================================================

# Ensure the 'date' column is in datetime format
crsp_m['date'] = pd.to_datetime(crsp_m['date'])

# Subset the data to only include dates between 2014 and 2023
crsp_subset = crsp_m[(crsp_m['date'] >= '2014-01-01') & (crsp_m['date'] <= '2024-06-30')]

# Count the number of unique permnos in the subset
unique_permnos_count = crsp_subset['permno'].nunique()

# Print the result
print(f"Number of unique permnos in CRSP from 2014 to 2023: {unique_permnos_count}")

# =============================================================================

# Ensure the 'date' column is in datetime format
crsp3['date'] = pd.to_datetime(crsp3['date'])

# Subset the data to only include dates between 2014 and 2023
crsp_subset = crsp3[(crsp3['date'] >= '2014-01-01') & (crsp3['date'] <= '2024-06-30')]

# Count the number of unique permnos in the subset
unique_permnos_count = crsp_subset['permno'].nunique()

# Print the result
print(f"Number of unique permnos in CRSP from 2014 to 2023: {unique_permnos_count}")