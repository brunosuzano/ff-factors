import sqlite3
import pandas as pd

# Connect to the SQLite database
tidy_finance = sqlite3.connect(database="data/tidy_finance_python.sqlite")

# Dictionary to store DataFrames with table information, where keys are table names
table_info_list = {}

# Get the list of all tables in the database
for table_name in tidy_finance.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
    table_name = table_name[0]
    
    # Collect number of rows
    row_count = tidy_finance.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
    
    # Get the column names, number of columns, and check if there's a primary key
    schema = tidy_finance.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    column_names = [col[1] for col in schema]  # Extract column names
    num_columns = len(column_names)  # Count the number of columns
    has_primary_key = any(col[-1] == 1 for col in schema)  # Check if any column is the primary key

    # Estimate table size in memory (simple assumption: 8 bytes per column per row)
    estimated_size_bytes = row_count * num_columns * 8
    
    # Create a DataFrame for the current table's information
    table_info_list[table_name] = pd.DataFrame({
        "table_name": [table_name],
        "num_rows": [row_count],
        "num_columns": [num_columns],
        "column_names": [", ".join(column_names)],  # Join column names into a string
        "has_primary_key": [has_primary_key],  # Indicates if there's a primary key
        "estimated_size_bytes": [estimated_size_bytes]  # Estimated size of the table in memory
    })

# Close the connection to the SQLite database
tidy_finance.close()

# Combine all table info DataFrames into a single DataFrame, sorted by table name
combined_table_info = pd.concat(table_info_list.values(), ignore_index=True).sort_values("table_name").reset_index(drop=True)

# Now, only `combined_table_info` and `table_info_list` exist as outputs

# Drop the specified variables
del column_names
del estimated_size_bytes
del has_primary_key
del num_columns
del row_count
del schema
del table_name
