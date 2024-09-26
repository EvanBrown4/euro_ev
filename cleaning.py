import pandas as pd
import sqlite3

results = pd.read_csv("data/results.csv")
rankings = pd.read_csv("data/rankings_with_norm.csv")

# Add data to SQLite memory
conn = sqlite3.connect(':memory:')
results.to_sql('results', conn, index=False, if_exists='replace')
rankings.to_sql('rankings', conn, index=False, if_exists='replace')

# Get the query
with open('sql/ranking_merge.sql', 'r') as file:
    main_query = file.read()

# Execute the query and save it
result = pd.read_sql_query(main_query, conn)
result.to_csv("data/results_with_rankings.csv", index=False)