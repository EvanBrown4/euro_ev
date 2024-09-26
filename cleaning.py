import pandas as pd
import sqlite3

results = pd.read_csv("results.csv")
rankings = pd.read_csv("rankings_with_norm.csv")

conn = sqlite3.connect(':memory:')
results.to_sql('results', conn, index=False, if_exists='replace')
rankings.to_sql('rankings', conn, index=False, if_exists='replace')

with open('sql/ranking_merge.sql', 'r') as file:
    # Read the content of the file
    main_query = file.read()

result = pd.read_sql_query(main_query, conn)
print(result)
result.to_csv("results_with_rankings.csv", index=False)