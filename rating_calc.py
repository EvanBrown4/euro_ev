import pandas as pd
import sqlite3

rankings = pd.read_csv("fifa_ranking_updated.csv")

conn = sqlite3.connect(':memory:')
rankings.to_sql('rankings', conn, index=False, if_exists='replace')

with open('sql/ranking_norm.sql', 'r') as file:
    query = file.read()

result = pd.read_sql_query(query, conn)
result.to_csv("rankings_with_norm.csv", index=False)