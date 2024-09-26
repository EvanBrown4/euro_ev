import pandas as pd
import sqlite3

results = pd.read_csv("results_with_rankings.csv")

results["date"] = pd.to_datetime(results["date"])
results = results[results["date"] >= pd.to_datetime("2022/06/11")]
results = results.sort_values(by="date")
# print(results)

points = {}

pd.Series(results["tournament"].unique()).to_csv("competitions.txt")

def calc_I(comp):
    if comp == "AFC Asian Cup qualification":
        I = 25
    if comp == "Friendly":
        I = 5
    if comp == "UEFA Nations League":
        I = 20
    if comp == "CONCACAF Nations League":
        I = 20
    if comp == "African Cup of Nations qualification":
        I = 25
    if comp == "FIFA World Cup qualification":
        I = 25
    if comp == "Kirin Cup":
        I = 10
    if comp == "COSAFA Cup":
        I = 10
    if comp == "EAFF Championship":
        I = 10
    if comp == "MSG Prime Minister's Cup":
        I = 10
    if comp == "King's Cup":
        I = 10
    if comp == "Jordan International Tournament":
        I = 10
    if comp == "Kirin Challenge Cup":
        I = 10
    if comp == "Baltic Cup":
        I = 10
    if comp == "FIFA World Cup":
        I = 55
    if comp == "AFF Championship":
        I = 10
    if comp == "Gulf Cup":
        I = 10
    if comp == "Tri Nation Tournament":
        I = 10
    if comp == "UEFA Euro qualification":
        I = 25
    if comp == "CAFA Nations Cup":
        I = 10
    if comp == "Mauritius Four Nations Cup":
        I = 10
    if comp == "Gold Cup qualification":
        I = 25
    if comp == "SAFF Cup":
        I = 10
    if comp == "Gold Cup":
        I = 37.5
    if comp == "Indian Ocean Island Games":
        I = 10
    if comp == "Pacific Games":
        I = 10
    if comp == "AFC Asian Cup":
        I = 37.5
    if comp == "African Cup of Nations":
        I = 37.5
    if comp == "FIFA Series":
        I = 10
    if comp == "UEFA Euro":
        I = 37.5
    return I

results["I"] = results["tournament"].apply(calc_I)

def update_points(row):
    # P = Pbefore + I * (W - We)
    home_team = row["home_team"]
    away_team = row["away_team"]
    home_score = row["home_score"]
    away_score = row["away_score"]
    I = row["I"]

    if home_team not in points:
        points[home_team] = 0
    if away_team not in points:
        points[away_team] = 0
    
    if home_score > away_score:
        home_W = 1
        away_W = 0
    elif home_score < away_score:
        home_W = 0
        away_W = 1
    else:
        home_W = 0.5
        away_W = 0.5

    home_dr = points[home_team] - points[away_team]
    away_dr = points[away_team] - points[home_team]
    home_We = 1 / (10**(-home_dr/600)+1)
    away_We = 1 / (10**(-away_dr/600)+1)

    points[home_team] = points[home_team] + I * (home_W - home_We)
    points[away_team] = points[away_team] + I * (away_W - away_We)
    return points[home_team], points[away_team]

results["home_recent_points"], results["away_recent_points"] = zip(*results.apply(update_points, axis=1))
home_data = results[['home_team', 'home_recent_points', 'date']].rename(columns={'home_team': 'team_name', 'home_recent_points': 'points'})
away_data = results[['away_team', 'away_recent_points', 'date']].rename(columns={'away_team': 'team_name', 'away_recent_points': 'points'})

# # Concatenate home and away data
results = pd.concat([home_data, away_data], ignore_index=True).sort_values(by="date", ascending=False)
results.to_csv("past_two_years.csv", index=False)

conn = sqlite3.connect(':memory:')
results.to_sql("past_two_years", conn, index=False, if_exists='replace')

with open('sql/past_two_years.sql', 'r') as file:
    query = file.read()

result = pd.read_sql_query(query, conn).sort_values(by="points", ascending=False)
result.to_csv("past_two_years.csv", index=False)

## TODO: STILL NEED TO NORMALIZE


    


# points are calculated including the result
# also a tweaked equation, putting different emphasis on different tournaments