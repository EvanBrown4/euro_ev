import pandas as pd
import sqlite3

euros = pd.read_csv("euro.csv")
print(euros)

# Do an artificial weight for home here, just a slight one
# For the ML algs, you can actually consider it
euros = euros[["home_team", "away_team", "tournament", "neutral", "round", "group"]]

rankings = pd.read_csv("rankings_with_norm.csv")
two_year_rankings = pd.read_csv("past_two_years.csv")
# two_year = 
# ^^ TODO: need to get just current scores from everyone for this too

conn = sqlite3.connect(':memory:')
euros.to_sql("euros", conn, index=False, if_exists='replace')
rankings.to_sql("rankings", conn, index=False, if_exists='replace')
two_year_rankings.to_sql("two_year", conn, index=False, if_exists='replace')

with open('sql/scoring_merge.sql', 'r') as file:
    query = file.read()

euro_probs = pd.read_sql_query(query, conn)
print(euro_probs)

def outcome_prob(row):
    home = row["home_norm_points"]
    away = row["away_norm_points"]
    neutral = row["neutral"]
    if neutral == 0:
        if row["home_team"] == "Germany":
            home = home * 8/7
        else:
            away = away * 8/7
    home_dr = home-away
    away_dr = away-home
    # TODO: PLAY WITH THE S VARIABLE (30 currently))
    # 1/(10^(-dr/S)+1)
    home_eq = 1/(10**(-home_dr/30)+1)
    away_eq = 1/(10**(-away_dr/30)+1)
    print (home_eq, away_eq)
    return home_eq, away_eq

euro_probs["home_win_prob"], euro_probs["away_win_prob"] = zip(*euro_probs.apply(outcome_prob, axis=1))

euro_probs.to_csv("outcome_probabilities_group_stage.csv")

def expected_points(xW):
    if xW < 0.5:
        return 2 * xW  # Scales linearly from 0 to 1
    elif xW == 0.5:
        return 1.0
    else:
        return 3 - 2 * (1 - xW)  # Scales linearly from 1 to 3
    
euro_probs["home_xP"] = euro_probs["home_win_prob"].apply(expected_points)
euro_probs["away_xP"] = euro_probs["away_win_prob"].apply(expected_points)

home_df = euro_probs[['home_team', 'home_xP', 'group', 'home_norm_points', 'neutral']].rename(columns={'home_team': 'team', 'home_xP': 'xP', 'home_norm_points': 'norm_points'})
away_df = euro_probs[['away_team', 'away_xP', 'group', 'away_norm_points', 'neutral']].rename(columns={'away_team': 'team', 'away_xP': 'xP', 'away_norm_points': 'norm_points'})

combined_df = pd.concat([home_df, away_df])

group_stage_xP = combined_df.groupby('team').agg({
    'xP': 'sum',
    'group': 'first',
    'norm_points': 'first',
    'neutral': 'first'
}).reset_index().sort_values(by='xP', ascending=False)

group_stage_xP['group_rank'] = group_stage_xP.groupby('group')['xP'].rank(ascending=False, method='min').astype(int)

# Sort DataFrame by 'group' and 'group_rank'
# group_stage_xP = group_stage_xP.sort_values(by=['group', 'group_rank'])



print (group_stage_xP)
group_stage_xP.to_csv("group_stage_xP2.csv", index=False)



third_place = group_stage_xP[group_stage_xP["group_rank"] == 3].sort_values(by=['xP'], ascending=False)
print(third_place)

# get probability of each team crashing out at each stage...
# get projected points of matchups
# then get, for each team, the probability of winning at each stage (if .6 in RO16 and .4 in QF, then .24 to win QF and get 6+8 points)

# this strat is just clean slating it in RO16

# with open('sql/scoring_merge2.sql', 'r') as file:
#     query = file.read()


# euros_two_year_probs = pd.read_sql_query(query, conn)
# euros_two_year_probs["home_win_prob"], euros_two_year_probs["away_win_prob"] = zip(*euros_two_year_probs.apply(outcome_prob, axis=1))
# euros_two_year_probs["home_xP"] = euros_two_year_probs["home_win_prob"].apply(expected_points)
# euros_two_year_probs["away_xP"] = euros_two_year_probs["away_win_prob"].apply(expected_points)

# home_df = euros_two_year_probs[['home_team', 'home_xP']].rename(columns={'home_team': 'team', 'home_xP': 'xP'})
# away_df = euros_two_year_probs[['away_team', 'away_xP']].rename(columns={'away_team': 'team', 'away_xP': 'xP'})

# combined_df = pd.concat([home_df, away_df])

# group_stage_xP = combined_df.groupby('team')['xP'].sum().reset_index().sort_values(by='xP', ascending=False)
# print (group_stage_xP)
# group_stage_xP.to_csv("group_stage_xP_two_years.csv", index=False)