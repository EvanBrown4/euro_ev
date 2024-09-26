import pandas as pd

df = pd.read_csv("euro_off_test.csv")

def outcome_prob(row):
    home = row["home_norm_points"]
    away = row["away_norm_points"]
    neutral = row["neutral"]
    if neutral == 0:
        if row["home"] == "Germany":
            home = home * 8/7
        else:
            away = away * 8/7
    home_dr = home-away
    away_dr = away-home
    # TODO: PLAY WITH THE S VARIABLE (25 currently))
    # 1/(10^(-dr/S)+1)
    home_eq = 1/(10**(-home_dr/50)+1)
    away_eq = 1/(10**(-away_dr/50)+1)
    print (home_eq, away_eq)
    return home_eq, away_eq

df["home_win_prob"], df["away_win_prob"] = zip(*df.apply(outcome_prob, axis=1))

def weighted_points(x):
    return x["home_norm_points"]*x["home_win_prob"] + x["away_norm_points"]*x["away_win_prob"]

df["weighted_next_round_points"] = df.apply(lambda x: weighted_points(x), axis=1)
print(df)
df.to_csv("euro_off_test.csv", index=False)