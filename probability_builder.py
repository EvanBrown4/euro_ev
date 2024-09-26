import pandas as pd
from itertools import combinations, permutations
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

df = pd.read_csv("data/rankings_with_norm.csv")
current = df[df["rank_date"] == '2024-04-04']
# print(current)

euro = current[current["confederation"] == "UEFA"].sort_values(by="rank")

copa = current[(current["confederation"] == "CONMEBOL") | (current["country_full"] == "Mexico") | (current["country_full"] == "Canada") | (current["country_full"] == "USA") | (current["country_full"] == "Jamaica") | (current["country_full"] == "Panama") | (current["country_full"] == "Costa Rica")].sort_values(by="rank")

# print(uefa)
# print(copa)
# uefa.to_csv("uefa_teams.csv")
# copa.to_csv("copa_teams.csv")

# NOTE: home and away don't really mean much here besides just splitting the teams up. The games are all at a neutral venue unless you are
# USA or Germany, and that is dealt with separately.

TIE_COUNT = 0
MATCH_COUNT = 0

def match_increment():
    global MATCH_COUNT
    MATCH_COUNT += 1

def tie_increment():
    global TIE_COUNT
    TIE_COUNT += 1

# Create all possible matchups (not just the ones that are known to be happening - this is so any possible knockout stage
# matchup can already have is prbabilities calculated)/
def matchups(df):
    teams = df["country_full"].unique()
    combos = list(combinations(teams, 2))
    matchups = pd.DataFrame(combos, columns=["home_team", "away_team"])
    # Combine and rename all home and away points.
    final = matchups.merge(df[["country_full", "norm_points"]], how="left", left_on="home_team", right_on="country_full")
    final = final.rename(columns={"norm_points": "home_norm_points"}).drop(columns=["country_full"])
    final = final.merge(df[["country_full", "norm_points"]], how="left", left_on="away_team", right_on="country_full")
    final = final.rename(columns={"norm_points": "away_norm_points"}).drop(columns=["country_full"])
    # Flag games with a host nation
    final["neutral"] = final.apply(lambda x: 0 if x["home_team"] == "Germany" or x["home_team"] == "USA" or x["away_team"] == "Germany" or x["away_team"] == "USA" else 1, axis=1)
    return final
euro_matchups = matchups(euro)
copa_matchups = matchups(copa)

def outcome_prob(row):
    home = row["home_norm_points"]
    away = row["away_norm_points"]
    neutral = row["neutral"]
    # Add a slight boost to a team if they are a host nation.
    if neutral == 0:
        if row["home_team"] == "Germany" or row["home_team"] == "USA":
            home = home * 8/7
        else:
            away = away * 8/7
    home_dr = home-away
    away_dr = away-home
    # Equation based on the logistic function (for example elo in Chess) to calculate home and away win probabilities.
    home_prob = 1/(10**(-home_dr/12)+1)
    away_prob = 1/(10**(-away_dr/12)+1)
    # Equation to get tie odds. higher odds closer to .5 win probability Shift to play with.
    # After normalizing it with the win probs, the average value is 0.18. The number of ties in the last euros and copa was .25.
    # We shifted a bit to adjust for the larger tie odds
    # Equation: home_prob = a, away_prob = b, c = probability shift, c(ab)/(abs(b-a)+1)
    tie_prob = 1.5 * (home_prob*away_prob) / (abs(away_prob-home_prob)+1)
    total_prob = home_prob+away_prob+tie_prob
    # Adjust the probabilities with the tie considered (keep in mind a tie is only possible during the group stage).
    adjusted_home_prob = home_prob/total_prob
    adjusted_away_prob = away_prob/total_prob
    adjusted_tie_prob = tie_prob/total_prob
    return home_prob, away_prob, adjusted_home_prob, adjusted_away_prob, adjusted_tie_prob

euro_matchups["knockout_home_win_prob"], euro_matchups["knockout_away_win_prob"], euro_matchups["home_win_prob"], euro_matchups["away_win_prob"], euro_matchups["tie_prob"] = zip(*euro_matchups.apply(outcome_prob, axis=1))
copa_matchups["knockout_home_win_prob"], copa_matchups["knockout_away_win_prob"], copa_matchups["home_win_prob"], copa_matchups["away_win_prob"], copa_matchups["tie_prob"] = zip(*copa_matchups.apply(outcome_prob, axis=1))

# euro_matchups.to_csv("euro_prob_matrix.csv", index=False)
# copa_matchups.to_csv("copa_prob_matrix.csv", index=False)


# Group Stage
euro_teams = {
    "Group A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "Group B": ["Spain", "Croatia", "Italy", "Albania"],
    "Group C": ["Slovenia", "Denmark", "Serbia", "England"],
    "Group D": ["Poland", "Netherlands", "Austria", "France"],
    "Group E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
    "Group F": ["Turkey", "Georgia", "Portugal", "Czechia"]
}

copa_teams = {
    "Group A": ["Argentina", "Peru", "Chile", "Canada"],
    "Group B": ["Mexico", "Ecuador", "Venezuela", "Jamaica"],
    "Group C": ["USA", "Uruguay", "Panama", "Bolivia"],
    "Group D": ["Brazil", "Colombia", "Paraguay", "Costa Rica"]
}

group_map = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F"
}

## Simulate tournament to get sample data for each team.

# Simulate group stage matches, returning the number of points obtained by each team in the match.
def sim_group_stage_match(team1, team2, matchups):
    # Keep track of the match number
    match_increment()

    # Ensure the matchup exists and get the win probabilities for each team.
    match = matchups[(matchups['home_team'] == team1) & (matchups['away_team'] == team2)]
    if match.empty:
        match = matchups[(matchups['home_team'] == team2) & (matchups['away_team'] == team1)]
        if match.empty:
            print(f"ERROR: Matchup does not exist: {team1} v {team2}")
            return (-1, -1)
        team1_prob = match['away_win_prob'].values[0]
        team2_prob = match['home_win_prob'].values[0]
    else:
        team1_prob = match['home_win_prob'].values[0]
        team2_prob = match['away_win_prob'].values[0]
    
    # Get a random number between 0 and 1 to simulate the match.
    match_outcome = np.random.rand()

    # Find the outcome of the match
    if match_outcome < team1_prob:
        return (3, 0)  # team1 wins
    elif match_outcome < team1_prob + team2_prob:
        return (0, 3)  # team2 wins
    else:
        tie_increment()
        return (1, 1)  # tie

# Simulate each group. Simulate every match in the group and return the total points for each team.
def sim_group(teams, matchups):
    group_points = {team: 0 for group in teams for team in teams[group]}
    for group in teams:
        for i in range(4):
            for j in range(i+1, 4):
                team1, team2 = teams[group][i], teams[group][j]
                team1_pts, team2_pts = sim_group_stage_match(team1, team2, matchups)
                # print(f"Matchup: {team1} v {team2}")
                # print(f"Outcome: {team1_pts} - {team2_pts}")
                if team1_pts == -1:
                    return -1
                group_points[team1] += team1_pts
                group_points[team2] += team2_pts
    return group_points

# Determine which teams from each group make it into the knockout round and who will play who. Return the matchups.
# Note that is based on the rules for the competition.
def determine_copa_knockout(teams, group_points, team_info):
    # Find the group standings and get a list of the first and second place teams (in order from A to D)
    group_standings = {group: sorted(teams[group], key=lambda team: group_points[team], reverse=True) for group in teams}
    first_place = [(team, group) for group, teams_in_group in teams.items() for team in group_standings[group][:1]]
    second_place = [(team, group) for group, teams_in_group in teams.items() for team in group_standings[group][1:2]]
    # print(f"Group Points: {group_points}")
    # print(f"Group Standings: {group_standings}")
    # print(f"First: {first_place}")
    # print(f"Second: {second_place}")
    # print("-----------")

    # R016 Matchups
    matchup = pd.DataFrame(columns=["home_team", "away_team", "home_norm_points", "away_norm_points", "game_num"])
    game_num = 9 # starts at nine to make integrating with RO16 from Euros easier (since there are 8 extra games before this round in the Euros)
    # keep in mind, game_num generally starts counting at 1, not 0.
    for team in first_place:
        home_team = team[0]
        if "Group A" in team:
            away_team = [team2 for team2 in second_place if "Group B" in team2][0][0]
        elif "Group B" in team:
            away_team = [team2 for team2 in second_place if "Group A" in team2][0][0]
        elif "Group C" in team:
            away_team = [team2 for team2 in second_place if "Group D" in team2][0][0]
        elif "Group D" in team:
            away_team = [team2 for team2 in second_place if "Group C" in team2][0][0]
        home_data = team_info[team_info["country_full"] == home_team]
        away_data = team_info[team_info["country_full"] == away_team]
        home_norm_points = home_data["norm_points"].values[0]
        away_norm_points = away_data["norm_points"].values[0]
        new_row = pd.DataFrame({"home_team": [home_team], "away_team": [away_team], "home_norm_points": [home_norm_points], "away_norm_points": [away_norm_points], "game_num": [game_num]})
        matchup = pd.concat([matchup, new_row], ignore_index=True)
        game_num += 1
    return matchup

# Determine which teams from each group make it into the knockout round and who will play who. Return the matchups.
# Note that is based on the rules for the competition.
def determine_euro_knockout(teams, group_points, team_info):
    # Find the group standings and get a list of the first, second, and third place teams (in order from A to F)
    group_standings = {group: sorted(teams[group], key=lambda team: group_points[team], reverse=True) for group in teams}
    first_place = [(team, group) for group, teams_in_group in teams.items() for team in group_standings[group][:1]]
    second_place = [(team, group) for group, teams_in_group in teams.items() for team in group_standings[group][1:2]]
    third_place = [(team, group) for group, teams_in_group in teams.items() for team in group_standings[group][2:3]]
    # Sort the third place teams to find the best four teams.
    third_place = sorted(third_place, key=lambda team_group: group_points[team_group[0]], reverse=True)[:4]

    # print(f"Group Points: {group_points}")
    # print(f"Group Standings: {group_standings}")
    # print(f"First: {first_place}")
    # print(f"Second: {second_place}")
    # print(f"Third: {third_place}")
    
    # Store which groups had a third place team qualify.
    third_groups = [t[1] for t in third_place]

    # Matchups based on which groups had a third place team qualify.
    condition_map = {
        ("Group A", "Group B", "Group C", "Group D"): "ABCD",
        ("Group A", "Group B", "Group C", "Group E"): "ABCE",
        ("Group A", "Group B", "Group C", "Group F"): "ABCF",
        ("Group A", "Group B", "Group D", "Group E"): "ABDE",
        ("Group A", "Group B", "Group D", "Group F"): "ABDF",
        ("Group A", "Group B", "Group E", "Group F"): "ABEF",
        ("Group A", "Group C", "Group D", "Group E"): "ACDE",
        ("Group A", "Group C", "Group D", "Group F"): "ACDF",
        ("Group A", "Group C", "Group E", "Group F"): "ACEF",
        ("Group A", "Group D", "Group E", "Group F"): "ADEF",
        ("Group B", "Group C", "Group D", "Group E"): "BCDE",
        ("Group B", "Group C", "Group D", "Group F"): "BCDF",
        ("Group B", "Group C", "Group E", "Group F"): "BCEF",
        ("Group B", "Group D", "Group E", "Group F"): "BDEF",
        ("Group C", "Group D", "Group E", "Group F"): "CDEF",
    }
    for condition, output in condition_map.items():
        # Check all permutations of condition to find a match
        for perm in permutations(condition):
            if all(group in third_groups for group in perm):
                break
        else:
            continue
        break

    # Build each matchup based on the outcomes of the group stage.
    matchup = pd.DataFrame(columns=["home_team", "away_team", "home_norm_points", "away_norm_points", "game_num"])
    game_num = 1
    home_team = None
    for team in second_place:
        away_team = team[0]
        if "Group B" in team:
            home_team = [team2 for team2 in second_place if "Group A" in team2][0][0]
            game_num = 8
        elif "Group C" in team:
            home_team = [team2 for team2 in first_place if "Group A" in team2][0][0]
            game_num = 2
        elif "Group E" in team:
            home_team = [team2 for team2 in second_place if "Group D" in team2][0][0]
            game_num = 4
        elif "Group F" in team:
            home_team = [team2 for team2 in first_place if "Group D" in team2][0][0]
            game_num = 6
        else:
            continue
        home_data = team_info[team_info["country_full"] == home_team]
        away_data = team_info[team_info["country_full"] == away_team]
        home_norm_points = home_data["norm_points"].values[0]
        away_norm_points = away_data["norm_points"].values[0]
        new_row = pd.DataFrame({"home_team": [home_team], "away_team": [away_team], "home_norm_points": [home_norm_points], "away_norm_points": [away_norm_points], "game_num": [game_num]})
        matchup = pd.concat([matchup, new_row], ignore_index=True)
    
    if output == "ABCD":
        order = ["Group A", "Group D", "Group B", "Group C"]
    elif output == "ABCE":
        order = ["Group A", "Group E", "Group B", "Group C"]
    elif output == "ABCF":
        order = ["Group A", "Group F", "Group B", "Group C"]
    elif output == "ABDE":
        order = ["Group D", "Group E", "Group A", "Group B"]
    elif output == "ABDF":
        order = ["Group D", "Group F", "Group A", "Group B"]
    elif output == "ABEF":
        order = ["Group E", "Group F", "Group B", "Group A"]
    elif output == "ACDE":
        order = ["Group E", "Group D", "Group C", "Group A"]
    elif output == "ACDF":
        order = ["Group F", "Group D", "Group C", "Group A"]
    elif output == "ACEF":
        order = ["Group E", "Group F", "Group C", "Group A"]
    elif output == "ADEF":
        order = ["Group E", "Group F", "Group D", "Group A"]
    elif output == "BCDE":
        order = ["Group E", "Group D", "Group B", "Group C"]
    elif output == "BCDF":
        order = ["Group F", "Group D", "Group C", "Group B"]
    elif output == "BCEF":
        order = ["Group F", "Group E", "Group C", "Group B"]
    elif output == "BDEF":
        order = ["Group F", "Group E", "Group D", "Group B"]
    elif output == "CDEF":
        order = ["Group F", "Group E", "Group D", "Group C"]
    away_team = None
    for team in first_place:
        home_team = team[0]
        if "Group B" in team:
            away_team = [team2 for team2 in third_place if order[0] in team2][0][0]
            game_num = 1
        elif "Group C" in team:
            away_team = [team2 for team2 in third_place if order[1] in team2][0][0]
            game_num = 7
        elif "Group E" in team:
            away_team = [team2 for team2 in third_place if order[2] in team2][0][0]
            game_num =  5
        elif "Group F" in team:
            away_team = [team2 for team2 in third_place if order[3] in team2][0][0]
            game_num = 3
        else:
            continue
        home_data = team_info[team_info["country_full"] == home_team]
        away_data = team_info[team_info["country_full"] == away_team]
        home_norm_points = home_data["norm_points"].values[0]
        away_norm_points = away_data["norm_points"].values[0]
        new_row = pd.DataFrame({"home_team": [home_team], "away_team": [away_team], "home_norm_points": [home_norm_points], "away_norm_points": [away_norm_points], "game_num": [game_num]})
        matchup = pd.concat([matchup, new_row], ignore_index=True)
    return matchup

# Simulate a knockout match (no ties). Return the outcome (points are different for each round and are therefore calculated separately).
def sim_knockout_match(team1, team2, matchups):
    # home and away are already set now
    match = matchups[(matchups['home_team'] == team1) & (matchups['away_team'] == team2)]
    if match.empty:
        match = matchups[(matchups['home_team'] == team2) & (matchups['away_team'] == team1)]
        if match.empty:
            print(f"ERROR: Matchup does not exist: {team1} v {team2}")
            return (-1, -1)
        team1_prob = match['away_win_prob'].values[0]
        team2_prob = match['home_win_prob'].values[0]
    else:
        team1_prob = match['home_win_prob'].values[0]
        team2_prob = match['away_win_prob'].values[0]
    
    match_outcome = np.random.rand()
    if match_outcome < team1_prob:
        return 1 # team 1 wins
    else:
        return 2 # team 2 wins

# Simulate an entire knockout round (eg. the quarterfinals). Return the winners. 
def sim_knockout_round(bracket, matchup_probs):
    winners = pd.DataFrame(columns=["team_name", "game_num"])
    for _, row in bracket.iterrows():
        team1, team2 = row["home_team"], row["away_team"]
        outcome = sim_knockout_match(team1, team2, matchup_probs)
        if outcome == 1:
            new_team = pd.DataFrame({"team_name": [team1], "game_num": row["game_num"]})
        else:
            new_team = pd.DataFrame({"team_name": [team2], "game_num": row["game_num"]})
        winners = pd.concat([winners, new_team])
    return winners

# Simulate the entire knockout competition (every round). Track how far each team makes it and how many points they get.
def sim_knockout(bracket, matchup_probs):
    # For the challenge, winning R016 is 4pts, winning QF is 6, SF is 8, and F is 10 (added on, not cumulative, so a winner would get 28pts)
    distance = pd.DataFrame(columns=["team_name", "points"])
    #RO16
    if len(bracket.index) == 8: #Euros
        winners = sim_knockout_round(bracket, matchup_probs)
        teams = winners["team_name"].unique()
        for team in teams:
            if team in distance["team_name"].unique():
                idx = distance.loc[distance["team_name"] == team].index[0]
                distance.at[idx, "points"] += 4
            else:
                new_row = pd.DataFrame({"team_name": [team], "points": [4]})
                distance = pd.concat([distance, new_row], ignore_index=True)
        qf_teams = [
            [winners[winners["game_num"] == 1].at[0, "team_name"], winners[winners["game_num"] == 2].at[0, "team_name"], 9],
            [winners[winners["game_num"] == 3].at[0, "team_name"], winners[winners["game_num"] == 4].at[0, "team_name"], 10],
            [winners[winners["game_num"] == 5].at[0, "team_name"], winners[winners["game_num"] == 6].at[0, "team_name"], 11],
            [winners[winners["game_num"] == 7].at[0, "team_name"], winners[winners["game_num"] == 8].at[0, "team_name"], 12]
        ]
        qf = pd.DataFrame(qf_teams, columns=["home_team", "away_team", "game_num"])
    else:
        qf = bracket
    #QF
    winners = sim_knockout_round(qf, matchup_probs)
    teams = winners["team_name"].unique()
    for team in teams:
        if team in distance["team_name"].unique():
            idx = distance.loc[distance["team_name"] == team].index[0]
            distance.at[idx, "points"] += 6
        else:
            new_row = pd.DataFrame({"team_name": [team], "points": [6]})
            distance = pd.concat([distance, new_row], ignore_index=True)
    sf_teams = [
            [winners[winners["game_num"] == 9].at[0, "team_name"], winners[winners["game_num"] == 10].at[0, "team_name"], 13],
            [winners[winners["game_num"] == 11].at[0, "team_name"], winners[winners["game_num"] == 12].at[0, "team_name"], 14]
        ]
    sf = pd.DataFrame(sf_teams, columns=["home_team", "away_team", "game_num"])

    #SF
    winners = sim_knockout_round(sf, matchup_probs)
    teams = winners["team_name"].unique()
    for team in teams:
        # Don't need to check here, as all teams in winners must be in distance by now, no matter the tournament.
        # A check would be good for safety, but I would rather an error throw than add an incorrect amount of points.
        idx = distance.loc[distance["team_name"] == team].index[0]
        distance.at[idx, "points"] += 8
    f_teams = [
            [winners[winners["game_num"] == 13].at[0, "team_name"], winners[winners["game_num"] == 14].at[0, "team_name"], 15]
        ]
    f = pd.DataFrame(f_teams, columns=["home_team", "away_team", "game_num"])

    #Final
    winner = sim_knockout_round(f, matchup_probs)
    teams = winner["team_name"].unique()
    if len(teams) != 1:
        print("ERROR: Too many winners")
        return -1
    for team in teams:
        # Only one team, but easier to do it in the same format for clarity and bug testing.
        idx = distance.loc[distance["team_name"] == team].index[0]
        distance.at[idx, "points"] += 10
    return distance

# Create a dataframe on the performance of each team in the simulation. Return the performance dataframe.
def performance(knockout, group):
    group_df = pd.DataFrame({"team_name": list(group.keys()), "points": list(group.values())})
    performance = pd.merge(group_df, knockout, how="left", on="team_name")
    performance["points_y"] = performance["points_y"].fillna(0)
    performance["points"] = performance["points_x"] + performance["points_y"]
    performance = performance.drop(columns=["points_x", "points_y"])
    return performance

# Run a single simulation. Return the performance of all teams (from both competitions).
def run_sim():
    copa_group_points = sim_group(copa_teams, copa_matchups)
    copa_knockout = determine_copa_knockout(copa_teams, copa_group_points, copa).sort_values(by="game_num")
    # copa_knockout.to_csv("copa_knockout_matchups.csv", index=False)
    copa_knockout_points = sim_knockout(copa_knockout, copa_matchups)
    copa_performance = performance(copa_knockout_points, copa_group_points)
    euro_group_points = sim_group(euro_teams, euro_matchups)
    euro_knockout = determine_euro_knockout(euro_teams, euro_group_points, euro).sort_values(by="game_num")
    # euro_knockout.to_csv("euro_knockout_matchups.csv", index=False)
    euro_knockout_points = sim_knockout(euro_knockout, euro_matchups)
    euro_performance = performance(euro_knockout_points, euro_group_points)
    team_performance = pd.concat([copa_performance, euro_performance], ignore_index=True)
    return team_performance

# Run 10000 simulations and calculate the expected points (xP) for each team.
# Weigh the xPs against the costs of each team, and maximize the amount of points
# based on the challenge's restrictions. Print the selected teams.
# Note that for lack of time I used xbar (the sample mean) as the expected points.
# This still works well as a simplified expected value as the E[xbar] = population mean.
def choose_teams():
    n = 10000
    all_performances = pd.DataFrame(columns=["team_name", "points"])
    for i in range(n):
        team_performance = run_sim()
        # print(team_performance)
        all_performances = pd.concat([all_performances, team_performance])
    # Calculate the mean performance of each team.
    xP = all_performances.groupby("team_name").mean().sort_values(by="points", ascending=False)

    # Combine the xPs and costs of each team.
    costs = pd.read_csv("data/team_costs.csv")
    xP_costs = pd.merge(xP, costs, how="inner", on="team_name")
    conference = pd.read_csv("data/conference.csv")
    xP_costs = xP_costs.merge(conference, how="left", on="team_name")
    print(xP_costs)
    points = xP_costs['points'].values
    costs = xP_costs['cost'].values
    team_names = xP_costs['team_name'].values
    tournament = xP_costs['tournament'].values

    # Challenge restrictions.
    BUDGET = 100
    MAX_TEAMS = 8   
    N = len(points)
    c = -points
    A = np.vstack([costs, np.ones(N)])
    b_upper = [BUDGET, MAX_TEAMS]
    b_lower = [0, 0]

    # Ensure at least one team from each conference is included.
    unique_tournaments = np.unique(tournament)
    for conf in unique_tournaments:
        conf_mask = (tournament == conf).astype(int)
        A = np.vstack([A, conf_mask])
        b_upper.append(np.inf)
        b_lower.append(1)  # At least one team from this conference

    # Find the combination of teams with maximized points.
    constraints = LinearConstraint(A, b_lower, b_upper)
    bounds = Bounds(0,1)
    res = milp(c, integrality=np.ones(N), constraints=constraints, bounds=bounds)

    selected_teams_indices = np.where(res.x > 0.99)[0]
    selected_teams = team_names[selected_teams_indices]
    max_expected_value = -res.fun

    # Print the selected teams.
    print(selected_teams, max_expected_value)



choose_teams()