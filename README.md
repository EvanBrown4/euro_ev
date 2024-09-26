# International Competitions Team Expected Values
An expected value optimization of the European Championship and Copa America.

Run main.py to execute.

*Note*
This was completed over a very short time period, as I created it close to the start of the tournaments and needed it finished by the start of the tournaments.
Also, in place of debugging programs for the sake of time, there are lines throughout that utilize print statement debugging. They are all currently commented out, but they can be helpful in needed.

*Challenge Rules*
This was based off of the rules for a certain challenge. They are as follows:
There are 40 teams participating between the Euros and Copa America, each of which has been assigned a cost based off of the Vegas betting odds. The costs can be seen on the Selection Page. You have 100 points to spend on up to 8 teams. You cannot select teams only from one tournament, and you cannot select the same team twice.

Scoring: 

- Points in the group stage translate 1:1 over to points in this pool
- Round of 16 win: 4 points
- Quarterfinals win: 6 points
- Semifinals win: 8 points
- Championship: 10 points

(Knockout stage points from wins add upon each other, (i.e., a Euros quarterfinal win would mean 10 points in total from the Euros knockout stage))"

What I did
wanted to weight every team's schedule and find who had the highest ev of points

What I know was an issue
Didn't take into account injuries

What I want to do with more


Data Credit:
https://www.kaggle.com/datasets/gabipana7/fifa-rankings-and-international-matches-1992-2022
    - results.csv (renamed from fifa_matches.csv)
    - rankings.csv (renamed from fifa_rankings.csv)
My data (obtained from the tournaments' official websites):
    - euro.csv: All of the upcoming group stage matches for the euros.
    - conference.csv: All teams in the tournaments and what conference they are in.
    - team_costs.csv: All teams costs based on the challenge's rules.