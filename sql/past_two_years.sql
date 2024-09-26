WITH TEAMS AS (
    SELECT p.date, p.team_name, p.points
    FROM past_two_years p
    JOIN (
        SELECT team_name, MAX(date) AS max_date
        FROM past_two_years
        GROUP BY team_name
    ) AS max_dates
    ON p.team_name = max_dates.team_name AND p.date = max_dates.max_date
), RankingStats AS (
    SELECT
        *,
        (SELECT MIN(points) FROM TEAMS) AS min_points,
        (SELECT MAX(points) FROM TEAMS) AS max_points
    FROM TEAMS
)

SELECT
    team_name,
    points,
    100 * (points - min_points) / NULLIF(max_points - min_points, 0) AS norm_points
FROM RankingStats
ORDER BY team_name, points