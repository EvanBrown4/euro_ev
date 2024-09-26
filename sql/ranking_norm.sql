-- Calculates a normalized score for teams for each date (since scores are cumulative from the beginning of scoring,
-- and all teams are updated at the same time)
WITH RankingStats AS (
    SELECT
        *,
        MIN(total_points) OVER (PARTITION by rank_date) AS min_points,
        MAX(total_points) OVER (PARTITION by rank_date) AS max_points
    FROM rankings
)

SELECT
    *,
    100 * (total_points - min_points) / NULLIF(max_points - min_points, 0) AS norm_points
FROM RankingStats
ORDER BY rank_date, country_full