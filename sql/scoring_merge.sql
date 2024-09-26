WITH RECENT_RANKINGS AS (
    SELECT
        *
    FROM rankings
    WHERE DATE(rank_date) = DATE('2024-04-04')
)
SELECT
    e.*,
    rh.norm_points AS home_norm_points,
    ra.norm_points AS away_norm_points,
FROM euros e
LEFT JOIN RECENT_RANKINGS rh ON e.home_team = rh.country_full
LEFT JOIN RECENT_RANKINGS ra ON e.away_team = ra.country_full
