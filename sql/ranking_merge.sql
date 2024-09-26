-- Combines the ranking information for the home and away team in each match.
SELECT
    res.*,
    ranA.total_points AS home_total_points,
    ranA.norm_points AS home_norm_points,
    ranA.confederation AS home_confederation,
    ranB.total_points AS away_total_points,
    ranB.norm_points AS away_norm_points,
    ranB.confederation AS away_confederation
FROM results res
LEFT JOIN rankings ranA ON res.home_team = ranA.country_full
LEFT JOIN rankings ranB ON res.away_team = ranB.country_full
WHERE
    ranA.rank_date = (
        SELECT MAX(ranA_sub.rank_date)
        FROM rankings AS ranA_sub
        WHERE ranA_sub.country_full = res.home_team
            AND ranA_sub.rank_date <= res.date
    )
    AND ranB.rank_date = (
        SELECT MAX(ranB_sub.rank_date)
        FROM rankings AS ranB_sub
        WHERE ranB_sub.country_full = res.away_team
            AND ranB_sub.rank_date <= res.date
    )