-- Supporting analytics only; these results are directional evidence, not causal proof.
-- 1. Monthly GSC impressions, clicks and CTR by branded vs non-branded query type.
SELECT substr(date, 1, 7) AS audit_month,
             query_type,
             SUM(impressions) AS impressions,
             SUM(clicks) AS clicks,
             ROUND(100.0 * SUM(clicks) / NULLIF(SUM(impressions), 0), 4) AS ctr_pct
FROM gsc_daily
GROUP BY substr(date, 1, 7), query_type
ORDER BY audit_month, query_type;

-- 2. Monthly non-branded GSC performance for cooling guide, comparison and product pages.
SELECT substr(date, 1, 7) AS audit_month,
             'non_branded' AS query_type,
             landing_page,
             page_type,
             SUM(impressions) AS impressions,
             SUM(clicks) AS clicks,
             ROUND(100.0 * SUM(clicks) / NULLIF(SUM(impressions), 0), 4) AS ctr_pct
FROM gsc_daily
WHERE query_type = 'non_branded'
    AND landing_page IN ('/guides/cooling-sheets', '/compare/cooling-sheet-brands', '/products/cooling-sheet-set')
GROUP BY substr(date, 1, 7), landing_page, page_type
ORDER BY audit_month, page_type, landing_page;

-- 3. Monthly GA4 sessions, engaged sessions, conversions and revenue.
SELECT substr(date, 1, 7) AS audit_month,
             SUM(sessions) AS sessions,
             SUM(engaged_sessions) AS engaged_sessions,
             SUM(conversions) AS conversions,
             SUM(revenue_usd) AS revenue_usd
FROM ga4_daily
GROUP BY substr(date, 1, 7)
ORDER BY audit_month;
