# Databricks notebook source

# MAGIC %md
# MAGIC # 02 · Silver Layer — Incremental Cleaning & Joining
# MAGIC
# MAGIC Reads from Bronze, applies cleaning rules, and produces three cleaned tables
# MAGIC plus a single joined table aligned on weekly drought release dates.
# MAGIC Only processes Bronze records newer than the current Silver watermark.
# MAGIC All writes use `MERGE INTO` — safe to re-run.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Clean NOAA data — pivot long → wide, cast types, fill nulls
# MAGIC 2. Clean USGS GWL — cast types, forward-fill sparse gaps
# MAGIC 3. Clean USDM — cast date, validate percentages sum to 100
# MAGIC 4. Join all three on weekly cadence
# MAGIC 5. Audit via `DESCRIBE HISTORY` (Delta provenance)

# COMMAND ----------

# MAGIC %run ../configs/00_config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Clean NOAA Climate Data
# MAGIC
# MAGIC Bronze stores NOAA in **long format** (one row per datatype per day).
# MAGIC Silver pivots to **wide format** (one row per day with PRCP, TMAX, TMIN columns)
# MAGIC and fills sparse nulls with a 7-day rolling median.

# COMMAND ----------

# Only reprocess the lookback window to catch any late API corrections
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TBL_SILVER_CLIMATE}
    USING DELTA AS
    SELECT
        CAST(NULL AS DATE)   AS date,
        CAST(NULL AS DOUBLE) AS prcp_in,
        CAST(NULL AS DOUBLE) AS tmax_f,
        CAST(NULL AS DOUBLE) AS tmin_f,
        CAST(NULL AS DOUBLE) AS temp_range_f,
        CAST(NULL AS TIMESTAMP) AS updated_at
    WHERE 1=0
""")

silver_climate_watermark = get_watermark(TBL_SILVER_CLIMATE, "date", HISTORY_START)
print(f"Silver climate watermark : {silver_climate_watermark}")

# Pivot NOAA long → wide using conditional aggregation, then clean
df_climate_clean = spark.sql(f"""
    WITH pivoted AS (
        SELECT
            CAST(SUBSTR(date, 1, 10) AS DATE)                               AS date,
            MAX(CASE WHEN datatype = 'PRCP' THEN value END)                  AS prcp_in,
            MAX(CASE WHEN datatype = 'TMAX' THEN value / 10.0 END)           AS tmax_f,
            MAX(CASE WHEN datatype = 'TMIN' THEN value / 10.0 END)           AS tmin_f
        FROM {TBL_BRONZE_CLIMATE}
        WHERE CAST(SUBSTR(date, 1, 10) AS DATE) >= '{silver_climate_watermark}'
        GROUP BY CAST(SUBSTR(date, 1, 10) AS DATE)
    ),
    with_rolling_fill AS (
        SELECT
            date,
            -- Fill nulls with 7-day rolling average (handles sparse missing days)
            COALESCE(prcp_in, AVG(prcp_in) OVER (
                ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )) AS prcp_in,
            COALESCE(tmax_f, AVG(tmax_f) OVER (
                ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )) AS tmax_f,
            COALESCE(tmin_f, AVG(tmin_f) OVER (
                ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )) AS tmin_f
        FROM pivoted
    )
    SELECT
        date,
        ROUND(prcp_in, 4)               AS prcp_in,
        ROUND(tmax_f,  1)               AS tmax_f,
        ROUND(tmin_f,  1)               AS tmin_f,
        ROUND(tmax_f - tmin_f, 1)       AS temp_range_f,
        current_timestamp()             AS updated_at
    FROM with_rolling_fill
    WHERE date IS NOT NULL
""")

df_climate_clean.createOrReplaceTempView("silver_climate_incoming")

spark.sql(f"""
    MERGE INTO {TBL_SILVER_CLIMATE} AS target
    USING silver_climate_incoming AS source
    ON target.date = source.date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Silver climate rows: {spark.table(TBL_SILVER_CLIMATE).count():,}")
display(spark.table(TBL_SILVER_CLIMATE).orderBy("date", ascending=False).limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Clean USGS Groundwater Level

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TBL_SILVER_GWL}
    USING DELTA AS
    SELECT
        CAST(NULL AS DATE)   AS date,
        CAST(NULL AS DOUBLE) AS gwl_ft,
        CAST(NULL AS DOUBLE) AS gwl_7d_avg_ft,
        CAST(NULL AS TIMESTAMP) AS updated_at
    WHERE 1=0
""")

silver_gwl_watermark = get_watermark(TBL_SILVER_GWL, "date", HISTORY_START)

df_gwl_clean = spark.sql(f"""
    WITH base AS (
        SELECT
            CAST(date AS DATE)  AS date,
            gwl_ft
        FROM {TBL_BRONZE_GWL}
        WHERE CAST(date AS DATE) >= '{silver_gwl_watermark}'
          AND gwl_ft IS NOT NULL
          AND gwl_ft BETWEEN 0 AND 1000   -- sanity bounds for feet below surface
    )
    SELECT
        date,
        ROUND(gwl_ft, 2)  AS gwl_ft,
        -- 7-day rolling average smooths out sensor noise
        ROUND(AVG(gwl_ft) OVER (
            ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2)             AS gwl_7d_avg_ft,
        current_timestamp() AS updated_at
    FROM base
""")

df_gwl_clean.createOrReplaceTempView("silver_gwl_incoming")

spark.sql(f"""
    MERGE INTO {TBL_SILVER_GWL} AS target
    USING silver_gwl_incoming AS source
    ON target.date = source.date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Silver GWL rows: {spark.table(TBL_SILVER_GWL).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Clean USDM Drought Data
# MAGIC
# MAGIC Validates that drought category percentages sum to ~100%,
# MAGIC and computes a composite `drought_severity_score` as a weighted index.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TBL_SILVER_DROUGHT}
    USING DELTA AS
    SELECT
        CAST(NULL AS DATE)   AS map_date,
        CAST(NULL AS DOUBLE) AS none_pct,
        CAST(NULL AS DOUBLE) AS d0_pct,
        CAST(NULL AS DOUBLE) AS d1_pct,
        CAST(NULL AS DOUBLE) AS d2_pct,
        CAST(NULL AS DOUBLE) AS d3_pct,
        CAST(NULL AS DOUBLE) AS d4_pct,
        CAST(NULL AS DOUBLE) AS drought_severity_score,
        CAST(NULL AS DOUBLE) AS extreme_drought_pct,
        CAST(NULL AS TIMESTAMP) AS updated_at
    WHERE 1=0
""")

silver_drought_watermark = get_watermark(TBL_SILVER_DROUGHT, "map_date", HISTORY_START)

df_drought_clean = spark.sql(f"""
    WITH base AS (
        SELECT
            CAST(map_date AS DATE)  AS map_date,
            COALESCE(none_pct, 0.0) AS none_pct,
            COALESCE(d0_pct,   0.0) AS d0_pct,
            COALESCE(d1_pct,   0.0) AS d1_pct,
            COALESCE(d2_pct,   0.0) AS d2_pct,
            COALESCE(d3_pct,   0.0) AS d3_pct,
            COALESCE(d4_pct,   0.0) AS d4_pct
        FROM {TBL_BRONZE_DROUGHT}
        WHERE CAST(map_date AS DATE) >= '{silver_drought_watermark}'
    ),
    validated AS (
        SELECT *,
            ROUND(none_pct + d0_pct + d1_pct + d2_pct + d3_pct + d4_pct, 1) AS pct_sum
        FROM base
    )
    SELECT
        map_date,
        none_pct, d0_pct, d1_pct, d2_pct, d3_pct, d4_pct,
        -- Weighted severity: D4 counts most, no-drought counts 0
        -- Weights: D0=1, D1=2, D2=3, D3=4, D4=5 (normalized to 0–100 scale)
        ROUND(
            (d0_pct * 1 + d1_pct * 2 + d2_pct * 3 + d3_pct * 4 + d4_pct * 5) / 5.0,
        2) AS drought_severity_score,
        ROUND(d3_pct + d4_pct, 2) AS extreme_drought_pct,
        current_timestamp()       AS updated_at
    FROM validated
    WHERE ABS(pct_sum - 100.0) <= 1.0   -- drop records with bad data (>1% rounding error)
""")

df_drought_clean.createOrReplaceTempView("silver_drought_incoming")

spark.sql(f"""
    MERGE INTO {TBL_SILVER_DROUGHT} AS target
    USING silver_drought_incoming AS source
    ON target.map_date = source.map_date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Silver drought rows: {spark.table(TBL_SILVER_DROUGHT).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Join All Three Sources
# MAGIC
# MAGIC USDM is the spine (weekly cadence). Climate and GWL are aggregated to the week
# MAGIC ending on each USDM `map_date` using `DATE_TRUNC` alignment.
# MAGIC
# MAGIC Showcasing: multi-table CTE join, window-based weekly aggregation, LEFT JOIN strategy.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TBL_SILVER_JOINED}
    USING DELTA AS
    SELECT
        CAST(NULL AS DATE)   AS week_end_date,
        CAST(NULL AS DOUBLE) AS none_pct,
        CAST(NULL AS DOUBLE) AS d0_pct,
        CAST(NULL AS DOUBLE) AS d1_pct,
        CAST(NULL AS DOUBLE) AS d2_pct,
        CAST(NULL AS DOUBLE) AS d3_pct,
        CAST(NULL AS DOUBLE) AS d4_pct,
        CAST(NULL AS DOUBLE) AS drought_severity_score,
        CAST(NULL AS DOUBLE) AS extreme_drought_pct,
        CAST(NULL AS DOUBLE) AS weekly_prcp_in,
        CAST(NULL AS DOUBLE) AS avg_tmax_f,
        CAST(NULL AS DOUBLE) AS avg_tmin_f,
        CAST(NULL AS DOUBLE) AS avg_temp_range_f,
        CAST(NULL AS DOUBLE) AS avg_gwl_ft,
        CAST(NULL AS DOUBLE) AS gwl_weekly_change_ft,
        CAST(NULL AS TIMESTAMP) AS updated_at
    WHERE 1=0
""")

silver_joined_watermark = get_watermark(TBL_SILVER_JOINED, "week_end_date", HISTORY_START)

df_joined = spark.sql(f"""
    WITH drought_spine AS (
        -- Drought is the spine — all joins hang off its weekly dates
        SELECT
            map_date          AS week_end_date,
            none_pct, d0_pct, d1_pct, d2_pct, d3_pct, d4_pct,
            drought_severity_score,
            extreme_drought_pct
        FROM {TBL_SILVER_DROUGHT}
        WHERE map_date >= '{silver_joined_watermark}'
    ),
    weekly_climate AS (
        -- Aggregate daily climate to the week ending on each USDM map_date
        -- USDM map_date is always a Tuesday; we align the preceding 7 days
        SELECT
            DATE_ADD(DATE_TRUNC('WEEK', date), 1)   AS week_end_date,  -- Mon start → Tue
            ROUND(SUM(prcp_in), 3)                   AS weekly_prcp_in,
            ROUND(AVG(tmax_f), 1)                    AS avg_tmax_f,
            ROUND(AVG(tmin_f), 1)                    AS avg_tmin_f,
            ROUND(AVG(temp_range_f), 1)              AS avg_temp_range_f
        FROM {TBL_SILVER_CLIMATE}
        GROUP BY DATE_ADD(DATE_TRUNC('WEEK', date), 1)
    ),
    weekly_gwl AS (
        SELECT
            DATE_ADD(DATE_TRUNC('WEEK', date), 1)    AS week_end_date,
            ROUND(AVG(gwl_ft), 2)                    AS avg_gwl_ft,
            -- Week-over-week change: positive = water table dropping (worse)
            ROUND(AVG(gwl_ft) - LAG(AVG(gwl_ft), 1) OVER (
                ORDER BY DATE_ADD(DATE_TRUNC('WEEK', date), 1)
            ), 3) AS gwl_weekly_change_ft
        FROM {TBL_SILVER_GWL}
        GROUP BY DATE_ADD(DATE_TRUNC('WEEK', date), 1)
    )
    SELECT
        d.week_end_date,
        d.none_pct, d.d0_pct, d.d1_pct, d.d2_pct, d.d3_pct, d.d4_pct,
        d.drought_severity_score,
        d.extreme_drought_pct,
        c.weekly_prcp_in,
        c.avg_tmax_f,
        c.avg_tmin_f,
        c.avg_temp_range_f,
        g.avg_gwl_ft,
        g.gwl_weekly_change_ft,
        current_timestamp() AS updated_at
    FROM drought_spine d
    LEFT JOIN weekly_climate c ON d.week_end_date = c.week_end_date
    LEFT JOIN weekly_gwl     g ON d.week_end_date = g.week_end_date
""")

df_joined.createOrReplaceTempView("silver_joined_incoming")

spark.sql(f"""
    MERGE INTO {TBL_SILVER_JOINED} AS target
    USING silver_joined_incoming AS source
    ON target.week_end_date = source.week_end_date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Silver joined rows: {spark.table(TBL_SILVER_JOINED).count():,}")
display(spark.table(TBL_SILVER_JOINED).orderBy("week_end_date", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Data Quality Checks

# COMMAND ----------

dq_sql = f"""
SELECT
    COUNT(*)                                        AS total_weeks,
    SUM(CASE WHEN weekly_prcp_in IS NULL THEN 1 END) AS missing_climate,
    SUM(CASE WHEN avg_gwl_ft IS NULL THEN 1 END)     AS missing_gwl,
    SUM(CASE WHEN extreme_drought_pct > 100 THEN 1 END) AS invalid_drought_pct,
    MIN(week_end_date)                               AS earliest_week,
    MAX(week_end_date)                               AS latest_week
FROM {TBL_SILVER_JOINED}
"""
display(spark.sql(dq_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Delta Provenance — DESCRIBE HISTORY
# MAGIC
# MAGIC Every `MERGE INTO` operation is recorded in the Delta transaction log.
# MAGIC Use this for auditing, debugging, and time travel.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {TBL_SILVER_JOINED}"))

# COMMAND ----------

# Time travel example: read the state of the joined table before the most recent merge
display(spark.sql(f"""
    SELECT week_end_date, drought_severity_score, extreme_drought_pct
    FROM {TBL_SILVER_JOINED} VERSION AS OF 0
    ORDER BY week_end_date DESC
    LIMIT 5
"""))
