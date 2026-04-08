# Databricks notebook source

# MAGIC %md
# MAGIC # 01 · Bronze Layer — Incremental API Ingestion
# MAGIC
# MAGIC Fetches **new data only** from three live sources and merges into raw Delta tables.
# MAGIC Safe to re-run at any time — idempotent via `MERGE INTO`.
# MAGIC
# MAGIC | Source | Cadence | API |
# MAGIC |---|---|---|
# MAGIC | NOAA CDO | Daily (1-2 day lag) | `ncei.noaa.gov/cdo-web/api/v2` |
# MAGIC | USGS NWIS | Daily | `dataretrieval` library |
# MAGIC | USDM | Weekly (every Tuesday) | `usdm.unl.edu/services/api` |
# MAGIC
# MAGIC **Recommended schedule:** Run daily via Databricks Workflows.

# COMMAND ----------

# Create token widget before pip install so it survives the kernel restart.
# Paste your NOAA token in the box above, then run all.
# NEVER hardcode your token in this file.
dbutils.widgets.text("noaa_token", "", "NOAA API Token")

# COMMAND ----------

# MAGIC %pip install dataretrieval requests --quiet

# COMMAND ----------

# MAGIC %run ../configs/00_config

# COMMAND ----------

import requests
import time
import io
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import dataretrieval.nwis as nwis
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, DateType, TimestampType,
)

INGESTED_AT = datetime.utcnow()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · NOAA CDO API — Daily Climate Observations
# MAGIC
# MAGIC Station **GHCND:USW00003145** (Yuma MCAS, AZ).
# MAGIC Fetches `PRCP`, `TMAX`, `TMIN` in standard units (inches / °F).
# MAGIC
# MAGIC Stores data in **long format** (one row per datatype per day) in the bronze layer.
# MAGIC Silver layer will pivot to wide format.

# COMMAND ----------

def _fetch_noaa_page(start: str, end: str, offset: int, token: str) -> List[Dict]:
    """Single paginated request to the NOAA CDO API."""
    params = {
        "datasetid":      NOAA_DATASET_ID,
        "stationid":      NOAA_STATION_ID,
        "datatypeid":     NOAA_DATATYPES,
        "startdate":      start,
        "enddate":        end,
        "units":          "standard",
        "limit":          NOAA_PAGE_LIMIT,
        "offset":         offset,
        "includemetadata": "false",
    }
    resp = requests.get(
        NOAA_BASE_URL,
        params=params,
        headers={"token": token},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_noaa_incremental(fetch_start: str, fetch_end: str, token: str) -> List[Dict]:
    """
    Fetch NOAA daily records between fetch_start and fetch_end.
    Handles CDO pagination (max 1 000 records/request) and the 1-year-per-request limit
    by splitting the window into annual chunks automatically.
    Rate-limits to stay within the 5 req/s CDO API limit.
    """
    from dateutil.relativedelta import relativedelta

    all_records: List[Dict] = []
    # Split into ≤1-year windows to satisfy the CDO API constraint
    window_start = datetime.strptime(fetch_start, "%Y-%m-%d").date()
    window_end   = datetime.strptime(fetch_end,   "%Y-%m-%d").date()

    chunk_start = window_start
    while chunk_start <= window_end:
        chunk_end = min(chunk_start.replace(year=chunk_start.year + 1) - timedelta(days=1), window_end)
        s, e = chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")

        offset = 1
        while True:
            page = _fetch_noaa_page(s, e, offset, token)
            all_records.extend(page)
            if len(page) < NOAA_PAGE_LIMIT:
                break
            offset += NOAA_PAGE_LIMIT
            time.sleep(0.25)   # 4 req/s to stay under the 5 req/s limit

        chunk_start = chunk_end + timedelta(days=1)

    return all_records


# ── Quick token validation ─────────────────────────────────────────────────────
assert NOAA_TOKEN, "NOAA_TOKEN is empty — paste your token in the widget above and re-run."
test_resp = requests.get(
    NOAA_BASE_URL,
    params={"datasetid": "GHCND", "stationid": NOAA_STATION_ID,
            "startdate": "2024-01-01", "enddate": "2024-01-03", "limit": 1},
    headers={"token": NOAA_TOKEN},
    timeout=30,
)
assert test_resp.status_code == 200, f"NOAA API error {test_resp.status_code}: {test_resp.text}"
print("NOAA token OK")

# ── Determine incremental fetch window ────────────────────────────────────────
climate_watermark = get_watermark(TBL_BRONZE_CLIMATE, "CAST(date AS DATE)", HISTORY_START)
print(f"NOAA fetch window : {climate_watermark} → {FETCH_END}")

noaa_records = fetch_noaa_incremental(climate_watermark, FETCH_END, NOAA_TOKEN)
print(f"Records fetched   : {len(noaa_records):,}")

# COMMAND ----------

# ── Define schema and load into Spark ─────────────────────────────────────────
schema_climate_raw = StructType([
    StructField("date",       StringType(),    nullable=False),  # raw ISO string from API
    StructField("datatype",   StringType(),    nullable=False),  # PRCP | TMAX | TMIN
    StructField("station",    StringType(),    nullable=False),
    StructField("value",      DoubleType(),    nullable=True),
    StructField("attributes", StringType(),    nullable=True),
    StructField("ingested_at",TimestampType(), nullable=False),
])

rows_climate = [
    (r["date"], r["datatype"], r["station"],
     float(r["value"]) if r.get("value") is not None else None,
     r.get("attributes"), INGESTED_AT)
    for r in noaa_records
]

df_climate_new = spark.createDataFrame(rows_climate, schema=schema_climate_raw)

# ── MERGE INTO bronze — idempotent upsert ─────────────────────────────────────
# Unique key: (date, datatype, station) — handles API corrections automatically
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TBL_BRONZE_CLIMATE}
    (date STRING, datatype STRING, station STRING,
     value DOUBLE, attributes STRING, ingested_at TIMESTAMP)
    USING DELTA
""")

df_climate_new.createOrReplaceTempView("climate_incoming")

spark.sql(f"""
    MERGE INTO {TBL_BRONZE_CLIMATE} AS target
    USING climate_incoming AS source
    ON  target.date     = source.date
    AND target.datatype = source.datatype
    AND target.station  = source.station
    WHEN MATCHED AND target.value != source.value THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Bronze climate current row count: {spark.table(TBL_BRONZE_CLIMATE).count():,}")
display(spark.table(TBL_BRONZE_CLIMATE).orderBy("date", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · USGS NWIS — Daily Groundwater Level
# MAGIC
# MAGIC Site **324003114235701** — Fortuna Wash Well, Yuma County, AZ.
# MAGIC Parameter **72019** = depth to water level (feet below land surface).
# MAGIC No API key required.

# COMMAND ----------

gwl_watermark = get_watermark(TBL_BRONZE_GWL, "CAST(date AS DATE)", HISTORY_START)
print(f"USGS fetch window : {gwl_watermark} → {FETCH_END}")

# dataretrieval returns a pandas DataFrame indexed by datetime
raw_gwl = (
    nwis.get_record(
        sites=USGS_SITE_NO,
        service=USGS_SERVICE,
        start=gwl_watermark,
        end=FETCH_END,
    )
    .reset_index()
)

# USGS column name varies: may be "72019_Mean", "72019_00003_Mean", etc.
# Find it dynamically by looking for a column containing the param code.
param_candidates = [c for c in raw_gwl.columns if USGS_PARAM_CODE in c and "Mean" in c]
print(f"USGS columns: {list(raw_gwl.columns)}")
assert param_candidates, f"No column found containing '{USGS_PARAM_CODE}' and 'Mean'. Columns: {list(raw_gwl.columns)}"
param_col = param_candidates[0]
print(f"Using column: {param_col}")
df_gwl_pd = raw_gwl[["datetime", param_col]].copy()
df_gwl_pd.columns = ["date", "gwl_ft"]
df_gwl_pd["date"]        = pd.to_datetime(df_gwl_pd["date"]).dt.date.astype(str)
df_gwl_pd["site_no"]     = USGS_SITE_NO
df_gwl_pd["ingested_at"] = INGESTED_AT
print(f"USGS records fetched: {len(df_gwl_pd):,}")

schema_gwl_raw = StructType([
    StructField("date",       StringType(),    nullable=False),
    StructField("gwl_ft",     DoubleType(),    nullable=True),
    StructField("site_no",    StringType(),    nullable=False),
    StructField("ingested_at",TimestampType(), nullable=False),
])

df_gwl_new = spark.createDataFrame(df_gwl_pd, schema=schema_gwl_raw)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TBL_BRONZE_GWL}
    (date STRING, gwl_ft DOUBLE, site_no STRING, ingested_at TIMESTAMP)
    USING DELTA
""")

df_gwl_new.createOrReplaceTempView("gwl_incoming")

spark.sql(f"""
    MERGE INTO {TBL_BRONZE_GWL} AS target
    USING gwl_incoming AS source
    ON target.date = source.date AND target.site_no = source.site_no
    WHEN MATCHED AND target.gwl_ft != source.gwl_ft THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Bronze GWL current row count: {spark.table(TBL_BRONZE_GWL).count():,}")
display(spark.table(TBL_BRONZE_GWL).orderBy("date", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · USDM Drought Monitor — Weekly County Statistics
# MAGIC
# MAGIC Yuma County FIPS **04027**.
# MAGIC Source: CSV uploaded to Unity Catalog volume (USDM API is blocked on Community Edition).
# MAGIC File: `/Volumes/drought_forecast/bronze/landing/usdm_yuma.csv`
# MAGIC To refresh: download latest CSV from droughtmonitor.unl.edu and re-upload.

# COMMAND ----------

USDM_CSV_PATH = "/Volumes/drought_forecast/bronze/landing/usdm_yuma.csv"

drought_watermark = get_watermark(TBL_BRONZE_DROUGHT, "CAST(map_date AS DATE)", HISTORY_START)
print(f"USDM watermark : {drought_watermark}")

df_drought_new = (
    spark.read.csv(USDM_CSV_PATH, header=True, inferSchema=True)
    .filter(F.col("Week") >= drought_watermark)
    .select(
        F.date_format(F.col("Week"), "yyyy-MM-dd").alias("map_date"),
        F.col("None").alias("none_pct"),
        F.col("D0").alias("d0_pct"),
        F.col("D1").alias("d1_pct"),
        F.col("D2").alias("d2_pct"),
        F.col("D3").alias("d3_pct"),
        F.col("D4").alias("d4_pct"),
        F.lit(None).cast("string").alias("valid_start"),
        F.lit(None).cast("string").alias("valid_end"),
        F.lit(USDM_COUNTY_FIPS).alias("county_fips"),
        F.lit(INGESTED_AT).alias("ingested_at"),
    )
)

print(f"USDM records loaded: {df_drought_new.count():,}")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TBL_BRONZE_DROUGHT}
    (map_date STRING, none_pct DOUBLE, d0_pct DOUBLE, d1_pct DOUBLE,
     d2_pct DOUBLE, d3_pct DOUBLE, d4_pct DOUBLE, valid_start STRING,
     valid_end STRING, county_fips STRING, ingested_at TIMESTAMP)
    USING DELTA
""")

df_drought_new.createOrReplaceTempView("drought_incoming")

spark.sql(f"""
    MERGE INTO {TBL_BRONZE_DROUGHT} AS target
    USING drought_incoming AS source
    ON target.map_date = source.map_date AND target.county_fips = source.county_fips
    WHEN MATCHED AND (
        target.d3_pct != source.d3_pct OR target.d4_pct != source.d4_pct
    ) THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

print(f"Bronze drought current row count: {spark.table(TBL_BRONZE_DROUGHT).count():,}")
display(spark.table(TBL_BRONZE_DROUGHT).orderBy("map_date", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Ingestion Audit
# MAGIC
# MAGIC Validates coverage and freshness across all three bronze tables.

# COMMAND ----------

audit_sql = f"""
SELECT 'NOAA Climate'    AS source,
       COUNT(*)          AS total_records,
       COUNT(DISTINCT CAST(date AS DATE)) AS distinct_dates,
       MIN(CAST(date AS DATE)) AS earliest,
       MAX(CAST(date AS DATE)) AS latest,
       MAX(ingested_at)  AS last_ingested_at
FROM {TBL_BRONZE_CLIMATE}

UNION ALL

SELECT 'USGS GWL',
       COUNT(*),
       COUNT(DISTINCT CAST(date AS DATE)),
       MIN(CAST(date AS DATE)),
       MAX(CAST(date AS DATE)),
       MAX(ingested_at)
FROM {TBL_BRONZE_GWL}

UNION ALL

SELECT 'USDM Drought',
       COUNT(*),
       COUNT(DISTINCT CAST(map_date AS DATE)),
       MIN(CAST(map_date AS DATE)),
       MAX(CAST(map_date AS DATE)),
       MAX(ingested_at)
FROM {TBL_BRONZE_DROUGHT}
"""
display(spark.sql(audit_sql))

# COMMAND ----------

# Confirm NOAA has all three data types present for the latest dates
display(spark.sql(f"""
    SELECT datatype,
           COUNT(*)                    AS records,
           MAX(CAST(date AS DATE))     AS latest_date
    FROM {TBL_BRONZE_CLIMATE}
    GROUP BY datatype
    ORDER BY datatype
"""))
