# Databricks notebook source

# MAGIC %md
# MAGIC # Pipeline Configuration
# MAGIC Central configuration for the Yuma AZ Extreme Drought Prediction pipeline.
# MAGIC Load in every pipeline notebook via: `%run ../configs/00_config`

# COMMAND ----------

from datetime import datetime, date, timedelta

# COMMAND ----------

# ── Run Date Widget ────────────────────────────────────────────────────────────
# Default = today. Override via Databricks Workflows job parameter or manually.
# Format: YYYY-MM-DD
dbutils.widgets.text("run_date", date.today().strftime("%Y-%m-%d"), "Run Date (YYYY-MM-DD)")
dbutils.widgets.text("lookback_days", "14", "Lookback Days (reprocess window)")

RUN_DATE     = dbutils.widgets.get("run_date")
LOOKBACK_DAYS = int(dbutils.widgets.get("lookback_days"))

# Derived dates
RUN_DATE_OBJ  = datetime.strptime(RUN_DATE, "%Y-%m-%d").date()
FETCH_END     = RUN_DATE                                         # fetch up to run date
REPROCESS_FROM = (RUN_DATE_OBJ - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

print(f"Run date      : {RUN_DATE}")
print(f"Lookback      : {LOOKBACK_DAYS} days  (reprocess from {REPROCESS_FROM})")

# COMMAND ----------

# ── Historical Backfill Start ──────────────────────────────────────────────────
# Used only when a table is empty / first run.
HISTORY_START = "2015-01-01"

# COMMAND ----------

# ── Catalog / Schema ───────────────────────────────────────────────────────────
# Unity Catalog: set CATALOG to your catalog name.
# Hive Metastore / Community Edition: set CATALOG = ""
CATALOG = "drought_forecast"
BRONZE  = "bronze"
SILVER  = "silver"
GOLD    = "gold"


def tbl(schema: str, table: str) -> str:
    """Return fully qualified table name."""
    return f"{CATALOG}.{schema}.{table}" if CATALOG else f"{schema}.{table}"


# Bronze  (raw, append-style via MERGE)
TBL_BRONZE_CLIMATE = tbl(BRONZE, "climate_raw")
TBL_BRONZE_GWL     = tbl(BRONZE, "groundwater_raw")
TBL_BRONZE_DROUGHT = tbl(BRONZE, "drought_raw")

# Silver  (cleaned and joined)
TBL_SILVER_CLIMATE = tbl(SILVER, "climate_clean")
TBL_SILVER_GWL     = tbl(SILVER, "groundwater_clean")
TBL_SILVER_DROUGHT = tbl(SILVER, "drought_weekly")
TBL_SILVER_JOINED  = tbl(SILVER, "drought_weather_joined")

# Gold  (ML-ready feature table)
TBL_GOLD_FEATURES  = tbl(GOLD, "drought_features")

# COMMAND ----------

# ── NOAA CDO API ───────────────────────────────────────────────────────────────
# Free token: https://www.ncdc.noaa.gov/cdo-web/token
# Store in Databricks Secrets:
#   databricks secrets create-scope --scope drought-secrets
#   databricks secrets put-secret --scope drought-secrets --key noaa-token
NOAA_TOKEN      = dbutils.secrets.get(scope="drought-secrets", key="noaa-token")
NOAA_BASE_URL   = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
NOAA_STATION_ID = "GHCND:USW00003145"   # Yuma MCAS, AZ
NOAA_DATASET_ID = "GHCND"
NOAA_DATATYPES  = "PRCP,TMAX,TMIN"
NOAA_PAGE_LIMIT = 1000                  # Max records per CDO API page

# ── USGS NWIS ─────────────────────────────────────────────────────────────────
# No API key required
USGS_SITE_NO    = "324003114235701"     # Fortuna Wash Well, Yuma County, AZ
USGS_PARAM_CODE = "72019"              # Depth to water level, feet below land surface
USGS_SERVICE    = "dv"                 # Daily values

# ── USDM Drought Monitor ───────────────────────────────────────────────────────
# No API key required — public data, updates every Tuesday
USDM_BASE_URL    = "https://usdm.unl.edu/services/api/usdm/statistics/byarea"
USDM_COUNTY_FIPS = "04027"             # Yuma County, Arizona

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "/Shared/yuma_drought_prediction"
MLFLOW_MODEL_NAME      = "YumaDroughtClassifier"

# COMMAND ----------

# ── Helper: watermark lookup ───────────────────────────────────────────────────
def get_watermark(table_name: str, date_col: str, default_date: str) -> str:
    """
    Return the day AFTER the latest date already stored in `table_name`.
    Falls back to `default_date` if the table does not exist or is empty.
    Used by bronze ingestion to determine incremental fetch window.
    """
    try:
        row = spark.sql(f"SELECT MAX({date_col}) AS max_date FROM {table_name}").collect()[0]
        if row["max_date"] is None:
            return default_date
        next_day = (row["max_date"] + timedelta(days=1)).strftime("%Y-%m-%d")
        return next_day
    except Exception:
        return default_date

# COMMAND ----------

# ── Create Schemas ─────────────────────────────────────────────────────────────
if CATALOG:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    prefix = f"{CATALOG}."
else:
    prefix = ""

for schema in [BRONZE, SILVER, GOLD]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {prefix}{schema}")

print(f"Catalog  : {CATALOG or 'hive_metastore'}")
print(f"Schemas  : {BRONZE} | {SILVER} | {GOLD}")
