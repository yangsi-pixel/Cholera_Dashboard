import os
from functools import lru_cache

import numpy as np
import pandas as pd
from supabase import Client, create_client

try:
    import joblib
except ImportError:
    joblib = None

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wvykyedcdloopzyfhlbe.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_AcbXoyx9KzVP4TOd06oMeA_oq5YYp-s")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ENSEMBLE_MODEL_PATH = "cholera_ensemble_model.pkl"
ENSEMBLE_FEATURES_PATH = "cholera_ensemble_features.pkl"
SAMPLE_DATA_PATH = "sample data.csv"

PREDICTION_FEATURE_COLUMNS = [
    "month",
    "month_sin",
    "month_cos",
    "cases",
    "cases_lag1",
    "cases_lag2",
    "cases_lag3",
    "cases_lag6",
    "cases_lag12",
    "cases_roll3",
    "cases_std3",
    "cases_growth",
    "temperature_lag1",
    "temperature_lag2",
    "temperature_lag3",
    "humidity_avg",
    "humidity_lag1",
    "humidity_lag2",
    "humidity_lag6",
    "humidity_roll3",
    "humidity_std3",
    "rainfall_avg",
    "rainfall_lag1",
    "rainfall_lag2",
    "rainfall_lag3",
    "rainfall_lag6",
    "rainfall_lag12",
    "rainfall_roll3",
    "rainfall_std3",
    "rainfall_change",
    "deaths",
    "rainfall_x_humidity",
    "cases_x_deaths",
    "rainfall_x_temp",
    "humidity_x_temp",
    "rainfall_cases",
]


@lru_cache(maxsize=1)
def load_cholera_ensemble_model():
    if joblib is None:
        raise RuntimeError("joblib is not installed. Run: pip install joblib")
    return joblib.load(ENSEMBLE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_cholera_ensemble_features():
    if joblib is None:
        raise RuntimeError("joblib is not installed. Run: pip install joblib")
    if os.path.exists(ENSEMBLE_FEATURES_PATH):
        return list(joblib.load(ENSEMBLE_FEATURES_PATH))

    model_features = getattr(load_cholera_ensemble_model(), "feature_names_in_", None)
    return list(model_features) if model_features is not None else PREDICTION_FEATURE_COLUMNS


def normalize_region_name(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return normalized.replace(
        {
            "adamawa": "adamaoua",
            "northwestregion": "northwest",
            "northwestprovince": "northwest",
            "southwestregion": "southwest",
            "southwestprovince": "southwest",
        }
    )


def normalize_single_region(region) -> str:
    return normalize_region_name(pd.Series([region])).iloc[0]


def pick_first_existing(df_input: pd.DataFrame, candidates):
    for candidate in candidates:
        if candidate in df_input.columns:
            return candidate
    return None


def normalize_column_token(column_name) -> str:
    return "".join(ch for ch in str(column_name).lower() if ch.isalnum())


def pick_first_flexible(df_input: pd.DataFrame, candidates):
    exact_match = pick_first_existing(df_input, candidates)
    if exact_match is not None:
        return exact_match

    normalized_columns = {normalize_column_token(column): column for column in df_input.columns}
    for candidate in candidates:
        normalized_candidate = normalize_column_token(candidate)
        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]
    return None


def standardize_prediction_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "region_en",
                "district",
                "suspected",
                "confirmed",
                "deaths",
                "rainfall_avg",
                "temperature_avg",
                "humidity_avg",
                "region_norm",
            ]
        )

    working_df = raw_df.copy()
    region_col = pick_first_flexible(working_df, ["region", "region_en", "Region", "regionName"])
    district_col = pick_first_flexible(working_df, ["district", "district_name", "District"])
    date_col = pick_first_flexible(working_df, ["date", "reporting_date", "report_date", "createdAt", "created_at", "timestamp", "TL"])
    suspected_col = pick_first_flexible(working_df, ["suspected", "suspected cases", "suspectedcases", "sCh", "suspectedCases"])
    confirmed_col = pick_first_flexible(working_df, ["confirmed", "confirmed cases", "confirmedcases", "cCh", "confirmedCases"])
    deaths_col = pick_first_flexible(working_df, ["deaths", "Deaths"])
    rainfall_col = pick_first_flexible(working_df, ["rainfall_avg", "rainfall", "rain_avg", "precipitation"])
    temperature_col = pick_first_flexible(working_df, ["temperature_avg", "temp_avg", "temperature", "temp"])
    humidity_col = pick_first_flexible(working_df, ["humidity_avg", "humid_avg", "humidity", "rh_avg"])

    working_df["region_en"] = "Unknown" if region_col is None else working_df[region_col].astype(str).replace("nan", "Unknown")
    working_df["district"] = "Unknown" if district_col is None else working_df[district_col].astype(str).replace("nan", "Unknown")
    working_df["date"] = pd.NaT if date_col is None else pd.to_datetime(working_df[date_col], errors="coerce")
    working_df["suspected"] = pd.to_numeric(working_df[suspected_col], errors="coerce").fillna(0) if suspected_col else 0
    working_df["confirmed"] = pd.to_numeric(working_df[confirmed_col], errors="coerce").fillna(0) if confirmed_col else 0
    working_df["deaths"] = pd.to_numeric(working_df[deaths_col], errors="coerce").fillna(0) if deaths_col else 0
    working_df["rainfall_avg"] = pd.to_numeric(working_df[rainfall_col], errors="coerce") if rainfall_col else np.nan
    working_df["temperature_avg"] = pd.to_numeric(working_df[temperature_col], errors="coerce") if temperature_col else np.nan
    working_df["humidity_avg"] = pd.to_numeric(working_df[humidity_col], errors="coerce") if humidity_col else np.nan
    working_df["region_norm"] = normalize_region_name(working_df["region_en"])

    return working_df[
        [
            "date",
            "region_en",
            "district",
            "suspected",
            "confirmed",
            "deaths",
            "rainfall_avg",
            "temperature_avg",
            "humidity_avg",
            "region_norm",
        ]
    ].dropna(subset=["date"])


def load_prediction_records_from_supabase(region: str) -> pd.DataFrame:
    """Load a region's report history from Supabase for prediction features."""
    columns = "date,region,district,confirmed,suspected,deaths,rainfall,temperature,humidity"
    rows = []
    page_size = 1000
    start = 0
    while True:
        response = (
            supabase.table("reports").select(columns).order("id")
            .range(start, start + page_size - 1).execute()
        )
        page = response.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    prediction_df = standardize_prediction_records(pd.DataFrame(rows))
    target_region_norm = normalize_single_region(region)
    return prediction_df[prediction_df["region_norm"] == target_region_norm].sort_values("date")


def load_sample_records(region: str) -> pd.DataFrame:
    if not os.path.exists(SAMPLE_DATA_PATH):
        return pd.DataFrame()

    sample_df = standardize_prediction_records(pd.read_csv(SAMPLE_DATA_PATH))
    region_norm = normalize_single_region(region)
    return sample_df[sample_df["region_norm"] == region_norm].sort_values("date")


def has_usable_environment_data(records_df: pd.DataFrame) -> bool:
    env_columns = ["rainfall_avg", "temperature_avg", "humidity_avg"]
    if records_df.empty or any(column not in records_df.columns for column in env_columns):
        return False
    return records_df[env_columns].notna().any().any()


def diagnose_region_history(region: str) -> dict:
    records_df = load_prediction_records_from_supabase(region)
    if records_df.empty:
        return {
            "region": region,
            "record_count": 0,
            "message": "No Supabase records matched this region after standardization.",
        }

    records_df = records_df.copy()
    records_df["month_date"] = pd.to_datetime(records_df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    valid_records = records_df.dropna(subset=["month_date"])

    if valid_records.empty:
        return {
            "region": region,
            "record_count": len(records_df),
            "valid_dated_record_count": 0,
            "message": "Supabase records matched the region, but none had a parseable date.",
        }

    month_counts = valid_records.groupby("month_date").size().sort_index()
    latest_month = month_counts.index.max()
    expected_months = pd.date_range(month_counts.index.min(), latest_month, freq="MS")
    missing_months = sorted(set(expected_months) - set(month_counts.index))

    return {
        "region": region,
        "record_count": len(records_df),
        "valid_dated_record_count": len(valid_records),
        "first_month": month_counts.index.min().strftime("%Y-%m"),
        "latest_month": latest_month.strftime("%Y-%m"),
        "monthly_record_counts": {month.strftime("%Y-%m"): int(count) for month, count in month_counts.items()},
        "missing_months": [month.strftime("%Y-%m") for month in missing_months],
        "has_12_month_lag": (latest_month - pd.DateOffset(months=12)) in set(month_counts.index),
        "environment_missing_counts": records_df[["rainfall_avg", "temperature_avg", "humidity_avg"]].isna().sum().to_dict(),
        "has_usable_environment_data": has_usable_environment_data(records_df),
    }


def fill_monthly_environment_gaps(monthly_df: pd.DataFrame) -> pd.DataFrame:
    # Environmental values are stored in Supabase. Fill gaps only from nearby
    # report periods; no external environmental API is used.
    monthly_df = monthly_df.copy()
    monthly_df[["rainfall_avg", "temperature_avg", "humidity_avg"]] = (
        monthly_df[["rainfall_avg", "temperature_avg", "humidity_avg"]].ffill().bfill().fillna(0)
    )
    return monthly_df


def resolve_feature_anchor_month(records_df: pd.DataFrame, region: str, target_month=None) -> pd.Timestamp:
    records_df = records_df.copy()
    records_df["month_date"] = pd.to_datetime(records_df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    records_df = records_df.dropna(subset=["month_date"])

    if records_df.empty:
        raise ValueError(f"Prediction cannot be generated: no dated Supabase records were found for region '{region}'.")

    if target_month is None:
        return records_df["month_date"].max()

    target_month_date = pd.to_datetime(target_month, errors="coerce")
    if pd.isna(target_month_date):
        raise ValueError("Prediction cannot be generated: target_month must be a valid date or YYYY-MM value.")

    target_month_date = target_month_date.to_period("M").to_timestamp()
    if target_month_date not in set(records_df["month_date"]):
        raise ValueError(f"Prediction cannot be generated: no report exists for {region} in target month {target_month_date:%Y-%m}.")

    return target_month_date


def aggregate_prediction_records_monthly(records_df: pd.DataFrame, region: str, anchor_month) -> pd.DataFrame:
    anchor_month_date = pd.to_datetime(anchor_month, errors="coerce")
    if pd.isna(anchor_month_date):
        raise ValueError("Prediction cannot be generated: anchor month must be a valid date or YYYY-MM value.")

    anchor_month_date = anchor_month_date.to_period("M").to_timestamp()
    records_df = records_df.copy()
    records_df["month_date"] = pd.to_datetime(records_df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    records_df = records_df[records_df["month_date"] <= anchor_month_date].sort_values("date")

    if records_df.empty:
        raise ValueError(f"Prediction cannot be generated: no historical records for {region} up to {anchor_month_date:%Y-%m}.")

    records_df["cases"] = records_df["suspected"] + records_df["confirmed"]
    monthly_df = (
        records_df.groupby("month_date", as_index=False)
        .agg(
            {
                "cases": "sum",
                "deaths": "sum",
                "rainfall_avg": "mean",
                "temperature_avg": "mean",
                "humidity_avg": "mean",
            }
        )
        .sort_values("month_date")
    )

    calendar_months = pd.date_range(monthly_df["month_date"].min(), anchor_month_date, freq="MS")
    monthly_df = (
        monthly_df.set_index("month_date")
        .reindex(calendar_months)
        .rename_axis("month_date")
        .reset_index()
    )
    monthly_df[["cases", "deaths"]] = monthly_df[["cases", "deaths"]].fillna(0)
    return fill_monthly_environment_gaps(monthly_df)


def add_prediction_feature_columns(monthly_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = monthly_df.copy()
    feature_df["month"] = feature_df["month_date"].dt.month
    feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
    feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)

    for lag in [1, 2, 3, 6, 12]:
        feature_df[f"cases_lag{lag}"] = feature_df["cases"].shift(lag)
    for lag in [1, 2, 3]:
        feature_df[f"temperature_lag{lag}"] = feature_df["temperature_avg"].shift(lag)
    for lag in [1, 2, 6]:
        feature_df[f"humidity_lag{lag}"] = feature_df["humidity_avg"].shift(lag)
    for lag in [1, 2, 3, 6, 12]:
        feature_df[f"rainfall_lag{lag}"] = feature_df["rainfall_avg"].shift(lag)

    previous_cases = feature_df["cases"].shift(1)
    feature_df["cases_roll3"] = previous_cases.rolling(window=3, min_periods=3).mean()
    feature_df["cases_std3"] = previous_cases.rolling(window=3, min_periods=3).std()
    feature_df["cases_growth"] = np.where(
        previous_cases.isna(),
        np.nan,
        np.where(previous_cases == 0, 0, (feature_df["cases"] - previous_cases) / previous_cases),
    )

    previous_humidity = feature_df["humidity_avg"].shift(1)
    feature_df["humidity_roll3"] = previous_humidity.rolling(window=3, min_periods=3).mean()
    feature_df["humidity_std3"] = previous_humidity.rolling(window=3, min_periods=3).std()

    previous_rainfall = feature_df["rainfall_avg"].shift(1)
    feature_df["rainfall_roll3"] = previous_rainfall.rolling(window=3, min_periods=3).mean()
    feature_df["rainfall_std3"] = previous_rainfall.rolling(window=3, min_periods=3).std()
    feature_df["rainfall_change"] = feature_df["rainfall_avg"] - previous_rainfall

    feature_df["rainfall_x_humidity"] = feature_df["rainfall_avg"] * feature_df["humidity_avg"]
    feature_df["cases_x_deaths"] = feature_df["cases"] * feature_df["deaths"]
    feature_df["rainfall_x_temp"] = feature_df["rainfall_avg"] * feature_df["temperature_avg"]
    feature_df["humidity_x_temp"] = feature_df["humidity_avg"] * feature_df["temperature_avg"]
    feature_df["rainfall_cases"] = feature_df["rainfall_avg"] * feature_df["cases"]
    return feature_df


def create_prediction_features_from_records(records_df: pd.DataFrame, region, target_month=None) -> pd.DataFrame:
    if records_df.empty:
        raise ValueError(f"Prediction cannot be generated: no records were found for region '{region}'.")

    anchor_month_date = resolve_feature_anchor_month(records_df, region, target_month)
    monthly_df = aggregate_prediction_records_monthly(records_df, region, anchor_month_date)
    feature_df = add_prediction_feature_columns(monthly_df)

    anchor_rows = feature_df[feature_df["month_date"] == anchor_month_date]
    if anchor_rows.empty:
        raise ValueError(f"Prediction cannot be generated: no feature row was created for {region} in {anchor_month_date:%Y-%m}.")

    feature_row = anchor_rows[PREDICTION_FEATURE_COLUMNS].copy()
    feature_row = feature_row.ffill()
    missing_features = feature_row.columns[feature_row.iloc[0].isna()].tolist()
    if missing_features:
        raise ValueError(
            "Prediction cannot be generated: insufficient historical monthly records "
            f"for {region} up to latest report month {anchor_month_date:%Y-%m}. "
            f"Missing features: {', '.join(missing_features)}"
        )

    feature_row = feature_row.replace([np.inf, -np.inf], 0).fillna(0)
    return feature_row.reset_index(drop=True)


def create_prediction_features(region, target_month=None) -> pd.DataFrame:
    """
    Build the model input row for one region from Supabase history.

    By default the feature row is anchored to the latest report month found for the
    region. That row is intended to be fed into the model for next-month prediction.
    If Supabase has no usable records for the requested region, local sample data
    is used as the fallback.

    The returned DataFrame contains only the columns in PREDICTION_FEATURE_COLUMNS.
    A ValueError is raised with a user-facing message when available history is insufficient.
    """
    try:
        records_df = load_prediction_records_from_supabase(region)
        if records_df.empty:
            raise ValueError(f"Prediction cannot be generated: no Supabase records were found for region '{region}'.")
        if not has_usable_environment_data(records_df) and not load_sample_records(region).empty:
            raise ValueError(
                f"Supabase {region} records do not include usable rainfall, temperature, or humidity values."
            )
        return create_prediction_features_from_records(records_df, region, target_month)
    except Exception as original_exc:
        sample_df = load_sample_records(region)
        if sample_df.empty:
            raise original_exc

        try:
            return create_prediction_features_from_records(sample_df, region, target_month)
        except Exception as sample_exc:
            raise ValueError(
                "Prediction cannot be generated from Supabase or sample data. "
                f"Supabase error: {original_exc}. Sample data error: {sample_exc}"
            ) from sample_exc


def prepare_ensemble_input(feature_row: pd.DataFrame) -> pd.DataFrame:
    model_features = load_cholera_ensemble_features()
    missing_features = [feature for feature in model_features if feature not in feature_row.columns]
    if missing_features:
        raise ValueError(f"Prediction cannot be generated: feature row is missing {', '.join(missing_features)}")

    return feature_row.reindex(columns=model_features)


def predict_next_month_risk(region) -> dict:
    """
    Generate latest-month features for a region and predict next-month risk.

    Returns a dictionary containing the raw model class, class probabilities when
    available, and the exact model input DataFrame used for prediction.
    """
    feature_row = create_prediction_features(region)
    model_input = prepare_ensemble_input(feature_row)
    model = load_cholera_ensemble_model()

    prediction = model.predict(model_input)[0]
    result = {
        "region": region,
        "prediction": int(prediction) if isinstance(prediction, (np.integer, int)) else prediction,
        "features": model_input,
        "model": model,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        classes = getattr(model, "classes_", range(len(probabilities)))
        result["probabilities"] = {
            int(class_label) if isinstance(class_label, (np.integer, int)) else class_label: float(probability)
            for class_label, probability in zip(classes, probabilities)
        }

    return result
