"""Live Supabase data, risk prediction, and SHAP explanation pipeline.

This module contains the non-UI work used by the Streamlit live dashboard.
Feature engineering and ensemble model loading are provided by
``prediction_features``; this module orchestrates them with Supabase report data.
"""

import os
import re
import hashlib

import numpy as np
import pandas as pd
import shap
import streamlit as st
from supabase import Client, create_client

from prediction_features import (
    PREDICTION_FEATURE_COLUMNS,
    create_prediction_features,
    load_cholera_ensemble_model,
    predict_next_month_risk,
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wvykyedcdloopzyfhlbe.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_AcbXoyx9KzVP4TOd06oMeA_oq5YYp-s")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
ENSEMBLE_RISK_LABELS = {0: "Low", 1: "High"}


def load_live_risk_model():
    """Return the joblib-loaded ensemble used by the live prediction pipeline."""
    return load_cholera_ensemble_model()


def fetch_supabase_data() -> pd.DataFrame:
    """Fetch the current ``reports`` rows without using Streamlit's data cache."""
    columns = [
        "id", "created_at",
        "date", "region", "district", "confirmed", "suspected", "deaths",
        "cfr", "rainfall", "temperature", "humidity",
    ]
    rows = []
    page_size = 1000
    start = 0
    while True:
        response = (
            supabase.table("reports").select(",".join(columns)).order("id")
            .range(start, start + page_size - 1).execute()
        )
        page = response.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    data = pd.DataFrame(rows, columns=columns)
    if data.empty:
        data["year"] = pd.Series(dtype="Int64")
        return data

    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_localize(None)
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    data["region"] = data["region"].fillna("Unknown").astype(str)
    data["district"] = data["district"].fillna("Unknown").astype(str)
    for column in ["confirmed", "suspected", "deaths", "cfr", "rainfall", "temperature", "humidity"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["year"] = data["date"].dt.year.astype("Int64")
    return data


@st.cache_data(ttl=3600)
def load_supabase_data() -> pd.DataFrame:
    """Return cached live reports for up to one hour."""
    return fetch_supabase_data()


def refresh_supabase_data() -> pd.DataFrame:
    """Fetch fresh reports; per-region prediction caches are not cleared."""
    return fetch_supabase_data()


def region_report_fingerprint(region_rows: pd.DataFrame) -> str:
    """Stable cache key that changes when a region receives or changes a report."""
    columns = [
        column for column in [
            "id", "created_at", "date", "region", "district", "confirmed", "suspected",
            "deaths", "cfr", "rainfall", "temperature", "humidity",
        ] if column in region_rows.columns
    ]
    rows = region_rows[columns].copy().sort_values(columns, kind="stable")
    row_hashes = pd.util.hash_pandas_object(rows, index=False).values.tobytes()
    return hashlib.sha256(row_hashes).hexdigest()


def explain_risk_with_shap(report_row, model) -> str:
    """Return a concise explanation of the prediction for one feature row."""
    if isinstance(report_row, pd.DataFrame):
        feature_row = report_row.iloc[[0]].copy()
    elif isinstance(report_row, pd.Series):
        feature_row = report_row.to_frame().T
    else:
        feature_row = pd.DataFrame([report_row])

    model_features = list(getattr(model, "feature_names_in_", feature_row.columns))
    feature_row = feature_row.reindex(columns=model_features, fill_value=0)
    feature_row = feature_row.apply(pd.to_numeric, errors="coerce").fillna(0)
    prediction = model.predict(feature_row)[0]
    classes = list(getattr(model, "classes_", []))
    risk_label = (
        ENSEMBLE_RISK_LABELS.get(int(prediction), str(prediction))
        if set(classes) == {0, 1}
        else RISK_LABELS.get(int(prediction), str(prediction))
    )

    try:
        explainer = shap.Explainer(model, feature_row)
    except TypeError:
        if not hasattr(model, "predict_proba"):
            raise
        explainer = shap.Explainer(model.predict_proba, feature_row)
    values = np.asarray(explainer(feature_row, max_evals=(2 * len(model_features)) + 1).values)
    contributions = (
        values[0, :, classes.index(prediction) if prediction in classes else 0]
        if values.ndim == 3 else values[0]
    )
    drivers = pd.Series(contributions, index=model_features)
    drivers = drivers.reindex(drivers.abs().sort_values(ascending=False).index)
    friendly_names = {
        "rainfall_avg": "rainfall", "rainfall": "rainfall",
        "humidity_avg": "humidity", "humidity": "humidity",
        "temperature_avg": "temperature", "temperature": "temperature",
        "month": "seasonal timing", "month_sin": "seasonal timing", "month_cos": "seasonal timing",
        "cases": "case counts", "confirmed": "confirmed cases", "suspected": "suspected cases",
        "deaths": "deaths", "cfr": "CFR",
    }

    def friendly_feature_name(feature: str) -> str:
        if feature in friendly_names:
            return friendly_names[feature]
        lag_match = re.fullmatch(r"(cases|rainfall|humidity|temperature)_lag(\d+)", str(feature))
        if lag_match:
            metric, months = lag_match.groups()
            metric_name = {"cases": "case counts", "rainfall": "rainfall", "humidity": "humidity", "temperature": "temperature"}[metric]
            unit = "month" if months == "1" else "months"
            return f"{metric_name} {months} {unit} ago"
        roll_match = re.fullmatch(r"(cases|rainfall|humidity)_roll(\d+)", str(feature))
        if roll_match:
            metric, months = roll_match.groups()
            metric_name = {"cases": "case counts", "rainfall": "rainfall", "humidity": "humidity"}[metric]
            return f"average {metric_name} over the previous {months} months"
        std_match = re.fullmatch(r"(cases|rainfall|humidity)_std(\d+)", str(feature))
        if std_match:
            metric, months = std_match.groups()
            return f"variation in {metric} over the previous {months} months"
        interaction_names = {
            "rainfall_x_humidity": "rainfall and humidity together",
            "cases_x_deaths": "case counts and deaths together",
            "rainfall_x_temp": "rainfall and temperature together",
            "humidity_x_temp": "humidity and temperature together",
            "rainfall_cases": "rainfall and case counts together",
            "cases_growth": "recent case-count growth",
            "rainfall_change": "recent rainfall change",
        }
        return interaction_names.get(str(feature), str(feature).replace("_", " "))

    ranked_drivers = []
    used_names = set()
    for feature, contribution in drivers.items():
        name = friendly_feature_name(feature)
        if name in used_names:
            continue
        used_names.add(name)
        direction = "supports" if contribution >= 0 else "opposes"
        ranked_drivers.append(f"{len(ranked_drivers) + 1}. {name} ({direction} {risk_label} risk)")
        if len(ranked_drivers) == 4:
            break

    if not ranked_drivers:
        return f"{risk_label} risk; SHAP could not identify influential features."
    return f"{risk_label} risk. Top SHAP influences: {'; '.join(ranked_drivers)}."


@st.cache_data(show_spinner=False)
def get_cached_region_prediction(region: str, report_fingerprint: str) -> dict:
    """Cache one region's feature build, prediction, and SHAP explanation until its reports change."""
    del report_fingerprint  # The fingerprint is intentionally part of Streamlit's cache key.
    try:
        result = predict_next_month_risk(region)
        prediction = result["prediction"]
        model = result.get("model")
        if model is None:
            model = load_live_risk_model()
        try:
            explanation = explain_risk_with_shap(result["features"], model)
        except Exception as exc:
            explanation = f"SHAP explanation unavailable: {exc}"
        return {
            "OutbreakRisk_NextMonth": ENSEMBLE_RISK_LABELS.get(prediction, str(prediction)),
            "OutbreakRisk_Class": prediction,
            "Prediction_Confidence": (
                float(result["probabilities"].get(prediction)) * 100
                if result.get("probabilities") and prediction in result["probabilities"]
                else None
            ),
            "Prediction_Error": "",
            "risk_explanation": explanation,
        }
    except Exception as exc:
        return {
            "OutbreakRisk_NextMonth": "Unavailable",
            "OutbreakRisk_Class": None,
            "Prediction_Confidence": None,
            "Prediction_Error": str(exc),
            "risk_explanation": "Explanation unavailable.",
        }


def build_live_regional_monthly_environment_table(
    live_region_df: pd.DataFrame,
    live_history_df: pd.DataFrame | None = None,
    latest_per_region: bool = False,
) -> pd.DataFrame:
    """Aggregate reports by region/month, build features, predict, and explain."""
    del live_history_df  # History is loaded by create_prediction_features per region.
    source = live_region_df.dropna(subset=["date"]).copy()
    if source.empty:
        return pd.DataFrame()
    source["report_month"] = pd.to_datetime(source["date"], errors="coerce").dt.to_period("M").astype(str)
    source = source[source["report_month"] != "NaT"]
    monthly = (
        source.groupby(["region", "report_month"], as_index=False)
        .agg(confirmed=("confirmed", "sum"), suspected=("suspected", "sum"), deaths=("deaths", "sum"),
             rainfall=("rainfall", "mean"), temperature=("temperature", "mean"), humidity=("humidity", "mean"))
        .sort_values(["report_month", "region"])
    )
    if latest_per_region:
        monthly = monthly.sort_values(["region", "report_month"]).groupby("region", as_index=False).tail(1).sort_values("region")

    fingerprints = {
        region: region_report_fingerprint(region_rows)
        for region, region_rows in source.groupby("region")
    }

    def predict_row(row):
        return pd.Series(get_cached_region_prediction(row["region"], fingerprints[row["region"]]))

    monthly[["OutbreakRisk_NextMonth", "OutbreakRisk_Class", "Prediction_Confidence", "Prediction_Error", "risk_explanation"]] = monthly.apply(predict_row, axis=1)
    return monthly
