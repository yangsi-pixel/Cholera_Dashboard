import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from prediction_features import PREDICTION_FEATURE_COLUMNS, create_prediction_features, predict_next_month_risk
from region_coords import DEFAULT_COORDS, REGION_COORDS

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
except ImportError:
    MongoClient = None
    ServerApi = None

try:
    import joblib
except ImportError:
    joblib = None


MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://yangsisonia9_db_user:64687@cluster0.295yjzd.mongodb.net/",
)

CMAP = {"Confirmed": "#1e5fd8", "Suspected": "#f9a825", "Deaths": "#d32f2f"}
RISK_COLOR_MAP = {"Low": "#2e7d32", "Medium": "#ef6c00", "High": "#c62828"}
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
ENSEMBLE_RISK_LABELS = {0: "Low", 1: "High"}


def normalize_region_name(series: pd.Series) -> pd.Series:
    # Normalize region labels and collapse known aliases to stable keys.
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


@st.cache_data(ttl=3600)
def load_historical_data():
    # Load and standardize the two historical datasets used by the dashboard.
    district_df = pd.read_csv("FINAL district_dataset_clean.csv")
    regional_env_df = pd.read_csv("FINAL regional_cholera_environment_dataset.csv")

    if "Region" in district_df.columns:
        district_df["region_en"] = district_df["Region"].astype(str)
    elif "region_en" not in district_df.columns:
        district_df["region_en"] = "Unknown"

    if "district" not in district_df.columns:
        if "Location" in district_df.columns:
            district_df["district"] = (
                district_df["Location"]
                .astype(str)
                .str.split("::")
                .str[-1]
                .str.replace(" Health District", "", regex=False)
                .str.strip()
            )
        else:
            district_df["district"] = "Unknown"

    if "Region" in regional_env_df.columns:
        regional_env_df["region_en"] = regional_env_df["Region"].astype(str)
    elif "region_en" not in regional_env_df.columns:
        regional_env_df["region_en"] = "Unknown"

    district_df["TL"] = pd.to_datetime(district_df["TL"], errors="coerce")
    district_df["year"] = district_df["TL"].dt.year
    regional_env_df["TL"] = pd.to_datetime(regional_env_df["TL"], errors="coerce")
    regional_env_df["year"] = regional_env_df["TL"].dt.year

    for col in ["sCh", "cCh", "deaths"]:
        district_df[col] = pd.to_numeric(district_df[col], errors="coerce").fillna(0)
        regional_env_df[col] = pd.to_numeric(regional_env_df[col], errors="coerce").fillna(0)

    # Keep only historical fields required for CFR-based analytics.
    regional_env_df = regional_env_df[["region_en", "TL", "year", "sCh", "cCh", "deaths"]].copy()

    district_df["region_norm"] = normalize_region_name(district_df["region_en"])
    regional_env_df["region_norm"] = normalize_region_name(regional_env_df["region_en"])

    years = sorted(
        pd.Index(district_df["year"].dropna().astype(int).unique())
        .union(pd.Index(regional_env_df["year"].dropna().astype(int).unique()))
        .tolist()
    )

    regions = ["All Regions"] + sorted(
        pd.Index(district_df["region_en"].dropna().astype(str).unique())
        .union(pd.Index(regional_env_df["region_en"].dropna().astype(str).unique()))
        .tolist()
    )

    return district_df, regional_env_df, years, regions


def aggregate_regional_all_regions_by_tl(df_input: pd.DataFrame) -> pd.DataFrame:
    # Aggregate regional records to one row per TL period for All Regions views.
    aggregated = (
        df_input.groupby("TL", as_index=False)
        .agg(
            {
                "sCh": "sum",
                "cCh": "sum",
                "deaths": "sum",
            }
        )
        .sort_values("TL")
    )
    aggregated["year"] = aggregated["TL"].dt.year
    return aggregated


@st.cache_data(ttl=3600)
def load_regional_hotspot_source():
    # Load the regional cholera source file used only for the All Regions hotspot map.
    candidate_files = [
        "FINAL regional_cholera_environment_data.csv",
        "FINAL regional_cholera_environment_dataset.csv",
    ]
    source_path = next((path for path in candidate_files if os.path.exists(path)), None)
    if source_path is None:
        raise FileNotFoundError("Regional cholera hotspot source file was not found.")

    hotspot_df = pd.read_csv(source_path)
    hotspot_df = hotspot_df[["Region", "reporting_date", "sCh", "cCh", "deaths"]].copy()

    # Standardize region labels to match the boundary source before merging.
    hotspot_df["Region"] = hotspot_df["Region"].replace(
        {
            "Adamaoua": "Adamawa",
            "Far North": "Far-North",
        }
    )

    # Build the map's year filter from reporting_date as requested.
    hotspot_df["reporting_date"] = pd.to_datetime(hotspot_df["reporting_date"], errors="coerce")
    hotspot_df["Year"] = hotspot_df["reporting_date"].dt.year

    for column in ["sCh", "cCh", "deaths"]:
        hotspot_df[column] = pd.to_numeric(hotspot_df[column], errors="coerce").fillna(0)

    return hotspot_df


@st.cache_data(ttl=3600)
def load_regional_boundaries():
    # Load ADM2 boundaries and dissolve them to ADM1 regional geometries.
    if gpd is None:
        raise RuntimeError("geopandas is not installed. Run: pip install geopandas")

    candidate_files = [
        "cmr_admin_boundaries.geojson",
        os.path.join("cmr_admin_boundaries", "cmr_admin2.geojson"),
    ]
    boundary_path = next((path for path in candidate_files if os.path.exists(path)), None)
    if boundary_path is None:
        raise FileNotFoundError("Cameroon administrative boundary GeoJSON was not found.")

    boundaries = gpd.read_file(boundary_path)
    regional_boundaries = boundaries[["adm1_name", "geometry"]].dissolve(by="adm1_name", as_index=False)
    return regional_boundaries


def render_regional_hotspot_map(selected_year, selected_metric):
    # Render the All Regions hotspot map from regional cholera totals and dissolved ADM1 boundaries.
    hotspot_source = load_regional_hotspot_source()
    regional_boundaries = load_regional_boundaries()

    # Keep only the selected year when provided, otherwise aggregate across all years.
    filtered_source = (
        hotspot_source
        if selected_year is None
        else hotspot_source[hotspot_source["Year"] == int(selected_year)]
    )
    yearly_totals = filtered_source.groupby("Region", as_index=False)[["sCh", "cCh", "deaths"]].sum()

    merged = regional_boundaries.merge(yearly_totals, how="left", left_on="adm1_name", right_on="Region")
    merged[["sCh", "cCh", "deaths"]] = merged[["sCh", "cCh", "deaths"]].fillna(0)
    merged["cfr"] = (merged["deaths"] / merged["cCh"].where(merged["cCh"] > 0, 1)) * 100
    merged.loc[merged["cCh"] == 0, "cfr"] = 0.0

    # Center the map from the dissolved regional bounds so northern regions are not clipped.
    min_lon, min_lat, max_lon, max_lat = merged.total_bounds
    map_center = {
        "lat": float((min_lat + max_lat) / 2),
        "lon": float((min_lon + max_lon) / 2),
    }

    geojson_data = json.loads(merged.to_json())
    year_label = "All Years" if selected_year is None else str(int(selected_year))
    map_title = f"Cameroon Regional Cholera Hotspot Map: {selected_metric} ({year_label})"

    fig = px.choropleth_mapbox(
        merged,
        geojson=geojson_data,
        locations="adm1_name",
        featureidkey="properties.adm1_name",
        color=selected_metric,
        hover_name="adm1_name",
        hover_data={"sCh": True, "cCh": True, "deaths": True, "cfr": ":.2f", "Region": False},
        mapbox_style="white-bg",
        center=map_center,
        zoom=4.35,
        opacity=0.7,
        title=map_title,
    )
    fig.update_traces(marker_line_width=1.0, marker_line_color="black")
    fig.update_layout(margin={"r": 0, "t": 60, "l": 0, "b": 0})
    apply_chart_layout(fig, height=620)
    st.plotly_chart(fig, use_container_width=True)


def build_cfr_hotspot_data(df_input: pd.DataFrame, selected_year=None) -> pd.DataFrame:
    # Build CFR hotspot summary per region from historical case/death data only.
    working_df = df_input[["region_en", "TL", "cCh", "deaths"]].copy()
    if selected_year is not None:
        working_df = working_df[working_df["TL"].dt.year == int(selected_year)]

    hotspot_df = (
        working_df.groupby("region_en", as_index=False)[["cCh", "deaths"]]
        .sum()
        .rename(columns={"region_en": "region"})
    )
    hotspot_df["cfr"] = (hotspot_df["deaths"] / hotspot_df["cCh"].where(hotspot_df["cCh"] > 0, 1)) * 100
    hotspot_df.loc[hotspot_df["cCh"] == 0, "cfr"] = 0.0
    hotspot_df["cfr_risk"] = hotspot_df["cfr"].apply(classify_cfr_risk)
    return hotspot_df.sort_values("region")


def render_historical_cfr_hotspot_summary(df_input: pd.DataFrame, selected_year=None):
    # Render regional CFR hotspot summary without any shapefile dependency.
    hotspot_df = build_cfr_hotspot_data(df_input, selected_year)
    if hotspot_df.empty:
        st.warning("No regional records available for CFR hotspot summary.")
        return

    summary_title = "Regional CFR Hotspots"
    if selected_year is not None:
        summary_title = f"Regional CFR Hotspots - {int(selected_year)}"

    ordered = hotspot_df.sort_values("cfr", ascending=False)
    fig = px.bar(
        ordered,
        x="region",
        y="cfr",
        color="cfr_risk",
        category_orders={"cfr_risk": ["High", "Medium", "Low"]},
        color_discrete_map=RISK_COLOR_MAP,
        labels={"region": "Region", "cfr": "CFR (%)", "cfr_risk": "Risk"},
        title=summary_title,
    )
    apply_chart_layout(fig, height=520)
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        ordered[["region", "cCh", "deaths", "cfr", "cfr_risk"]]
        .rename(columns={"region": "Region", "cCh": "Confirmed", "deaths": "Deaths", "cfr": "CFR (%)", "cfr_risk": "Risk"}),
        use_container_width=True,
    )


def pick_first_existing(df_input: pd.DataFrame, candidates):
    # Return the first matching column name from a candidate list.
    for candidate in candidates:
        if candidate in df_input.columns:
            return candidate
    return None


def normalize_region_for_feature(region_name) -> str:
    # Match the one-hot naming style used by the training dataset.
    return str(region_name or "Unknown").replace(" ", "").replace("-", "")


def prepare_history_for_features(history_df: pd.DataFrame) -> pd.DataFrame:
    # Standardize historical monthly rows used to build lag features.
    history = history_df.copy()
    if history.empty:
        return pd.DataFrame(columns=["region_key", "month_date", "log_cases", "rainfall_avg"])

    if "region" not in history.columns and "Region" in history.columns:
        history["region"] = history["Region"]
    if "month_date" not in history.columns:
        date_col = pick_first_existing(history, ["reporting_date", "date", "TL"])
        history["month_date"] = pd.to_datetime(history[date_col], errors="coerce") if date_col else pd.NaT
    else:
        history["month_date"] = pd.to_datetime(history["month_date"], errors="coerce")

    if "log_cases" not in history.columns:
        if "cases" in history.columns:
            total_cases = pd.to_numeric(history["cases"], errors="coerce").fillna(0)
        else:
            confirmed_source = history["confirmed"] if "confirmed" in history.columns else history.get("cCh", pd.Series(0, index=history.index))
            suspected_source = history["suspected"] if "suspected" in history.columns else history.get("sCh", pd.Series(0, index=history.index))
            confirmed = pd.to_numeric(confirmed_source, errors="coerce").fillna(0)
            suspected = pd.to_numeric(suspected_source, errors="coerce").fillna(0)
            total_cases = confirmed + suspected
        history["log_cases"] = np.log1p(total_cases)

    rainfall_source = history["rainfall_avg"] if "rainfall_avg" in history.columns else pd.Series(0, index=history.index)
    region_source = history["region"] if "region" in history.columns else pd.Series("Unknown", index=history.index)
    history["rainfall_avg"] = pd.to_numeric(rainfall_source, errors="coerce").fillna(0)
    history["region_key"] = region_source.apply(normalize_region_for_feature)
    history["month_date"] = history["month_date"].dt.to_period("M").dt.to_timestamp()
    history = history.dropna(subset=["month_date"]).sort_values(["region_key", "month_date"])
    return history.drop_duplicates(subset=["region_key", "month_date"], keep="last")


def build_live_history_rows(monthly_env: pd.DataFrame) -> pd.DataFrame:
    # Convert monthly MongoDB aggregates into temporary history rows for lag features.
    if monthly_env.empty:
        return pd.DataFrame(columns=["Region", "month_date", "cases", "rainfall_avg"])

    live_history = monthly_env[["region_en", "report_month", "cCh", "sCh", "rainfall_avg"]].copy()
    live_history["Region"] = live_history["region_en"]
    live_history["month_date"] = pd.to_datetime(live_history["report_month"] + "-01", errors="coerce")
    live_history["cases"] = (
        pd.to_numeric(live_history["cCh"], errors="coerce").fillna(0)
        + pd.to_numeric(live_history["sCh"], errors="coerce").fillna(0)
    )
    live_history["rainfall_avg"] = pd.to_numeric(live_history["rainfall_avg"], errors="coerce").fillna(0)
    return live_history[["Region", "month_date", "cases", "rainfall_avg"]].dropna(subset=["month_date"])


def build_features(report, history_df, region_cols):
    report_date = pd.to_datetime(report["reporting_date"], errors="coerce")
    report_month = report_date.to_period("M").to_timestamp() if not pd.isna(report_date) else pd.NaT
    region_key = normalize_region_for_feature(report["region"])

    confirmed = pd.to_numeric(report.get("confirmed", 0), errors="coerce")
    suspected = pd.to_numeric(report.get("suspected", 0), errors="coerce")
    deaths = pd.to_numeric(report.get("deaths", 0), errors="coerce")
    rainfall_avg = pd.to_numeric(report.get("rainfall_avg", 0), errors="coerce")
    temperature_avg = pd.to_numeric(report.get("temperature_avg", 0), errors="coerce")
    humidity_avg = pd.to_numeric(report.get("humidity_avg", 0), errors="coerce")

    confirmed = 0 if pd.isna(confirmed) else confirmed
    suspected = 0 if pd.isna(suspected) else suspected
    deaths = 0 if pd.isna(deaths) else deaths
    rainfall_avg = 0 if pd.isna(rainfall_avg) else rainfall_avg
    temperature_avg = 0 if pd.isna(temperature_avg) else temperature_avg
    humidity_avg = 0 if pd.isna(humidity_avg) else humidity_avg

    total_cases = confirmed + suspected
    log_cases = np.log1p(total_cases)
    history = prepare_history_for_features(history_df)
    region_history = history[history["region_key"] == region_key].copy()

    previous_month = report_month - pd.DateOffset(months=1) if not pd.isna(report_month) else pd.NaT
    two_months_ago = report_month - pd.DateOffset(months=2) if not pd.isna(report_month) else pd.NaT

    def get_history_value(month_date, column_name):
        if pd.isna(month_date) or region_history.empty:
            return 0
        matches = region_history[region_history["month_date"] == month_date]
        if matches.empty:
            return 0
        return float(matches.sort_values("month_date").iloc[-1][column_name])

    feature_row = {
        "log_cases": log_cases,
        "deaths": deaths,
        "rainfall_avg": rainfall_avg,
        "temperature_avg": temperature_avg,
        "humidity_avg": humidity_avg,
        "log_cases_lag1": get_history_value(previous_month, "log_cases"),
        "log_cases_lag2": get_history_value(two_months_ago, "log_cases"),
        "rainfall_lag1": get_history_value(previous_month, "rainfall_avg"),
    }

    for region_col in region_cols:
        encoded_region = normalize_region_for_feature(region_col.replace("Region_", "", 1))
        feature_row[region_col] = 1 if encoded_region == region_key else 0

    return pd.DataFrame([feature_row], columns=list(feature_row.keys()))


@st.cache_resource
def load_cholera_risk_model():
    if joblib is None:
        raise RuntimeError("joblib is not installed. Run: pip install joblib")
    return joblib.load("cholera_risk_model.pkl")


@st.cache_data(ttl=3600)
def load_model_history_and_regions():
    history_path = "monthly_cholera_model_dataset_onehot.csv"
    history_df = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()
    region_cols = [column for column in history_df.columns if column.startswith("Region_")]
    return history_df, region_cols


def predict_occurrence_risk(report, history_df, region_cols):
    model = load_cholera_risk_model()
    model_features = getattr(model, "feature_names_in_", None)
    if not region_cols and model_features is not None:
        region_cols = [feature for feature in model_features if str(feature).startswith("Region_")]

    feature_row = build_features(report, history_df, region_cols)
    if model_features is not None:
        feature_row = feature_row.reindex(columns=list(model_features), fill_value=0)

    prediction = int(model.predict(feature_row)[0])
    return RISK_LABELS.get(prediction, "Unknown")


def standardize_mongo_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Map flexible Mongo fields into the dashboard's canonical schema.
    if raw_df.empty:
        return pd.DataFrame(columns=["date", "region_en", "district", "sCh", "cCh", "deaths"])

    region_col = pick_first_existing(raw_df, ["region", "region_en", "Region", "regionName"])
    district_col = pick_first_existing(raw_df, ["district", "district_name", "District"])
    date_col = pick_first_existing(raw_df, ["date", "report_date", "createdAt", "created_at", "timestamp", "TL"])
    suspected_col = pick_first_existing(raw_df, ["suspected", "sCh", "suspectedCases"])
    confirmed_col = pick_first_existing(raw_df, ["confirmed", "cCh", "confirmedCases"])
    deaths_col = pick_first_existing(raw_df, ["deaths", "Deaths"])

    raw_df["region_en"] = "Unknown" if region_col is None else raw_df[region_col].astype(str).replace("nan", "Unknown")
    raw_df["district"] = "Unknown" if district_col is None else raw_df[district_col].astype(str).replace("nan", "Unknown")
    raw_df["date"] = pd.Timestamp.utcnow() if date_col is None else pd.to_datetime(raw_df[date_col], errors="coerce")

    raw_df["sCh"] = pd.to_numeric(raw_df[suspected_col], errors="coerce").fillna(0) if suspected_col else 0
    raw_df["cCh"] = pd.to_numeric(raw_df[confirmed_col], errors="coerce").fillna(0) if confirmed_col else 0
    raw_df["deaths"] = pd.to_numeric(raw_df[deaths_col], errors="coerce").fillna(0) if deaths_col else 0

    result = raw_df[["date", "region_en", "district", "sCh", "cCh", "deaths"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["year"] = result["date"].dt.year
    result["region_norm"] = normalize_region_name(result["region_en"])
    return result


def build_live_regional_monthly_environment_table(
    live_region_df: pd.DataFrame,
    live_history_df: pd.DataFrame | None = None,
    latest_per_region: bool = False,
) -> pd.DataFrame:
    # Build one row per region/month for the risk table. The ensemble feature
    # pipeline handles environmental values separately, so this table should not
    # block on NASA API calls.
    env_source = live_region_df.dropna(subset=["date"]).copy()
    if env_source.empty:
        return pd.DataFrame()

    env_source["report_month"] = pd.to_datetime(env_source["date"], errors="coerce").dt.to_period("M").astype(str)
    env_source["report_month"] = env_source["report_month"].replace("NaT", pd.NA)
    env_source = env_source.dropna(subset=["report_month"])
    if env_source.empty:
        return pd.DataFrame()

    monthly_env = (
        env_source.groupby(["region_en", "report_month"], as_index=False)[["cCh", "sCh", "deaths"]]
        .sum()
        .sort_values(["report_month", "region_en"])
    )
    if monthly_env.empty:
        return pd.DataFrame()

    if latest_per_region:
        monthly_env = (
            monthly_env.sort_values(["region_en", "report_month"])
            .groupby("region_en", as_index=False)
            .tail(1)
            .sort_values("region_en")
        )

    def predict_row_output(row):
        try:
            prediction_result = predict_next_month_risk(row["region_en"])
            prediction = prediction_result["prediction"]
            return pd.Series(
                {
                    "OutbreakRisk_NextMonth": ENSEMBLE_RISK_LABELS.get(prediction, str(prediction)),
                    "OutbreakRisk_Class": prediction,
                    "Prediction_Error": "",
                }
            )
        except Exception as exc:
            return pd.Series(
                {
                    "OutbreakRisk_NextMonth": "Unavailable",
                    "OutbreakRisk_Class": pd.NA,
                    "Prediction_Error": str(exc),
                }
            )

    monthly_env[["OutbreakRisk_NextMonth", "OutbreakRisk_Class", "Prediction_Error"]] = monthly_env.apply(predict_row_output, axis=1)
    return monthly_env


@st.cache_data(ttl=300)
def load_mongo_live_data():
    # Load reports from MongoDB, trying preferred databases/collections first.
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed. Run: pip install pymongo")

    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
    client.admin.command("ping")

    preferred_dbs = [
        "cholera_monitoring_dashboard",
        "cholera_monitoring_dashboad_project",
        "cholera_monitoring_dashboard_project",
        "cholera monitoring dashboad project",
        "test",
    ]
    preferred_collections = ["reports", "report", "cholera_reports", "records"]

    db_names = client.list_database_names()
    ordered_dbs = [name for name in preferred_dbs if name in db_names] + [
        name for name in db_names if name not in preferred_dbs and name not in {"admin", "local", "config"}
    ]

    for db_name in ordered_dbs:
        db = client[db_name]
        col_names = db.list_collection_names()
        ordered_cols = [name for name in preferred_collections if name in col_names] + [
            name for name in col_names if name not in preferred_collections
        ]

        for col_name in ordered_cols:
            docs = list(db[col_name].find({}, {"_id": 0}).limit(10000))
            if not docs:
                continue
            mongo_df = standardize_mongo_records(pd.DataFrame(docs))
            if not mongo_df.empty and mongo_df[["sCh", "cCh", "deaths"]].sum().sum() > 0:
                return mongo_df, f"{db_name}.{col_name}"

    return pd.DataFrame(columns=["date", "region_en", "district", "sCh", "cCh", "deaths", "year", "region_norm"]), ""


def get_coords(region: str):
    # Resolve a region/district string to a representative map coordinate.
    for key, value in REGION_COORDS.items():
        if key.lower() in region.lower() or region.lower() in key.lower():
            return value
    return DEFAULT_COORDS


@st.cache_data(ttl=3600)
def fetch_nasa_conditions_for_date(lat, lon, report_date_str):
    # Fetch NASA POWER daily rainfall, temperature, and humidity for one date.
    report_dt = pd.to_datetime(report_date_str, errors="coerce")
    if pd.isna(report_dt):
        return {"rainfall_avg": None, "temp_avg": None, "humidity_avg": None, "date": None}

    date_token = report_dt.strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOT,T2M,RH2M&start={date_token}&end={date_token}"
        f"&latitude={lat}&longitude={lon}&community=AG&format=JSON"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    data = resp.json().get("properties", {}).get("parameter", {})
    rain_raw = data.get("PRECTOT", {}).get(date_token, -999)
    temp_raw = data.get("T2M", {}).get(date_token, -999)
    hum_raw = data.get("RH2M", {}).get(date_token, -999)

    return {
        "rainfall_avg": None if rain_raw == -999 else float(rain_raw),
        "temp_avg": None if temp_raw == -999 else float(temp_raw),
        "humidity_avg": None if hum_raw == -999 else float(hum_raw),
        "date": report_dt.strftime("%Y-%m-%d"),
    }


@st.cache_data(ttl=3600)
def fetch_nasa_conditions_for_month(lat, lon, report_month_str):
    # Fetch NASA POWER values once for the full month and return monthly averages.
    month_dt = pd.to_datetime(f"{report_month_str}-01", errors="coerce")
    if pd.isna(month_dt):
        return {"rainfall_avg": None, "temp_avg": None, "humidity_avg": None, "month": None}

    start_token = month_dt.strftime("%Y%m%d")
    end_token = month_dt.to_period("M").end_time.strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOT,T2M,RH2M&start={start_token}&end={end_token}"
        f"&latitude={lat}&longitude={lon}&community=AG&format=JSON"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    data = resp.json().get("properties", {}).get("parameter", {})

    def monthly_average(parameter_name):
        values = [
            float(value)
            for value in data.get(parameter_name, {}).values()
            if value != -999
        ]
        return sum(values) / len(values) if values else None

    return {
        "rainfall_avg": monthly_average("PRECTOT"),
        "temp_avg": monthly_average("T2M"),
        "humidity_avg": monthly_average("RH2M"),
        "month": month_dt.strftime("%Y-%m"),
    }


def classify_environmental_risk(rainfall_avg, humidity_avg, temp_avg):
    # Assign a placeholder High/Medium/Low environmental risk label.
    if rainfall_avg is None or humidity_avg is None or temp_avg is None:
        return "Medium"

    high_signals = 0
    medium_signals = 0

    if rainfall_avg >= 15:
        high_signals += 1
    elif rainfall_avg >= 6:
        medium_signals += 1

    if humidity_avg >= 80:
        high_signals += 1
    elif humidity_avg >= 65:
        medium_signals += 1

    if 24 <= temp_avg <= 34:
        high_signals += 1
    elif 20 <= temp_avg < 24 or 34 < temp_avg <= 37:
        medium_signals += 1

    if high_signals >= 2:
        return "High"
    if high_signals == 1 or medium_signals >= 2:
        return "Medium"
    return "Low"

def classify_cfr_risk(cfr_value):
    # Classify CFR risk using WHO-inspired thresholds: <1 low, 1-3 medium, >=3 high.
    if cfr_value >= 3:
        return "High"
    if cfr_value >= 1:
        return "Medium"
    return "Low"


def apply_chart_layout(fig, height=None):
    # Apply the shared dashboard visual style to Plotly figures.
    layout = {
        "plot_bgcolor": "#f7fbff",
        "paper_bgcolor": "#ffffff",
        "title_font_color": "#184fb6",
        "xaxis": {"gridcolor": "#dce9ff"},
        "yaxis": {"gridcolor": "#dce9ff"},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)


def render_metric_cards(confirmed, suspected, deaths, cfr):
    # Render KPI cards with totals and a color-coded CFR badge.
    cfr_risk = classify_cfr_risk(cfr)
    cfr_risk_color = RISK_COLOR_MAP.get(cfr_risk, "#455a64")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class=\"card\"><h3>Confirmed Cases</h3><p>{int(confirmed):,}</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class=\"card\"><h3>Suspected Cases</h3><p>{int(suspected):,}</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class=\"card\"><h3>Total Deaths</h3><p>{int(deaths):,}</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(
            f"""
            <div class=\"card\">
                <h3>CFR (%)</h3>
                <p style=\"margin-bottom:8px;\">{cfr:.2f}</p>
                <div style=\"display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap;\">
                    <span style=\"background:{cfr_risk_color}; color:#fff; padding:3px 9px; border-radius:999px; font-size:0.78rem; font-weight:700;\">
                        {cfr_risk}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_main_historical_chart(case_filtered_df, chart_type, chart_title):
    # Render the primary cases chart (Line/Bar/Pie) for historical slices.
    case_plot_df = case_filtered_df.groupby("TL", as_index=False)[["cCh", "sCh", "deaths"]].sum().sort_values("TL")

    if chart_type == "Line":
        melted = case_plot_df.melt(
            id_vars="TL",
            value_vars=["cCh", "sCh", "deaths"],
            var_name="Metric",
            value_name="Count",
        )
        melted["Metric"] = melted["Metric"].map({"cCh": "Confirmed", "sCh": "Suspected", "deaths": "Deaths"})
        fig = px.line(
            melted,
            x="TL",
            y="Count",
            color="Metric",
            title=chart_title,
            markers=True,
            labels={"TL": "Time Period Start", "Count": "Cases"},
            color_discrete_map=CMAP,
        )
    elif chart_type == "Bar":
        bar_df = case_plot_df.melt(
            id_vars="TL",
            value_vars=["cCh", "sCh", "deaths"],
            var_name="Metric",
            value_name="Count",
        )
        bar_df["Metric"] = bar_df["Metric"].map({"cCh": "Confirmed", "sCh": "Suspected", "deaths": "Deaths"})
        bar_df["TL"] = bar_df["TL"].dt.strftime("%Y-%m-%d")
        fig = px.bar(
            bar_df,
            x="TL",
            y="Count",
            color="Metric",
            barmode="group",
            title=chart_title,
            labels={"TL": "Time Period Start", "Count": "Number of Records"},
            color_discrete_map=CMAP,
        )
        fig.update_layout(bargap=0.08, bargroupgap=0.03)
    else:
        pie_df = pd.DataFrame(
            {
                "Metric": ["Confirmed", "Suspected", "Deaths"],
                "Count": [
                    int(case_filtered_df["cCh"].sum()),
                    int(case_filtered_df["sCh"].sum()),
                    int(case_filtered_df["deaths"].sum()),
                ],
            }
        )
        pie_df = pie_df[pie_df["Count"] > 0]
        if pie_df.empty:
            st.info("No data to display as a pie chart for this selection.")
            return
        fig = px.pie(
            pie_df,
            values="Count",
            names="Metric",
            title=chart_title,
            color="Metric",
            color_discrete_map=CMAP,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")

    apply_chart_layout(fig, height=520 if chart_type == "Bar" else None)
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def get_historical_selection(sidebar_regions, years):
    # Collect sidebar filters for historical dashboards.
    selected_region = st.sidebar.selectbox("Select Region", sidebar_regions)
    selected_region_norm = None if selected_region == "All Regions" else normalize_region_name(pd.Series([selected_region])).iloc[0]

    if selected_region == "All Regions":
        selected_district = st.sidebar.selectbox("Select District", ["All Districts"], disabled=True)
    else:
        district_options = sorted(
            district_df[district_df["region_norm"] == selected_region_norm]["district"].dropna().unique().tolist()
        )
        selected_district = st.sidebar.selectbox("Select District", ["All Districts"] + district_options)

    chart_type = st.sidebar.radio("Chart Type", ["Line", "Bar", "Pie"], index=0)
    selected_year = st.sidebar.selectbox("Select Year", years) if view_mode == "By Year" else None
    return selected_region, selected_region_norm, selected_district, chart_type, selected_year


def filter_historical_data(view_mode, selected_region, selected_region_norm, selected_district, selected_year):
    # Build historical case dataset from selected filters.
    if view_mode == "By Year":
        if selected_district == "All Districts":
            if selected_region == "All Regions":
                case_filtered_df = aggregate_regional_all_regions_by_tl(regional_env_df[regional_env_df["year"] == selected_year])
            else:
                case_filtered_df = regional_env_df[
                    (regional_env_df["region_norm"] == selected_region_norm) &
                    (regional_env_df["year"] == selected_year)
                ].sort_values("TL")
        else:
            base_df = district_df if selected_region == "All Regions" else district_df[district_df["region_norm"] == selected_region_norm]
            case_filtered_df = base_df[base_df["year"] == selected_year]
            case_filtered_df = case_filtered_df[case_filtered_df["district"] == selected_district]
        title_suffix = f" - {int(selected_year)}"
    else:
        if selected_district == "All Districts":
            if selected_region == "All Regions":
                case_filtered_df = aggregate_regional_all_regions_by_tl(regional_env_df)
            else:
                case_filtered_df = regional_env_df[regional_env_df["region_norm"] == selected_region_norm].sort_values("TL")
        else:
            base_df = district_df if selected_region == "All Regions" else district_df[district_df["region_norm"] == selected_region_norm]
            case_filtered_df = base_df[base_df["district"] == selected_district]
        title_suffix = " - All Years Trend"

    return case_filtered_df, title_suffix


def render_historical_dashboard(view_mode, regions, years):
    # Render the complete historical dashboard from filter to charts.
    selected_region, selected_region_norm, selected_district, chart_type, selected_year = get_historical_selection(regions, years)
    case_filtered_df, title_suffix = filter_historical_data(
        view_mode,
        selected_region,
        selected_region_norm,
        selected_district,
        selected_year,
    )

    st.subheader(f"{selected_region} - {selected_district}{title_suffix}")
    if case_filtered_df.empty:
        st.warning("No records found for the selected district.")
        return

    total_confirmed = int(case_filtered_df["cCh"].sum())
    total_suspected = int(case_filtered_df["sCh"].sum())
    total_deaths = int(case_filtered_df["deaths"].sum())
    cfr = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0

    render_metric_cards(total_confirmed, total_suspected, total_deaths, cfr)

    chart_title = f"Cholera Cases in {selected_region} - {selected_district}{title_suffix}"
    if chart_type == "Pie":
        chart_title = f"Distribution of Cholera Metrics in {selected_region} - {selected_district}{title_suffix}"
    render_main_historical_chart(case_filtered_df, chart_type, chart_title)

    hotspot_year = selected_year if view_mode == "By Year" else None
    if selected_region == "All Regions":
        st.subheader("Regional Cholera Hotspot Map")
        control_col, map_col = st.columns([1, 5])
        with control_col:
            hotspot_metric = st.radio(
                "Hotspot Metric",
                ["sCh", "cCh", "deaths", "cfr"],
                index=1,
                key=f"hotspot_metric_{'all_years' if hotspot_year is None else int(hotspot_year)}",
            )
        with map_col:
            render_regional_hotspot_map(hotspot_year, hotspot_metric)

    # Persist historical regional CFR hotspot table.
    export_df = build_cfr_hotspot_data(regional_env_df, hotspot_year)
    export_name = (
        f"historical_cfr_hotspot_by_region_{int(hotspot_year)}.csv"
        if hotspot_year is not None
        else "historical_cfr_hotspot_by_region_all_years.csv"
    )
    export_df.to_csv(export_name, index=False)


def render_live_dashboard():
    # Render the complete dashboard sourced from MongoDB records.
    st.subheader(" Cholera Reports - MongoDB(Demo)")

    rcol, _ = st.columns([1, 5])
    with rcol:
        if st.button("Refresh Data"):
            st.cache_data.clear()

    try:
        live_df, source_collection = load_mongo_live_data()
    except Exception as exc:
        st.error(f"Unable to load MongoDB data: {exc}")
        return

    if live_df.empty:
        st.warning("No records were found in MongoDB collections for visualization.")
        return

    st.caption(f"Source: MongoDB ({source_collection})")

    region_options = ["All Regions"] + sorted(live_df["region_en"].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Region", region_options)

    if selected_region == "All Regions":
        region_filtered = live_df.copy()
        selected_district = st.sidebar.selectbox("District", ["All Districts"], disabled=True)
    else:
        selected_live_norm = normalize_region_name(pd.Series([selected_region])).iloc[0]
        region_filtered = live_df[live_df["region_norm"] == selected_live_norm].copy()
        district_options = ["All Districts"] + sorted(region_filtered["district"].dropna().unique().tolist())
        selected_district = st.sidebar.selectbox("District", district_options)

    live_filtered = region_filtered if selected_district == "All Districts" else region_filtered[region_filtered["district"] == selected_district].copy()
    if live_filtered.empty:
        st.warning("No MongoDB records for the selected filters.")
        return

    live_filtered["report_month"] = pd.to_datetime(live_filtered["date"], errors="coerce").dt.to_period("M").astype(str)
    live_filtered["report_month"] = live_filtered["report_month"].replace("NaT", pd.NA)
    live_month_options = sorted(live_filtered["report_month"].dropna().unique().tolist(), reverse=True)
    if not live_month_options:
        st.warning("No valid report dates are available in MongoDB records.")
        return

    live_month_labels = ["All Months"] + live_month_options
    selected_live_month_label = st.sidebar.selectbox(" Report Month", live_month_labels, index=0)
    live_chart_type = st.sidebar.radio(" Chart Type", ["Line", "Bar", "Pie"], index=0)

    if selected_live_month_label != "All Months":
        live_filtered = live_filtered[live_filtered["report_month"] == selected_live_month_label].copy()
        if live_filtered.empty:
            st.warning("No MongoDB records for the selected month.")
            return

    st.caption(
        " month selected: All Months (aggregated)"
        if selected_live_month_label == "All Months"
        else f" month selected: {selected_live_month_label}"
    )

    total_confirmed = int(live_filtered["cCh"].sum())
    total_suspected = int(live_filtered["sCh"].sum())
    total_deaths = int(live_filtered["deaths"].sum())
    cfr = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0
    render_metric_cards(total_confirmed, total_suspected, total_deaths, cfr)

    live_case_df = live_filtered.dropna(subset=["date"]).copy()
    if not live_case_df.empty:
        live_case_df["TL"] = pd.to_datetime(live_case_df["date"], errors="coerce")
        live_title_suffix = " - All Months" if selected_live_month_label == "All Months" else f" - {selected_live_month_label}"
        live_chart_title = f" Cholera Cases in {selected_region} - {selected_district}{live_title_suffix}"
        if live_chart_type == "Pie":
            live_chart_title = f"Distribution of Cholera Metrics in {selected_region} - {selected_district}{live_title_suffix}"
        render_main_historical_chart(live_case_df, live_chart_type, live_chart_title)

    env_table = build_live_regional_monthly_environment_table(
        region_filtered,
        live_history_df=region_filtered,
        latest_per_region=True,
    )
    if not env_table.empty:

        def risk_cell_style(value):
            color = RISK_COLOR_MAP.get(str(value), "#455a64")
            return f"background-color: {color}; color: white; font-weight: 700"

        st.subheader("Risk Prediction Table")
        risk_table_columns = ["region_en", "report_month", "OutbreakRisk_Class", "OutbreakRisk_NextMonth"]
        if "Prediction_Error" in env_table.columns and env_table["Prediction_Error"].astype(str).str.len().gt(0).any():
            risk_table_columns.append("Prediction_Error")

        display_env_table = env_table[risk_table_columns].rename(
            columns={
                "region_en": "Region",
                "report_month": "Last Report Month",
                "OutbreakRisk_Class": "Model Output",
                "Prediction_Error": "Prediction Error",
            }
        )
        st.dataframe(
            display_env_table.style.applymap(risk_cell_style, subset=["OutbreakRisk_NextMonth"]),
            use_container_width=True,
        )


def main():
    # Application entrypoint: load data, configure mode, and render the active view.
    global district_df, regional_env_df, view_mode

    district_df, regional_env_df, years, regions = load_historical_data()

    with open("app_styles.css", "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

    st.title("Cameroon Cholera Monitoring Dashboard")

    data_mode = st.sidebar.radio("Data Mode", ["Historical Data", " Recent Data"], index=0)
    view_mode = (
        st.sidebar.radio("Historical Data", ["By Year", "All Years Trend"], index=0)
        if data_mode == "Historical Data"
        else " Data"
    )

    if view_mode in ["By Year", "All Years Trend"]:
        render_historical_dashboard(view_mode, regions, years)
    else:
        render_live_dashboard()


if __name__ == "__main__":
    main()
