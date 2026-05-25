import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import json
from datetime import datetime, timedelta

try:
    import pydeck as pdk
except ImportError:
    pdk = None

try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
except ImportError:
    MongoClient = None
    ServerApi = None

# Load your dataset
df = pd.read_csv("region_period_summary_0_5380.csv")
district_df = pd.read_csv("dataset_with_only_district.csv")

# Extract year from TL column (starting date)
df['TL'] = pd.to_datetime(df['TL'], errors='coerce')
df['year'] = df['TL'].dt.year
district_df['TL'] = pd.to_datetime(district_df['TL'], errors='coerce')
district_df['year'] = district_df['TL'].dt.year

# Ensure numeric aggregation works even when source CSV has blanks
for col in ['sCh', 'cCh', 'deaths']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    district_df[col] = pd.to_numeric(district_df[col], errors='coerce').fillna(0)

def normalize_region_name(series):
    return series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

def normalize_single_region_name(name):
    return normalize_region_name(pd.Series([name])).iloc[0]

df['region_norm'] = normalize_region_name(df['region_en'])
district_df['region_norm'] = normalize_region_name(district_df['region_en'])

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://yangsisonia9_db_user:64687@cluster0.295yjzd.mongodb.net/"
)

def canonicalize_disease_name(value):
    raw = str(value or "").strip()
    norm = "".join(ch for ch in raw.lower() if ch.isalnum())

    aliases = {
        "cholera": "Cholera",
        "covid": "COVID-19",
        "covid19": "COVID-19",
        "coronavirus": "COVID-19",
        "sarscov2": "COVID-19",
        "measles": "Measles",
        "mpox": "Mpox",
        "monkeypox": "Mpox",
        "influenza": "Influenza",
        "flu": "Influenza",
    }

    return aliases.get(norm, raw if raw else "Unknown")

def pick_first_existing(df_input, candidates):
    for candidate in candidates:
        if candidate in df_input.columns:
            return candidate
    return None

def standardize_mongo_records(raw_df):
    if raw_df.empty:
        return pd.DataFrame(columns=["date", "region_en", "district", "disease_type", "sCh", "cCh", "deaths"])

    region_col = pick_first_existing(raw_df, ["region", "region_en", "Region", "regionName"])
    district_col = pick_first_existing(raw_df, ["district", "district_name", "District"])
    date_col = pick_first_existing(raw_df, ["date", "report_date", "createdAt", "created_at", "timestamp", "TL"])
    disease_col = pick_first_existing(raw_df, ["pandemic", "disease_type", "disease", "diseaseType", "condition", "pathology"])
    suspected_col = pick_first_existing(raw_df, ["suspected", "sCh", "suspectedCases"])
    confirmed_col = pick_first_existing(raw_df, ["confirmed", "cCh", "confirmedCases"])
    deaths_col = pick_first_existing(raw_df, ["deaths", "Deaths"])

    if region_col is None:
        raw_df["region_en"] = "Unknown"
    else:
        raw_df["region_en"] = raw_df[region_col].astype(str).replace("nan", "Unknown")

    if district_col is None:
        raw_df["district"] = "Unknown"
    else:
        raw_df["district"] = raw_df[district_col].astype(str).replace("nan", "Unknown")

    if date_col is None:
        raw_df["date"] = pd.Timestamp.utcnow()
    else:
        raw_df["date"] = pd.to_datetime(raw_df[date_col], errors="coerce")

    if disease_col is None:
        raw_df["disease_type"] = "Unknown"
    else:
        raw_df["disease_type"] = raw_df[disease_col].astype(str).replace("nan", "Unknown").str.strip()
        raw_df.loc[raw_df["disease_type"] == "", "disease_type"] = "Unknown"
    raw_df["disease_type"] = raw_df["disease_type"].apply(canonicalize_disease_name)

    raw_df["sCh"] = pd.to_numeric(raw_df[suspected_col], errors="coerce").fillna(0) if suspected_col else 0
    raw_df["cCh"] = pd.to_numeric(raw_df[confirmed_col], errors="coerce").fillna(0) if confirmed_col else 0
    raw_df["deaths"] = pd.to_numeric(raw_df[deaths_col], errors="coerce").fillna(0) if deaths_col else 0

    result = raw_df[["date", "region_en", "district", "disease_type", "sCh", "cCh", "deaths"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["year"] = result["date"].dt.year
    result["region_norm"] = normalize_region_name(result["region_en"])
    return result

@st.cache_data(ttl=3600)
def load_region_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_geo_mapping(geojson_obj):
    mapping = {}
    for feature in geojson_obj.get("features", []):
        shape_name = feature.get("properties", {}).get("shapeName")
        if shape_name:
            mapping[normalize_single_region_name(shape_name)] = shape_name

    aliases = {
        "adamawa": "Adamaoua",
        "farnorth": "Far North",
        "northwest": "North-West",
        "southwest": "South-West",
    }
    for norm_name, shape_name in aliases.items():
        if shape_name in [f.get("properties", {}).get("shapeName") for f in geojson_obj.get("features", [])]:
            mapping[norm_name] = shape_name
    return mapping

@st.cache_data(ttl=300)
def load_mongo_live_data():
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed. Run: pip install pymongo")

    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
    client.admin.command("ping")

    db_names = client.list_database_names()
    preferred_dbs = [
        "cholera_monitoring_dashboard",
        "cholera_monitoring_dashboad_project",
        "cholera_monitoring_dashboard_project",
        "cholera monitoring dashboad project",
        "test",
    ]
    preferred_collections = ["reports", "report", "cholera_reports", "records"]

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

    return pd.DataFrame(columns=["date", "region_en", "district", "disease_type", "sCh", "cCh", "deaths", "year", "region_norm"]), ""

# -------------------------------------------------------
# Coordinates for all 10 regions of Cameroon (lat, lon)
# -------------------------------------------------------
REGION_COORDS = {
    "Adamawa": (7.3167, 13.5833), "Adamaoua": (7.3167, 13.5833),
    "Centre": (3.8667, 11.5167), "Center": (3.8667, 11.5167),
    "Yaounde": (3.8667, 11.5167), "Yaoundé": (3.8667, 11.5167),
    "East": (4.5833, 13.6833), "Est": (4.5833, 13.6833), "Bertoua": (4.5833, 13.6833),
    "Far North": (10.5959, 14.3159), "Extreme Nord": (10.5959, 14.3159),
    "Extrême-Nord": (10.5959, 14.3159), "Maroua": (10.5959, 14.3159),
    "Littoral": (4.0500, 9.7000), "Douala": (4.0500, 9.7000),
    "North": (9.3000, 13.3833), "Nord": (9.3000, 13.3833), "Garoua": (9.3000, 13.3833),
    "North West": (5.9631, 10.1597), "Northwest": (5.9631, 10.1597),
    "Nord-Ouest": (5.9631, 10.1597), "Bamenda": (5.9631, 10.1597),
    "South": (2.9000, 11.1500), "Sud": (2.9000, 11.1500), "Ebolowa": (2.9000, 11.1500),
    "South West": (4.1527, 9.2403), "Southwest": (4.1527, 9.2403),
    "Sud-Ouest": (4.1527, 9.2403), "Buea": (4.1527, 9.2403),
    "Fako": (4.1527, 9.2403), "Meme": (4.5167, 9.3833),
    "Manyu": (5.9333, 9.3667), "Ndian": (4.6667, 8.8333),
    "Kupe-Muanenguba": (4.7833, 9.6833), "Lebialem": (5.5000, 9.9167),
    "West": (5.4667, 10.4167), "Ouest": (5.4667, 10.4167), "Bafoussam": (5.4667, 10.4167),
}
DEFAULT_COORDS = (5.6968, 12.3547)

def get_coords(region):
    for key in REGION_COORDS:
        if key.lower() in region.lower() or region.lower() in key.lower():
            return REGION_COORDS[key]
    return DEFAULT_COORDS

@st.cache_data(ttl=3600)
def fetch_nasa_conditions(lat, lon):
    end_dt   = datetime.today() - timedelta(days=2)
    start_dt = end_dt - timedelta(days=29)
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOT,T2M,RH2M&start={start_dt.strftime('%Y%m%d')}&end={end_dt.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&community=AG&format=JSON"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]
    rain = [v for v in data["PRECTOT"].values() if v != -999]
    temp = [v for v in data["T2M"].values()     if v != -999]
    hum  = [v for v in data["RH2M"].values()    if v != -999]
    return {
        "rainfall_avg": sum(rain) / len(rain) if rain else None,
        "temp_avg":     sum(temp) / len(temp) if temp else None,
        "humidity_avg": sum(hum)  / len(hum)  if hum  else None,
        "start": start_dt.strftime("%Y-%m-%d"),
        "end":   end_dt.strftime("%Y-%m-%d"),
    }

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
regions = ["All Regions"] + sorted(df['region_en'].dropna().unique().tolist())
years   = sorted(df['year'].dropna().unique())

data_mode = st.sidebar.radio(
    "Data Mode",
    ["Historical Data", "Live Data"],
    index=0
)

if data_mode == "Historical Data":
    historical_mode = st.sidebar.radio("Historical Data", ["By Year", "All Years Trend"], index=0)
    view_mode = historical_mode
else:
    view_mode = "Live Data"

if view_mode != "Live Data":
    selected_region = st.sidebar.selectbox("Select Region", regions)
    if selected_region == "All Regions":
        selected_region_norm = None
        district_options = ["All Districts"]
        selected_district = st.sidebar.selectbox("Select District", district_options, disabled=True)
    else:
        selected_region_norm = normalize_region_name(pd.Series([selected_region])).iloc[0]
        district_options = sorted(
            district_df[district_df['region_norm'] == selected_region_norm]['district'].dropna().unique().tolist()
        )
        district_options = ["All Districts"] + district_options
        selected_district = st.sidebar.selectbox("Select District", district_options)

    if selected_district == "All Districts":
        selected_env = st.sidebar.multiselect(
            "Select Environmental Variables",
            ["rainfall_avg", "humidity_avg", "temp_avg"],
            default=["rainfall_avg"]
        )
    else:
        selected_env = []

    chart_type = st.sidebar.radio(
        "Chart Type",
        ["Line", "Bar", "Pie"],
        index=0
    )

if view_mode == "By Year":
    selected_year = st.sidebar.selectbox("Select Year", years)
    if selected_region == "All Regions":
        case_filtered_df = district_df[district_df['year'] == selected_year]
    else:
        case_filtered_df = district_df[
            (district_df['region_norm'] == selected_region_norm) &
            (district_df['year'] == selected_year)
        ]

    if selected_district != "All Districts":
        case_filtered_df = case_filtered_df[case_filtered_df['district'] == selected_district]

    if selected_district == "All Districts":
        if selected_region == "All Regions":
            filtered_df = df[df['year'] == selected_year].sort_values('TL')
        else:
            filtered_df = df[
                (df['region_norm'] == selected_region_norm) & (df['year'] == selected_year)
            ].sort_values('TL')
    else:
        if selected_region == "All Regions":
            filtered_df = district_df[
                (district_df['district'] == selected_district) &
                (district_df['year'] == selected_year)
            ].sort_values('TL')
        else:
            filtered_df = district_df[
                (district_df['region_norm'] == selected_region_norm) &
                (district_df['district'] == selected_district) &
                (district_df['year'] == selected_year)
            ].sort_values('TL')
    title_suffix  = f" - {int(selected_year)}"
elif view_mode == "All Years Trend":
    selected_year = None
    if selected_region == "All Regions":
        case_filtered_df = district_df.copy()
    else:
        case_filtered_df = district_df[
            (district_df['region_norm'] == selected_region_norm)
        ]

    if selected_district != "All Districts":
        case_filtered_df = case_filtered_df[case_filtered_df['district'] == selected_district]

    if selected_district == "All Districts":
        if selected_region == "All Regions":
            filtered_df = df.sort_values('TL')
        else:
            filtered_df = df[df['region_norm'] == selected_region_norm].sort_values('TL')
    else:
        if selected_region == "All Regions":
            filtered_df = district_df[
                (district_df['district'] == selected_district)
            ].sort_values('TL')
        else:
            filtered_df = district_df[
                (district_df['region_norm'] == selected_region_norm) &
                (district_df['district'] == selected_district)
            ].sort_values('TL')
    title_suffix  = " - All Years Trend"

df['period'] = df['TL'].astype(str) + ' to ' + df['TR'].astype(str)

# -------------------------------------------------------
# Shared CSS
# -------------------------------------------------------
st.markdown("""
    <style>
    :root {
        --blue-50: #f4f8ff;
        --blue-100: #e6f0ff;
        --blue-200: #cfe0ff;
        --blue-400: #4f8cff;
        --blue-600: #1e5fd8;
        --blue-700: #184fb6;
        --blue-900: #0f2d66;
    }
    .stApp {
        background: linear-gradient(180deg, var(--blue-50) 0%, #ffffff 55%, var(--blue-50) 100%);
    }
    h1, h2, h3, .stMarkdown, .stCaption {
        color: var(--blue-900);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eaf3ff 0%, #dfefff 100%);
        border-right: 1px solid var(--blue-200);
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: var(--blue-700) !important;
        font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(180deg, var(--blue-400), var(--blue-600));
        color: white;
        border: 1px solid var(--blue-700);
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, var(--blue-600), var(--blue-700));
    }
    [data-baseweb="select"] {
        background-color: #eef5ff !important;
        border: 1px solid var(--blue-200) !important;
        border-radius: 10px !important;
    }
    .card {
        border: 2px solid #1E90FF;
        border-radius: 10px;
        padding: 0px;
        text-align: center;
        transition: 0.3s;
        height: 150px;
        background: linear-gradient(180deg, #ffffff 0%, #eef5ff 100%);
        display: flex;
        flex-direction: column;
        justify-content: left;
        align-items: left;
    }
    .card:hover { background-color: #F0F8FF; box-shadow: 0px 4px 8px rgba(0,0,0,0.2); }
    .card h3 { margin: 0; color: #1E90FF; }
    .card p  { font-size: 22px; font-weight: bold; margin: 0px 0 0 0; }
    .env-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
        background: linear-gradient(180deg, #ffffff 0%, #eef5ff 100%);
        border-left: 6px solid;
        border-top: 1px solid #d8e8ff;
        border-right: 1px solid #d8e8ff;
        border-bottom: 1px solid #d8e8ff;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .env-card .label { font-size: 16px; font-weight: 600; color: #333; }
    .env-card .value { font-size: 15px; color: #555; }
    .env-card .badge {
        font-size: 14px; font-weight: bold;
        padding: 4px 14px; border-radius: 20px; color: white;
    }
    .risk-banner {
        border-radius: 12px; padding: 18px 24px;
        text-align: center; font-size: 20px;
        font-weight: bold; color: white; margin-top: 12px;
        box-shadow: 0 4px 16px rgba(24, 79, 182, 0.2);
    }
    /* Blue background for selected multiselect tags */
    div[data-baseweb="tag"] {
        background-color: #d9e8ff !important;
        color: #0f2d66 !important;
        border: 1px solid #a9c7ff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Cameroon Cholera Monitoring Dashboard")

# -------------------------------------------------------
# VIEW: By Year / All Years Trend
# -------------------------------------------------------
if view_mode in ["By Year", "All Years Trend"]:
    st.subheader(f"{selected_region} - {selected_district}{title_suffix}")

    if case_filtered_df.empty:
        st.warning("No records found for the selected district.")
        st.stop()

    # Cases/deaths should always come from district-level data
    case_plot_df = case_filtered_df.groupby('TL', as_index=False)[['cCh', 'sCh', 'deaths']].sum().sort_values('TL')

    total_confirmed_hist = int(case_filtered_df['cCh'].sum())
    total_suspected_hist = int(case_filtered_df['sCh'].sum())
    total_deaths_hist = int(case_filtered_df['deaths'].sum())
    cfr_hist = (total_deaths_hist / total_confirmed_hist * 100) if total_confirmed_hist > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="card"><h3>Confirmed Cases</h3>
            <p>{total_confirmed_hist}</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="card"><h3>Suspected Cases</h3>
            <p>{total_suspected_hist}</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="card"><h3>Total Deaths</h3>
            <p>{total_deaths_hist}</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="card"><h3>CFR (%)</h3>
            <p>{cfr_hist:.2f}</p></div>""", unsafe_allow_html=True)

    CMAP = {"Confirmed": "#1e5fd8", "Suspected": "#f9a825", "Deaths": "#d32f2f"}
    chart_title = f"Cholera Cases in {selected_region} - {selected_district}{title_suffix}"

    if chart_type == "Line":
        melted = case_plot_df.melt(
            id_vars="TL",
            value_vars=["cCh", "sCh", "deaths"],
            var_name="Metric", value_name="Count"
        )
        melted["Metric"] = melted["Metric"].map({"cCh": "Confirmed", "sCh": "Suspected", "deaths": "Deaths"})
        fig_main = px.line(
            melted, x="TL", y="Count", color="Metric",
            title=chart_title, markers=True,
            labels={"TL": "Time Period Start", "Count": "Cases"},
            color_discrete_map=CMAP
        )
        fig_main.update_layout(
            plot_bgcolor="#f7fbff", paper_bgcolor="#ffffff",
            title_font_color="#184fb6",
            xaxis=dict(gridcolor="#dce9ff"), yaxis=dict(gridcolor="#dce9ff"),
            legend_title_text=""
        )
    elif chart_type == "Bar":
        bar_df = case_plot_df.melt(
            id_vars="TL",
            value_vars=["cCh", "sCh", "deaths"],
            var_name="Metric", value_name="Count"
        )
        bar_df["Metric"] = bar_df["Metric"].map({"cCh": "Confirmed", "sCh": "Suspected", "deaths": "Deaths"})
        bar_df["TL"] = bar_df["TL"].dt.strftime("%Y-%m-%d")
        fig_main = px.bar(
            bar_df, x="TL", y="Count", color="Metric", barmode="group",
            title=chart_title,
            labels={"TL": "Time Period Start", "Count": "Number of Records"},
            color_discrete_map=CMAP
        )
        fig_main.update_layout(
            plot_bgcolor="#f7fbff", paper_bgcolor="#ffffff",
            title_font_color="#184fb6",
            xaxis=dict(gridcolor="#dce9ff"), yaxis=dict(gridcolor="#dce9ff"),
            legend_title_text="",
            height=520,
            bargap=0.08,
            bargroupgap=0.03
        )
    else:  # Pie
        pie_df = pd.DataFrame({
            "Metric": ["Confirmed", "Suspected", "Deaths"],
            "Count": [
                int(case_filtered_df["cCh"].sum()),
                int(case_filtered_df["sCh"].sum()),
                int(case_filtered_df["deaths"].sum()),
            ]
        })
        pie_df = pie_df[pie_df["Count"] > 0]
        if pie_df.empty:
            st.info("No data to display as a pie chart for this selection.")
            fig_main = None
        else:
            fig_main = px.pie(
                pie_df, values="Count", names="Metric",
                title=f"Distribution of Cholera Metrics in {selected_region} - {selected_district}{title_suffix}",
                color="Metric", color_discrete_map=CMAP
            )
            fig_main.update_traces(textposition="inside", textinfo="percent+label")
            fig_main.update_layout(
                paper_bgcolor="#ffffff",
                title_font_color="#184fb6",
                legend_title_text=""
            )

    if fig_main is not None:
        st.plotly_chart(fig_main, use_container_width=True)

    if selected_district == "All Districts":
        for var in selected_env:
            if chart_type == "Bar":
                env_bar_df = filtered_df.copy()
                env_bar_df["TL"] = env_bar_df["TL"].dt.strftime("%Y-%m-%d")
                fig_env = px.bar(
                    env_bar_df, x="TL", y=var,
                    title=f"{var} in {selected_region}{title_suffix}",
                    labels={"TL": "Time Period Start"},
                    color_discrete_sequence=["#1e5fd8"]
                )
            elif chart_type == "Pie":
                # Pie doesn't suit time-series env data; fall back to bar
                fig_env = px.bar(
                    filtered_df, x="TL", y=var,
                    title=f"{var} in {selected_region}{title_suffix}",
                    labels={"TL": "Time Period Start"},
                    color_discrete_sequence=["#1e5fd8"]
                )
            else:
                fig_env = px.line(
                    filtered_df, x="TL", y=var,
                    title=f"{var} in {selected_region}{title_suffix}",
                    markers=True, color_discrete_sequence=["blue"],
                    labels={"TL": "Time Period Start"}
                )
                fig_env.update_traces(line_color="#1e5fd8", marker_color="#4f8cff")
            fig_env.update_layout(
                plot_bgcolor="#f7fbff",
                paper_bgcolor="#ffffff",
                title_font_color="#184fb6",
                xaxis=dict(gridcolor="#dce9ff"),
                yaxis=dict(gridcolor="#dce9ff"),
                height=460,
                bargap=0.08,
                bargroupgap=0.03
            )
            st.plotly_chart(fig_env)

        for var in selected_env:
            fig_scatter = px.scatter(
                filtered_df, x=var, y="cCh",
                title=f"Confirmed Cases vs {var} in {selected_region}{title_suffix}",
                trendline="ols", labels={"cCh": "Confirmed Cases"}
            )
            fig_scatter.update_traces(marker_color="#4f8cff")
            fig_scatter.update_layout(
                plot_bgcolor="#f7fbff",
                paper_bgcolor="#ffffff",
                title_font_color="#184fb6",
                xaxis=dict(gridcolor="#dce9ff"),
                yaxis=dict(gridcolor="#dce9ff")
            )
            st.plotly_chart(fig_scatter)

# -------------------------------------------------------
# VIEW: Live Data — all regions
# -------------------------------------------------------
else:
    st.subheader("Live Cholera Reports — MongoDB")

    rcol, _ = st.columns([1, 5])
    with rcol:
        if st.button("Refresh Live Data"):
            st.cache_data.clear()

    try:
        live_df, source_collection = load_mongo_live_data()
    except Exception as exc:
        st.error(f"Unable to load MongoDB data: {exc}")
        st.stop()

    if live_df.empty:
        st.warning("No records were found in MongoDB collections for live visualization.")
        st.stop()

    st.caption(f"Source: MongoDB ({source_collection})")

    live_disease_options = sorted(
        [
            d
            for d in live_df["disease_type"].dropna().astype(str).str.strip().unique().tolist()
            if d and d.lower() not in {"unknown", "nan"}
        ]
    )

    if not live_disease_options:
        st.warning("No valid disease/pandemic values found in MongoDB records.")
        st.stop()

    selected_disease = st.sidebar.selectbox("Disease Type", live_disease_options)
    disease_filtered = live_df[live_df["disease_type"] == selected_disease].copy()

    if disease_filtered.empty:
        st.warning(f"No MongoDB records found for {selected_disease}.")
        st.stop()

    region_options_live = ["All Regions"] + sorted(disease_filtered["region_en"].dropna().unique().tolist())
    selected_live_region = st.sidebar.selectbox("Region", region_options_live)

    if selected_live_region == "All Regions":
        region_filtered = disease_filtered.copy()
    else:
        selected_live_norm = normalize_region_name(pd.Series([selected_live_region])).iloc[0]
        region_filtered = disease_filtered[disease_filtered["region_norm"] == selected_live_norm].copy()

    if selected_live_region == "All Regions":
        district_options_live = ["All Districts"]
        selected_live_district = st.sidebar.selectbox(
            "District",
            district_options_live,
            disabled=True
        )
    else:
        district_options_live = ["All Districts"] + sorted(
            region_filtered["district"].dropna().unique().tolist()
        )
        selected_live_district = st.sidebar.selectbox("District", district_options_live)

    if selected_live_district == "All Districts":
        live_filtered = region_filtered.copy()
    else:
        live_filtered = region_filtered[region_filtered["district"] == selected_live_district].copy()

    if live_filtered.empty:
        st.warning("No MongoDB records for the selected live filters.")
        st.stop()

    total_confirmed = int(live_filtered["cCh"].sum())
    total_suspected = int(live_filtered["sCh"].sum())
    total_deaths = int(live_filtered["deaths"].sum())
    cfr = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="card"><h3>Confirmed Cases</h3>
            <p>{total_confirmed:,}</p></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="card"><h3>Suspected Cases</h3>
            <p>{total_suspected:,}</p></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="card"><h3>Total Deaths</h3>
            <p>{total_deaths:,}</p></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="card"><h3>CFR (%)</h3>
            <p>{cfr:.2f}</p></div>""", unsafe_allow_html=True)

    region_agg = (
        live_filtered.groupby("region_en", as_index=False)[["cCh", "sCh", "deaths"]]
        .sum()
        .sort_values("cCh", ascending=False)
    )
    region_agg["region_norm"] = normalize_region_name(region_agg["region_en"])

    fig_region = px.bar(
        region_agg,
        x="region_en",
        y=["cCh", "sCh", "deaths"],
        barmode="group",
        title="Regional Breakdown (Live MongoDB)",
        labels={"region_en": "Region", "value": "Cases", "variable": "Metric"},
        color_discrete_map={"cCh": "#1e5fd8", "sCh": "#f9a825", "deaths": "#d32f2f"}
    )
    fig_region.update_layout(
        plot_bgcolor="#f7fbff",
        paper_bgcolor="#ffffff",
        title_font_color="#184fb6",
        xaxis=dict(gridcolor="#dce9ff"),
        yaxis=dict(gridcolor="#dce9ff")
    )
    st.plotly_chart(fig_region, use_container_width=True)

    try:
        region_geojson = load_region_geojson("geoBoundaries-CMR-ADM1.geojson")
        geo_regions = []
        for feature in region_geojson.get("features", []):
            shape_name = feature.get("properties", {}).get("shapeName")
            if shape_name:
                geo_regions.append(
                    {
                        "shapeName": shape_name,
                        "region_norm": normalize_single_region_name(shape_name),
                    }
                )

        geo_df = pd.DataFrame(geo_regions)
        if not geo_df.empty:
            st.subheader("Complete Cameroon Map (GeoJSON ADM1)")
            if pdk is None:
                st.warning("pydeck is not installed. Run: pip install pydeck")
            else:
                geo_layer = pdk.Layer(
                    "GeoJsonLayer",
                    region_geojson,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    get_fill_color=[219, 234, 254, 170],
                    get_line_color=[24, 79, 182, 220],
                    get_line_width=2,
                    line_width_min_pixels=1,
                    auto_highlight=True,
                )

                deck = pdk.Deck(
                    layers=[geo_layer],
                    initial_view_state=pdk.ViewState(
                        latitude=5.7,
                        longitude=12.3,
                        zoom=5,
                        pitch=0,
                        bearing=0,
                    ),
                    map_style=None,
                    tooltip={"text": "{shapeName}"},
                )
                st.pydeck_chart(deck, use_container_width=True)

            st.caption(
                "GeoJSON regions loaded: "
                + ", ".join(geo_df["shapeName"].tolist())
            )
        else:
            st.info("No mappable region names found in the GeoJSON file.")
    except Exception as geo_exc:
        st.warning(f"GeoJSON map unavailable: {geo_exc}")

    monthly_source = live_filtered.dropna(subset=["date"]).copy()
    if monthly_source.empty:
        monthly_agg = pd.DataFrame(columns=["month", "cCh", "sCh", "deaths"])
    else:
        monthly_agg = (
            monthly_source.set_index("date")[["cCh", "sCh", "deaths"]]
            .resample("MS")
            .sum()
            .reset_index()
            .rename(columns={"date": "month"})
            .sort_values("month")
        )

    if not monthly_agg.empty:
        monthly_plot = monthly_agg.melt(
            id_vars="month",
            value_vars=["cCh", "sCh", "deaths"],
            var_name="Metric",
            value_name="Count"
        )
        monthly_plot["Metric"] = monthly_plot["Metric"].map(
            {"cCh": "Confirmed", "sCh": "Suspected", "deaths": "Deaths"}
        )
        fig_trend = px.line(
            monthly_plot,
            x="month",
            y="Count",
            color="Metric",
            markers=True,
            title="Monthly Trend (Live MongoDB)",
            color_discrete_map={"Confirmed": "#1e5fd8", "Suspected": "#f9a825", "Deaths": "#d32f2f"}
        )
        fig_trend.update_layout(
            plot_bgcolor="#f7fbff",
            paper_bgcolor="#ffffff",
            title_font_color="#184fb6",
            xaxis=dict(gridcolor="#dce9ff"),
            yaxis=dict(gridcolor="#dce9ff")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

