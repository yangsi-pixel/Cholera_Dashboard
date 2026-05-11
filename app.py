import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, timedelta

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

df['region_norm'] = normalize_region_name(df['region_en'])
district_df['region_norm'] = normalize_region_name(district_df['region_en'])

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

def classify_rainfall(val):
    if val >= 150: return ("HIGH",   "#D32F2F")
    elif val >= 60: return ("MEDIUM", "#F57C00")
    else:           return ("LOW",    "#388E3C")

def classify_temperature(val):
    if val >= 30:   return ("HIGH",   "#D32F2F")
    elif val >= 24: return ("MEDIUM", "#F57C00")
    else:           return ("LOW",    "#388E3C")

def classify_humidity(val):
    if val >= 80:   return ("HIGH",   "#D32F2F")
    elif val >= 60: return ("MEDIUM", "#F57C00")
    else:           return ("LOW",    "#388E3C")

def classify_cholera_risk(rain_lbl, hum_lbl, temp_lbl):
    score = sum(1 for l in [rain_lbl, hum_lbl, temp_lbl] if l == "HIGH")
    if score >= 2: return ("HIGH RISK",   "#D32F2F")
    elif score == 1: return ("MEDIUM RISK", "#F57C00")
    else:            return ("LOW RISK",    "#388E3C")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
regions = ["All Regions"] + sorted(df['region_en'].dropna().unique().tolist())
years   = sorted(df['year'].dropna().unique())

view_mode = st.sidebar.radio("View Mode", ["By Year", "All Years Trend", "Live Data"])

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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="card"><h3>Confirmed Cases</h3>
            <p>{int(case_filtered_df['cCh'].sum())}</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="card"><h3>Suspected Cases</h3>
            <p>{int(case_filtered_df['sCh'].sum())}</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="card"><h3>Total Deaths</h3>
            <p>{int(case_filtered_df['deaths'].sum())}</p></div>""", unsafe_allow_html=True)

    fig_cases = px.line(
        case_plot_df, x="TL", y="cCh",
        title=f"Confirmed Cholera Cases in {selected_region} - {selected_district}{title_suffix}",
        markers=True, labels={"cCh": "Confirmed Cases", "TL": "Time Period Start"}
    )
    fig_cases.update_traces(line_color="#1e5fd8", marker_color="#4f8cff")
    fig_cases.update_layout(
        plot_bgcolor="#f7fbff",
        paper_bgcolor="#ffffff",
        title_font_color="#184fb6",
        xaxis=dict(gridcolor="#dce9ff"),
        yaxis=dict(gridcolor="#dce9ff")
    )
    st.plotly_chart(fig_cases)

    if selected_district == "All Districts":
        for var in selected_env:
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
                yaxis=dict(gridcolor="#dce9ff")
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
    st.subheader("Live Environmental Conditions — All Regions")

    end_dt   = datetime.today() - timedelta(days=2)
    start_dt = end_dt - timedelta(days=29)
    st.caption(f"Data averaged over last 30 days ({start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}) | Source: NASA POWER API")

    rcol, _ = st.columns([1, 5])
    with rcol:
        if st.button("Refresh Live Data"):
            st.cache_data.clear()

    all_regions = sorted(df['region_en'].unique())

    for region in all_regions:
        lat, lon = get_coords(region)
        with st.spinner(f"Fetching {region}..."):
            try:
                nasa = fetch_nasa_conditions(lat, lon)
                rain_val = nasa["rainfall_avg"]
                temp_val = nasa["temp_avg"]
                hum_val  = nasa["humidity_avg"]
                source   = "NASA POWER"
            except Exception:
                latest   = df[df['region_en'] == region].sort_values('TL').iloc[-1]
                rain_val = latest['rainfall_avg']
                temp_val = latest['temp_avg']
                hum_val  = latest['humidity_avg']
                source   = ""

        rain_lbl, rain_col = classify_rainfall(rain_val)
        temp_lbl, temp_col = classify_temperature(temp_val)
        hum_lbl,  hum_col  = classify_humidity(hum_val)
        risk_lbl, risk_col = classify_cholera_risk(rain_lbl, hum_lbl, temp_lbl)

        st.markdown(f"#### {region}", unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.markdown(f"""<div class="env-card" style="border-color:{rain_col};">
                <div><div class="label">Rainfall</div>
                <div class="value">{rain_val:.1f} mm/day</div></div>
                <span class="badge" style="background-color:{rain_col};">{rain_lbl}</span>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""<div class="env-card" style="border-color:{temp_col};">
                <div><div class="label">Temperature</div>
                <div class="value">{temp_val:.1f} °C</div></div>
                <span class="badge" style="background-color:{temp_col};">{temp_lbl}</span>
            </div>""", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"""<div class="env-card" style="border-color:{hum_col};">
                <div><div class="label">Humidity</div>
                <div class="value">{hum_val:.1f} %</div></div>
                <span class="badge" style="background-color:{hum_col};">{hum_lbl}</span>
            </div>""", unsafe_allow_html=True)
        with rc4:
            st.markdown(f"""<div class="risk-banner" style="background-color:{risk_col};font-size:15px;padding:12px;">
                {risk_lbl}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("""
        **Thresholds:** Rainfall — LOW < 60 mm/day | MEDIUM 60–149 mm/day | HIGH ≥ 150 mm/day  
        Temperature — LOW < 24 °C | MEDIUM 24–29 °C | HIGH ≥ 30 °C  
        Humidity — LOW < 60 % | MEDIUM 60–79 % | HIGH ≥ 80 %  
        Overall risk is HIGH if ≥ 2 variables are HIGH, MEDIUM if 1 is HIGH, LOW otherwise.
    """)
