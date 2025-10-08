import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="WSGT_BPVis_THE 0.0.1", page_icon="Pamo_Icon_White.png", layout="wide")


col1, col2 = st.columns(2)
with col2:
    logo_path = Path("WS_Logo.jpg")
    if logo_path.exists():
        st.image(str(logo_path), width=900)


st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## BPVis THE")
st.sidebar.write("Version 0.0.1")

st.title("Summer Overheating")

COLOR_OK = "#22c55e";
COLOR_BAD = "#ef4444";
COLOR_PURPLE = "#833fd1";
COLOR_RED = "#c02419"


# ------------------ Helpers ------------------
@st.cache_data(show_spinner=False)
def load_xlsx(bytes_):
    """Read all sheets into a dict of DataFrames."""
    x = pd.ExcelFile(io.BytesIO(bytes_))
    return {s: pd.read_excel(x, s) for s in x.sheet_names}


def _reconstruct_timestamp_from_doy_hour(df: pd.DataFrame) -> pd.Series:
    """If a sheet has columns 'doy' and 'hour' but no timestamp, reconstruct a datetime index.
    DIN 4108 simulations must start on a Monday -> use 2010-01-01 (Monday)."""
    base = pd.Timestamp("2010-01-01 00:00:00")  # Monday
    doy = pd.to_numeric(df["doy"], errors="coerce")
    hour = pd.to_numeric(df["hour"], errors="coerce")
    ts = base + pd.to_timedelta(doy - 1, unit="D") + pd.to_timedelta(hour, unit="H")
    return ts


def normalize_from_room_sheets(sheets: dict) -> pd.DataFrame:
    """
    Build a long dataframe from 'one-sheet-per-room' layout.

    Supports two input patterns per room sheet:
      A) Columns: [timestamp, t_op_C]  (case-insensitive variants tolerated)
      B) Columns: [doy, hour, t_op_C]  (reconstruct timestamp; Jan 1 is Monday)

    Optional 'Rooms' sheet maps room_id -> room_name, use_type (Residential / Non-Residential).
    """
    special = {"rooms", "project_data", "comfort_settings", "occupancy", "outdoor_temps"}
    rows = []
    for name, df in sheets.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if name.lower() in special:
            continue
        cols_lower = {c.lower(): c for c in df.columns}

        # try pattern A first: explicit timestamp
        col_ts = None
        for k in ("timestamp", "time", "datetime"):
            if k in cols_lower:
                col_ts = cols_lower[k];
                break

        # temperature column
        col_t = None
        for k in ("t_op_c", "t_op", "operative_temperature", "top_c", "operative temp", "operative", "top"):
            if k in cols_lower:
                col_t = cols_lower[k];
                break

        if col_t is None:
            # not a room sheet
            continue

        d = df.copy()

        if col_ts is not None:
            # Use provided timestamp
            d = d[[col_ts, col_t]].rename(columns={col_ts: "timestamp", col_t: "t_op_C"}).copy()
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        elif ("doy" in cols_lower) and ("hour" in cols_lower):
            # Reconstruct timestamp from doy/hour, force Monday start year
            d = d[[cols_lower["doy"], cols_lower["hour"], col_t]].rename(
                columns={cols_lower["doy"]: "doy", cols_lower["hour"]: "hour", col_t: "t_op_C"}
            ).copy()
            d["timestamp"] = _reconstruct_timestamp_from_doy_hour(d)
        else:
            # missing required timing columns
            continue

        d = d.dropna(subset=["timestamp"])
        d["t_op_C"] = pd.to_numeric(d["t_op_C"], errors="coerce")
        d = d.dropna(subset=["t_op_C"])
        d["room_id"] = str(name)
        rows.append(d[["timestamp", "t_op_C", "room_id"]])

    if not rows:
        raise ValueError("No valid room sheets found. Provide either [timestamp, t_op_C] or [doy, hour, t_op_C].")

    out = pd.concat(rows, ignore_index=True)

    # Attach metadata from Rooms sheet (explicit None checks)
    rooms_df = sheets.get("Rooms")
    if rooms_df is None:
        rooms_df = sheets.get("rooms")

    if isinstance(rooms_df, pd.DataFrame):
        r = rooms_df.copy()
        rcols = {c.lower(): c for c in r.columns}
        rid = rcols.get("room_id") or list(r.columns)[0]
        r = r.rename(columns={rid: "room_id"})
        out = out.merge(r, on="room_id", how="left")

    # Fallbacks
    if "room_name" not in out.columns:
        out["room_name"] = out["room_id"]
    if "use_type" not in out.columns:
        out["use_type"] = np.where(
            out["room_id"].str.lower().str.contains("bed|res|apt|flat|sleep"),
            "Residential",
            "Non-Residential",
        )
    return out


def season_mask(idx: pd.DatetimeIndex, start_md: str, end_md: str) -> pd.Series:
    """Return boolean mask for Month-Day window (inclusive) across any year."""

    def parse_md(s):
        parts = str(s).split("-")
        if len(parts) == 3: return int(parts[1]), int(parts[2])
        if len(parts) == 2: return int(parts[0]), int(parts[1])
        raise ValueError("Season dates must be MM-DD or YYYY-MM-DD")

    sm, sd = parse_md(start_md);
    em, ed = parse_md(end_md)
    flags = []
    for t in idx:
        md = (t.month, t.day)
        if (sm, sd) <= (em, ed):
            flags.append((md >= (sm, sd)) and (md <= (em, ed)))
        else:
            flags.append((md >= (sm, sd)) or (md <= (em, ed)))
    return pd.Series(flags, index=idx)


def occ_series(idx: pd.DatetimeIndex, use_type: str) -> pd.Series:
    """Two use types per DIN 4108-2:
       - Residential: 24/7
       - Non-Residential: Mon–Fri 07:00–18:00 (hours 7..17)
    """
    if use_type == "Residential":
        return pd.Series(True, index=idx)
    wk = idx.weekday;
    hr = idx.hour
    occ = (wk <= 4) & (hr >= 7) & (hr <= 18)
    return pd.Series(occ, index=idx)


def limits_for_zone_and_use(zone: str, use_type: str):
    """DIN 4108-2 mapping:
       Zone A: limit 25 °C
       Zone B: limit 26 °C
       Zone C: limit 27 °C
       Degree-Hours caps: Residential=1200, Non-Residential=500
       Season: May 1 – Sep 30
    """
    zone = (zone or "B").upper()
    if zone == "A":
        limit = 25.0
    elif zone == "B":
        limit = 26.0
    else:  # "C"
        limit = 27.0
    cap_dh = 1200.0 if use_type == "Residential" else 500.0
    season_start, season_end = "05-01", "09-30"
    return limit, cap_dh, season_start, season_end


def compute_4108(df_room: pd.DataFrame, limit_c: float, season_start: str, season_end: str, occ: pd.Series):
    """Return degree-hours and hours above limit for OCCUPIED hours in SEASON window."""
    s = df_room.set_index("timestamp")["t_op_C"].astype(float).sort_index()
    occ = occ.reindex(s.index, fill_value=False)
    sea = season_mask(s.index, season_start, season_end)
    sel = s[occ & sea]
    dh_series = np.maximum(sel - limit_c, 0.0)  # K·h per hour
    hours_above = int((sel > limit_c).sum())
    degree_hours = float(dh_series.sum())
    # monthly aggregation
    m = pd.DataFrame({"t": sel, "dh": dh_series})
    m["month"] = m.index.month_name()
    order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
    m["month"] = pd.Categorical(m["month"], categories=order, ordered=True)
    monthly = m.groupby("month", as_index=False).agg(
        hours_above=("t", lambda x: int((x > limit_c).sum())),
        dh=("dh", "sum")
    ).sort_values("month")
    return {"hours": hours_above, "dh": degree_hours, "monthly": monthly, "dh_series": dh_series}


# ------------------ Sidebar ------------------

if Path('sample_thermal_database.xlsx').exists():
    st.sidebar.markdown("### Download Template")
    st.sidebar.download_button('Download Excel Template', data=Path('sample_thermal_database.xlsx').read_bytes(),
                         file_name='sample_thermal_database.xlsx')

st.sidebar.markdown("---")

st.sidebar.markdown("### Upload Data")
up = st.sidebar.file_uploader(
    "Upload Excel File", type=["xlsx"],
    help="One sheet per room with columns: timestamp + t_op_C OR doy + hour + t_op_C"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Information")
with st.sidebar.expander("Project Data", expanded=False):
    proj_name = st.text_input("Project Name", "Example Building — Comfort")
    use_type_global = st.selectbox("Default Use Type (fallback)", ["Residential", "Non-Residential"], index=1)

with st.sidebar.expander("DIN 4108 Settings", expanded=False):
    cz = st.selectbox("Climate Zone (DIN 4108-2)", ["A", "B", "C"], index=1)
    # derive defaults from selected zone + global use type (override option removed)
    limit_C, cap_DH, season_start, season_end = limits_for_zone_and_use(cz, use_type_global)


st.sidebar.markdown("---")

if not up:
    st.write("### ← Please upload data on sidebar")
else:
    # Load and normalize
    try:
        sheets = load_xlsx(up.getvalue())
    except Exception as e:
        st.error(f"Could not read workbook: {e}")
        st.stop()

    try:
        df = normalize_from_room_sheets(sheets)
    except Exception as e:
        st.error(f"Input normalization error: {e}")
        st.stop()

    st.title(proj_name)

    # Room selector
    rooms = df["room_id"].unique().tolist()
    room_choice = st.selectbox("Room to visualize", rooms, index=0)
    dfr = df[df["room_id"] == room_choice].copy()

    # Determine use_type per room
    use_type_room = dfr["use_type"].dropna().astype(str).iloc[0] if "use_type" in dfr.columns else use_type_global
    if use_type_room not in ("Residential", "Non-Residential"):
        use_type_room = use_type_global

    # Recompute limits for this room + zone
    limit_r, cap_dh_r, season_start_r, season_end_r = limits_for_zone_and_use(cz, use_type_room)

    # Build hourly index and align
    idx = pd.date_range(dfr["timestamp"].min().floor("H"), dfr["timestamp"].max().ceil("H"), freq="H")
    s_room = dfr.set_index("timestamp")["t_op_C"].astype(float).reindex(idx).interpolate(limit=3).bfill().ffill()
    dfrA = pd.DataFrame({"timestamp": idx, "t_op_C": s_room.values})

    # Occupancy series
    occ = occ_series(idx, use_type_room)

    # Compute metrics
    out = compute_4108(dfrA, limit_r, season_start_r, season_end_r, occ)

    st.markdown("---")
    # KPIs
    st.metric("Room", str(room_choice))
    st.metric("DIN 4108 Climate Zone", cz)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Limit (°C)", f"{limit_r:.1f}")
    k2.metric("Hours Above Threshold", f"{out['hours']:}")
    k3.metric("Degree-Hours (°K.h)", f"{out['dh']:.1f}")
    status = "PASS" if out["dh"] <= cap_dh_r else "FAIL"
    margin = cap_dh_r - out["dh"]
    k4.metric(label=f"Compliance (cap {cap_dh_r:.0f} K·h)", value=status, delta=f"{margin:+.0f} K·h vs cap")

    # Heatmaps (no aggregation) with fixed temp color scale 15..35 °C
    st.markdown("---")
    st.subheader("Hourly Results")

    dfh = dfrA.copy()
    dfh["doy"] = dfh["timestamp"].dt.dayofyear
    dfh["hour"] = dfh["timestamp"].dt.hour

    hours = np.arange(0, 24)
    days = np.arange(1, 367)  # 1..366

    # 1) Temperature imshow (no aggregation) fixed z range
    grid_T = (
        dfh.pivot_table(index="hour", columns="doy", values="t_op_C", aggfunc="first")
        .reindex(index=hours, columns=days)
    )
    fig_t = px.imshow(
        grid_T,
        origin="lower",
        aspect="auto",
        color_continuous_scale="thermal",
        zmin=16, zmax=30,
        title="Operative Temperature (°C)",
        labels=dict(x="Day of Year", y="Hour", color="°C")
    )
    fig_t.update_layout(height=600)
    st.plotly_chart(fig_t, use_container_width=True)

    # 2) Degree-hours imshow (0 outside occupied+season)
    in_season = season_mask(idx, season_start_r, season_end_r)
    mask = occ & in_season
    dh_hourly = (np.maximum(s_room - limit_r, 0.0)).astype(float)
    dfh["dh"] = np.where(mask.values, dh_hourly, 0.0)

    grid_DH = (
        dfh.pivot_table(index="hour", columns="doy", values="dh", aggfunc="first")
        .reindex(index=hours, columns=days)
        .fillna(0.0)
    )
    fig_e = px.imshow(
        grid_DH,
        origin="lower",
        aspect="auto",
        color_continuous_scale="Reds",
        title="Hourly Degree-Hours Above Limit (°K.h)",
        labels=dict(x="Day of Year", y="Hour", color="K·h")
    )
    fig_e.update_layout(height=600)
    st.plotly_chart(fig_e, use_container_width=True)


    st.markdown("---")
    st.write("### Monthly Results")
    # Monthly bars
    a, b = st.columns(2)
    with a:
        fig_h = px.bar(out["monthly"], x="month", y="hours_above",
                       title="Monthly Hours Above Limit",
                       text_auto=True, height=600)
        fig_h.update_traces(marker_color=COLOR_RED, textfont_color="white")
        fig_h.update_layout(xaxis_title="", yaxis_title="hours", showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)
    with b:
        fig_dh = px.bar(out["monthly"], x="month", y="dh",
                        title="Monthly Degree-Hours Above Limit",
                        text_auto=".1f", height=600)
        fig_dh.update_traces(marker_color=COLOR_PURPLE, textfont_color="white")
        fig_dh.update_layout(xaxis_title="", yaxis_title="K·h", showlegend=False)
        st.plotly_chart(fig_dh, use_container_width=True)

    # ------------------ Temp distribution bars (occupied hours only) ------------------
    st.markdown("---")
    st.write("### Operative Temperature Distribution")
    temp_occ = s_room[occ]
    bins = [-np.inf, 20, 22, 24, 26, 28, 30, np.inf]
    labels = ["<20°C", "20–22°C", "22–24°C", "24–26°C", "26–28°C", "28–30°C", ">30°C"]
    cats = pd.cut(temp_occ, bins=bins, labels=labels, right=False, include_lowest=True)
    dist = cats.value_counts().reindex(labels, fill_value=0).reset_index()
    dist.columns = ["Temp Range", "Hours"]

    fig_dist = px.bar(
        dist, x="Temp Range", y="Hours",
        title="Hours per Operative Temperature Range",
        text_auto=True, height=600
    )
    fig_dist.update_traces(marker_color="#5A73A5", textfont_color="white")
    fig_dist.update_layout(xaxis_title="", yaxis_title="hours", showlegend=False)


    st.plotly_chart(fig_dist, use_container_width=True)

    # Save Project (writes settings back)

    st.markdown("---")
    with st.sidebar:
        if st.button("Save Project", use_container_width=True):
            wb = load_xlsx(up.getvalue())
            proj = pd.DataFrame({"Key": ["Project_Name", "Climate_Zone_4108"],
                                 "Value": [proj_name, cz]})
            cfg = pd.DataFrame(
                {"Key": ["Default_Use_Type", "Limit_C", "Cap_Degree_Hours", "Season_Start", "Season_End"],
                 "Value": [use_type_global, float(limit_C), float(cap_DH), str(season_start), str(season_end)]})
            wb["Project_Data"] = proj
            wb["Comfort_Settings"] = cfg
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                for n, dfn in wb.items():
                    if isinstance(dfn, pd.DataFrame):
                        dfn.to_excel(w, sheet_name=n, index=False)
            st.download_button("Download Updated Workbook", data=buf.getvalue(),
                               file_name=Path(up.name).stem + "_comfort_saved.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown("---")

        with st.sidebar:

            st.caption("*A product of*")
            st.image("WS_Logo.png", width=300)
            st.caption("Werner Sobek Green Technologies GmbH")
            st.caption("Fachgruppe Simulation")
            st.markdown("---")
            st.caption("*Coded by*")
            st.caption("Rodrigo Carvalho")
            st.caption("*Need help? Contact me under:*")
            st.caption("*email:* rodrigo.carvalho@wernersobek.com")
            st.caption("*Tel* +49.40.6963863-14")
            st.caption("*Mob* +49.171.964.7850")



















