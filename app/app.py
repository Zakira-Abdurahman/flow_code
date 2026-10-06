"""Addis Ababa ride-demand forecast demo (Deliverable E).

Run from the project root:  streamlit run app/app.py
Reads only the bundled tables in app/assets/ (written by notebooks/04_modeling_and_evaluation.ipynb).
"""
import datetime as dt
import json
from pathlib import Path

import altair as alt
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
TRIPS_PER_DRIVER = 1.3
FIRST_DAY, LAST_DAY = dt.date(2025, 11, 1), dt.date(2025, 11, 14)

st.set_page_config(page_title="Addis Ride Demand", page_icon=":material/local_taxi:", layout="wide")
st.html("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: 650;}
  [data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.75;}
  h1 {font-weight: 750; letter-spacing: -0.02em;}
</style>
""")


# ------------------------------------------------------------------ bundled data
@st.cache_resource
def load_model():
    """Load the exported model; fall back to LightGBM's text format if the joblib file cannot be read."""
    try:
        bundle = joblib.load(ASSETS / "final_model.joblib")
        return bundle["model"].predict, {k: v for k, v in bundle.items() if k != "model"}, "final_model.joblib"
    except Exception:
        meta = json.loads((ASSETS / "model_metadata.json").read_text())
        booster = lgb.Booster(model_file=str(ASSETS / "final_model_lightgbm.txt"))
        return booster.predict, meta, "final_model_lightgbm.txt"


@st.cache_data
def load_forecast():
    """Predict every zone-hour of 1–14 Nov from the bundled feature rows."""
    predict, meta, _ = load_model()
    x = pd.read_csv(ASSETS / "test_features.csv", parse_dates=["pickup_hour"])
    x["zone_code"] = x["zone"].map(meta["zone_codes"])
    x["zone_type"] = x["zone"].map(meta["zone_type"])
    x["zone_type_code"] = x["zone_type"].map(meta["zone_type_codes"])
    x["forecast"] = np.clip(predict(x[meta["features"]]), 0, None)
    x["date"] = x["pickup_hour"].dt.date
    x["hour"] = x["pickup_hour"].dt.hour
    x["zone_type_short"] = x["zone_type"].str.split(" \\(").str[0]
    fares = pd.read_csv(ASSETS / "zone_fares.csv", index_col="zone")["avg_fare_birr"]
    x["avg_fare"] = x["zone"].map(fares)
    x["drivers_needed"] = np.ceil(x["forecast"] / TRIPS_PER_DRIVER).astype(int)
    x["gross_fares"] = x["forecast"] * x["avg_fare"]
    return x


@st.cache_data
def load_events():
    """Cleaned, non-cancelled events (one row per event and zone)."""
    return pd.read_csv(ASSETS / "events.csv", parse_dates=["start_datetime", "end_datetime"])


@st.cache_data
def load_typical():
    """Mean trips by zone, weekday and hour over the last 8 weeks of history."""
    return pd.read_csv(ASSETS / "typical_profile.csv")


@st.cache_data
def load_recent():
    """Actual trips for the last 14 days of history."""
    return pd.read_csv(ASSETS / "recent_actuals.csv", parse_dates=["pickup_hour"])


@st.cache_data
def load_results():
    """Validation scores per model and fold, and permutation importance."""
    return pd.read_csv(ASSETS / "model_results.csv"), pd.read_csv(ASSETS / "feature_importance.csv")


def chart(c, height=None):
    """Render an Altair chart with the app theme (light or dark)."""
    st.altair_chart(c.properties(height=height) if height else c, width="stretch", theme="streamlit")


def kpi_row(items):
    """A row of bordered metric cards: (label, value, note)."""
    for col, (label, value, note) in zip(st.columns(len(items)), items):
        with col.container(border=True):
            st.metric(label, value)
            st.caption(note)


def describe_rain(day):
    """One line summarising the looked-up rain for a day."""
    wet = day[day["rain_mm"] > 0]
    if wet.empty:
        return "dry all day"
    peak = wet.loc[wet["rain_mm"].idxmax()]
    return (f"rain in {len(wet)} hour{'s' if len(wet) > 1 else ''} ({wet['rain_mm'].sum():.1f} mm total), "
            f"heaviest {peak['rain_mm']:.1f} mm at {int(peak['hour']):02d}:00")


def day_events(zone, day):
    """Events in this zone whose window [start − 2 h, end + 2 h) touches the day."""
    e = load_events()
    start, end = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
    near = e[(e["zone"] == zone) & (e["start_datetime"] - pd.Timedelta(hours=2) < end)
             & (e["end_datetime"] + pd.Timedelta(hours=2) > start)]
    return near.sort_values("start_datetime")


def describe_event(r):
    """One readable line for an event."""
    kind = r.event_type.replace("_", " ")
    where = f" at {r.venue}" if isinstance(r.venue, str) and r.venue != "none" else ""
    if r.event_type in ("public_holiday", "school_break"):
        return f"{r.event_name} ({kind}, citywide)"
    crowd = f", ~{r.expected_attendance:,.0f} expected" if r.expected_attendance > 0 else ""
    return f"{r.event_name}{where} {r.start_datetime:%H:%M}–{r.end_datetime:%H:%M} ({kind}{crowd})"


# ------------------------------------------------------------------ pages
def zone_forecast_page():
    fc = load_forecast()
    zones = sorted(fc["zone"].unique())

    st.title("Hourly demand forecast")
    st.caption("Pick a zone and a day. Weather and events are looked up for you from the bundled forecast and calendar.")
    c1, c2 = st.columns([1, 1])
    zone = c1.selectbox("Zone", zones, index=zones.index("Bole") if "Bole" in zones else 0)
    day = c2.date_input("Date", value=FIRST_DAY, format="DD/MM/YYYY")

    if not isinstance(day, dt.date) or not FIRST_DAY <= day <= LAST_DAY:
        st.warning(f"Forecasts are available for **1–14 November 2025** only. "
                   f"Please pick a date in that range (you chose {day:%d %b %Y}).", icon=":material/event_busy:")
        return

    view = fc[(fc["zone"] == zone) & (fc["date"] == day)].sort_values("hour").copy()
    typical = load_typical()
    typ = typical[(typical["zone"] == zone) & (typical["dow"] == pd.Timestamp(day).dayofweek)].set_index("hour")["typical_trips"]
    view["typical"] = view["hour"].map(typ)
    events = day_events(zone, day)
    peak = view.loc[view["forecast"].idxmax()]
    fare = view["avg_fare"].iloc[0]

    rain_line = describe_rain(view)
    temps = f"{view['temp_c'].min():.0f}–{view['temp_c'].max():.0f} °C"
    source = "weather forecast" if view["is_forecast"].mean() > 0.5 else "observed weather"
    event_line = "; ".join(describe_event(r) for r in events.itertuples()) or "no events in or near this zone"
    st.info(f"**Looked up for {zone}, {day:%A %d %B %Y}:** {source} {temps}, {rain_line} · {event_line}.",
            icon=":material/travel_explore:")

    kpi_row([
        ("Forecast trips", f"{view['forecast'].sum():,.0f}", f"vs {view['typical'].sum():,.0f} on a typical {day:%A}"),
        ("Peak hour", f"{int(peak['hour']):02d}:00", f"{peak['forecast']:.0f} trips · {int(peak['drivers_needed'])} drivers"),
        ("Driver-hours needed", f"{view['drivers_needed'].sum():,}", f"forecast ÷ {TRIPS_PER_DRIVER} trips per driver-hour"),
        ("Expected gross fares", f"{view['gross_fares'].sum():,.0f} ETB", f"at {fare:.0f} ETB, {zone}'s average fare"),
    ])

    with st.container(border=True):
        st.subheader(f"{zone}, {day:%a %d %b}: forecast vs typical {day:%A}")
        base = view.assign(time=pd.to_datetime(view["pickup_hour"]))
        lines = base.melt(id_vars=["time"], value_vars=["forecast", "typical"], var_name="series", value_name="trips")
        lines["series"] = lines["series"].map({"forecast": "Forecast", "typical": f"Typical {day:%A} (last 8 weeks)"})
        layers = []
        if not events.empty:
            local_ev = events[~events["event_type"].isin(["public_holiday", "school_break"])]
            if not local_ev.empty:
                day_start, day_end = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(hours=23)
                shade = pd.DataFrame({
                    "start": (local_ev["start_datetime"] - pd.Timedelta(hours=2)).clip(lower=day_start),
                    "end": (local_ev["end_datetime"] + pd.Timedelta(hours=2)).clip(upper=day_end),
                    "event": local_ev["event_name"]})
                layers.append(alt.Chart(shade).mark_rect(opacity=0.15, color="#eb6834").encode(
                    x="start:T", x2="end:T", tooltip=[alt.Tooltip("event:N", title="Event window (±2 h)")]))
        layers.append(alt.Chart(lines).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("time:T", title="Hour", axis=alt.Axis(format="%H:00")),
            y=alt.Y("trips:Q", title="Trips per hour"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
            strokeDash=alt.StrokeDash("series:N", legend=None),
            tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("time:T", title="Hour", format="%H:00"),
                     alt.Tooltip("trips:Q", title="Trips", format=".0f")]))
        chart(alt.layer(*layers), height=320)
        rain = base[["time", "rain_mm"]]
        chart(alt.Chart(rain).mark_bar(color="#3987e5", opacity=0.6).encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:00")),
            y=alt.Y("rain_mm:Q", title="Rain (mm)"),
            tooltip=[alt.Tooltip("time:T", title="Hour", format="%H:00"), alt.Tooltip("rain_mm:Q", title="Rain (mm)")]), height=90)
        st.caption("Shaded: event windows (2 h before start to 2 h after end), where the model expects extra demand. "
                   "Bottom: looked-up rain per hour.")

    with st.container(border=True):
        st.subheader("24-hour forecast")
        table = view[["hour", "forecast", "typical", "drivers_needed", "gross_fares", "temp_c", "rain_mm", "ev_in_window"]].copy()
        table["hour"] = table["hour"].map(lambda h: f"{h:02d}:00")
        table["forecast"] = table["forecast"].round().astype(int)
        table["ev_in_window"] = np.where(table["ev_in_window"] == 1, "yes", "")
        st.dataframe(table, hide_index=True, width="stretch", height=880, column_config={
            "hour": "Hour",
            "forecast": st.column_config.ProgressColumn("Forecast trips", format="%d", min_value=0,
                                                        max_value=float(max(view["forecast"].max(), 1))),
            "typical": st.column_config.NumberColumn("Typical", format="%.0f"),
            "drivers_needed": st.column_config.NumberColumn("Drivers needed", format="%d"),
            "gross_fares": st.column_config.NumberColumn("Expected fares (ETB)", format="%,.0f"),
            "temp_c": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
            "rain_mm": st.column_config.NumberColumn("Rain (mm)", format="%.1f"),
            "ev_in_window": "Event window",
        })
        st.download_button("Download this day (CSV)", table.to_csv(index=False).encode(),
                           f"forecast_{zone.lower().replace(' ', '_')}_{day:%Y%m%d}.csv", "text/csv", icon=":material/download:")


def city_overview_page():
    fc = load_forecast()
    st.title("City overview, 1–14 November")
    st.caption("All 12 zones · forecast trips, drivers and fares")
    by_zone = fc.groupby(["zone", "zone_type_short"], as_index=False).agg(
        trips=("forecast", "sum"), per_hour=("forecast", "mean"), fares=("gross_fares", "sum"))
    by_hour = fc.groupby("hour")["forecast"].mean()
    by_day = fc.groupby("date")["forecast"].sum()
    kpi_row([
        ("Forecast trips", f"{fc['forecast'].sum():,.0f}", "14 days · 12 zones"),
        ("Busiest zone", by_zone.sort_values("per_hour").iloc[-1]["zone"], f"{by_zone['per_hour'].max():.0f} trips per hour on average"),
        ("Peak hour", f"{by_hour.idxmax():02d}:00", f"{by_hour.max():.0f} trips per zone on average"),
        ("Expected gross fares", f"{fc['gross_fares'].sum() / 1e6:,.1f} M ETB", f"busiest day {by_day.idxmax():%a %d %b}"),
    ])
    with st.container(border=True):
        st.subheader("Citywide hourly forecast after the last two weeks of actual demand")
        recent = load_recent().groupby("pickup_hour", as_index=False)["trips"].sum().assign(series="Actual (18–31 Oct)")
        future = fc.groupby("pickup_hour", as_index=False)["forecast"].sum().rename(columns={"forecast": "trips"}).assign(series="Forecast (1–14 Nov)")
        both = pd.concat([recent, future])
        chart(alt.Chart(both).mark_line(strokeWidth=1.4).encode(
            x=alt.X("pickup_hour:T", title=None), y=alt.Y("trips:Q", title="Trips per hour, all zones"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("pickup_hour:T", title="Hour", format="%a %d %b %H:00"),
                     alt.Tooltip("trips:Q", title="Trips", format=",.0f")]).interactive(bind_y=False), height=320)
    left, right = st.columns([1.1, 1])
    with left.container(border=True):
        st.subheader("Zones ranked")
        chart(alt.Chart(by_zone).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
            x=alt.X("per_hour:Q", title="Average trips per hour"), y=alt.Y("zone:N", sort="-x", title=None),
            color=alt.Color("zone_type_short:N", title="Zone type", legend=alt.Legend(orient="bottom", columns=3)),
            tooltip=[alt.Tooltip("zone:N"), alt.Tooltip("per_hour:Q", title="Trips per hour", format=".1f"),
                     alt.Tooltip("fares:Q", title="Fares (ETB)", format=",.0f")]), height=360)
    with right.container(border=True):
        st.subheader("When each zone is busy")
        heat = fc.groupby(["zone", "hour"], as_index=False)["forecast"].mean()
        chart(alt.Chart(heat).mark_rect().encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("zone:N", title=None, sort=list(by_zone.sort_values("per_hour", ascending=False)["zone"])),
            color=alt.Color("forecast:Q", title="Trips per hour"),
            tooltip=[alt.Tooltip("zone:N"), alt.Tooltip("hour:O"), alt.Tooltip("forecast:Q", format=".1f")]), height=360)


def model_page():
    _, meta, source = load_model()
    results, importance = load_results()
    summary = (results.groupby("model").agg(rmse=("rmse", "mean"), sd=("rmse", "std"), mae=("mae", "mean"))
               .reset_index().sort_values("rmse"))
    final = summary[summary["model"] == meta["model_name"]].iloc[0]
    naive = summary[summary["model"].str.startswith("Baseline: seasonal")].iloc[0]
    st.title("How the model works")
    st.caption(f"{meta['model_name']} · validated on four 14-day windows (Sep–Oct 2025) before predicting 1–14 Nov")
    kpi_row([
        ("Typical error (MAE)", f"{final['mae']:.1f} trips", f"per zone-hour · about {final['mae'] / TRIPS_PER_DRIVER:.0f} drivers"),
        ("RMSE", f"{final['rmse']:.2f}", f"± {final['sd']:.2f} across 4 validation fortnights"),
        ("Better than baseline", f"{100 * (1 - final['rmse'] / naive['rmse']):.0f}%", f"RMSE vs seasonal naive ({naive['rmse']:.2f})"),
        ("Training data", f"{meta['n_train_rows']:,}", "cleaned zone-hours, 1 Jan – 31 Oct 2025"),
    ])
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Models compared")
        summary["kind"] = np.where(summary["model"] == meta["model_name"], "Final model",
                                   np.where(summary["model"].str.startswith("Baseline"), "Baseline", "Other model"))
        summary["low"], summary["high"] = summary["rmse"] - summary["sd"], summary["rmse"] + summary["sd"]
        base = alt.Chart(summary).encode(y=alt.Y("model:N", sort=list(summary["model"]), title=None, axis=alt.Axis(labelLimit=260)))
        bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
            x=alt.X("rmse:Q", title="Validation RMSE (lower is better)"),
            color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["Final model", "Other model", "Baseline"])),
            tooltip=[alt.Tooltip("model:N"), alt.Tooltip("rmse:Q", format=".2f"), alt.Tooltip("mae:Q", format=".2f")])
        chart(bars + base.mark_rule().encode(x="low:Q", x2="high:Q"), height=330)
    with right.container(border=True):
        st.subheader("What drives the forecast")
        top = importance.head(15).assign(label=lambda t: t["feature"].str.replace("_", " "))
        chart(alt.Chart(top).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
            x=alt.X("mean:Q", title="Error increase when shuffled (trips)"), y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color("group:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("mean:Q", title="Importance", format=".2f")]), height=330)
    with st.expander("Key assumptions"):
        st.markdown("""
- **Clocks:** trips and events are in Addis time; both weather formats are UTC and were shifted +3 h.
- **Horizon:** forecasts are made 14 days ahead, so demand lags are at least 14 days old.
- **Weather:** November uses the weather forecast in the data; validation used observed weather as a stand-in.
- **Events:** the calendar is complete for November; unlisted events are the main cause of large misses.
- **Fares:** expected gross fares = forecast trips × the zone's average fare over Jan–Oct.
""")
    with st.expander("Model details"):
        st.json({k: meta[k] for k in ("params", "trained_on", "validation", "versions")})
        st.caption(f"Loaded from `{source}`.")


pg = st.navigation([st.Page(zone_forecast_page, title="Zone forecast", icon=":material/schedule:", default=True),
                    st.Page(city_overview_page, title="City overview", icon=":material/insights:"),
                    st.Page(model_page, title="Model", icon=":material/model_training:")])
with st.sidebar:
    st.markdown("## :material/local_taxi: Addis Ride Demand")
    st.caption("Hourly trip forecast for 12 zones, 1–14 November 2025 · team flowcode")
pg.run()
