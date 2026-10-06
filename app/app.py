"""Addis Ababa ride-demand intelligence — forecast demo (Deliverable E) and interactive explorer (stretch goal).

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
HISTORY_START = dt.date(2025, 1, 1)
FONT = "Inter"
TYPE_ORDER = ["Business / office", "Mixed hub", "Market", "Residential", "Leisure / nightlife"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
RAIN_BINS, RAIN_LABELS = [-0.01, 0, 1, 5, np.inf], ["Dry", "Light ≤1 mm", "Moderate 1–5 mm", "Heavy >5 mm"]
RAW_ROWS = {"trips": 85_460, "weather": 7_538, "events": 165, "test": 4_032}

st.set_page_config(page_title="Addis Ride Demand", page_icon=":material/local_taxi:", layout="wide")


# ------------------------------------------------------------------ theme
def palette():
    """Colours for the active light or dark theme."""
    dark = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
    if dark:
        return dict(dark=True, bg="#1a1a19", card="#232321", card2="#2a2a28", border="#383835", ink="#f5f5f2",
                    muted="#c3c2b7", faint="#8a8984", grid="#33332f", accent="#3987e5", accent_soft="rgba(57,135,229,.16)",
                    series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
                    seq=["#173b66", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"])
    return dict(dark=False, bg="#fcfcfb", card="#ffffff", card2="#f6f5f2", border="#e6e5e1", ink="#0b0b0b",
                muted="#52514e", faint="#8a8984", grid="#eeede9", accent="#2a78d6", accent_soft="rgba(42,120,214,.09)",
                series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
                seq=["#e4effc", "#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


P = palette()
st.html(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea, select {{ font-family: '{FONT}', sans-serif !important; }}
  .stApp {{ background: {P['bg']}; }}
  .block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1320px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stDecoration"], footer, .stDeployButton, [data-testid="stAppDeployButton"] {{ display: none !important; }}
  h1, h2, h3 {{ font-family: '{FONT}', sans-serif !important; letter-spacing: -0.02em; color: {P['ink']}; }}
  .hero {{ border: 1px solid {P['border']}; border-radius: 20px; padding: 34px 36px 30px; margin-bottom: 18px;
           background: {P['card']}; }}
  .eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: {P['accent']}; }}
  .hero h1 {{ font-size: 38px; font-weight: 800; margin: 8px 0 8px; line-height: 1.1; }}
  .hero p {{ font-size: 15.5px; color: {P['muted']}; max-width: 820px; margin: 0; line-height: 1.55; }}
  .kpi-grid {{ display: grid; gap: 12px; margin: 6px 0 18px; }}
  .kpi {{ background: {P['card']}; border: 1px solid {P['border']}; border-radius: 16px; padding: 16px 18px; }}
  .kpi-label {{ font-size: 12.5px; color: {P['muted']}; font-weight: 500; }}
  .kpi-value {{ font-size: 28px; font-weight: 750; color: {P['ink']}; margin: 4px 0 2px; letter-spacing: -0.02em; }}
  .kpi-note {{ font-size: 12px; color: {P['faint']}; line-height: 1.4; }}
  .section {{ margin: 26px 0 10px; }}
  .section h2 {{ font-size: 21px; font-weight: 700; margin: 0; }}
  .section p {{ font-size: 13.5px; color: {P['muted']}; margin: 4px 0 0; }}
  .card-grid {{ display: grid; gap: 12px; margin-bottom: 8px; }}
  .card {{ background: {P['card']}; border: 1px solid {P['border']}; border-radius: 16px; padding: 18px 20px; }}
  .card h4 {{ font-size: 15px; font-weight: 650; margin: 0 0 6px; color: {P['ink']}; }}
  .card .big {{ font-size: 26px; font-weight: 750; color: {P['accent']}; letter-spacing: -0.02em; margin: 2px 0 6px; }}
  .card p {{ font-size: 13px; color: {P['muted']}; margin: 0; line-height: 1.5; }}
  .card ul {{ font-size: 13px; color: {P['muted']}; margin: 6px 0 0 16px; padding: 0; line-height: 1.6; }}
  .tag {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
          background: {P['accent_soft']}; color: {P['accent']}; margin-right: 6px; }}
  .lookup {{ border-left: 3px solid {P['accent']}; background: {P['card2']}; border-radius: 10px; padding: 12px 16px;
             font-size: 13.5px; color: {P['ink']}; margin: 4px 0 16px; line-height: 1.55; }}
  .note {{ font-size: 12.5px; color: {P['faint']}; margin-top: -4px; }}
  [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px !important; background: {P['card']}; }}
  @media (max-width: 900px) {{ .kpi-grid, .card-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }}
                               .hero h1 {{ font-size: 28px; }} }}
</style>
""")


# ------------------------------------------------------------------ html helpers
def hero(eyebrow, title, text):
    """Page banner."""
    st.html(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p></div>')


def kpis(items):
    """A responsive row of metric cards: (label, value, note)."""
    cells = "".join(f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div>'
                    f'<div class="kpi-note">{n}</div></div>' for l, v, n in items)
    st.html(f'<div class="kpi-grid" style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">{cells}</div>')


def section(title, text=""):
    """Section heading with an optional one-line description."""
    st.html(f'<div class="section"><h2>{title}</h2>{f"<p>{text}</p>" if text else ""}</div>')


def cards(items, columns=3):
    """Grid of description cards: dicts with title, big?, text?, bullets?, tags?."""
    html = ""
    for c in items:
        tags = "".join(f'<span class="tag">{t}</span>' for t in c.get("tags", []))
        big = f'<div class="big">{c["big"]}</div>' if c.get("big") else ""
        text = f'<p>{c["text"]}</p>' if c.get("text") else ""
        bullets = "<ul>" + "".join(f"<li>{b}</li>" for b in c["bullets"]) + "</ul>" if c.get("bullets") else ""
        html += f'<div class="card">{tags}<h4>{c["title"]}</h4>{big}{text}{bullets}</div>'
    st.html(f'<div class="card-grid" style="grid-template-columns:repeat({columns},minmax(0,1fr))">{html}</div>')


def chart(c, height=300):
    """Render an Altair chart styled for the active theme."""
    c = (c.properties(height=height, background="transparent")
          .configure_view(stroke=None)
          .configure_axis(labelColor=P["muted"], titleColor=P["muted"], gridColor=P["grid"], domainColor=P["border"],
                          tickColor=P["border"], labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11,
                          titleFontWeight=500)
          .configure_legend(labelColor=P["muted"], titleColor=P["muted"], labelFont=FONT, titleFont=FONT,
                            labelFontSize=11, titleFontSize=11, orient="top", symbolType="circle")
          .configure_range(category=P["series"], ordinal=P["seq"], ramp=P["seq"], heatmap=P["seq"])
          .configure_text(font=FONT, color=P["muted"]))
    st.altair_chart(c, width="stretch", theme=None)


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


def short_type(s):
    """'Business / office (weekday commute)' -> 'Business / office'."""
    return s.str.split(" \\(").str[0]


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
    x["zone_type_short"] = short_type(x["zone_type"])
    fares = pd.read_csv(ASSETS / "zone_fares.csv", index_col="zone")["avg_fare_birr"]
    x["avg_fare"] = x["zone"].map(fares)
    x["drivers_needed"] = np.ceil(x["forecast"] / TRIPS_PER_DRIVER).astype(int)
    x["gross_fares"] = x["forecast"] * x["avg_fare"]
    return x


@st.cache_data
def load_weather():
    """Cleaned hourly weather (Addis time)."""
    return pd.read_csv(ASSETS / "weather.csv", parse_dates=["timestamp"])


@st.cache_data
def load_combined():
    """History (actual, Jan–Oct) and forecast (1–14 Nov) in one table, with rain and calendar columns."""
    _, meta, _ = load_model()
    hist = pd.read_csv(ASSETS / "history.csv", parse_dates=["pickup_hour"]).assign(source="Actual")
    w = load_weather()[["timestamp", "rain_mm"]].rename(columns={"timestamp": "pickup_hour"})
    hist = hist.merge(w, on="pickup_hour", how="left")
    fc = load_forecast()[["zone", "pickup_hour", "forecast", "rain_mm"]].rename(columns={"forecast": "trips"}).assign(source="Forecast")
    d = pd.concat([hist, fc], ignore_index=True)
    d["zone_type"] = short_type(d["zone"].map(meta["zone_type"]))
    d["date"] = d["pickup_hour"].dt.normalize()
    d["hour"] = d["pickup_hour"].dt.hour
    d["dow"] = d["pickup_hour"].dt.dayofweek
    d["rain_class"] = pd.cut(d["rain_mm"].fillna(0), RAIN_BINS, labels=RAIN_LABELS)
    return d


@st.cache_data
def load_table(name, **kw):
    """Read one bundled CSV."""
    return pd.read_csv(ASSETS / name, **kw)


# ------------------------------------------------------------------ pages
def overview_page():
    d = load_combined()
    hist, fc = d[d["source"] == "Actual"], d[d["source"] == "Forecast"]
    _, meta, _ = load_model()
    results = load_table("model_results.csv")
    final = results[results["model"] == meta["model_name"]]
    weekly = hist[hist["zone"] != "Ayat"].groupby(hist["date"].dt.to_period("W-SUN")).agg(t=("trips", "sum"), n=("trips", "size"))
    weekly = weekly[weekly["n"] >= 0.9 * 11 * 168]
    weekly["t"] = weekly["t"] / weekly["n"] * 11 * 168
    growth = weekly["t"].iloc[-4:].mean() / weekly["t"].iloc[:4].mean() - 1

    hero("Addis Ababa · ride-demand intelligence", "Every zone, every hour — forecast two weeks ahead",
         "Ten months of trips, hourly weather and a city events calendar, cleaned onto one clock and turned into an "
         "hourly forecast for 12 zones for 1–14 November 2025 — with the drivers and fares it implies.")
    kpis([
        ("Trips analysed", f"{hist['trips'].sum() / 1e6:.2f} M", f"{len(hist):,} clean zone-hours, Jan–Oct 2025"),
        ("Demand growth", f"{growth:+.0%}", "weekly trips, first vs last 4 weeks (11 zones)"),
        ("Forecast, 1–14 Nov", f"{fc['trips'].sum():,.0f}", f"trips · {fc['trips'].sum() / TRIPS_PER_DRIVER:,.0f} driver-hours"),
        ("Typical forecast error", f"±{final['mae'].mean():.1f}", "trips per zone-hour (MAE, 4 validation fortnights)"),
        ("Zones", "12", "5 demand types found by clustering"),
    ])

    section("The data", "Three raw exports plus the test grid, cleaned in code onto one Addis-time clock.")
    events = load_table("events.csv")
    weather = load_weather()
    cards([
        {"tags": ["Target"], "title": "Trips", "big": f"{len(hist):,}",
         "text": f"zone-hours kept of {RAW_ROWS['trips']:,} raw rows (97%), 1 Jan – 31 Oct 2025.",
         "bullets": ["55 zone spellings → 12 zones", "3 timestamp formats, 2 clocks → Addis time",
                     "×8 trip spikes fixed, −1 placeholders removed", "362 conflicting duplicate hours set aside"]},
        {"tags": ["Weather"], "title": "Hourly weather", "big": f"{len(weather):,}",
         "text": f"continuous hours from {RAW_ROWS['weather']:,} raw rows, one citywide station.",
         "bullets": ["UTC timestamps shifted +3 h", "353 °F readings converted", "193 missing hours filled and flagged",
                     "November = forecast rows"]},
        {"tags": ["Events"], "title": "Events calendar", "big": f"{events['event_id'].nunique():,}",
         "text": f"confirmed events from {RAW_ROWS['events']} raw rows; one row per affected zone.",
         "bullets": ["20 type spellings → 8 types", "citywide and two-zone events expanded",
                     "6 duplicates removed, 6 cancelled set aside", "attendance text → numbers"]},
        {"tags": ["Forecast"], "title": "Test grid", "big": f"{RAW_ROWS['test']:,}",
         "text": "zone-hours to predict: 12 zones × 336 hours, 1–14 November 2025.",
         "bullets": ["same cleaning rules as train", "same feature pipeline, train statistics only",
                     "lags ≥ 14 days old", "7 scheduled events in the period"]},
    ], columns=4)

    section("What the data says", "Headline findings from the analysis report (B), each measured against comparable normal hours.")
    cards([
        {"title": "Rain sends people to rides", "big": "+26%", "text": "demand in rainy hours (> 1 mm) vs the same zone, month, weekday and hour when dry — except Merkato, where it falls to 0.78×."},
        {"title": "The final whistle matters most", "big": "2.2×", "text": "demand in the stadium zone in the 2 hours after a football match ends, vs 1.6× before kick-off and 1.3× during."},
        {"title": "Weekends split the city", "big": "0.47× ↔ 1.34×", "text": "Business zones (Piassa, Arat Kilo, Kazanchis) halve at weekends; Bole, the nightlife zone, is 34% busier."},
        {"title": "Holidays depend on the day", "big": "73–85%", "text": "of a normal day on weekday holidays (Good Friday lowest); weekend holidays stay near normal."},
        {"title": "Payday lifts demand", "big": "+5.9%", "text": "on the 26th–2nd of each month after removing trend and weekday effects (p < 0.001)."},
        {"title": "Steady growth", "big": f"{growth:+.0%}", "text": "January to October in weekly trips — about +517 trips every week, so November sits above the yearly average."},
    ], columns=3)

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Weekly trips — history and forecast**")
        steady = d[d["zone"] != "Ayat"]
        wk = steady.groupby(steady["date"].dt.to_period("W-SUN")).agg(
            trips=("trips", "sum"), hours=("trips", "size"), fc=("source", lambda s: (s == "Forecast").mean())).reset_index()
        wk = wk[wk["hours"] >= 0.5 * 11 * 168]
        wk["trips"] = wk["trips"] / wk["hours"] * 11 * 168
        wk["week"] = wk["date"].dt.start_time
        wk["source"] = np.where(wk["fc"] > 0.5, "Forecast", "Actual")
        base = alt.Chart(wk).encode(x=alt.X("week:T", title=None),
                                    y=alt.Y("trips:Q", title="Trips per week (11 zones)", axis=alt.Axis(format="~s"),
                                            scale=alt.Scale(zero=True)))
        chart(base.mark_line(strokeWidth=2, color=P["faint"])
              + base.mark_circle(size=55, opacity=1).encode(
                  color=alt.Color("source:N", title=None, scale=alt.Scale(domain=["Actual", "Forecast"], range=P["series"][:2])),
                  tooltip=[alt.Tooltip("week:T", title="Week of", format="%d %b"), "source:N",
                           alt.Tooltip("trips:Q", format=",.0f")]), 280)
        st.html('<div class="note">The 11 zones open all year; weeks with missing hours (e.g. the 12–13 May outage) are scaled to a full week.</div>')
    with right.container(border=True):
        st.markdown("**Weekday hourly profile by zone type**")
        prof = hist[hist["dow"] < 5].groupby(["zone_type", "hour"])["trips"].mean().reset_index()
        chart(alt.Chart(prof).mark_line(strokeWidth=2.2).encode(
            x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y("trips:Q", title="Mean trips per zone-hour"),
            color=alt.Color("zone_type:N", title=None, sort=TYPE_ORDER),
            tooltip=["zone_type:N", "hour:Q", alt.Tooltip("trips:Q", format=".1f")]), 280)
        st.html('<div class="note">Business zones peak at 08:00 and 18:00; the market at midday; nightlife late evening.</div>')


def zone_forecast_page():
    fc = load_forecast()
    zones = sorted(fc["zone"].unique())
    hero("Hourly forecast", "Pick a zone and a day",
         "Weather and events are looked up for you from the bundled forecast and calendar. You get the 24-hour forecast, "
         "the peak hour, the drivers needed each hour and the expected gross fares.")
    c1, c2, _ = st.columns([1, 1, 1.4])
    zone = c1.selectbox("Zone", zones, index=zones.index("Kazanchis") if "Kazanchis" in zones else 0)
    day = c2.date_input("Date", value=dt.date(2025, 11, 10), format="DD/MM/YYYY")
    if not isinstance(day, dt.date) or not FIRST_DAY <= day <= LAST_DAY:
        st.warning(f"Forecasts are available for **1–14 November 2025** only. "
                   f"Please pick a date in that range (you chose {day:%d %b %Y}).", icon=":material/event_busy:")
        return

    view = fc[(fc["zone"] == zone) & (fc["date"] == day)].sort_values("hour").copy()
    typical = load_table("typical_profile.csv")
    typ = typical[(typical["zone"] == zone) & (typical["dow"] == pd.Timestamp(day).dayofweek)].set_index("hour")["typical_trips"]
    view["typical"] = view["hour"].map(typ)
    events = load_table("events.csv", parse_dates=["start_datetime", "end_datetime"])
    start, end = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
    events = events[(events["zone"] == zone) & (events["start_datetime"] - pd.Timedelta(hours=2) < end)
                    & (events["end_datetime"] + pd.Timedelta(hours=2) > start)].sort_values("start_datetime")
    peak = view.loc[view["forecast"].idxmax()]

    wet = view[view["rain_mm"] > 0]
    rain = ("dry all day" if wet.empty else
            f"rain in {len(wet)} hour{'s' if len(wet) > 1 else ''}, heaviest {wet['rain_mm'].max():.1f} mm at "
            f"{int(wet.loc[wet['rain_mm'].idxmax(), 'hour']):02d}:00")
    source = "weather forecast" if view["is_forecast"].mean() > 0.5 else "observed weather"

    def describe(r):
        kind = r.event_type.replace("_", " ")
        if r.event_type in ("public_holiday", "school_break"):
            return f"{r.event_name} ({kind}, citywide)"
        venue = f" at {r.venue}" if isinstance(r.venue, str) and r.venue != "none" else ""
        crowd = f", ~{r.expected_attendance:,.0f} expected" if r.expected_attendance > 0 else ""
        return f"{r.event_name}{venue} {r.start_datetime:%H:%M}–{r.end_datetime:%H:%M} ({kind}{crowd})"

    ev_line = "; ".join(describe(r) for r in events.itertuples()) or "no events in or near this zone"
    st.html(f'<div class="lookup"><b>Looked up for {zone}, {day:%A %d %B %Y}</b> — {source} '
            f'{view["temp_c"].min():.0f}–{view["temp_c"].max():.0f} °C, {rain} · {ev_line}.</div>')

    kpis([
        ("Forecast trips", f"{view['forecast'].sum():,.0f}", f"vs {view['typical'].sum():,.0f} on a typical {day:%A}"),
        ("Peak hour", f"{int(peak['hour']):02d}:00", f"{peak['forecast']:.0f} trips · {int(peak['drivers_needed'])} drivers"),
        ("Driver-hours needed", f"{view['drivers_needed'].sum():,}", f"forecast ÷ {TRIPS_PER_DRIVER} trips per driver-hour"),
        ("Expected gross fares", f"{view['gross_fares'].sum():,.0f} ETB", f"at {view['avg_fare'].iloc[0]:.0f} ETB, the zone's average fare"),
    ])

    with st.container(border=True):
        st.markdown(f"**{zone}, {day:%a %d %b} — forecast vs a typical {day:%A}**")
        base = view.assign(time=pd.to_datetime(view["pickup_hour"]))
        lines = base.melt(id_vars=["time"], value_vars=["forecast", "typical"], var_name="series", value_name="trips")
        lines["series"] = lines["series"].map({"forecast": "Forecast", "typical": f"Typical {day:%A} (last 8 weeks)"})
        layers = []
        local = events[~events["event_type"].isin(["public_holiday", "school_break"])]
        if not local.empty:
            shade = pd.DataFrame({"start": (local["start_datetime"] - pd.Timedelta(hours=2)).clip(lower=start),
                                  "end": (local["end_datetime"] + pd.Timedelta(hours=2)).clip(upper=start + pd.Timedelta(hours=23)),
                                  "event": local["event_name"]})
            layers.append(alt.Chart(shade).mark_rect(opacity=0.14, color=P["series"][1]).encode(
                x="start:T", x2="end:T", tooltip=[alt.Tooltip("event:N", title="Event window (±2 h)")]))
        layers.append(alt.Chart(lines).mark_line(point=True, strokeWidth=2.4).encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:00", tickCount=12, labelOverlap=True)), y=alt.Y("trips:Q", title="Trips per hour"),
            color=alt.Color("series:N", title=None), strokeDash=alt.StrokeDash("series:N", legend=None),
            tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("time:T", title="Hour", format="%H:00"),
                     alt.Tooltip("trips:Q", title="Trips", format=".0f")]))
        chart(alt.layer(*layers), 320)
        st.html('<div class="note">Shaded: event windows (2 h before start to 2 h after end), where the model expects extra demand.</div>')

    left, right = st.columns([1.5, 1])
    with left.container(border=True):
        st.markdown("**24-hour forecast**")
        table = view[["hour", "forecast", "typical", "drivers_needed", "gross_fares", "temp_c", "rain_mm", "ev_in_window"]].copy()
        table["hour"] = table["hour"].map(lambda h: f"{h:02d}:00")
        table["forecast"] = table["forecast"].round().astype(int)
        table["ev_in_window"] = np.where(table["ev_in_window"] == 1, "yes", "")
        st.dataframe(table, hide_index=True, width="stretch", height=878, column_config={
            "hour": "Hour",
            "forecast": st.column_config.ProgressColumn("Forecast trips", format="%d", min_value=0,
                                                        max_value=float(max(view["forecast"].max(), 1))),
            "typical": st.column_config.NumberColumn("Typical", format="%.0f"),
            "drivers_needed": st.column_config.NumberColumn("Drivers", format="%d"),
            "gross_fares": st.column_config.NumberColumn("Fares (ETB)", format="%,.0f"),
            "temp_c": st.column_config.NumberColumn("Temp °C", format="%.1f"),
            "rain_mm": st.column_config.NumberColumn("Rain mm", format="%.1f"),
            "ev_in_window": "Event"})
        st.download_button("Download this day (CSV)", table.to_csv(index=False).encode(),
                           f"forecast_{zone.lower().replace(' ', '_')}_{day:%Y%m%d}.csv", "text/csv", icon=":material/download:")
    with right.container(border=True):
        st.markdown("**Drivers needed by hour**")
        chart(alt.Chart(view).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=P["accent"]).encode(
            x=alt.X("hour:O", title="Hour"), y=alt.Y("drivers_needed:Q", title="Drivers"),
            tooltip=["hour:O", "drivers_needed:Q", alt.Tooltip("forecast:Q", format=".0f")]), 250)
        st.markdown("**Rain looked up (mm per hour)**")
        chart(alt.Chart(view).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=P["series"][2]).encode(
            x=alt.X("hour:O", title="Hour"), y=alt.Y("rain_mm:Q", title="Rain (mm)", scale=alt.Scale(domainMin=0)),
            tooltip=["hour:O", "rain_mm:Q"]), 160)


def explorer_page():
    d = load_combined()
    zones = sorted(d["zone"].unique())
    hero("Explorer", "Slice ten months of demand and the forecast",
         "Filter by zone, dates and hours; every chart, statistic and observation below updates. History (actual trips, "
         "1 Jan – 31 Oct) and forecast (1–14 Nov) are shown side by side.")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.6, 1.2, 1.1, 0.8])
        picked = f1.multiselect("Zones", zones, default=zones, placeholder="All zones") or zones
        dates = f2.date_input("Date range", value=(dt.date(2025, 9, 1), LAST_DAY), min_value=HISTORY_START,
                              max_value=LAST_DAY, format="DD/MM/YYYY")
        hours = f3.slider("Hours of day", 0, 23, (0, 23))
        show = f4.radio("Data", ["Both", "Actual", "Forecast"], horizontal=False)
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info("Pick a start and an end date.")
        return
    sel = d[d["zone"].isin(picked) & d["date"].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
            & d["hour"].between(*hours)]
    if show != "Both":
        sel = sel[sel["source"] == show]
    if sel.empty:
        st.warning("No data for this selection — widen the dates or hours.")
        return

    by_zone = sel.groupby("zone")["trips"].mean().sort_values(ascending=False)
    by_hour = sel.groupby("hour")["trips"].mean()
    wk = sel.groupby(sel["dow"] >= 5)["trips"].mean()
    wk_ratio = wk.get(True, np.nan) / wk.get(False, np.nan)
    days = sel.groupby("date")["trips"].sum()
    kpis([
        ("Trips in selection", f"{sel['trips'].sum():,.0f}", f"{len(sel):,} zone-hours · {sel['date'].nunique()} days"),
        ("Mean per zone-hour", f"{sel['trips'].mean():.1f}", f"median {sel['trips'].median():.0f} · p90 {sel['trips'].quantile(.9):.0f}"),
        ("Busiest zone", by_zone.index[0], f"{by_zone.iloc[0]:.1f} trips per hour"),
        ("Peak hour", f"{by_hour.idxmax():02d}:00", f"{by_hour.max():.1f} trips per zone"),
        ("Weekend ÷ weekday", "—" if np.isnan(wk_ratio) else f"{wk_ratio:.2f}", f"busiest day {days.idxmax():%a %d %b}"),
    ])

    with st.container(border=True):
        st.markdown("**Daily trips over time**")
        daily = sel.groupby(["date", "source"])["trips"].sum().reset_index()
        chart(alt.Chart(daily).mark_line(strokeWidth=1.8).encode(
            x=alt.X("date:T", title=None), y=alt.Y("trips:Q", title="Trips per day", axis=alt.Axis(format="~s")),
            color=alt.Color("source:N", title=None, scale=alt.Scale(domain=["Actual", "Forecast"], range=P["series"][:2])),
            tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), "source:N", alt.Tooltip("trips:Q", format=",.0f")]
        ).interactive(bind_y=False), 260)

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**When demand happens — weekday × hour**")
        heat = sel.groupby(["dow", "hour"])["trips"].mean().reset_index()
        heat["day"] = heat["dow"].map(dict(enumerate(DOW)))
        chart(alt.Chart(heat).mark_rect(cornerRadius=2).encode(
            x=alt.X("hour:O", title="Hour of day"), y=alt.Y("day:N", sort=DOW, title=None),
            color=alt.Color("trips:Q", title="Mean trips", scale=alt.Scale(range=P["seq"])),
            tooltip=["day:N", "hour:O", alt.Tooltip("trips:Q", format=".1f")]), 260)
    with right.container(border=True):
        st.markdown("**Zones ranked**")
        zr = sel.groupby(["zone", "zone_type"])["trips"].mean().reset_index()
        chart(alt.Chart(zr).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("trips:Q", title="Mean trips per zone-hour"), y=alt.Y("zone:N", sort="-x", title=None),
            color=alt.Color("zone_type:N", title=None, sort=TYPE_ORDER),
            tooltip=["zone:N", "zone_type:N", alt.Tooltip("trips:Q", format=".1f")]), 260)

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Rain and demand** (actual trips, same hours of day)")
        act = sel[sel["source"] == "Actual"]
        if act.empty:
            st.info("Rain effect needs actual history — include dates before 1 November.")
        else:
            base = act[act["rain_class"] == "Dry"].groupby(["zone", "hour"])["trips"].mean().rename("dry")
            rr = act.join(base, on=["zone", "hour"]).dropna(subset=["dry"])
            rc = rr.groupby("rain_class", observed=True).agg(trips=("trips", "sum"), dry=("dry", "sum"), hours=("pickup_hour", "nunique")).reset_index()
            rc["ratio"] = rc["trips"] / rc["dry"]
            chart(alt.Chart(rc).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=P["series"][2]).encode(
                x=alt.X("rain_class:N", sort=RAIN_LABELS, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("ratio:Q", title="Demand ÷ dry-hour level", scale=alt.Scale(domainMin=0)),
                tooltip=["rain_class:N", alt.Tooltip("ratio:Q", format=".2f"), alt.Tooltip("hours:Q", title="Hours")]), 240)
    with right.container(border=True):
        st.markdown("**Spread of hourly trips by zone type**")
        chart(alt.Chart(sel).mark_boxplot(extent=1.5, size=26, median={"color": P["ink"]}).encode(
            x=alt.X("zone_type:N", title=None, sort=TYPE_ORDER, axis=alt.Axis(labelAngle=0, labelLimit=120)),
            y=alt.Y("trips:Q", title="Trips per zone-hour"),
            color=alt.Color("zone_type:N", legend=None, sort=TYPE_ORDER)), 240)

    section("Descriptive statistics", "Per zone, for the current selection.")
    stats = sel.groupby("zone").agg(zone_type=("zone_type", "first"), zone_hours=("trips", "size"), total=("trips", "sum"),
                                     mean=("trips", "mean"), median=("trips", "median"),
                                     p90=("trips", lambda s: s.quantile(.9)), max=("trips", "max"))
    stats["share"] = 100 * stats["total"] / stats["total"].sum()
    wkz = sel.groupby(["zone", sel["dow"] >= 5])["trips"].mean().unstack()
    stats["weekend_ratio"] = (wkz.get(True) / wkz.get(False)) if True in wkz and False in wkz else np.nan
    stats = stats.sort_values("total", ascending=False).reset_index()
    st.dataframe(stats, hide_index=True, width="stretch", column_config={
        "zone": "Zone", "zone_type": "Type", "zone_hours": st.column_config.NumberColumn("Zone-hours", format="%,d"),
        "total": st.column_config.NumberColumn("Total trips", format="%,.0f"),
        "mean": st.column_config.NumberColumn("Mean", format="%.1f"), "median": st.column_config.NumberColumn("Median", format="%.0f"),
        "p90": st.column_config.NumberColumn("P90", format="%.0f"), "max": st.column_config.NumberColumn("Max", format="%.0f"),
        "share": st.column_config.ProgressColumn("Share of trips", format="%.1f%%", min_value=0, max_value=float(stats["share"].max())),
        "weekend_ratio": st.column_config.NumberColumn("Weekend ÷ weekday", format="%.2f")})

    obs = [f"<b>{by_zone.index[0]}</b> is the busiest zone in this selection ({by_zone.iloc[0]:.1f} trips per hour), "
           f"<b>{by_zone.index[-1]}</b> the quietest ({by_zone.iloc[-1]:.1f}).",
           f"Demand peaks at <b>{by_hour.idxmax():02d}:00</b> and is lowest at <b>{by_hour.idxmin():02d}:00</b> "
           f"({by_hour.max() / max(by_hour.min(), 0.1):.0f}× difference)."]
    if not np.isnan(wk_ratio):
        obs.append(f"Weekends run at <b>{wk_ratio:.2f}×</b> weekdays across the selected zones.")
    if sel["source"].nunique() == 2:
        a = sel[sel["source"] == "Actual"]["trips"].mean()
        f = sel[sel["source"] == "Forecast"]["trips"].mean()
        obs.append(f"The forecast averages <b>{f:.1f}</b> trips per zone-hour vs <b>{a:.1f}</b> in the selected history ({f / a - 1:+.0%}).")
    cards([{"title": "Observations", "bullets": obs}], columns=1)


def data_page():
    hero("Data", "What went in, and what was fixed",
         "Descriptions of each table, every cleaning step with the rows it touched, and the dictionary of every column "
         "in the modelling tables.")
    tab1, tab2, tab3, tab4 = st.tabs(["Datasets", "Cleaning log", "Data dictionary", "Zone types"])
    with tab1:
        hist = load_combined().query("source == 'Actual'")
        weather = load_weather()
        events = load_table("events.csv")
        cards([
            {"title": "ride_demand_train.csv", "big": f"{RAW_ROWS['trips']:,} → {len(hist):,}",
             "text": "Hourly trips per zone, 1 Jan – 31 Oct 2025. Columns: record_id, zone, pickup_hour, trips, avg_fare_birr, avg_wait_min, active_drivers."},
            {"title": "weather_hourly.csv", "big": f"{RAW_ROWS['weather']:,} → {len(weather):,}",
             "text": "One citywide station, 31 Dec 2024 – 14 Nov 2025. Columns: timestamp, temp_c, rain_mm, humidity_pct, wind_kmh, data_type (observed / forecast)."},
            {"title": "events_calendar.csv", "big": f"{RAW_ROWS['events']} → {events['event_id'].nunique()} events",
             "text": "Matches, concerts, conferences, exhibitions, road closures, runs, holidays, school breaks; with zone, start, end, attendance and status."},
            {"title": "ride_demand_test.csv", "big": f"{RAW_ROWS['test']:,}",
             "text": "The zone-hours to forecast, 1–14 Nov 2025. Columns: row_id, zone, pickup_hour (no target)."},
        ], columns=2)
        left, right = st.columns(2)
        with left.container(border=True):
            st.markdown("**Hourly trips — distribution (history)**")
            chart(alt.Chart(hist).mark_bar(color=P["accent"]).encode(
                x=alt.X("trips:Q", bin=alt.Bin(maxbins=50), title="Trips per zone-hour"),
                y=alt.Y("count():Q", title="Zone-hours")), 230)
        with right.container(border=True):
            st.markdown("**Events by type**")
            et = events.drop_duplicates("event_id")["event_type"].str.replace("_", " ").value_counts().reset_index()
            et.columns = ["type", "events"]
            chart(alt.Chart(et).mark_bar(cornerRadiusEnd=4, color=P["series"][1]).encode(
                x=alt.X("events:Q", title="Events"), y=alt.Y("type:N", sort="-x", title=None),
                tooltip=["type:N", "events:Q"]), 230)
    with tab2:
        log = load_table("cleaning_log.csv")
        files = ["All"] + sorted(log["file"].unique())
        which = st.segmented_control("File", files, default="All")
        view = log if which in (None, "All") else log[log["file"] == which]
        st.dataframe(view.drop(columns=[c for c in ["#"] if c in view]), hide_index=True, width="stretch", height=560,
                     column_config={"pct_rows": st.column_config.NumberColumn("% rows", format="%.2f"),
                                    "rows_affected": st.column_config.NumberColumn("Rows", format="%,d")})
    with tab3:
        dic = load_table("data_dictionary_master.csv")
        st.dataframe(dic, hide_index=True, width="stretch", height=620)
    with tab4:
        zt = load_table("zone_types.csv")
        zt["zone_type"] = short_type(zt["zone_type"])
        hist = load_combined().query("source == 'Actual'")
        summary = hist.groupby("zone").agg(mean=("trips", "mean")).join(
            hist.groupby(["zone", hist["dow"] >= 5])["trips"].mean().unstack().pipe(lambda t: (t[True] / t[False]).rename("weekend_ratio")))
        st.dataframe(zt.join(summary, on="zone").sort_values(["zone_type", "zone"]), hide_index=True, width="stretch",
                     column_config={"mean": st.column_config.NumberColumn("Mean trips per hour", format="%.1f"),
                                    "weekend_ratio": st.column_config.NumberColumn("Weekend ÷ weekday", format="%.2f")})
        st.html('<div class="note">Zones were grouped by the shape of their weekday hourly profile (k-means, k = 5 chosen by silhouette score).</div>')


def model_page():
    _, meta, source = load_model()
    results, importance = load_table("model_results.csv"), load_table("feature_importance.csv")
    summary = (results.groupby("model").agg(rmse=("rmse", "mean"), sd=("rmse", "std"), mae=("mae", "mean"))
               .reset_index().sort_values("rmse"))
    final = summary[summary["model"] == meta["model_name"]].iloc[0]
    naive = summary[summary["model"].str.startswith("Baseline: seasonal")].iloc[0]
    hero("Model", "How the forecast is made and how good it is",
         f"{meta['model_name']}, chosen on four rolling 14-day validation windows (Sep–Oct 2025), then retrained on all "
         "data from 1 January to 31 October.")
    kpis([
        ("Typical error (MAE)", f"{final['mae']:.1f} trips", f"per zone-hour · about {final['mae'] / TRIPS_PER_DRIVER:.0f} drivers"),
        ("RMSE", f"{final['rmse']:.2f}", f"± {final['sd']:.2f} across 4 validation fortnights"),
        ("Better than baseline", f"{100 * (1 - final['rmse'] / naive['rmse']):.0f}%", f"RMSE vs seasonal naive ({naive['rmse']:.2f})"),
        ("Features", f"{len(meta['features'])}", "calendar, zone, trend & lags, weather, events"),
    ])
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Models compared** — validation RMSE, lower is better")
        summary["kind"] = np.where(summary["model"] == meta["model_name"], "Final model",
                                   np.where(summary["model"].str.startswith("Baseline"), "Baseline", "Other model"))
        summary["low"], summary["high"] = summary["rmse"] - summary["sd"], summary["rmse"] + summary["sd"]
        base = alt.Chart(summary).encode(y=alt.Y("model:N", sort=list(summary["model"]), title=None, axis=alt.Axis(labelLimit=320)))
        chart(base.mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("rmse:Q", title="RMSE (trips per zone-hour)"),
            color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Final model", "Other model", "Baseline"],
                                                                  range=[P["accent"], P["seq"][2], P["faint"]])),
            tooltip=["model:N", alt.Tooltip("rmse:Q", format=".2f"), alt.Tooltip("mae:Q", format=".2f")])
              + base.mark_rule(color=P["muted"]).encode(x="low:Q", x2="high:Q"), 330)
    with right.container(border=True):
        st.markdown("**What drives the forecast** — permutation importance")
        top = importance.head(15).assign(label=lambda t: t["feature"].str.replace("_", " "))
        chart(alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("mean:Q", title="Error increase when shuffled (trips)"), y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color("group:N", title=None, scale=alt.Scale(domain=["Calendar / zone / trend", "Weather", "Event"],
                                                                   range=[P["faint"], P["accent"], P["series"][1]])),
            tooltip=["feature:N", alt.Tooltip("mean:Q", format=".2f")]), 330)
    cards([
        {"title": "1 · Clean", "text": "55 zone spellings, 3 timestamp formats and 2 clocks, ×8 trip spikes, °F temperatures, placeholder codes and duplicates fixed across trips, weather and events."},
        {"title": "2 · Join", "text": "Every zone-hour gets its weather hour and any event within 2 hours of its start or end. Trips is the left table; the row count never changes."},
        {"title": "3 · Features", "text": f"{len(meta['features'])} features known 14 days ahead: calendar, zone, trend and 14–28-day lags, weather, events."},
        {"title": "4 · Model", "text": "LightGBM tuned with 30 random trials on an earlier window; validated on four rolling fortnights; retrained on all Jan–Oct data."},
    ], columns=4)
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


pg = st.navigation([
    st.Page(overview_page, title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(zone_forecast_page, title="Zone forecast", icon=":material/schedule:", url_path="forecast"),
    st.Page(explorer_page, title="Explorer", icon=":material/query_stats:", url_path="explorer"),
    st.Page(data_page, title="Data", icon=":material/database:", url_path="data"),
    st.Page(model_page, title="Model", icon=":material/model_training:", url_path="model"),
], position="top")
pg.run()
