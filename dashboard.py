import os
import json
import requests
import datetime as dt
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, no_update
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv

# ============================== Config & persistence ==============================
FILE = "tickers.json" # fix this
POLYGON_API_KEY = getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY. Set it in your environment (Heroku Config Vars).")

def get_default_data():
    # Seed lists (unchanged)
    shorts_df = pd.DataFrame({
        "Ticker": ["AAPL", "TSLA", "AMZN"],
        "Short %": [1.2, 3.4, 2.1],
    })
    ratings_dict = {
        "AAPL": pd.DataFrame({"Rating": ["Buy", "Hold", "Sell"], "Count": [10, 4, 2]}),
        "TSLA": pd.DataFrame({"Rating": ["Buy", "Hold", "Sell"], "Count": [6, 8, 3]}),
        "AMZN": pd.DataFrame({"Rating": ["Buy", "Hold", "Sell"], "Count": [12, 3, 1]}),
    }
    return {
        "shorts": shorts_df.to_dict("records"),
        "ratings": {k: v.to_dict("records") for k, v in ratings_dict.items()},
        # Cache of plotted points: {ticker: [{"date": "YYYY-MM-DD", "short_volume": int}, ...]}
        "shorts_history": {},
        # Diagnostics: {ticker: [{"date": "YYYY-MM-DD", "status": "OK|ERROR", "message": str, "short_volume": int|None}], ...}
        "shorts_diag": {},
    }

def load_data():
    if os.path.exists(FILE):
        try:
            with open(FILE, "r") as f:
                data = json.load(f)
                data.setdefault("shorts_history", {})
                data.setdefault("shorts_diag", {})
                return data
        except Exception:
            pass
    return get_default_data()

def save_data(data):
    with open(FILE, "w") as f:
        json.dump(data, f)

# ============================== Date helpers ==============================
def last_business_day(d: dt.date | None = None) -> dt.date:
    d = d or dt.date.today()
    # Snap weekends back to Friday
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= dt.timedelta(days=1)
    return d

def weekly_schedule_last_3_months(end_day: dt.date) -> list[dt.date]:
    """
    Build ~weekly dates (~12–14 points) for ~3 months, ending at last business day.
    Uses Fridays for consistency, plus today's last business day if different.
    """
    end_bday = last_business_day(end_day)
    start = end_bday - dt.timedelta(days=92)
    fridays = pd.date_range(start, end_bday, freq="W-FRI").date.tolist()
    if not fridays or fridays[-1] != end_bday:
        fridays.append(end_bday)
    if len(fridays) > 14:
        fridays = fridays[-14:]
    return sorted(set(fridays))

# ============================== Polygon fetch (concurrent, tight timeouts) ==============================
def _fetch_short_for_date(session: requests.Session, ticker: str, d: dt.date):
    url = "https://api.polygon.io/stocks/v1/short-volume"
    params = {
        "ticker": ticker.upper(),
        "date": d.isoformat(),
        # send both to be resilient to API param name variants
        "apiKey": POLYGON_API_KEY,
        "apikey": POLYGON_API_KEY,
    }
    try:
        r = session.get(url, params=params, timeout=(2, 4))  # (connect, read)
        if r.status_code != 200:
            txt = (r.text or "")[:160]
            return {"date": d, "ticker": ticker.upper(), "short_volume": None,
                    "status": "ERROR", "message": f"{r.status_code}: {txt}"}
        j = r.json()
        if not j.get("results"):
            return {"date": d, "ticker": ticker.upper(), "short_volume": None,
                    "status": "ERROR", "message": "no results"}
        sv = sum(item.get("short_volume", 0) for item in j["results"])
        return {"date": d, "ticker": ticker.upper(), "short_volume": int(sv),
                "status": "OK", "message": ""}
    except Exception as e:
        return {"date": d, "ticker": ticker.upper(), "short_volume": None,
                "status": "ERROR", "message": str(e)}

def fetch_short_volume_for_dates(ticker: str, dates: list[dt.date]) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    with requests.Session() as sess:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_fetch_short_for_date, sess, ticker, d): d for d in dates}
            for fut in as_completed(futs):
                rows.append(fut.result())
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "OK"]
    df_ok = pd.DataFrame(columns=["date", "ticker", "short_volume"]) if ok.empty else \
        ok.sort_values("date").drop_duplicates(subset="date", keep="last")[["date", "ticker", "short_volume"]]
    # Diagnostics payload (dicts with iso dates)
    diag = []
    for _, r in df.sort_values("date").iterrows():
        diag.append({
            "date": pd.to_datetime(r["date"]).date().isoformat(),
            "status": r["status"],
            "message": str(r["message"]) if pd.notna(r["message"]) else "",
            "short_volume": (int(r["short_volume"]) if pd.notna(r["short_volume"]) else None),
        })
    return df_ok, diag

# ============================== Cache manager ==============================
def ensure_cached_short_volume(ticker: str, data: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    Ensure we have a cached weekly series for last ~3 months.
    Only fetches missing dates; writes cache/diag to disk. Returns (df, diag_list, data).
    """
    ticker = ticker.upper()
    cache = data.get("shorts_history", {})
    diag_store = data.get("shorts_diag", {})

    cached_series = cache.get(ticker, [])
    cached_df = pd.DataFrame(cached_series)
    if not cached_df.empty:
        cached_df["date"] = pd.to_datetime(cached_df["date"]).dt.date

    target_dates = weekly_schedule_last_3_months(dt.date.today())
    cached_dates = set(cached_df["date"].tolist()) if not cached_df.empty else set()
    missing = [d for d in target_dates if d not in cached_dates]

    diag_for_this_run: list[dict] = []
    if missing:
        fetched_df, diag = fetch_short_volume_for_dates(ticker, missing)
        diag_for_this_run = diag
        if not fetched_df.empty:
            fetched_df["date"] = pd.to_datetime(fetched_df["date"]).dt.date
            combined = fetched_df if cached_df.empty else pd.concat([cached_df, fetched_df], ignore_index=True)
        else:
            combined = cached_df
        # Update diagnostics store (keep last 30 rows per ticker)
        old_diag = diag_store.get(ticker, [])
        diag_store[ticker] = (old_diag + diag)[-30:]
    else:
        combined = cached_df
        # No new fetch, but still expose last known diagnostics
        diag_for_this_run = diag_store.get(ticker, [])

    if combined.empty:
        data["shorts_history"] = cache
        data["shorts_diag"] = diag_store
        save_data(data)
        return combined, diag_for_this_run, data

    newest = max(combined["date"])
    cutoff = newest - dt.timedelta(days=92)
    combined = combined[(combined["date"] >= cutoff) & (combined["date"] <= newest)]
    combined = combined.sort_values("date").drop_duplicates(subset="date", keep="last")

    cache[ticker] = [{"date": d.isoformat(), "short_volume": int(v)}
                     for d, v in zip(combined["date"], combined["short_volume"])]
    data["shorts_history"] = cache
    data["shorts_diag"] = diag_store
    save_data(data)
    return combined, diag_for_this_run, data

# ============================== Dash app ==============================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Finance Dashboard"

server = app.server

app.layout = html.Div(
    [
        html.H1("Finance Dashboard", style={"textAlign": "center"}),
        # Single source of truth for user state
        dcc.Store(id="app-store", data=load_data()),
        dcc.Tabs(id="tabs", value="shorts", children=[
            dcc.Tab(label="Short Interest", value="shorts"),
            dcc.Tab(label="Analyst Ratings", value="ratings"),
        ]),
        html.Div(id="tabs-content"),
        # Light periodic check to append new weekly point when a week rolls over
        dcc.Interval(id="weekly-refresh", interval=60*60*1000, n_intervals=0),  # hourly
    ],
    style={"maxWidth": "1100px", "margin": "0 auto"},
)

# ============================== Tab renderer ==============================
@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value"), Input("app-store", "data")],
)
def render_content(tab, data):
    shorts_df = pd.DataFrame(data.get("shorts", []))
    ratings_dict = {k: pd.DataFrame(v) for k, v in (data.get("ratings") or {}).items()}

    if tab == "shorts":
        tickers = list(shorts_df["Ticker"]) if not shorts_df.empty and "Ticker" in shorts_df.columns else []
        default_ticker = tickers[0] if tickers else None

        return html.Div(
            [
                # Left: tickers + add/delete
                html.Div(
                    [
                        html.H3("Tickers", style={"marginBottom": "8px"}),
                        dcc.RadioItems(
                            id="shorts-radio",
                            options=[{"label": t, "value": t} for t in tickers],
                            value=default_ticker,
                            labelStyle={"display": "block", "margin": "6px 0"},
                            inputStyle={"marginRight": "8px"},
                        ),
                        html.Div(style={"height": "12px"}),
                        dcc.Input(id="ticker-input", type="text", placeholder="Enter ticker (e.g., RKLB)"),
                        html.Button("Add", id="add-btn", n_clicks=0, style={"marginLeft": "8px"}),
                        html.Button("Delete", id="del-btn", n_clicks=0, style={"marginLeft": "8px"}),
                    ],
                    style={"width": "220px", "padding": "10px 12px",
                           "borderRight": "1px solid #e5e7eb", "flexShrink": 0},
                ),
                # Right: chart + diagnostics
                html.Div(
                    [
                        dcc.Graph(id="shorts-line", style={"height": "520px"}),
                        html.Div(id="shorts-diag", style={"fontSize": "12px", "color": "#555", "marginTop": "6px"}),
                    ],
                    style={"flex": 1, "padding": "10px 16px"},
                ),
            ],
            style={"display": "flex", "alignItems": "stretch"},
        )

    # Ratings tab
    options = [{"label": t, "value": t} for t in ratings_dict.keys()]
    default_value = options[0]["value"] if options else None
    return html.Div([
        dcc.Dropdown(id="ratings-dropdown", options=options, value=default_value, placeholder="Select a ticker"),
        dcc.Graph(id="ratings-graph", style={"marginTop": "10px"}),
    ])

# ============================== Add/Delete tickers (persisted) ==============================
@app.callback(
    [Output("app-store", "data"), Output("ticker-input", "value")],
    [Input("add-btn", "n_clicks"), Input("del-btn", "n_clicks")],
    [State("ticker-input", "value"), State("app-store", "data")],
    prevent_initial_call=True,
)
def modify_tickers(add_clicks, del_clicks, ticker, data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if not ticker:
        return dash.no_update, dash.no_update

    ticker = ticker.strip().upper()
    if not ticker:
        return dash.no_update, dash.no_update

    shorts = pd.DataFrame(data.get("shorts", []))
    ratings = {k: pd.DataFrame(v) for k, v in (data.get("ratings") or {}).items()}
    shorts_history = data.get("shorts_history", {})
    shorts_diag = data.get("shorts_diag", {})

    if button_id == "add-btn":
        if shorts.empty or ticker not in (shorts["Ticker"].values if "Ticker" in shorts.columns else []):
            new_row = pd.DataFrame({"Ticker": [ticker], "Short %": [0.0]})
            shorts = pd.concat([shorts, new_row], ignore_index=True)
        if ticker not in ratings:
            ratings[ticker] = pd.DataFrame({"Rating": ["Buy", "Hold", "Sell"], "Count": [0, 0, 0]})
        shorts_history.setdefault(ticker, [])
        shorts_diag.setdefault(ticker, [])

    elif button_id == "del-btn":
        if not shorts.empty and "Ticker" in shorts.columns:
            shorts = shorts[shorts["Ticker"] != ticker]
        ratings.pop(ticker, None)
        shorts_history.pop(ticker, None)
        shorts_diag.pop(ticker, None)

    new_data = {
        "shorts": shorts.to_dict("records"),
        "ratings": {k: v.to_dict("records") for k, v in ratings.items()},
        "shorts_history": shorts_history,
        "shorts_diag": shorts_diag,
    }
    save_data(new_data)
    return new_data, ""

# ============================== Keep the vertical list in sync ==============================
@app.callback(
    [Output("shorts-radio", "options"), Output("shorts-radio", "value")],
    Input("app-store", "data"),
    State("shorts-radio", "value"),
    prevent_initial_call=True,
)
def sync_vertical_list(data, current_value):
    shorts_df = pd.DataFrame(data.get("shorts", []))
    tickers = list(shorts_df["Ticker"]) if not shorts_df.empty and "Ticker" in shorts_df.columns else []
    opts = [{"label": t, "value": t} for t in tickers]
    val = current_value if current_value in tickers else (tickers[0] if tickers else None)
    return opts, val

# ============================== Weekly refresh (duplicate writer) ==============================
# Requires Dash >= 2.9 for allow_duplicate=True
@app.callback(
    Output("app-store", "data", allow_duplicate=True),
    [Input("weekly-refresh", "n_intervals"), Input("shorts-radio", "value")],
    State("app-store", "data"),
    prevent_initial_call=True,
)
def maybe_refresh_weekly(_, selected_ticker, data):
    if not selected_ticker:
        return no_update
    # Only attempt to append if the expected end moved
    hist = (data.get("shorts_history", {}) or {}).get(selected_ticker.upper(), [])
    last_cached = None
    if hist:
        try:
            last_cached = max(pd.to_datetime([h["date"] for h in hist]).date)
        except Exception:
            last_cached = None
    expected_end = last_business_day(dt.date.today())
    if last_cached and expected_end <= last_cached:
        return no_update
    _, _, updated = ensure_cached_short_volume(selected_ticker, data)
    return updated if updated != data else no_update

# ============================== Line chart + diagnostics ==============================
@app.callback(
    [Output("shorts-line", "figure"), Output("shorts-diag", "children")],
    [Input("shorts-radio", "value"), Input("app-store", "data")],
)
def draw_short_volume_line(selected_ticker, data):
    if not selected_ticker:
        return px.line(title="Select a ticker"), ""

    df, diag, _ = ensure_cached_short_volume(selected_ticker, data)
    if df.empty:
        fig = px.line(title=f"No short-volume data for {selected_ticker}. "
                            f"Check Polygon key/plan or try another ticker.")
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
        return fig, _render_diag(diag)

    fig = px.line(
        df, x="date", y="short_volume", markers=True,
        title=f"Short Volume – {selected_ticker} ({df['date'].min()} to {df['date'].max()})",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Short Volume",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(tickformat="%b %d")
    return fig, _render_diag(diag)

def _render_diag(diag_rows: list[dict]):
    if not diag_rows:
        return html.Div("Diagnostics: (none yet)")
    # Show most recent first, keep last 20
    rows = sorted(diag_rows, key=lambda r: r.get("date", ""), reverse=True)[:20]
    return html.Details([
        html.Summary("Diagnostics (last fetches)"),
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Date"), html.Th("Status"), html.Th("Short Vol"), html.Th("Message")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(r.get("date", "")),
                        html.Td(r.get("status", "")),
                        html.Td("" if r.get("short_volume") is None else f'{r.get("short_volume"):,}'),
                        html.Td(r.get("message", "")),
                    ]) for r in rows
                ])
            ],
            style={"borderCollapse": "collapse", "width": "100%"},
        )
    ], open=False)

# ============================== Ratings (unchanged) ==============================
@app.callback(
    Output("ratings-graph", "figure"),
    [Input("ratings-dropdown", "value"), Input("app-store", "data")],
)
def update_ratings_chart(ticker, data):
    if not ticker:
        return px.bar(title="No ticker selected")
    series = (data.get("ratings") or {}).get(ticker, [])
    df = pd.DataFrame(series)
    if df.empty:
        return px.bar(title=f"No ratings for {ticker}")
    return px.pie(df, names="Rating", values="Count", title=f"Analyst Ratings for {ticker}")

# ============================== Main ==============================
if __name__ == "__main__":
    app.run(debug=True)
