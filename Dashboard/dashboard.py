import os
import json
import requests
import datetime as dt
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, no_update
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv() # Load local .env
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
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= dt.timedelta(days=1)
    return d

def prev_business_day(d: dt.date | None = None) -> dt.date:
    """Always return the most recent completed trading day (T-1)."""
    d = (d or dt.date.today()) - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d

def weekly_schedule_last_3_months(end_day: dt.date) -> list[dt.date]:
    """
    ~weekly series for ~3 months, ending at the previous business day (T-1).
    Uses Fridays for consistency, plus T-1 if that’s not a Friday.
    """
    end_bday = prev_business_day(end_day)   # <-- was last_business_day
    start = end_bday - dt.timedelta(days=92)
    fridays = pd.date_range(start, end_bday, freq="W-FRI").date.tolist()
    if not fridays or fridays[-1] != end_bday:
        fridays.append(end_bday)
    if len(fridays) > 14:
        fridays = fridays[-14:]
    return sorted(set(fridays))

# -------- Analyst from Polygon News (primary) + parser --------
ACTION_WORDS = r"(upgrades?|downgrades?|initiates|maintains|reiterates|assumes|resumes)"
RATING_WORDS = r"(Strong Buy|Outperform|Overweight|Equal[- ]?Weight|Market Perform|Peer Perform|Buy|Neutral|Hold|Reduce|Underweight|Sell|Negative|Positive|Sector Perform|Sector Weight)"
_PT_NUM = r"\$?\s*(\d+(?:\.\d+)?)"

def _parse_analyst_note(text: str) -> dict:
    import re
    s = re.sub(r"\s+", " ", text or "").strip()
    # Firm + action
    m_firm = re.search(rf"^(?P<firm>.+?)\s+(?P<action>{ACTION_WORDS})\b", s, flags=re.I)
    firm = (m_firm.group("firm").strip() if m_firm else "").rstrip(":,.- ")
    action = (m_firm.group("action").capitalize() if m_firm else "")
    # Rating (from → to) in either order
    m_to_from = re.search(rf"\bto\s+(?P<to>{RATING_WORDS})\s+(?:from|vs\.?)\s+(?P<from>{RATING_WORDS})", s, re.I)
    m_from_to = re.search(rf"\bfrom\s+(?P<from>{RATING_WORDS})\s+to\s+(?P<to>{RATING_WORDS})", s, re.I)
    rating = ""
    if m_to_from or m_from_to:
        g = (m_to_from or m_from_to).groupdict()
        rating = f"{g['from']} → {g['to']}"

    # Price target changes
    pa, pt = "—", "—"
    m_pt_to_from = re.search(rf"price target.*?\bto\s+{_PT_NUM}\s+(?:from|vs\.|previously)\s+{_PT_NUM}", s, re.I)
    m_pt_from_to = re.search(rf"price target.*?\bfrom\s+{_PT_NUM}\s+to\s+{_PT_NUM}", s, re.I)
    m_pt_sets    = re.search(rf"price target.*?\b(to|at|set at|sets at)\s+{_PT_NUM}", s, re.I)

    def _fmt(oldv, newv):
        try:
            oldf, newf = float(oldv), float(newv)
            return ("Raises" if newf > oldf else ("Lowers" if newf < oldf else "Reiterates")), f"{oldf:g} → {newf:g}"
        except Exception:
            return "—", f"{oldv} → {newv}"

    if m_pt_to_from:
        newv, oldv = m_pt_to_from.groups()[0], m_pt_to_from.groups()[2]
        pa, pt = _fmt(oldv, newv)
    elif m_pt_from_to:
        oldv, newv = m_pt_from_to.groups()[0], m_pt_from_to.groups()[2]
        pa, pt = _fmt(oldv, newv)
    elif m_pt_sets:
        newv = m_pt_sets.groups()[-1]
        try:
            pa, pt = "Sets", f"{float(newv):g}"
        except Exception:
            pa, pt = "Sets", f"{newv}"

    return {"analyst": firm, "rating_action": action, "rating": rating, "price_action": pa, "price_target": pt}

def _fetch_analyst_polygon_news(ticker: str, limit: int = 30) -> list[dict]:
    cutoff = prev_business_day(dt.date.today())
    start = cutoff - dt.timedelta(days=6)  # small buffer window
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker.upper(),
        "published_utc.gte": start.isoformat(),
        "published_utc.lte": (cutoff + dt.timedelta(days=1)).isoformat(),
        "order": "desc",
        "limit": limit,
        "apiKey": POLYGON_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=(3, 8))
        r.raise_for_status()
        results = (r.json() or {}).get("results", []) or []
    except Exception:
        return []

    rows = []
    for it in results:
        title = it.get("title") or ""
        desc  = it.get("description") or ""
        txt   = f"{title}. {desc}"
        parsed = _parse_analyst_note(txt)
        if parsed["rating_action"] or ("price target" in txt.lower()):
            d = pd.to_datetime(it.get("published_utc")).date() if it.get("published_utc") else None
            if not d or d > cutoff:
                continue
            parsed["date"] = d.isoformat()
            # If firm wasn’t captured, try title prefix (e.g., "B of A Securities:")
            if not parsed["analyst"]:
                head = title.split(":")[0][:60]
                if any(w in head.lower() for w in ["securities","research","capital","markets","partners"]):
                    parsed["analyst"] = head
            rows.append(parsed)
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows

# ---------- Analyst helpers (Yahoo Finance Upgrade/Downgrade feed) ----------
def _yahoo_ud_url(ticker: str) -> str:
    # Unofficial but stable JSON module Yahoo exposes
    return f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=upgradeDowngradeHistory"

def _safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def fetch_recent_analyst_actions(ticker: str, limit: int = 3) -> list[dict]:
    # 1) Polygon News -> parsed (includes price-target deltas when present)
    pn = _fetch_analyst_polygon_news(ticker, limit=30)
    if pn:
        return pn[:limit]

    # 2) Fallback: Yahoo JSON (your previous code path)
    url = _yahoo_ud_url(ticker.upper())
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(3, 6))
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        return [{"date": "", "analyst": "", "rating_action": "ERROR", "rating": "", "price_action": "",
                 "price_target": f"Failed to fetch: {e}"}]

    history = _safe_get(j, "quoteSummary", "result", default=[]) or []
    history = _safe_get(history[0] if history else {}, "upgradeDowngradeHistory", "history", default=[]) or []

    cutoff = prev_business_day(dt.date.today())
    rows = []
    for h in history:
        epoch = h.get("epochGradeDate") or h.get("gradeDate") or 0
        d = dt.date.fromtimestamp(epoch) if epoch else None
        if not d or d > cutoff:
            continue
        firm = h.get("firm") or h.get("company") or ""
        action = (h.get("action") or "").capitalize()
        from_g, to_g = h.get("fromGrade") or "", h.get("toGrade") or ""
        rating_str = f"{from_g} → {to_g}".strip(" →")

        pt_old = (h.get("priceTargetPrior") or h.get("priceTargetOld") or h.get("priceTargetFrom"))
        pt_new = (h.get("priceTargetCurrent") or h.get("priceTargetNew") or h.get("priceTargetTo"))
        if pt_old is not None and pt_new is not None:
            try:
                oldf, newf = float(pt_old), float(pt_new)
                pa = "Raises" if newf > oldf else ("Lowers" if newf < oldf else "Reiterates")
                pt = f"{oldf:g} → {newf:g}"
            except Exception:
                pa, pt = "—", f"{pt_old} → {pt_new}"
        elif pt_new is not None:
            pa, pt = "Sets", f"{pt_new}"
        else:
            pa, pt = "—", "—"

        rows.append({
            "date": d.isoformat(),
            "analyst": firm,
            "rating_action": action,
            "rating": rating_str,
            "price_action": pa,
            "price_target": pt,
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows[:limit]

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
        dcc.Store(id="app-store", storage_type="local", data=load_data()),
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
    # Common: get current tickers from store
    shorts_df = pd.DataFrame(data.get("shorts", []))
    tickers = list(shorts_df["Ticker"]) if not shorts_df.empty and "Ticker" in shorts_df.columns else []
    default_ticker = tickers[0] if tickers else None

    # ===== Short Interest tab =====
    if tab == "shorts":
        return html.Div(
            [
                # LEFT: tickers + add/delete
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
                # RIGHT: chart + diagnostics
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

    # ===== Analyst Ratings tab =====
    if tab == "ratings":
        return html.Div(
            [
                # LEFT: vertical ticker list (separate selection for ratings)
                html.Div(
                    [
                        html.H3("Tickers", style={"marginBottom": "8px"}),
                        dcc.RadioItems(
                            id="ratings-radio",
                            options=[{"label": t, "value": t} for t in tickers],
                            value=default_ticker,
                            labelStyle={"display": "block", "margin": "6px 0"},
                            inputStyle={"marginRight": "8px"},
                        ),
                    ],
                    style={"width": "220px", "padding": "10px 12px",
                           "borderRight": "1px solid #e5e7eb", "flexShrink": 0},
                ),
                # RIGHT: header + refresh + analyst cards + diagnostics
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Most Recent Analyst Notes (prev trading day or earlier)", style={"margin": "0"}),
                                html.Button("Refresh", id="analyst-refresh-btn", n_clicks=0, style={"marginLeft": "10px"}),
                            ],
                            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"},
                        ),
                        html.Div(id="analyst-cards"),
                        html.Div(id="analyst-diag", style={"fontSize": "12px", "color": "#555", "marginTop": "6px"}),
                    ],
                    style={"flex": 1, "padding": "10px 16px"},
                ),
            ],
            style={"display": "flex", "alignItems": "stretch"},
        )

    # Fallback (no tab matched)
    return html.Div()

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
    hist = (data.get("shorts_history", {}) or {}).get(selected_ticker.upper(), [])
    last_cached = None
    if hist:
        try:
            last_cached = max(pd.to_datetime([h["date"] for h in hist]).date)
        except Exception:
            pass
    expected_end = prev_business_day(dt.date.today())   # <-- was last_business_day
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

def _analyst_cards_layout(rows: list[dict]):
    if not rows:
        return html.Div("No analyst notes found for the selected ticker (prior to today).")

    def _card(r):
        return html.Div(
            [
                html.Div([
                    html.Div(r.get("analyst", "—"), style={"fontWeight": 600, "fontSize": "16px"}),
                    html.Div(r.get("date", "—"), style={"fontSize": "12px", "color": "#666"}),
                ], style={"marginBottom": "6px"}),
                html.Table([
                    html.Tbody([
                        html.Tr([html.Td("Rating Action", style={"fontWeight": 600, "paddingRight": "10px"}),
                                 html.Td(r.get("rating_action", "—"))]),
                        html.Tr([html.Td("Rating", style={"fontWeight": 600, "paddingRight": "10px"}),
                                 html.Td(r.get("rating", "—"))]),
                        html.Tr([html.Td("Price Action", style={"fontWeight": 600, "paddingRight": "10px"}),
                                 html.Td(r.get("price_action", "—"))]),
                        html.Tr([html.Td("Price Target", style={"fontWeight": 600, "paddingRight": "10px"}),
                                 html.Td(r.get("price_target", "—"))]),
                    ])
                ], style={"width": "100%"}),
            ],
            style={
                "border": "1px solid #e5e7eb",
                "borderRadius": "12px",
                "padding": "12px 14px",
                "marginBottom": "10px",
                "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
            },
        )

    return html.Div([_card(r) for r in rows])

@app.callback(
    [Output("analyst-cards", "children"), Output("analyst-diag", "children")],
    [Input("ratings-radio", "value"), Input("tabs", "value"), Input("analyst-refresh-btn", "n_clicks")],
    prevent_initial_call=False,
)
def update_analyst_cards(ticker, active_tab, _n):
    if active_tab != "ratings" or not ticker:
        return no_update, no_update
    rows = fetch_recent_analyst_actions(ticker, limit=3)
    diag = ""
    if rows and rows[0].get("rating_action") == "ERROR":
        diag = rows[0].get("price_target", "")
        rows = []
    return _analyst_cards_layout(rows), diag

# ============================== Main ==============================
if __name__ == "__main__":
    app.run(debug=True)