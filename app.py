# app.py
# Streamlit Meeting Room Dashboard (CSV-based, no SQL)
# Rooms: 3 Meeting Rooms + 1 Board Room
# Views: Now/Next per room, Today timeline, quick booking form
# Timezone: Asia/Kolkata | Business hours: 09:00–19:00 | Slot: 30 min | No overlaps

import streamlit as st
import pandas as pd
from datetime import datetime, date, time, timedelta
from dateutil import tz
import io
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Room Availability Dashboard", layout="wide")
IST = tz.gettz("Asia/Kolkata")

ROOMS = [
    {"id": "mr1", "name": "Meeting Room 1"},
    {"id": "mr2", "name": "Meeting Room 2"},
    {"id": "mr3", "name": "Meeting Room 3"},
    {"id": "br",  "name": "Board Room"},
]

BOOKINGS_CSV = "bookings.csv"
BUSINESS_HOURS = (time(9, 0), time(19, 0))  # 09:00–19:00
SLOT_MINUTES = 30
STARTING_SOON_MIN = 15

# -----------------------------
# STORAGE HELPERS (CSV)
# -----------------------------
def ensure_csv():
    if not os.path.exists(BOOKINGS_CSV):
        df = pd.DataFrame(columns=[
            "room_id", "room_name",
            "date", "start_time", "end_time",
            "booked_by", "title", "created_at"
        ])
        df.to_csv(BOOKINGS_CSV, index=False)

@st.cache_data(show_spinner=False)
def load_bookings() -> pd.DataFrame:
    ensure_csv()
    df = pd.read_csv(BOOKINGS_CSV, dtype=str)
    if df.empty:
        return df
    # normalize
    for col in ["date", "start_time", "end_time", "created_at"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def save_bookings(df: pd.DataFrame):
    # NOTE: Streamlit is single-process; this is fine for a small team app.
    df.to_csv(BOOKINGS_CSV, index=False)
    load_bookings.clear()  # invalidate cache

# -----------------------------
# TIME UTILS
# -----------------------------
def dt_ist(d: date, t: time) -> datetime:
    # Create IST-aware datetime
    return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=IST)

def parse_time_str(s: str) -> time:
    # "HH:MM"
    hh, mm = s.split(":")
    return time(int(hh), int(mm))

def fmt_time(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"

def now_ist() -> datetime:
    return datetime.now(IST)

# -----------------------------
# BOOKING LOGIC
# -----------------------------
def overlaps(s1: time, e1: time, s2: time, e2: time) -> bool:
    # [s1, e1) intersects [s2, e2)
    return (s1 < e2) and (s2 < e1)

def room_status_for_today(df: pd.DataFrame, room_id: str, today: date):
    """Return (state, label, now_meeting_row or None, next_meeting_row or None)"""
    now = now_ist()
    today_rows = df[(df["room_id"] == room_id) & (df["date"] == today.isoformat())].copy()
    # sort by start_time
    if not today_rows.empty:
        today_rows["start_time_t"] = today_rows["start_time"].apply(parse_time_str)
        today_rows["end_time_t"]   = today_rows["end_time"].apply(parse_time_str)
        today_rows.sort_values(by="start_time_t", inplace=True)

    current_row = None
    next_row = None
    for _, r in today_rows.iterrows():
        s = parse_time_str(r["start_time"])
        e = parse_time_str(r["end_time"])
        sdt = dt_ist(today, s); edt = dt_ist(today, e)
        if sdt <= now < edt:
            current_row = r
            break
        if sdt > now and next_row is None:
            next_row = r

    if current_row is not None:
        label = f"Occupied · {current_row['start_time']}–{current_row['end_time']}"
        return "occupied", label, current_row, next_row

    if next_row is not None:
        label = f"Available · Next: {next_row['start_time']}–{next_row['end_time']}"
        return "available", label, None, next_row

    return "available", "Available all day", None, None

def validate_booking(new_row: dict, df: pd.DataFrame) -> str | None:
    """Return None if OK, else error string."""
    # Hours
    start_t = parse_time_str(new_row["start_time"])
    end_t   = parse_time_str(new_row["end_time"])
    if end_t <= start_t:
        return "End time must be after start time."
    if start_t < BUSINESS_HOURS[0] or end_t > BUSINESS_HOURS[1]:
        return f"Booking must be within business hours {fmt_time(BUSINESS_HOURS[0])}–{fmt_time(BUSINESS_HOURS[1])}."

    # Slot granularity
    for t in (start_t, end_t):
        if (t.minute % SLOT_MINUTES) != 0:
            return f"Time must align with {SLOT_MINUTES}-minute slots (e.g., 09:00, 09:30, 10:00)."

    # No overlap
    same_day = df[(df["room_id"] == new_row["room_id"]) & (df["date"] == new_row["date"])]
    for _, r in same_day.iterrows():
        s2 = parse_time_str(r["start_time"])
        e2 = parse_time_str(r["end_time"])
        if overlaps(start_t, end_t, s2, e2):
            return f"Conflict with existing booking {r['start_time']}–{r['end_time']}."

    return None

# -----------------------------
# UI HELPERS
# -----------------------------
def status_pill(state: str, starting_soon: bool) -> str:
    if state == "occupied":
        return '<span class="pill pill-red">Occupied</span>'
    if starting_soon:
        return '<span class="pill pill-amber">Starting Soon</span>'
    return '<span class="pill pill-green">Available</span>'

def inject_css():
    st.markdown(
        """
        <style>
        .pill {
            display:inline-flex; align-items:center; gap:6px;
            border-radius:999px; padding:4px 10px; font-size:12px; 
            border:1px solid rgba(0,0,0,0.08); font-weight:600;
        }
        .pill-green { background:#DCFCE7; color:#065F46; border-color:#A7F3D0; }
        .pill-red   { background:#FFE4E6; color:#9F1239; border-color:#FECDD3; }
        .pill-amber { background:#FEF3C7; color:#92400E; border-color:#FDE68A; }
        .card {
            border:1px solid #eee; border-radius:16px; padding:16px; 
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .muted { color:#6B7280; font-size:13px; }
        .title { font-weight:700; }
        .block { position:absolute; top:50%; transform:translateY(-50%); height:28px; 
                 border:1px solid; border-radius:8px; padding:0 8px; display:flex; align-items:center; font-size:12px;}
        .blk-occupied { background:#FFE4E6; color:#9F1239; border-color:#FECDD3; }
        .blk-open     { background:#DCFCE7; color:#065F46; border-color:#A7F3D0; }
        </style>
        """,
        unsafe_allow_html=True
    )

def room_card(df: pd.DataFrame, room: dict, today: date):
    state, label, now_row, next_row = room_status_for_today(df, room["id"], today)

    starting_soon = False
    if state == "available" and next_row is not None:
        ns = parse_time_str(next_row["start_time"])
        soon_dt = dt_ist(today, ns)
        diff = soon_dt - now_ist()
        starting_soon = diff <= timedelta(minutes=STARTING_SOON_MIN) and diff.total_seconds() > 0

    # Room card container
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(room["name"])
    st.markdown(status_pill(state, starting_soon), unsafe_allow_html=True)
    st.markdown(f"<div class='muted' style='margin-top:6px;'>{label}</div>", unsafe_allow_html=True)

    # Split into two main columns: Left (status) | Right (quick book)
    col_left, col_right = st.columns([0.6, 0.4])

    # LEFT SECTION — Now / Next info
    with col_left:
        st.caption("Now")
        if state == "occupied" and now_row is not None:
            st.markdown(f"**{now_row['title']}**")
            st.markdown(
                f"<span class='muted'>{now_row['start_time']}–{now_row['end_time']} · Host: {now_row['booked_by']}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span class='muted'>Free</span>", unsafe_allow_html=True)

        st.caption("Next")
        if next_row is not None:
            st.markdown(f"**{next_row['title']}**")
            st.markdown(
                f"<span class='muted'>{next_row['start_time']}–{next_row['end_time']} · Host: {next_row['booked_by']}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span class='muted'>No upcoming meetings</span>", unsafe_allow_html=True)

    # RIGHT SECTION — Quick Booking form
    with col_right:
        st.caption("Quick Book")
        with st.form(f"quick_book_{room['id']}", clear_on_submit=True):
            qb_title = st.text_input("Title / Purpose", placeholder="e.g., Marketing Sync")
            qb_host = st.text_input("Booked By", placeholder="Your name")
            qb_date = st.date_input("Date", value=today)
            now_time = datetime.now().time().replace(second=0, microsecond=0)
            start_default = (
                datetime.combine(today, now_time) + timedelta(minutes=SLOT_MINUTES)
            ).time().replace(minute=(0 if now_time.minute < 30 else 30))

            qb_start = st.time_input("Start", value=start_default, step=60 * SLOT_MINUTES)
            qb_end = st.time_input(
                "End",
                value=(datetime.combine(date.today(), start_default) + timedelta(minutes=SLOT_MINUTES)).time(),
                step=60 * SLOT_MINUTES,
            )
            submitted = st.form_submit_button("Book")

        if submitted:
            if not qb_title or not qb_host:
                st.error("Please provide Title and Booked By.")
            else:
                new_row = {
                    "room_id": room["id"],
                    "room_name": room["name"],
                    "date": qb_date.isoformat(),
                    "start_time": fmt_time(qb_start),
                    "end_time": fmt_time(qb_end),
                    "booked_by": qb_host.strip(),
                    "title": qb_title.strip(),
                    "created_at": datetime.now(IST).isoformat(),
                }
                err = validate_booking(new_row, load_bookings())
                if err:
                    st.error(err)
                else:
                    df2 = load_bookings().copy()
                    df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)
                    save_bookings(df2)
                    st.success(
                        f"Booked {room['name']} · {new_row['date']} {new_row['start_time']}–{new_row['end_time']}"
                    )

    st.markdown("</div>", unsafe_allow_html=True)


def today_timeline(df: pd.DataFrame, today: date):
    st.markdown("#### Today’s Timeline")
    st.caption("09:00 — 19:00")
    # container
    for room in ROOMS:
        st.markdown(f"**{room['name']}**")
        # base bar
        timeline = st.container()
        with timeline:
            # Render as simple HTML bar with absolute blocks
            start_dt = dt_ist(today, BUSINESS_HOURS[0])
            end_dt   = dt_ist(today, BUSINESS_HOURS[1])
            span_ms  = (end_dt - start_dt).total_seconds() * 1000

            # Make row
            st.markdown(
                """
                <div style="position:relative;height:40px;border-radius:12px;background:#F3F4F6;overflow:hidden;margin-bottom:8px;"></div>
                """,
                unsafe_allow_html=True
            )
            # We need a delta to position absolutely inside the last inserted div.
            # Streamlit doesn't allow direct child targeting, so we instead render blocks immediately after with fixed widths using a wrapper.
            # We'll simulate by printing a block row with inline-block widths.

        # compute blocks
        rows = df[(df["room_id"] == room["id"]) & (df["date"] == today.isoformat())].copy()
        if not rows.empty:
            rows["s"] = rows["start_time"].apply(parse_time_str)
            rows["e"] = rows["end_time"].apply(parse_time_str)
            rows.sort_values(by="s", inplace=True)

        # Build an HTML bar with segments
        html = io.StringIO()
        html.write('<div style="position:relative;height:40px;border-radius:12px;background:#F3F4F6;overflow:hidden;margin-bottom:16px;">')
        for _, r in rows.iterrows():
            sdt = dt_ist(today, r["s"])
            edt = dt_ist(today, r["e"])
            left = max(0.0, (sdt - start_dt).total_seconds() / (end_dt - start_dt).total_seconds()) * 100.0
            width = max(0.0, (edt - sdt).total_seconds() / (end_dt - start_dt).total_seconds()) * 100.0
            occupied = True  # blocks represent bookings
            cls = "blk-occupied" if occupied else "blk-open"
            html.write(
                f'<div class="block {cls}" style="left:{left:.4f}%; width:{width:.4f}%;">{r["title"]}</div>'
            )
        html.write("</div>")
        st.markdown(html.getvalue(), unsafe_allow_html=True)

# -----------------------------
# PAGE
# -----------------------------
inject_css()

# Header
today = now_ist().date()
date_str = now_ist().strftime("%A, %d %B %Y")
time_str = now_ist().strftime("%H:%M")

left, right = st.columns([0.7, 0.3])
with left:
    st.title("Room Availability Dashboard")
    st.markdown(f"<span class='muted'>{date_str} · {time_str} IST</span>", unsafe_allow_html=True)
with right:
    view = st.segmented_control("View", options=["Today", "Tomorrow", "Week"], default="Today")

# Data
df = load_bookings()

# Quick filters (non-persistent demo; Timeline is for today view)
if view == "Today":
    # Grid cards (2 columns on wide screens)
    rows = []
    for i in range(0, len(ROOMS), 2):
        rows.append(ROOMS[i:i+2])
    for row_rooms in rows:
        cols = st.columns(len(row_rooms))
        for col, room in zip(cols, row_rooms):
            with col:
                room_card(df, room, today)
    st.divider()
    today_timeline(df, today)

elif view == "Tomorrow":
    tomorrow = today + timedelta(days=1)
    st.caption(f"Showing {tomorrow.strftime('%A, %d %B %Y')}")
    rows = []
    for i in range(0, len(ROOMS), 2):
        rows.append(ROOMS[i:i+2])
    for row_rooms in rows:
        cols = st.columns(len(row_rooms))
        for col, room in zip(cols, row_rooms):
            with col:
                room_card(df, room, tomorrow)

elif view == "Week":
    st.caption("This week overview (use the Quick Book in each card to add items).")
    for offset in range(0, 7):
        d = today + timedelta(days=offset)
        st.markdown(f"### {d.strftime('%A, %d %B %Y')}")
        rows = []
        for i in range(0, len(ROOMS), 2):
            rows.append(ROOMS[i:i+2])
        for row_rooms in rows:
            cols = st.columns(len(row_rooms))
            for col, room in zip(cols, row_rooms):
                with col:
                    room_card(df, room, d)
        st.divider()

# -----------------------------
# Download / Template
# -----------------------------
with st.expander("📄 Download / Upload bookings CSV"):
    st.caption("Columns: room_id, room_name, date (YYYY-MM-DD), start_time (HH:MM), end_time (HH:MM), booked_by, title, created_at (ISO).")
    buf = io.StringIO()
    (load_bookings() if not load_bookings().empty else pd.DataFrame(columns=[
        "room_id","room_name","date","start_time","end_time","booked_by","title","created_at"
    ])).to_csv(buf, index=False)
    st.download_button("Download current bookings.csv", data=buf.getvalue(), file_name="bookings.csv", mime="text/csv")

    up = st.file_uploader("Upload bookings.csv", type=["csv"])
    if up:
        up_df = pd.read_csv(up, dtype=str)
        # basic schema check
        required = {"room_id","room_name","date","start_time","end_time","booked_by","title","created_at"}
        if not required.issubset(set(up_df.columns)):
            st.error(f"Invalid CSV. Required columns: {', '.join(sorted(required))}")
        else:
            # also block overlaps within each room/day for the uploaded file
            err_msgs = []
            for rid in [r["id"] for r in ROOMS]:
                sub = up_df[up_df["room_id"] == rid]
                for d in sub["date"].unique():
                    sd = sub[sub["date"] == d].copy()
                    for i, r1 in sd.iterrows():
                        for j, r2 in sd.iterrows():
                            if i >= j: 
                                continue
                            if overlaps(parse_time_str(r1["start_time"]), parse_time_str(r1["end_time"]),
                                        parse_time_str(r2["start_time"]), parse_time_str(r2["end_time"])):
                                err_msgs.append(f"Overlap in {rid} on {d}: {r1['start_time']}-{r1['end_time']} vs {r2['start_time']}-{r2['end_time']}")
            if err_msgs:
                st.error("Upload blocked due to overlaps:\n- " + "\n- ".join(err_msgs[:10]) + ("\n..." if len(err_msgs) > 10 else ""))
            else:
                save_bookings(up_df)
                st.success("Bookings uploaded and saved.")

# Footer
st.caption("Powered by Streamlit · CSV storage · IST timezone · No SQL required")
