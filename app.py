import re
import os
import math
from pathlib import Path
from io import BytesIO

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
import plotly.graph_objects as go

#
# PAGE CONFIG
#
st.set_page_config(layout="wide", page_title="Hudson Cicala — Pass Dashboard")

#
# OPTIONAL DOCX IMPORT
#
DOCX_AVAILABLE = True
try:
    from docx import Document
except Exception:
    DOCX_AVAILABLE = False

#
# STYLE
#
st.markdown("""
<style>
    /* Mantendo apenas o estilo dos cartões de métricas e títulos das linhas para não quebrar a fonte nativa do Streamlit */
    .row-label-blue {
        font-size: 14px; font-weight: 700; color: #60a5fa;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 8px; border-bottom: 1px solid #1e3a8a; padding-bottom: 4px;
    }
    .row-label-green {
        font-size: 14px; font-weight: 700; color: #34d399;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 8px; border-bottom: 1px solid #064e3b; padding-bottom: 4px;
    }
    .row-label-amber {
        font-size: 14px; font-weight: 700; color: #fbbf24;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 8px; border-bottom: 1px solid #78350f; padding-bottom: 4px;
    }
    
    .metric-box {
        background: #1a1a2e; border-radius: 8px; padding: 12px 16px;
        margin-bottom: 12px; border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title { font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .metric-value-row { display: flex; align-items: baseline; gap: 8px; }
    .metric-value { font-size: 24px; font-weight: 800; color: #ffffff; line-height: 1; }
    .metric-arrow { font-size: 13px; font-weight: 700; padding: 2px 6px; border-radius: 4px; }
    .arrow-up { color: #10b981; background: rgba(16, 185, 129, 0.15); }
    .arrow-down { color: #ef4444; background: rgba(239, 68, 68, 0.15); }
    .metric-sub { font-size: 11px; color: #64748b; margin-top: 4px; display: flex; justify-content: space-between; }
</style>
""", unsafe_allow_html=True)

#
# CONSTANTS
#
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
GOAL_X = 120.0
GOAL_Y = 40.0

FIG_W, FIG_H = 7.0, 4.7
FIG_DPI = 180

COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07

C_BLUE = "#2F80ED"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"

CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)

NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60
LATERAL_MIN_DIST = 12.0

#
# BASE PASSES
#
BASE_MATCHES_DATA = {
    "Connecticut United (03-27)": [
        ("PASS WON", 26.75, 68.34, 8.97, 51.05, None),
        ("PASS WON", 31.24, 51.22, 34.57, 72.50, None),
        ("PASS WON", 36.06, 46.90, 44.37, 57.04, None),
        ("PASS WON", 48.36, 64.02, 58.17, 51.72, None),
        ("PASS WON", 58.17, 64.02, 62.49, 55.21, None),
        ("PASS WON", 54.51, 49.72, 64.82, 61.69, None),
        ("PASS WON", 42.21, 70.84, 34.90, 76.49, None),
        ("PASS WON", 43.54, 75.32, 36.73, 67.84, None),
        ("PASS WON", 32.24, 53.96, 6.81, 38.50, None),
        ("PASS WON", 33.57, 65.77, 36.56, 75.57, None),
        ("PASS WON", 37.39, 61.11, 43.04, 75.41, None),
        ("PASS WON", 65.49, 53.63, 56.18, 70.42, None),
        ("PASS WON", 55.68, 48.15, 46.87, 30.86, None),
        ("PASS WON", 52.02, 22.05, 46.70, 41.99, None),
        ("PASS WON", 62.16, 35.51, 71.80, 35.18, None),
        ("PASS WON", 54.02, 33.35, 63.99, 22.55, None),
        ("PASS WON", 60.00, 22.21, 76.62, 32.85, None),
        ("PASS WON", 87.10, 9.41, 77.45, 16.23, None),
        ("PASS WON", 62.66, 20.05, 117.18, 8.25, None),
        ("PASS WON", 98.90, 43.49, 103.22, 47.15, None),
        ("PASS WON", 70.31, 45.98, 82.28, 60.11, None),
        ("PASS WON", 85.10, 75.24, 101.39, 74.08, None),
        ("PASS WON", 53.18, 67.59, 39.05, 59.62, None),
        ("PASS WON", 55.18, 49.64, 54.85, 13.07, None),
        ("PASS WON", 68.64, 19.22, 49.03, 24.37, None),
        ("PASS WON", 53.35, 22.71, 59.34, 30.19, None),
        ("PASS WON", 44.37, 24.71, 40.05, 46.82, None),
        ("PASS WON", 43.88, 39.34, 41.38, 73.08, None),
        ("PASS WON", 56.84, 53.46, 70.81, 76.24, None),
        ("PASS WON", 82.77, 12.24, 91.42, 4.59, None),
        ("PASS WON", 108.04, 11.74, 115.69, 58.29, None),
        ("PASS WON", 93.08, 3.93, 111.03, 13.74, None),
        ("PASS WON", 84.60, 17.89, 96.74, 22.05, None),
        ("PASS WON", 58.34, 16.06, 65.65, 2.43, None),
        ("PASS WON", 52.02, 8.58, 44.37, 15.73, None),
        ("PASS WON", 61.00, 23.21, 49.36, 15.23, None),
        ("PASS WON", 32.74, 30.69, 50.03, 33.02, None),
        ("PASS WON", 51.85, 33.68, 60.66, 40.00, None),
        ("PASS WON", 79.95, 60.45, 98.23, 60.28, None),
        ("PASS WON", 31.24, 52.14, 39.05, 72.08, None),
        ("PASS WON", 39.72, 48.98, 33.40, 57.62, None),
        ("PASS WON", 70.64, 51.47, 61.00, 51.64, None),
        ("PASS LOST", 53.35, 19.55, 73.96, 11.24, None),
        ("PASS LOST", 63.82, 20.55, 88.76, 22.55, None),
        ("PASS LOST", 85.60, 27.86, 94.41, 37.17, None),
        ("PASS LOST", 77.79, 27.53, 96.41, 25.37, None),
        ("PASS LOST", 91.09, 27.86, 109.54, 50.47, None),
        ("PASS LOST", 58.17, 26.04, 95.41, 40.33, None),
        ("PASS LOST", 53.35, 28.53, 73.80, 27.86, None),
        ("PASS LOST", 53.35, 34.02, 84.60, 58.62, None),
        ("PASS LOST", 56.18, 49.48, 97.07, 62.11, None),
        ("PASS LOST", 34.23, 74.91, 65.65, 78.57, None),
    ],
    "Nashville SC (03-28)": [
        ("PASS WON", 21.27, 14.23, 29.25, 31.02, None),
        ("PASS WON", 29.41, 23.38, 34.40, 64.60, None),
        ("PASS WON", 41.55, 39.67, 41.88, 6.92, None),
        ("PASS WON", 44.54, 32.52, 43.54, 14.23, None),
        ("PASS WON", 23.59, 56.46, 34.57, 47.48, None),
        ("PASS WON", 30.58, 64.44, 21.10, 49.48, None),
        ("PASS WON", 33.07, 56.79, 49.53, 69.59, None),
        ("PASS WON", 33.24, 59.78, 44.04, 71.75, None),
        ("PASS WON", 61.50, 71.58, 54.68, 75.57, None),
        ("PASS WON", 63.16, 50.81, 78.45, 67.26, None),
        ("PASS WON", 63.49, 76.90, 84.44, 62.77, None),
        ("PASS WON", 76.96, 56.96, 86.93, 57.79, None),
        ("PASS WON", 82.61, 59.12, 96.41, 68.43, None),
        ("PASS WON", 79.78, 35.35, 106.21, 11.74, None),
        ("PASS WON", 45.37, 49.64, 40.72, 32.02, None),
        ("PASS LOST", 78.62, 64.94, 96.57, 67.10, None),
        ("PASS LOST", 85.43, 68.76, 106.05, 77.74, None),
    ],
    "Seongnam FC (03-29)": [
        ("PASS WON", 28.08, 28.53, 29.75, 8.25, None),
        ("PASS WON", 33.74, 26.54, 29.41, 43.82, None),
        ("PASS WON", 28.08, 47.15, 31.57, 64.60, None),
        ("PASS WON", 39.39, 43.82, 51.69, 53.46, None),
        ("PASS WON", 43.88, 46.15, 55.84, 40.66, None),
        ("PASS WON", 47.03, 49.97, 44.04, 28.03, None),
        ("PASS WON", 47.53, 50.81, 71.97, 33.18, None),
        ("PASS WON", 67.65, 52.63, 64.32, 33.85, None),
        ("PASS WON", 73.63, 65.10, 69.31, 73.25, None),
        ("PASS WON", 77.29, 63.27, 79.12, 72.91, None),
        ("PASS WON", 81.61, 56.62, 93.91, 73.75, None),
        ("PASS WON", 86.43, 66.43, 81.78, 54.96, None),
        ("PASS WON", 111.03, 71.42, 99.56, 67.59, None),
        ("PASS WON", 89.76, 59.62, 97.74, 48.98, None),
        ("PASS WON", 88.43, 52.47, 96.41, 74.24, None),
        ("PASS WON", 87.93, 50.97, 77.12, 27.70, None),
        ("PASS WON", 81.61, 53.63, 74.30, 27.03, None),
        ("PASS WON", 79.28, 51.14, 94.91, 70.42, None),
        ("PASS WON", 52.85, 32.85, 65.49, 25.37, None),
        ("PASS WON", 82.77, 33.18, 69.31, 47.65, None),
        ("PASS LOST", 72.14, 16.56, 78.45, 1.60, None),
        ("PASS LOST", 79.62, 27.53, 97.07, 47.98, None),
        ("PASS LOST", 91.75, 50.14, 109.70, 65.77, None),
        ("PASS LOST", 96.41, 56.79, 107.04, 67.26, None),
    ],
    "NY Red Bulls (03-31)": [
        ("PASS WON", 39.39, 19.39, 52.35, 4.76, None),
        ("PASS WON", 63.82, 7.92, 72.63, 1.43, None),
        ("PASS WON", 70.47, 11.91, 80.95, 13.74, None),
        ("PASS WON", 64.49, 22.55, 97.24, 10.24, None),
        ("PASS WON", 32.07, 35.51, 43.04, 28.20, None),
        ("PASS WON", 53.52, 46.32, 54.02, 33.68, None),
        ("PASS WON", 77.12, 48.64, 84.94, 50.14, None),
        ("PASS WON", 78.12, 52.47, 117.52, 69.42, None),
        ("PASS WON", 88.76, 65.93, 97.40, 76.74, None),
        ("PASS WON", 82.61, 69.26, 86.60, 77.40, None),
        ("PASS WON", 78.62, 66.26, 79.62, 78.40, None),
        ("PASS WON", 83.61, 75.91, 62.49, 57.12, None),
        ("PASS WON", 34.40, 50.14, 88.76, 75.41, None),
        ("PASS WON", 56.68, 64.27, 78.29, 64.27, None),
        ("PASS WON", 51.85, 73.25, 54.18, 78.07, None),
        ("PASS WON", 41.05, 57.45, 46.04, 74.91, None),
        ("PASS WON", 37.39, 60.61, 41.71, 73.91, None),
        ("PASS WON", 30.41, 63.44, 36.89, 77.40, None),
        ("PASS WON", 26.09, 63.94, 28.42, 76.74, None),
        ("PASS WON", 22.43, 56.62, 22.10, 76.41, None),
        ("PASS WON", 33.90, 64.77, 25.42, 73.58, None),
        ("PASS LOST", 41.88, 42.49, 56.18, 52.97, None),
        ("PASS LOST", 37.56, 41.16, 46.37, 53.96, None),
        ("PASS LOST", 54.68, 56.96, 54.85, 64.44, None),
        ("PASS LOST", 51.69, 68.43, 66.15, 76.57, None),
    ],
}

#
# HELPERS
#
def apply_date_mapping(name: str) -> str:
    mapping = {
        "Connecticut United": "Connecticut United (03-27)",
        "Nashville SC": "Nashville SC (03-28)",
        "Seongnam FC": "Seongnam FC (03-29)",
        "NY Red Bulls": "NY Red Bulls (03-31)",
        "Real Salt Lake": "Real Salt Lake (04-26)",
        "Real Futbol": "Real Futbol (05-23)",
        "San Jose": "San Jose (05-24)",
        "Houston Dynamo": "Houston Dynamo (05-26)"
    }
    for k, v in mapping.items():
        if k.lower() == name.lower().strip():
            return v
    return name

def get_match_minutes(match_name: str) -> float:
    """Returns the minutes played for a given match to calculate p90."""
    name_lower = match_name.lower()
    if "connecticut" in name_lower: return 60.0
    if "nashville" in name_lower: return 60.0
    if "seongnam" in name_lower: return 32.0
    if "red bulls" in name_lower: return 60.0
    if "houston" in name_lower: return 63.0
    if "vardar" in name_lower: return 65.0
    return 90.0

def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)

def get_lane(y):
    if y >= LANE_LEFT_MIN: return "left"
    elif y < LANE_RIGHT_MAX: return "right"
    return "center"

def is_progressive_pass(x_start, y_start, x_end, y_end) -> bool:
    if x_start < 35: return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist == 0: return False
    return ((start_dist - end_dist) / start_dist) >= 0.25

def classify_pass_direction(x_start, y_start, x_end, y_end) -> str:
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx**2 + dy**2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0: return "forward"
    if angle_deg >= 135.0: return "backward"
    if dist > LATERAL_MIN_DIST:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"

def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24):
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    xp = 0.01 + (Xc / FIELD_X) * 0.99
    yc = 1.0 - np.abs((Yc / FIELD_Y) - 0.5) * 2.0
    base = xp * (0.8 + 0.2 * yc)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    XT = base.copy()
    XT = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)
    XTc = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            XTc[iy, ix] = XT[iy*sub:(iy+1)*sub, ix*sub:(ix+1)*sub].mean()
    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)
    return XTc

XT_GRID = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX_XT, 0, NX_XT - 1))
    iy = int(np.clip((y / FIELD_Y) * NY_XT, 0, NY_XT - 1))
    return float(XT_GRID[iy, ix])

#
# DOCX PARSER
#
def read_docx_text(docx_path: Path) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed.")
    doc = Document(str(docx_path))
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())

def parse_docx_events(raw_text: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    matches = {}
    current_match = None
    current_state = None
    re_match = re.compile(r"^Vs\s+(.+)$", re.IGNORECASE)
    re_success = re.compile(r"^Sucesso$", re.IGNORECASE)
    re_fail = re.compile(r"^Errado[s]?$", re.IGNORECASE)
    re_arrow = re.compile(
        r"^Seta\s+\d+:\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*->\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$",
        re.IGNORECASE
    )
    for ln in lines:
        m_match = re_match.match(ln)
        if m_match:
            current_match = m_match.group(1).strip()
            matches.setdefault(current_match, [])
            current_state = None
            continue
        if re_success.match(ln):
            current_state = "PASS WON"
            continue
        if re_fail.match(ln):
            current_state = "PASS LOST"
            continue
        m_arrow = re_arrow.match(ln)
        if m_arrow and current_match and current_state:
            x1, y1, x2, y2 = map(float, m_arrow.groups())
            matches[current_match].append((current_state, x1, y1, x2, y2, None))
    return {k: v for k, v in matches.items() if len(v) > 0}

def load_docx_matches(docx_filename="Passes - Hudson Cicala.docx") -> dict:
    p = Path(docx_filename)
    if not p.exists(): return {}
    txt = read_docx_text(p)
    return parse_docx_events(txt)

#
# DATA LOADING
#
docx_matches_data = {}
try:
    docx_matches_data = load_docx_matches()
except Exception:
    pass

combined_matches_data = {}
for k, v in docx_matches_data.items():
    mapped_k = apply_date_mapping(k)
    name = mapped_k if mapped_k not in combined_matches_data else f"DOCX - {mapped_k}"
    combined_matches_data[name] = v

for k, v in BASE_MATCHES_DATA.items():
    combined_matches_data[k] = v

if len(combined_matches_data) == 0:
    st.error("Could not load data.")
    st.stop()

#
# BUILD DATAFRAMES & REORDER MATCHES
#
dfs_by_match = {}
for match_name, events in combined_matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["progressive"] = dfm.apply(
        lambda r: r["is_won"] and is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]), axis=1
    )
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r["x_start"], r["y_start"], r["x_end"], r["y_end"]), axis=1
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"])**2 + (dfm["y_end"] - dfm["y_start"])**2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"] = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"], dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    dfs_by_match[match_name] = dfm

# REORDER LOGIC: Move Match 15-18 to be after Match 6
items = list(dfs_by_match.items())
if len(items) >= 18:
    part1 = items[:6]   # Matches 1 to 6
    part2 = items[14:18] # Matches 15 to 18
    part3 = items[6:14]  # Matches 7 to 14
    part4 = items[18:]   # Matches 19+
    dfs_by_match = dict(part1 + part2 + part3 + part4)

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)

#
# STATS & SCORES
#
def compute_stats(df: pd.DataFrame, match_name: str) -> dict:
    total = len(df)
    mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    
    if total == 0:
        return {
            "total_passes": 0, "successful_passes": 0, "unsuccessful_passes": 0, "accuracy_pct": 0.0,
            "progressive_attempted": 0, "progressive_successful": 0, "progressive_accuracy_pct": 0.0,
            "to_final_third_total": 0, "to_final_third_success": 0, "to_final_third_accuracy_pct": 0.0,
            "fwd": 0, "fwd_pct": 0.0, "bwd": 0, "bwd_pct": 0.0, "lat": 0, "lat_pct": 0.0,
            "pos_count": 0, "pos_pct": 0.0, "high_xt_pct": 0.0, "sum_dxt": 0.0,
            "total_p90": 0.0, "prog_p90": 0.0, "f3_p90": 0.0, "xt_p90": 0.0, "neg_xt_p90": 0.0,
            "minutes": mins, "long_acc_pct": 0.0, "high_xt_p90": 0.0, "dz_p90": 0.0
        }

    successful = int(df["is_won"].sum())
    unsuccessful = total - successful
    accuracy = successful / total * 100.0

    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int((~df["is_won"] & df.apply(
        lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]), axis=1
    )).sum())
    progressive_attempted = progressive_total + progressive_unsuccessful
    progressive_accuracy = (progressive_total / progressive_attempted * 100.0) if progressive_attempted else 0.0

    to_final_third = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    to_final_third_total = int(to_final_third.sum())
    to_final_third_success = int((to_final_third & df["is_won"]).sum())
    to_final_third_accuracy = (to_final_third_success / to_final_third_total * 100.0) if to_final_third_total else 0.0

    # Long passes (> 25m)
    long_passes = df[df["pass_distance"] > 25.0]
    long_total = len(long_passes)
    long_success = int(long_passes["is_won"].sum())
    long_acc_pct = (long_success / long_total * 100.0) if long_total > 0 else 0.0

    # Dangerous Zone Passes (Last column OR center of the 5th column)
    dz_mask = df["is_won"] & (
        (df["x_end"] >= 100.0) |
        ((df["x_end"] >= 80.0) & (df["x_end"] < 100.0) & (df["y_end"] >= LANE_RIGHT_MAX) & (df["y_end"] < LANE_LEFT_MIN))
    )
    dz_passes = int(dz_mask.sum())

    fwd = int(df["is_forward"].sum())
    bwd = int(df["is_backward"].sum())
    lat = int(df["is_lateral"].sum())

    pos_count = int((df["is_won"] & (df["delta_xt_adj"] > 0)).sum())
    pos_pct = (pos_count / total * 100.0) if total > 0 else 0.0

    high_xt = int((df["delta_xt_adj"] > 0.1).sum())
    sum_dxt = float(df.loc[df["is_won"], "delta_xt_adj"].sum())
    
    # Negative Pass Impact (passes that lost threat)
    neg_xt = float(df.loc[df["is_won"] & (df["delta_xt_adj"] < 0), "delta_xt_adj"].sum())

    return {
        "total_passes": total,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "progressive_attempted": progressive_attempted,
        "progressive_successful": progressive_total,
        "progressive_accuracy_pct": round(progressive_accuracy, 2),
        "to_final_third_total": to_final_third_total,
        "to_final_third_success": to_final_third_success,
        "to_final_third_accuracy_pct": round(to_final_third_accuracy, 2),
        "fwd": fwd, "fwd_pct": round(fwd / total * 100.0, 1),
        "bwd": bwd, "bwd_pct": round(bwd / total * 100.0, 1),
        "lat": lat, "lat_pct": round(lat / total * 100.0, 1),
        "pos_count": pos_count,
        "pos_pct": round(pos_pct, 1),
        "high_xt_pct": round(high_xt / total * 100.0, 1),
        "sum_dxt": round(sum_dxt, 3),
        "total_p90": round(total * p90_factor, 1),
        "prog_p90": round(progressive_total * p90_factor, 2),
        "f3_p90": round(to_final_third_success * p90_factor, 2),
        "xt_p90": round(sum_dxt * p90_factor, 3),
        "neg_xt_p90": round(neg_xt * p90_factor, 3),
        "minutes": mins,
        "long_acc_pct": round(long_acc_pct, 1),
        "high_xt_p90": round(high_xt * p90_factor, 2),
        "dz_p90": round(dz_passes * p90_factor, 2)
    }

def compute_match_scores(dfs_dict):
    records = []
    for m_name, df_m in dfs_dict.items():
        s = compute_stats(df_m, m_name)
        total_passes = s['total_passes']
        if total_passes == 0: continue
        records.append({
            'match': m_name,
            'xt_p90': s['xt_p90'],
            'prog_p90': s['prog_p90'],
            'f3_p90': s['f3_p90'],
            'pos_pct': s['pos_pct'],
            'total_p90': s['total_p90'],
            'neg_xt_p90': s['neg_xt_p90'],
            'accuracy_pct': s['accuracy_pct'],
            'long_acc_pct': s['long_acc_pct'],
            'high_xt_p90': s['high_xt_p90'],
            'dz_p90': s['dz_p90'],
            'prog_acc_pct': s['progressive_accuracy_pct']
        })
    df_scores = pd.DataFrame(records)
    if df_scores.empty: return df_scores

    # Absolute scale normalization with fixed floor and ceiling
    def normalize_fixed(series, val_min, val_max):
        clipped_series = series.clip(lower=val_min, upper=val_max)
        if val_max == val_min: 
            return pd.Series([70.0] * len(series))
        return 40 + ((clipped_series - val_min) / (val_max - val_min)) * 60

    # Apply absolute limits
    df_scores['xt_norm'] = normalize_fixed(df_scores['xt_p90'], val_min=0.05, val_max=0.45)
    df_scores['prog_norm'] = normalize_fixed(df_scores['prog_p90'], val_min=1.0, val_max=15.0)
    df_scores['f3_norm'] = normalize_fixed(df_scores['f3_p90'], val_min=1.0, val_max=15.0)
    df_scores['pos_pct_norm'] = normalize_fixed(df_scores['pos_pct'], val_min=25.0, val_max=75.0)
    df_scores['total_p90_norm'] = normalize_fixed(df_scores['total_p90'], val_min=10.0, val_max=85.0)
    df_scores['neg_xt_norm'] = normalize_fixed(df_scores['neg_xt_p90'], val_min=-0.15, val_max=0.00)

    # Weights
    df_scores['Grade'] = (df_scores['xt_norm'] * 0.30) + \
                         (df_scores['prog_norm'] * 0.20) + \
                         (df_scores['f3_norm'] * 0.20) + \
                         (df_scores['pos_pct_norm'] * 0.10) + \
                         (df_scores['total_p90_norm'] * 0.10) + \
                         (df_scores['neg_xt_norm'] * 0.10)
    df_scores['Grade'] = df_scores['Grade'].round(1)
    return df_scores

#
# UI HELPERS
#
def _safe_pct_diff(a: float, b: float) -> float:
    base = max(abs(b), 1.0)
    pct = (abs(a - b) / base) * 100.0
    return min(pct, 999.0)

def _arrow_html(val_game: float, val_avg: float) -> str:
    if np.isclose(val_game, val_avg, atol=1e-9): return ""
    if abs(val_game) < 1 and abs(val_avg) < 1: return ""
    if val_game > val_avg:
        pct = _safe_pct_diff(val_game, val_avg)
        return f'<span class="metric-arrow arrow-up">↑ {pct:.0f}%</span>'
    else:
        pct = _safe_pct_diff(val_avg, val_game)
        return f'<span class="metric-arrow arrow-down">↓ {pct:.0f}%</span>'

def cmp_box(label, val_game, val_avg, disp_game=None, disp_avg=None, sub_game="", border="#3b82f6"):
    disp_game = str(val_game) if disp_game is None else disp_game
    disp_avg = str(val_avg) if disp_avg is None else disp_avg
    arrow = _arrow_html(float(val_game), float(val_avg))
    sub_game_html = f'<span>{sub_game}</span>' if sub_game else "<span></span>"
    
    html = (
        f'<div class="metric-box" style="border-left-color: {border};">'
        f'<div class="metric-title">{label}</div>'
        f'<div class="metric-value-row">'
        f'<span class="metric-value">{disp_game}</span>'
        f'{arrow}'
        f'</div>'
        f'<div class="metric-sub">'
        f'{sub_game_html}'
        f'<span>AVG: {disp_avg}</span>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def summary_box(label, value, sub_value="", border="#3b82f6"):
    """A simplified metric box for overall stats without comparison arrows"""
    sub_html = f'<div style="font-size: 12px; color: #64748b; margin-top: 4px;">{sub_value}</div>' if sub_value else ""
    html = (
        f'<div class="metric-box" style="border-left-color: {border};">'
        f'<div class="metric-title">{label}</div>'
        f'<div class="metric-value-row">'
        f'<span class="metric-value">{value}</span>'
        f'</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def row_label(text, cls="row-label-blue"):
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

#
# DRAW HELPERS (PITCH)
#
def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg)
    fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#ffffff", lw=1.2, alpha=0.40, linestyle="--")
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.7, alpha=0.12, linestyle="--")
    return fig, ax, pitch

def _attack_arrow(fig, has_cbar=False):
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(FancyArrowPatch(
        (0.44 + ox, 0.045), (0.56 + ox, 0.045),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11, linewidth=1.6, color="#aaaaaa"
    ))
    fig.text(0.50 + ox, 0.012, "Attacking Direction", ha="center", va="bottom", transform=fig.transFigure, fontsize=7.5, color="#aaaaaa")

def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

def draw_pass_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_prog = bool(row["progressive"])
        if is_lost: color, alpha = COLOR_FAIL, 0.72
        elif is_prog: color, alpha = COLOR_PROGRESSIVE, 0.88
        else: color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS
        
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.3, headwidth=2.0, headlength=2.0, ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row["x_start"], row["y_start"], s=32, marker="o", color=color,
                      edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=alpha)

    leg = ax.legend(handles=[
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
    ], loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, facecolor="#1a1a2e", edgecolor="#444466", fontsize=6.5, labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90)

    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_corridor_heatmap(df):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left": (LANE_LEFT_MIN, FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right": (0.0, LANE_RIGHT_MAX),
    }
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            arr[i] = int(((df_s["x_end"] >= x0_) & (df_s["x_end"] < x1_) &
                          (df_s["y_end"] >= y0) & (df_s["y_end"] < y1)).sum())
        counts[cname] = arr

    all_vals = np.concatenate([counts[c] for c in counts])
    vmax = max(1, int(all_vals.max()))
    cmap = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)

    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            value = counts[cname][i]
            ax.add_patch(Rectangle((x0_, y0), x1_ - x0_, y1 - y0,
                                   facecolor=cmap(norm(value)), edgecolor=(1, 1, 1, 0.12), lw=0.5, alpha=0.95, zorder=2))
            ax.text((x0_ + x1_) / 2, (y0 + y1) / 2, str(value),
                    ha="center", va="center", color="#000000" if value <= threshold else "#ffffff",
                    fontsize=9, fontweight="700" if value >= vmax * 0.5 else "600", zorder=4)

    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)

    _attack_arrow(fig)
    return _save_fig(fig), fig

def _draw_comet_arrow(ax, x0, y0, x1, y1, color):
    segs = 12
    ts = np.linspace(0.0, 1.0, segs + 1)
    for i in range(segs):
        t0, t1 = ts[i], ts[i + 1]
        xa, ya = x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0
        xb, yb = x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1
        alpha = 0.85 * (0.15 + 0.85 * t1)
        lw = 2.5 * (0.80 + 0.20 * t1)
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=4, solid_capstyle="round")
    ax.scatter(x0, y0, s=20, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, zorder=5, alpha=0.85)
    ax.scatter(x1, y1, s=32, marker="o", facecolors=color, edgecolors="white", linewidths=0.9, zorder=6, alpha=0.85)

def draw_top5_xt_map(df):
    fig, ax, pitch = _base_pitch()
    top5 = (df[(df["is_won"]) & (df["delta_xt_adj"] > 0)]
            .sort_values("delta_xt_adj", ascending=False)
            .head(5).copy().reset_index(drop=True))

    if not top5.empty:
        for _, row in top5.iterrows():
            val = float(row["delta_xt_adj"])
            color = CMAP_TOP10(NORM_TOP10(np.clip(val, 0.05, 0.40)))
            _draw_comet_arrow(ax, float(row["x_start"]), float(row["y_start"]), float(row["x_end"]), float(row["y_end"]), color)

        sm = plt.cm.ScalarMappable(cmap=CMAP_TOP10, norm=NORM_TOP10)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
        cbar.set_label("Pass Impact", color="#ffffff", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")

    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

#
# PLOTLY CHARTS
#
def draw_total_passes_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["total_p90"]
    mean_val = y.mean()

    fig.add_trace(go.Scatter(
        x=x_labels, y=y, customdata=df_scores["match"], mode='lines+markers',
        line=dict(color="#00d2ff", width=3, shape='spline'),
        marker=dict(size=8, color="#00d2ff"),
        fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.05)',
        name="Total Passes", hovertemplate="<b>%{customdata}</b><br>Total Passes: %{y:.1f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_val]*len(x_labels), mode='lines',
        line=dict(color="#ffd700", width=1.5, dash='dash'),
        name=f"Avg: {mean_val:.1f}", hoverinfo='skip'
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Total Passes", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_grade_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["Grade"]
    mean_grade = y.mean()

    # Metrics for Tooltip Feature
    metrics = {
        'Σ Pass Impact': ('xt_p90', df_scores['xt_p90'].mean()),
        'Prog Passes': ('prog_p90', df_scores['prog_p90'].mean()),
        'Final 3rd': ('f3_p90', df_scores['f3_p90'].mean()),
        '% Pos Impact': ('pos_pct', df_scores['pos_pct'].mean()),
        'Total Passes': ('total_p90', df_scores['total_p90'].mean())
    }
    
    hover_texts = []
    for _, row in df_scores.iterrows():
        diffs = {}
        for name, (col, avg) in metrics.items():
            if avg == 0:
                diffs[name] = 0
            else:
                diffs[name] = ((row[col] - avg) / avg) * 100
        
        if row['Grade'] >= mean_grade:
            best_metric = max(diffs, key=diffs.get)
            val = diffs[best_metric]
            hover_texts.append(f"<br><span style='font-size:11px; color:#94a3b8'>Highlight:</span> <b>{best_metric}</b> <span style='color:#10b981'>(+{val:.1f}%)</span>")
        else:
            worst_metric = min(diffs, key=diffs.get)
            val = diffs[worst_metric]
            hover_texts.append(f"<br><span style='font-size:11px; color:#94a3b8'>Issue:</span> <b>{worst_metric}</b> <span style='color:#ef4444'>({val:.1f}%)</span>")
            
    customdata = np.stack((df_scores["match"], hover_texts), axis=-1)

    fig.add_trace(go.Scatter(
        x=x_labels, y=y, customdata=customdata, mode='lines+markers',
        line=dict(color="#2F80ED", width=3, shape='spline'),
        marker=dict(size=8, color="#2F80ED"),
        fill='tozeroy', fillcolor='rgba(47, 128, 237, 0.05)',
        name="Grade", hovertemplate="<b>%{customdata[0]}</b><br>Grade: %{y:.1f}%{customdata[1]}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_grade]*len(x_labels), mode='lines',
        line=dict(color="#ffd700", width=1.5, dash='dash'),
        name=f"Avg: {mean_grade:.1f}", hoverinfo='skip'
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=400, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[40, 100], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Grade Evolution", font=dict(size=14, color="#ffffff"))
    )
    return fig

def draw_progressive_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["prog_p90"]
    mean_prog = y.mean()

    fig.add_trace(go.Scatter(
        x=x_labels, y=y, customdata=df_scores["match"], mode='lines+markers',
        line=dict(color="#10b981", width=3, shape='spline'),
        marker=dict(size=8, color="#10b981"),
        fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.05)',
        name="Progressive Passes", hovertemplate="<b>%{customdata}</b><br>Progressive: %{y:.1f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_prog]*len(x_labels), mode='lines',
        line=dict(color="#ffd700", width=1.5, dash='dash'),
        name=f"Avg: {mean_prog:.1f}", hoverinfo='skip'
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Progressive Passes", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_final_third_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["f3_p90"]
    mean_f3 = y.mean()

    fig.add_trace(go.Scatter(
        x=x_labels, y=y, customdata=df_scores["match"], mode='lines+markers',
        line=dict(color="#8b5cf6", width=3, shape='spline'),
        marker=dict(size=8, color="#8b5cf6"),
        fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.05)',
        name="Final Third Passes", hovertemplate="<b>%{customdata}</b><br>Final Third: %{y:.1f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_f3]*len(x_labels), mode='lines',
        line=dict(color="#ffd700", width=1.5, dash='dash'),
        name=f"Avg: {mean_f3:.1f}", hoverinfo='skip'
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Final Third Passes", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_xt_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["xt_p90"]
    mean_xt = y.mean()

    fig.add_trace(go.Scatter(
        x=x_labels, y=y, customdata=df_scores["match"], mode='lines+markers',
        line=dict(color="#f59e0b", width=3, shape='spline'),
        marker=dict(size=8, color="#f59e0b"),
        fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.05)',
        name="Σ Pass Impact", hovertemplate="<b>%{customdata}</b><br>Σ Pass Impact: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_xt]*len(x_labels), mode='lines',
        line=dict(color="#ffd700", width=1.5, dash='dash'),
        name=f"Avg: {mean_xt:.2f}", hoverinfo='skip'
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Σ Pass Impact", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_comparison_bar(title, val_first, val_last, suffix=""):
    """Helper function to draw clean comparison bars for Evolution tab"""
    color_last = "#10b981" if val_last >= val_first else "#E07070"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["First 9 Matches", "Last 9 Matches"],
        y=[val_first, val_last],
        marker_color=["#444466", color_last],
        text=[f"{val_first:.2f}{suffix}", f"{val_last:.2f}{suffix}"],
        textposition='auto',
        width=[0.35, 0.35],
        hovertemplate="%{x}<br>" + title + ": %{y:.2f}" + suffix + "<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=250, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, max(val_first, val_last) * 1.2], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        title=dict(text=title, font=dict(size=14, color="#a0a0b5")),
        showlegend=False,
        bargap=0.2
    )
    return fig

#
# SIDEBAR
#
st.sidebar.title("Pass Dashboard")
st.sidebar.markdown("### 2026 Matches")
st.sidebar.markdown("#### Hudson Cicala")

# Add image if exists
img_path = "Captura de tela 2026-06-02 154425.png"
if os.path.exists(img_path):
    st.sidebar.image(img_path, use_container_width=True)

st.sidebar.markdown("---")

# Calculate averages across all loaded matches globally
num_matches = len(dfs_by_match)
all_match_stats = [compute_stats(dfs_by_match[m], m) for m in dfs_by_match]

#
# TABS & LAYOUT
#
tab_graf, tab_dash, tab_evo = st.tabs(["Charts & Analysis", "Detailed Dashboard", "Evolution"])

with tab_graf:
    st.markdown("### Overall Performance Summary")
    
    if num_matches > 0:
        total_passes_all = sum(s['total_passes'] for s in all_match_stats)
        total_succ_all = sum(s['successful_passes'] for s in all_match_stats)
        total_prog_all = sum(s['progressive_successful'] for s in all_match_stats)
        total_f3_all = sum(s['to_final_third_success'] for s in all_match_stats)
        total_pos_all = sum(s['pos_count'] for s in all_match_stats)
        total_xt_all = sum(s['sum_dxt'] for s in all_match_stats)

        avg_acc = sum(s['accuracy_pct'] for s in all_match_stats) / num_matches
        avg_prog_p90 = sum(s['prog_p90'] for s in all_match_stats) / num_matches
        avg_f3_p90 = sum(s['f3_p90'] for s in all_match_stats) / num_matches
        avg_pos_pct = sum(s['pos_pct'] for s in all_match_stats) / num_matches
        avg_xt_p90 = sum(s['xt_p90'] for s in all_match_stats) / num_matches
        avg_total_p90 = sum(s['total_p90'] for s in all_match_stats) / num_matches
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            row_label("📋 Passes (Avg p90)", "row-label-blue")
            summary_box("Passes p90", f"{avg_total_p90:.1f}", f"Total: {total_passes_all}", border=C_BLUE)
            summary_box("Successful %", f"{avg_acc:.1f}%", f"Total: {total_succ_all}", border=C_BLUE)
        with col_s2:
            row_label("📊 Advanced (Avg p90)", "row-label-green")
            summary_box("Progressive p90", f"{avg_prog_p90:.1f}", f"Total: {total_prog_all}", border=C_GREEN)
            summary_box("Final Third p90", f"{avg_f3_p90:.1f}", f"Total: {total_f3_all}", border=C_GREEN)
        with col_s3:
            row_label("⚡ Pass Impact (Avg p90)", "row-label-amber")
            summary_box("% Positive Impact", f"{avg_pos_pct:.1f}%", f"Total: {total_pos_all}", border=C_AMBER)
            summary_box("Σ Pass Impact", f"{avg_xt_p90:.3f}", f"Total: {total_xt_all:.3f}", border=C_AMBER)
            
        st.markdown(f"<div style='text-align: right; color: #94a3b8; font-size: 14px; margin-top: 8px;'><b>{num_matches} matches collected</b></div>", unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)

    df_scores = compute_match_scores(dfs_by_match)
    if not df_scores.empty:
        # Chart 1: General Grade (Blue) - FIRST CHART
        fig_scores = draw_grade_chart(df_scores)
        st.plotly_chart(fig_scores, use_container_width=True)

        # Grade explanation expander - NOW BELOW the grade chart, simplified text
        with st.expander("How is the Grade calculated?"):
            st.markdown("""
            **Metrics evaluated:**
            - **Pass Impact:** Measures actual danger created by passes.
            - **Progressive Passes:** Line-breaking ability.
            - **Final Third Passes:** Attacking presence in dangerous zones.
            - **% Positive Pass Impact:** Efficiency of threat generation.
            - **Total Passes:** Overall involvement.
            - **Negative Pass Impact:** Penalty for passes that lose threat.
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Stats")

        # Chart: Total Passes (Cyan)
        fig_total = draw_total_passes_chart(df_scores)
        st.plotly_chart(fig_total, use_container_width=True)

        # Chart: Progressive Passes (Green)
        fig_prog = draw_progressive_chart(df_scores)
        st.plotly_chart(fig_prog, use_container_width=True)

        # Chart: Final Third Passes (Purple)
        fig_f3 = draw_final_third_chart(df_scores)
        st.plotly_chart(fig_f3, use_container_width=True)

        # Chart: Σ Pass Impact (Amber)
        fig_xt = draw_xt_chart(df_scores)
        st.plotly_chart(fig_xt, use_container_width=True)
    else:
        st.warning("Not enough data to generate charts.")

with tab_dash:
    st.markdown("### Match Filters")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        all_match_names = list(dfs_by_match.keys())
        selected_match = st.selectbox("Select Match", options=all_match_names, index=0)
    with col_f2:
        pass_filter = st.radio(
            "Pass Type",
            ["All", "Successful", "Unsuccessful", "Progressive", "Final Third"],
            index=0,
            horizontal=True
        )

    def apply_filter(df):
        if pass_filter == "Successful": return df[df["is_won"]].copy()
        if pass_filter == "Unsuccessful": return df[~df["is_won"]].copy()
        if pass_filter == "Progressive": return df[df["progressive"]].copy()
        if pass_filter == "Final Third": return df[(df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)].copy()
        return df.copy()

    df_game = apply_filter(dfs_by_match[selected_match].copy())
    s_game = compute_stats(df_game, selected_match)

    s_avg = {}
    if num_matches > 0:
        for k in all_match_stats[0].keys():
            if isinstance(all_match_stats[0][k], (int, float)):
                s_avg[k] = sum(s[k] for s in all_match_stats) / num_matches
            else:
                s_avg[k] = 0
    else:
        s_avg = s_game.copy()

    st.markdown("---")

    # --- ROW 1: MAPS ---
    img_pm_game, fig_pm_game = draw_pass_map(df_game); plt.close(fig_pm_game)
    img_ht_game, fig_ht_game = draw_corridor_heatmap(df_game); plt.close(fig_ht_game)
    img_xt_game, fig_xt_game = draw_top5_xt_map(df_game); plt.close(fig_xt_game)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        row_label(f"🟦 Pass Map", "row-label-blue")
        st.image(img_pm_game, use_container_width=True)
    with col_m2:
        row_label(f"🟩 Zone Heatmap", "row-label-green")
        st.image(img_ht_game, use_container_width=True)
    with col_m3:
        row_label(f"🟡 Top 5 Pass Impact", "row-label-amber")
        st.image(img_xt_game, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 2: STATS (2 Metrics per Column) ---
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        row_label("📋 Pass Overview", "row-label-blue")
        cmp_box("Total Passes", s_game["total_p90"], f"{s_avg['total_p90']:.1f}", border=C_BLUE)
        cmp_box("Successful %", s_game["accuracy_pct"], s_avg["accuracy_pct"], 
                disp_game=f"{s_game['accuracy_pct']:.1f}%", disp_avg=f"{s_avg['accuracy_pct']:.1f}%", border=C_BLUE)

    with col_s2:
        row_label("📊 Advanced", "row-label-green")
        cmp_box("Progressive", s_game["prog_p90"], f"{s_avg['prog_p90']:.1f}", border=C_GREEN)
        cmp_box("Final Third", s_game["f3_p90"], f"{s_avg['f3_p90']:.1f}", border=C_GREEN)

    with col_s3:
        row_label("⚡ Pass Impact", "row-label-amber")
        cmp_box("% Positive Impact", s_game["pos_pct"], s_avg["pos_pct"], 
                disp_game=f"{s_game['pos_pct']:.1f}%", disp_avg=f"{s_avg['pos_pct']:.1f}%", border=C_AMBER)
        cmp_box("Σ Pass Impact", s_game["xt_p90"], f"{s_avg['xt_p90']:.3f}", 
                disp_game=f"{s_game['xt_p90']:.3f}", disp_avg=f"{s_avg['xt_p90']:.3f}", border=C_AMBER)

with tab_evo:
    st.markdown("### First 9 vs Last 9 Matches")
    st.markdown("Comparing the average performance between the first 9 and the last 9 matches to analyze player evolution.")
    
    df_scores = compute_match_scores(dfs_by_match)
    if len(df_scores) > 0:
        if len(df_scores) < 18:
            st.info(f"Note: Only {len(df_scores)} matches available. The comparison will overlap or use available data.")
        
        first_9 = df_scores.head(9)
        last_9 = df_scores.tail(9)

        # Layout 3x3
        # Row 1
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            fig_grade = draw_comparison_bar("Grade", first_9["Grade"].mean(), last_9["Grade"].mean())
            st.plotly_chart(fig_grade, use_container_width=True)
        with r1c2:
            fig_xt_evo = draw_comparison_bar("Σ Pass Impact", first_9["xt_p90"].mean(), last_9["xt_p90"].mean())
            st.plotly_chart(fig_xt_evo, use_container_width=True)
        with r1c3:
            fig_high_xt_evo = draw_comparison_bar("High Impact Passes", first_9["high_xt_p90"].mean(), last_9["high_xt_p90"].mean())
            st.plotly_chart(fig_high_xt_evo, use_container_width=True)

        # Row 2
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            fig_prog_evo = draw_comparison_bar("Progressive Passes", first_9["prog_p90"].mean(), last_9["prog_p90"].mean())
            st.plotly_chart(fig_prog_evo, use_container_width=True)
        with r2c2:
            fig_f3_evo = draw_comparison_bar("Final Third Passes", first_9["f3_p90"].mean(), last_9["f3_p90"].mean())
            st.plotly_chart(fig_f3_evo, use_container_width=True)
        with r2c3:
            fig_dz_evo = draw_comparison_bar("Dangerous Zone Passes", first_9["dz_p90"].mean(), last_9["dz_p90"].mean())
            st.plotly_chart(fig_dz_evo, use_container_width=True)

        # Row 3
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            fig_acc_evo = draw_comparison_bar("Successful Passes %", first_9["accuracy_pct"].mean(), last_9["accuracy_pct"].mean(), suffix="%")
            st.plotly_chart(fig_acc_evo, use_container_width=True)
        with r3c2:
            fig_long_evo = draw_comparison_bar("Long Pass Accuracy %", first_9["long_acc_pct"].mean(), last_9["long_acc_pct"].mean(), suffix="%")
            st.plotly_chart(fig_long_evo, use_container_width=True)
        with r3c3:
            fig_prog_acc_evo = draw_comparison_bar("Progressive Accuracy %", first_9["prog_acc_pct"].mean(), last_9["prog_acc_pct"].mean(), suffix="%")
            st.plotly_chart(fig_prog_acc_evo, use_container_width=True)
    else:
        st.warning("Not enough data to generate evolution charts.")
