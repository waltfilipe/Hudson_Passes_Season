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

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Generic Player — Dashboard")

# OPTIONAL DOCX IMPORT
DOCX_AVAILABLE = True
try:
    from docx import Document
except Exception:
    DOCX_AVAILABLE = False

# STYLE
st.markdown("""
""", unsafe_allow_html=True)

# CONSTANTS
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
C_BLUE_DARK = "#1a56db"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_PURPLE_LIGHT = "#a78bfa"
C_BLUE_PASTEL = "#5b9bd5"
C_GREEN_PASTEL = "#70ad47"
C_AMBER_PASTEL = "#d4a843"
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)
NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60
LATERAL_MIN_DIST = 12.0
PENALTY_AREA_X = 18.0
FUNNEL_X_EXTEND = 33.0
PENALTY_AREA_Y_MIN = 18.0
PENALTY_AREA_Y_MAX = 62.0

def _hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

def get_lane(y):
    if y >= LANE_LEFT_MIN:
        return "left"
    elif y < LANE_RIGHT_MAX:
        return "right"
    return "center"

def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)

def is_progressive_pass(x_start, y_start, x_end, y_end):
    if x_start < 35:
        return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist == 0:
        return False
    return ((start_dist - end_dist) / start_dist) >= 0.25

def classify_pass_direction(x_start, y_start, x_end, y_end):
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:
        return "forward"
    if angle_deg >= 135.0:
        return "backward"
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
            XTc[iy, ix] = XT[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)
    return XTc

XT_GRID = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX_XT, 0, NX_XT - 1))
    iy = int(np.clip((y / FIELD_Y) * NY_XT, 0, NY_XT - 1))
    return float(XT_GRID[iy, ix])

def is_in_funnel_zone(x, y):
    """Check if a defensive action is in the penalty area + 15m extended zone."""
    return x <= FUNNEL_X_EXTEND and PENALTY_AREA_Y_MIN <= y <= PENALTY_AREA_Y_MAX

# BASE PASSES
BASE_MATCHES_DATA = {
    "Opponent 01 (03-05)": [
        ("PASS WON", 54.42, 80.0, 84.65, 67.57, None),
        ("PASS WON", 81.41, 44.95, 100.93, 51.68, None),
        ("PASS WON", 37.89, 39.13, 21.25, 47.77, None),
        ("PASS WON", 46.77, 19.84, 76.0, 23.74, None),
        ("PASS WON", 39.92, 60.84, 35.04, 73.46, None),
        ("PASS WON", 44.05, 45.1, 31.42, 44.03, None),
        ("PASS WON", 34.33, 31.32, 52.04, 25.13, None),
        ("PASS WON", 10.99, 67.01, 4.41, 69.53, None),
        ("PASS WON", 70.49, 47.26, 82.33, 50.94, None),
        ("PASS WON", 42.09, 4.93, 65.6, 18.35, None),
        ("PASS WON", 30.21, 49.29, 42.78, 80.0, None),
        ("PASS WON", 37.99, 29.21, 17.61, 36.25, None),
        ("PASS WON", 82.54, 42.99, 73.54, 34.25, None),
        ("PASS WON", 41.02, 78.55, 35.18, 78.33, None),
        ("PASS WON", 72.53, 30.85, 78.7, 43.03, None),
        ("PASS WON", 86.43, 51.03, 83.64, 40.87, None),
        ("PASS WON", 29.75, 41.68, 47.64, 35.8, None),
        ("PASS WON", 28.78, 47.17, 48.66, 21.37, None),
        ("PASS WON", 10.16, 29.5, 24.71, 17.32, None),
        ("PASS WON", 63.68, 40.46, 64.19, 14.41, None),
        ("PASS WON", 30.95, 42.91, 73.14, 33.85, None),
        ("PASS WON", 68.79, 25.9, 87.2, 30.85, None),
        ("PASS WON", 45.08, 45.09, 53.87, 28.62, None),
        ("PASS WON", 51.49, 46.05, 77.63, 46.08, None),
        ("PASS WON", 63.34, 41.56, 75.01, 29.94, None),
        ("PASS WON", 28.39, 21.98, 55.99, 15.54, None),
        ("PASS WON", 46.59, 36.28, 48.91, 55.79, None),
        ("PASS LOST", 73.97, 0.0, 90.58, 0.0, None),
        ("PASS LOST", 79.33, 71.25, 109.14, 56.25, None),
        ("PASS LOST", 102.23, 60.04, 120.0, 20.95, None),
        ("PASS LOST", 64.3, 64.35, 73.11, 62.85, None),
        ("PASS LOST", 83.7, 45.84, 113.96, 51.11, None),
        ("PASS LOST", 37.59, 72.28, 43.13, 71.24, None),
        ("PASS LOST", 55.35, 61.77, 90.65, 54.35, None),
        ("PASS LOST", 72.1, 34.12, 78.51, 40.72, None),
        ("PASS LOST", 34.01, 27.56, 44.57, 34.59, None),
    ],
    "Opponent 02 (03-09)": [
        ("PASS WON", 14.61, 34.39, 2.8, 42.39, None),
        ("PASS WON", 68.06, 74.19, 81.98, 68.35, None),
        ("PASS WON", 17.08, 38.44, 0.0, 34.2, None),
        ("PASS WON", 77.65, 77.62, 76.11, 80.0, None),
        ("PASS WON", 47.36, 49.16, 52.32, 62.99, None),
        ("PASS WON", 52.82, 68.38, 64.37, 52.09, None),
        ("PASS WON", 35.87, 48.58, 38.33, 49.49, None),
        ("PASS WON", 66.34, 10.82, 85.52, 0.0, None),
        ("PASS WON", 61.38, 80.0, 83.92, 64.08, None),
        ("PASS WON", 22.78, 64.11, 43.11, 44.78, None),
        ("PASS WON", 74.89, 40.5, 93.12, 57.63, None),
        ("PASS WON", 30.42, 34.08, 55.34, 50.23, None),
        ("PASS WON", 28.23, 36.1, 39.64, 49.14, None),
        ("PASS WON", 51.82, 70.31, 68.99, 77.76, None),
        ("PASS WON", 67.95, 41.57, 68.18, 16.83, None),
        ("PASS WON", 52.56, 32.38, 50.99, 18.51, None),
        ("PASS WON", 47.98, 80.0, 54.09, 80.0, None),
        ("PASS WON", 54.46, 52.19, 39.85, 70.21, None),
        ("PASS WON", 58.81, 24.6, 73.64, 25.3, None),
        ("PASS WON", 66.16, 46.47, 68.47, 27.18, None),
        ("PASS LOST", 90.5, 37.74, 92.55, 21.36, None),
        ("PASS LOST", 27.23, 14.88, 48.58, 34.48, None),
        ("PASS LOST", 115.07, 39.63, 120.0, 18.79, None),
        ("PASS LOST", 93.59, 35.85, 97.44, 58.89, None),
        ("PASS LOST", 65.99, 46.69, 75.31, 80.0, None),
        ("PASS LOST", 74.98, 35.28, 80.3, 70.57, None),
        ("PASS LOST", 48.97, 40.21, 65.76, 31.55, None),
    ],
    "Opponent 03 (03-13)": [
        ("PASS WON", 69.48, 58.28, 67.12, 39.56, None),
        ("PASS WON", 83.43, 49.73, 84.53, 62.12, None),
        ("PASS WON", 56.54, 65.86, 66.82, 73.99, None),
        ("PASS WON", 23.79, 37.16, 50.87, 27.47, None),
        ("PASS WON", 41.54, 54.26, 52.24, 57.05, None),
        ("PASS WON", 50.79, 67.51, 58.81, 80.0, None),
        ("PASS WON", 82.73, 35.79, 100.53, 18.61, None),
        ("PASS WON", 22.08, 3.69, 20.92, 2.59, None),
        ("PASS WON", 30.27, 20.14, 31.44, 41.49, None),
        ("PASS WON", 115.67, 48.96, 90.8, 56.77, None),
        ("PASS WON", 36.14, 9.07, 70.51, 0.0, None),
        ("PASS WON", 6.55, 77.39, 24.56, 73.87, None),
        ("PASS WON", 64.62, 28.86, 77.34, 43.25, None),
        ("PASS WON", 36.91, 25.86, 23.29, 30.44, None),
        ("PASS WON", 42.59, 38.74, 46.38, 32.08, None),
        ("PASS WON", 97.36, 14.66, 115.99, 0.0, None),
        ("PASS WON", 43.44, 10.96, 48.94, 5.81, None),
        ("PASS LOST", 37.21, 68.23, 51.83, 2.31, None),
        ("PASS LOST", 88.02, 46.61, 90.16, 35.0, None),
        ("PASS LOST", 95.15, 39.6, 93.13, 36.91, None),
        ("PASS LOST", 75.61, 64.98, 92.23, 48.86, None),
        ("PASS LOST", 68.43, 50.81, 88.36, 40.4, None),
        ("PASS LOST", 58.04, 69.06, 84.96, 67.08, None),
    ],
    "Opponent 04 (03-17)": [
        ("PASS WON", 23.81, 7.1, 19.03, 28.78, None),
        ("PASS WON", 62.46, 4.62, 51.23, 0.0, None),
        ("PASS WON", 73.8, 48.21, 84.71, 51.24, None),
        ("PASS WON", 79.26, 64.87, 81.99, 38.68, None),
        ("PASS WON", 59.35, 38.93, 63.15, 57.58, None),
        ("PASS WON", 41.53, 36.16, 59.6, 44.96, None),
        ("PASS WON", 87.73, 68.81, 105.82, 74.34, None),
        ("PASS WON", 51.5, 54.48, 45.42, 57.5, None),
        ("PASS WON", 39.43, 39.84, 33.35, 62.66, None),
        ("PASS WON", 72.6, 35.9, 113.72, 54.21, None),
        ("PASS WON", 68.7, 12.61, 75.86, 36.07, None),
        ("PASS WON", 66.5, 63.55, 64.87, 58.22, None),
        ("PASS WON", 80.81, 27.52, 78.54, 28.12, None),
        ("PASS WON", 63.09, 66.05, 91.46, 39.16, None),
        ("PASS LOST", 73.61, 26.46, 83.42, 19.8, None),
        ("PASS LOST", 44.48, 54.27, 51.12, 60.99, None),
    ],
    "Opponent 05 (03-21)": [
        ("PASS WON", 67.93, 26.78, 88.09, 44.56, None),
        ("PASS WON", 60.8, 80.0, 63.63, 76.92, None),
        ("PASS WON", 48.43, 66.32, 55.21, 68.27, None),
        ("PASS WON", 48.11, 15.71, 46.94, 36.34, None),
        ("PASS WON", 97.58, 24.63, 120.0, 0.0, None),
        ("PASS WON", 41.31, 23.15, 53.33, 37.58, None),
        ("PASS WON", 30.88, 80.0, 39.58, 77.7, None),
        ("PASS WON", 39.36, 39.48, 38.95, 59.36, None),
        ("PASS WON", 22.18, 62.74, 4.22, 67.79, None),
        ("PASS WON", 51.78, 49.69, 62.02, 57.67, None),
        ("PASS WON", 65.4, 32.31, 83.25, 33.54, None),
        ("PASS WON", 37.92, 49.29, 43.32, 44.55, None),
        ("PASS WON", 45.31, 21.72, 60.12, 22.24, None),
        ("PASS WON", 47.54, 57.34, 39.26, 62.31, None),
        ("PASS WON", 64.04, 60.12, 54.36, 49.72, None),
        ("PASS WON", 43.19, 21.76, 60.24, 29.75, None),
        ("PASS WON", 42.68, 51.2, 47.84, 3.62, None),
        ("PASS WON", 103.2, 41.8, 105.71, 29.56, None),
        ("PASS LOST", 46.56, 67.36, 84.44, 44.33, None),
        ("PASS LOST", 85.18, 50.31, 107.06, 37.47, None),
        ("PASS LOST", 64.56, 31.69, 90.31, 16.47, None),
        ("PASS LOST", 67.62, 51.54, 106.2, 62.91, None),
        ("PASS LOST", 53.85, 19.32, 67.02, 39.03, None),
    ],
    "Opponent 06 (03-25)": [
        ("PASS WON", 84.23, 40.7, 74.17, 42.29, None),
        ("PASS WON", 49.34, 51.21, 36.84, 40.34, None),
        ("PASS WON", 58.32, 0.0, 82.76, 0.0, None),
        ("PASS WON", 76.01, 24.22, 117.37, 4.09, None),
        ("PASS WON", 71.07, 47.68, 57.0, 61.26, None),
        ("PASS WON", 65.62, 54.83, 89.09, 33.66, None),
        ("PASS WON", 52.5, 34.4, 50.37, 5.98, None),
        ("PASS WON", 34.08, 66.91, 22.6, 61.66, None),
        ("PASS WON", 66.23, 70.35, 73.13, 61.49, None),
        ("PASS WON", 79.29, 73.09, 93.45, 53.21, None),
        ("PASS WON", 69.1, 7.05, 76.02, 20.3, None),
        ("PASS WON", 60.11, 35.08, 88.55, 29.43, None),
        ("PASS WON", 64.26, 53.75, 65.66, 28.75, None),
        ("PASS WON", 45.31, 26.3, 33.74, 1.68, None),
        ("PASS WON", 72.94, 40.53, 54.85, 37.47, None),
        ("PASS WON", 82.83, 61.5, 68.58, 80.0, None),
        ("PASS WON", 70.98, 63.89, 75.56, 64.9, None),
        ("PASS WON", 29.43, 75.74, 30.81, 47.79, None),
        ("PASS LOST", 65.05, 49.2, 87.19, 43.47, None),
        ("PASS LOST", 65.03, 33.59, 84.18, 60.04, None),
        ("PASS LOST", 46.82, 45.66, 80.17, 24.74, None),
    ],
    "Opponent 07 (03-01)": [
        ("PASS WON", 38.88, 0.0, 38.25, 0.0, None),
        ("PASS WON", 83.46, 36.22, 115.69, 54.82, None),
        ("PASS WON", 54.25, 35.1, 81.37, 47.14, None),
        ("PASS WON", 45.71, 36.85, 63.46, 57.07, None),
        ("PASS WON", 74.91, 80.0, 87.13, 79.58, None),
        ("PASS WON", 86.27, 39.76, 114.93, 25.18, None),
        ("PASS WON", 50.15, 63.55, 74.9, 64.71, None),
        ("PASS WON", 86.92, 19.97, 88.86, 20.62, None),
        ("PASS WON", 62.58, 20.33, 85.66, 16.98, None),
        ("PASS WON", 32.1, 28.1, 58.65, 18.05, None),
        ("PASS WON", 47.22, 58.61, 50.23, 42.42, None),
        ("PASS WON", 31.68, 75.81, 24.9, 72.68, None),
        ("PASS WON", 69.42, 54.31, 95.31, 59.27, None),
        ("PASS WON", 99.99, 64.33, 97.19, 58.1, None),
        ("PASS WON", 37.43, 24.28, 32.13, 29.16, None),
        ("PASS WON", 101.58, 50.88, 114.65, 56.52, None),
        ("PASS WON", 77.81, 58.5, 85.73, 80.0, None),
        ("PASS WON", 54.94, 37.32, 58.68, 32.49, None),
        ("PASS WON", 101.52, 35.23, 108.24, 21.82, None),
        ("PASS WON", 35.92, 48.22, 60.78, 33.4, None),
        ("PASS WON", 33.61, 59.12, 50.47, 45.54, None),
        ("PASS WON", 25.46, 46.87, 13.86, 57.18, None),
        ("PASS WON", 30.02, 30.91, 36.67, 31.22, None),
        ("PASS WON", 75.42, 38.37, 98.64, 42.17, None),
        ("PASS WON", 65.39, 56.46, 64.23, 54.77, None),
        ("PASS WON", 22.38, 58.57, 48.8, 62.14, None),
        ("PASS WON", 0.99, 38.8, 18.96, 49.45, None),
        ("PASS WON", 56.81, 52.78, 72.19, 67.08, None),
        ("PASS WON", 62.97, 51.07, 85.1, 50.05, None),
        ("PASS WON", 69.55, 45.48, 76.69, 74.11, None),
        ("PASS WON", 7.06, 38.18, 22.85, 43.09, None),
        ("PASS WON", 68.12, 37.59, 89.95, 13.44, None),
        ("PASS WON", 11.42, 28.64, 17.09, 27.31, None),
        ("PASS WON", 114.16, 31.16, 114.55, 2.37, None),
        ("PASS WON", 3.95, 40.47, 39.96, 31.26, None),
        ("PASS WON", 86.23, 39.1, 99.2, 38.2, None),
        ("PASS WON", 81.87, 60.74, 90.58, 62.64, None),
        ("PASS WON", 50.15, 40.97, 63.6, 38.17, None),
        ("PASS WON", 46.57, 21.04, 40.58, 15.38, None),
        ("PASS LOST", 35.89, 39.15, 43.5, 57.05, None),
        ("PASS LOST", 44.26, 48.34, 57.69, 53.46, None),
        ("PASS LOST", 57.28, 33.83, 59.82, 26.08, None),
        ("PASS LOST", 60.43, 58.81, 81.44, 65.73, None),
        ("PASS LOST", 41.56, 72.54, 85.51, 59.39, None),
        ("PASS LOST", 74.52, 8.41, 85.1, 0.0, None),
    ],
    "Opponent 08 (03-05)": [
        ("PASS WON", 43.57, 41.62, 54.79, 35.24, None),
        ("PASS WON", 45.05, 18.32, 48.41, 5.53, None),
        ("PASS WON", 63.58, 42.06, 71.47, 39.06, None),
        ("PASS WON", 54.84, 54.91, 86.24, 28.26, None),
        ("PASS WON", 31.12, 30.31, 48.3, 29.93, None),
        ("PASS WON", 54.93, 42.1, 92.7, 48.52, None),
        ("PASS WON", 65.99, 64.76, 90.73, 78.31, None),
        ("PASS WON", 55.04, 33.22, 93.02, 22.74, None),
        ("PASS WON", 46.85, 46.02, 71.76, 46.51, None),
        ("PASS WON", 86.32, 80.0, 112.98, 80.0, None),
        ("PASS WON", 75.43, 38.76, 74.49, 26.47, None),
        ("PASS WON", 66.42, 54.93, 64.7, 80.0, None),
        ("PASS WON", 85.83, 4.19, 81.77, 10.7, None),
        ("PASS WON", 72.2, 38.84, 64.1, 46.27, None),
        ("PASS WON", 12.17, 52.55, 23.67, 45.81, None),
        ("PASS WON", 40.78, 40.99, 60.62, 40.06, None),
        ("PASS WON", 120.0, 36.29, 120.0, 60.55, None),
        ("PASS WON", 36.98, 76.93, 46.84, 80.0, None),
        ("PASS WON", 41.08, 66.33, 54.39, 80.0, None),
        ("PASS WON", 78.27, 31.94, 79.94, 40.62, None),
        ("PASS WON", 71.03, 28.6, 85.0, 40.71, None),
        ("PASS WON", 51.12, 7.84, 52.28, 37.34, None),
        ("PASS WON", 49.53, 35.94, 76.96, 34.04, None),
        ("PASS WON", 42.27, 20.04, 49.5, 0.0, None),
        ("PASS WON", 26.85, 42.77, 40.65, 32.97, None),
        ("PASS WON", 38.57, 6.46, 70.84, 0.0, None),
        ("PASS WON", 15.56, 42.55, 33.31, 23.27, None),
        ("PASS LOST", 31.97, 80.0, 47.69, 79.52, None),
        ("PASS LOST", 51.89, 41.82, 47.7, 74.5, None),
        ("PASS LOST", 66.44, 71.49, 83.21, 35.05, None),
        ("PASS LOST", 79.34, 53.67, 59.31, 80.0, None),
        ("PASS LOST", 80.13, 65.04, 60.09, 50.71, None),
        ("PASS LOST", 83.3, 58.5, 91.64, 76.27, None),
        ("PASS LOST", 70.89, 38.83, 88.46, 40.22, None),
        ("PASS LOST", 98.67, 23.49, 102.63, 28.01, None),
        ("PASS LOST", 37.34, 15.61, 63.54, 0.0, None),
    ],
    "Opponent 09 (03-09)": [
        ("PASS WON", 69.38, 58.01, 86.99, 72.48, None),
        ("PASS WON", 61.17, 37.76, 93.33, 37.4, None),
        ("PASS WON", 7.67, 67.41, 33.29, 80.0, None),
        ("PASS WON", 0.0, 73.72, 4.68, 75.76, None),
        ("PASS WON", 18.94, 50.07, 41.25, 49.83, None),
        ("PASS WON", 47.53, 48.29, 52.87, 29.32, None),
        ("PASS WON", 36.83, 53.47, 58.32, 80.0, None),
        ("PASS WON", 23.86, 23.74, 31.67, 52.48, None),
        ("PASS WON", 62.58, 14.96, 57.23, 35.03, None),
        ("PASS WON", 28.62, 60.12, 21.71, 49.69, None),
        ("PASS WON", 36.7, 80.0, 53.21, 69.75, None),
        ("PASS WON", 53.43, 17.78, 68.1, 32.93, None),
        ("PASS WON", 84.49, 50.82, 82.88, 48.56, None),
        ("PASS WON", 51.79, 4.77, 68.67, 40.62, None),
        ("PASS WON", 51.7, 23.61, 50.87, 38.66, None),
        ("PASS WON", 73.17, 39.94, 88.97, 31.27, None),
        ("PASS WON", 80.91, 31.58, 83.96, 43.52, None),
        ("PASS WON", 82.58, 31.44, 84.54, 17.6, None),
        ("PASS WON", 33.23, 55.19, 37.83, 60.52, None),
        ("PASS WON", 26.45, 71.74, 17.67, 80.0, None),
        ("PASS WON", 30.36, 31.88, 25.53, 52.91, None),
        ("PASS WON", 39.55, 57.17, 48.45, 75.47, None),
        ("PASS WON", 39.27, 62.25, 27.59, 42.96, None),
        ("PASS WON", 38.65, 80.0, 49.38, 72.34, None),
        ("PASS WON", 46.66, 58.61, 47.62, 80.0, None),
        ("PASS WON", 54.63, 43.71, 64.67, 50.02, None),
        ("PASS WON", 70.39, 73.13, 109.79, 56.97, None),
        ("PASS WON", 45.05, 42.7, 42.14, 30.17, None),
        ("PASS WON", 41.55, 33.89, 61.08, 44.35, None),
        ("PASS WON", 113.08, 80.0, 120.0, 64.43, None),
        ("PASS LOST", 97.44, 56.64, 109.64, 62.95, None),
        ("PASS LOST", 88.13, 40.21, 107.6, 25.53, None),
        ("PASS LOST", 97.3, 33.03, 85.13, 48.05, None),
        ("PASS LOST", 58.83, 50.82, 85.2, 49.81, None),
        ("PASS LOST", 61.24, 48.81, 56.03, 33.8, None),
        ("PASS LOST", 42.61, 55.29, 54.92, 80.0, None),
        ("PASS LOST", 70.73, 24.69, 87.7, 2.8, None),
    ],
    "Opponent 10 (03-13)": [
        ("PASS WON", 76.35, 75.27, 77.77, 80.0, None),
        ("PASS WON", 62.58, 63.52, 50.01, 80.0, None),
        ("PASS WON", 56.95, 42.78, 60.88, 39.5, None),
        ("PASS WON", 41.31, 40.74, 49.89, 54.35, None),
        ("PASS WON", 17.08, 30.78, 34.4, 35.85, None),
        ("PASS WON", 18.9, 32.42, 33.32, 25.26, None),
        ("PASS WON", 60.76, 59.31, 76.98, 67.63, None),
        ("PASS WON", 42.37, 29.86, 54.48, 39.12, None),
        ("PASS WON", 49.89, 50.64, 54.41, 44.0, None),
        ("PASS WON", 94.33, 43.36, 104.56, 38.51, None),
        ("PASS WON", 14.89, 31.77, 25.44, 44.49, None),
        ("PASS WON", 51.18, 41.05, 56.67, 52.94, None),
        ("PASS WON", 24.66, 67.07, 39.63, 65.75, None),
        ("PASS WON", 49.04, 59.49, 52.15, 44.61, None),
        ("PASS WON", 37.18, 47.17, 67.26, 75.14, None),
        ("PASS WON", 60.79, 41.67, 89.68, 52.81, None),
        ("PASS WON", 67.46, 52.19, 81.22, 71.26, None),
        ("PASS WON", 73.8, 48.27, 107.35, 53.58, None),
        ("PASS WON", 76.5, 58.07, 72.22, 55.91, None),
        ("PASS LOST", 44.87, 73.56, 49.69, 80.0, None),
        ("PASS LOST", 56.73, 0.0, 57.04, 10.89, None),
        ("PASS LOST", 56.68, 60.89, 74.18, 80.0, None),
        ("PASS LOST", 60.42, 61.83, 96.09, 74.48, None),
        ("PASS LOST", 67.21, 40.91, 76.91, 17.38, None),
        ("PASS LOST", 81.86, 47.76, 81.67, 63.99, None),
    ],
    "Opponent 11 (04-17)": [
        ("PASS WON", 53.54, 59.86, 63.89, 70.65, None),
        ("PASS WON", 79.34, 43.08, 85.71, 35.61, None),
        ("PASS WON", 53.31, 20.56, 54.59, 33.44, None),
        ("PASS WON", 65.24, 17.86, 82.21, 0.0, None),
        ("PASS WON", 48.99, 61.26, 45.64, 68.33, None),
        ("PASS WON", 33.44, 53.32, 53.13, 61.6, None),
        ("PASS WON", 44.66, 55.27, 36.79, 49.43, None),
        ("PASS WON", 61.15, 0.0, 69.31, 13.94, None),
        ("PASS WON", 47.04, 13.38, 65.01, 0.0, None),
        ("PASS WON", 50.39, 52.54, 62.9, 64.42, None),
        ("PASS WON", 67.02, 50.94, 74.38, 39.68, None),
        ("PASS WON", 46.89, 64.04, 57.99, 60.07, None),
        ("PASS WON", 46.53, 80.0, 63.63, 71.28, None),
        ("PASS WON", 60.68, 48.36, 80.98, 24.34, None),
        ("PASS WON", 34.36, 76.21, 45.21, 42.76, None),
        ("PASS WON", 18.73, 42.7, 22.0, 46.64, None),
        ("PASS WON", 50.65, 35.73, 84.15, 46.62, None),
        ("PASS WON", 55.35, 62.75, 54.29, 56.24, None),
        ("PASS WON", 59.59, 72.29, 66.1, 80.0, None),
        ("PASS WON", 79.03, 31.21, 85.81, 49.33, None),
        ("PASS WON", 29.27, 51.04, 33.99, 25.24, None),
        ("PASS WON", 56.22, 34.55, 69.57, 19.36, None),
        ("PASS WON", 79.28, 80.0, 78.53, 69.0, None),
        ("PASS WON", 52.46, 44.11, 56.86, 41.9, None),
        ("PASS WON", 100.92, 7.68, 120.0, 11.38, None),
        ("PASS WON", 58.08, 48.03, 69.41, 48.98, None),
        ("PASS WON", 77.36, 37.51, 106.97, 45.91, None),
        ("PASS WON", 33.74, 31.98, 46.34, 31.15, None),
        ("PASS WON", 40.96, 48.88, 35.66, 80.0, None),
        ("PASS WON", 21.49, 49.35, 28.49, 42.58, None),
        ("PASS WON", 24.59, 60.08, 50.42, 32.88, None),
        ("PASS WON", 49.44, 58.86, 59.57, 59.67, None),
        ("PASS WON", 80.92, 49.26, 93.85, 47.6, None),
        ("PASS LOST", 49.47, 71.51, 75.92, 71.98, None),
        ("PASS LOST", 69.1, 65.44, 81.31, 68.9, None),
        ("PASS LOST", 80.44, 38.18, 103.4, 34.36, None),
        ("PASS LOST", 83.55, 44.78, 94.64, 14.62, None),
        ("PASS LOST", 73.89, 5.71, 104.53, 7.8, None),
    ],
    "Opponent 12 (04-21)": [
        ("PASS WON", 71.06, 72.91, 88.52, 80.0, None),
        ("PASS WON", 19.21, 43.29, 35.06, 36.89, None),
        ("PASS WON", 65.71, 48.95, 48.98, 51.61, None),
        ("PASS WON", 43.76, 44.22, 46.62, 41.33, None),
        ("PASS WON", 71.74, 8.37, 75.63, 4.13, None),
        ("PASS WON", 48.35, 67.37, 70.41, 69.73, None),
        ("PASS WON", 63.54, 29.47, 61.69, 25.6, None),
        ("PASS WON", 3.87, 69.4, 14.34, 80.0, None),
        ("PASS WON", 37.03, 57.73, 49.9, 50.08, None),
        ("PASS WON", 70.77, 35.06, 70.06, 37.82, None),
        ("PASS WON", 53.11, 34.23, 82.36, 57.23, None),
        ("PASS WON", 107.62, 7.25, 113.19, 0.0, None),
        ("PASS WON", 65.42, 80.0, 87.46, 79.68, None),
        ("PASS WON", 78.34, 74.77, 50.44, 52.5, None),
        ("PASS WON", 79.62, 17.47, 95.95, 34.86, None),
        ("PASS WON", 95.54, 15.72, 101.36, 2.5, None),
        ("PASS WON", 51.51, 41.56, 56.89, 57.21, None),
        ("PASS WON", 2.52, 38.5, 10.48, 35.11, None),
        ("PASS WON", 83.49, 20.38, 84.22, 9.02, None),
        ("PASS WON", 21.76, 72.63, 20.98, 56.65, None),
        ("PASS WON", 22.5, 41.09, 21.09, 45.28, None),
        ("PASS WON", 45.57, 65.57, 31.09, 58.93, None),
        ("PASS WON", 58.83, 78.11, 84.04, 80.0, None),
        ("PASS WON", 39.37, 57.67, 47.64, 31.82, None),
        ("PASS WON", 65.81, 72.07, 50.88, 67.22, None),
        ("PASS WON", 70.72, 62.33, 73.4, 40.72, None),
        ("PASS WON", 77.45, 67.72, 70.88, 55.87, None),
        ("PASS WON", 61.5, 56.58, 83.28, 52.63, None),
        ("PASS WON", 50.76, 46.09, 71.05, 60.54, None),
        ("PASS WON", 88.64, 44.2, 75.8, 65.71, None),
        ("PASS WON", 59.38, 58.58, 45.91, 60.19, None),
        ("PASS WON", 85.07, 16.28, 102.59, 18.56, None),
        ("PASS WON", 74.54, 38.38, 102.72, 46.42, None),
        ("PASS WON", 62.75, 24.79, 86.65, 7.07, None),
        ("PASS WON", 89.53, 38.22, 117.79, 24.88, None),
        ("PASS WON", 83.41, 57.67, 106.22, 80.0, None),
        ("PASS WON", 52.97, 59.54, 52.97, 71.96, None),
        ("PASS WON", 49.43, 43.22, 70.99, 25.71, None),
        ("PASS LOST", 65.49, 44.57, 83.34, 53.18, None),
        ("PASS LOST", 82.25, 51.06, 108.88, 52.82, None),
        ("PASS LOST", 83.19, 45.11, 120.0, 45.69, None),
        ("PASS LOST", 86.4, 55.76, 95.85, 47.64, None),
        ("PASS LOST", 82.69, 61.01, 102.66, 63.26, None),
        ("PASS LOST", 35.34, 38.45, 37.03, 53.15, None),
        ("PASS LOST", 71.8, 27.79, 106.66, 46.62, None),
        ("PASS LOST", 91.42, 80.0, 118.87, 44.52, None),
        ("PASS LOST", 66.59, 32.35, 100.6, 24.04, None),
    ],
    "Opponent 13 (04-25)": [
        ("PASS WON", 54.17, 48.01, 60.47, 57.24, None),
        ("PASS WON", 69.71, 24.81, 69.81, 46.31, None),
        ("PASS WON", 53.62, 78.58, 40.27, 80.0, None),
        ("PASS WON", 86.08, 58.39, 74.14, 35.92, None),
        ("PASS WON", 37.29, 28.27, 45.7, 50.9, None),
        ("PASS WON", 49.86, 48.22, 72.97, 16.68, None),
        ("PASS WON", 39.67, 47.93, 51.6, 73.36, None),
        ("PASS WON", 44.43, 29.49, 50.62, 45.12, None),
        ("PASS WON", 58.07, 44.07, 56.98, 51.61, None),
        ("PASS WON", 53.55, 59.04, 66.75, 76.18, None),
        ("PASS WON", 62.72, 15.93, 79.87, 33.29, None),
        ("PASS WON", 44.47, 45.96, 32.55, 20.58, None),
        ("PASS WON", 41.28, 53.91, 36.11, 66.66, None),
        ("PASS WON", 57.32, 39.32, 66.63, 49.35, None),
        ("PASS WON", 44.38, 67.95, 65.67, 35.51, None),
        ("PASS WON", 65.09, 27.39, 73.57, 26.09, None),
        ("PASS WON", 41.18, 65.07, 61.57, 59.86, None),
        ("PASS WON", 39.31, 54.17, 63.92, 61.52, None),
        ("PASS WON", 69.71, 71.22, 91.75, 57.9, None),
        ("PASS WON", 61.62, 56.1, 56.6, 46.51, None),
        ("PASS WON", 28.19, 54.38, 37.5, 80.0, None),
        ("PASS WON", 49.84, 22.0, 61.11, 24.38, None),
        ("PASS WON", 28.01, 33.69, 17.67, 33.72, None),
        ("PASS WON", 91.38, 51.19, 103.38, 42.24, None),
        ("PASS WON", 80.57, 46.82, 106.68, 38.83, None),
        ("PASS WON", 73.3, 0.0, 93.37, 17.51, None),
        ("PASS WON", 54.01, 57.81, 55.78, 57.91, None),
        ("PASS WON", 44.35, 13.91, 57.27, 1.97, None),
        ("PASS WON", 54.18, 52.16, 49.51, 55.7, None),
        ("PASS WON", 19.64, 61.38, 27.59, 70.92, None),
        ("PASS WON", 61.14, 65.52, 56.34, 68.26, None),
        ("PASS WON", 47.38, 40.22, 66.9, 48.63, None),
        ("PASS WON", 96.78, 33.87, 106.43, 20.65, None),
        ("PASS WON", 18.68, 57.58, 40.57, 55.39, None),
        ("PASS WON", 68.69, 58.82, 85.21, 76.22, None),
        ("PASS WON", 50.05, 42.08, 61.01, 51.49, None),
        ("PASS WON", 71.31, 60.25, 69.59, 70.34, None),
        ("PASS LOST", 11.48, 15.47, 34.52, 4.56, None),
        ("PASS LOST", 76.78, 76.75, 73.61, 64.89, None),
        ("PASS LOST", 39.26, 46.35, 64.36, 44.15, None),
        ("PASS LOST", 62.08, 17.18, 86.91, 36.17, None),
        ("PASS LOST", 75.14, 62.35, 84.91, 57.45, None),
        ("PASS LOST", 91.09, 62.78, 102.24, 79.72, None),
        ("PASS LOST", 83.41, 14.97, 101.34, 16.54, None),
        ("PASS LOST", 71.17, 38.74, 100.14, 34.22, None),
        ("PASS LOST", 113.11, 53.84, 118.38, 73.42, None),
        ("PASS LOST", 66.04, 77.04, 103.52, 80.0, None),
    ],
    "Opponent 14 (04-01)": [
        ("PASS WON", 77.78, 30.93, 93.01, 29.56, None),
        ("PASS WON", 57.58, 38.12, 69.33, 21.53, None),
        ("PASS WON", 53.36, 75.26, 39.17, 65.15, None),
        ("PASS WON", 75.93, 30.47, 99.57, 17.97, None),
        ("PASS WON", 73.15, 14.45, 94.43, 7.28, None),
        ("PASS WON", 58.39, 43.28, 83.12, 51.27, None),
        ("PASS WON", 95.18, 35.43, 96.8, 0.0, None),
        ("PASS WON", 72.52, 38.55, 87.06, 41.48, None),
        ("PASS WON", 39.42, 63.3, 47.95, 51.26, None),
        ("PASS WON", 47.04, 37.08, 73.57, 47.08, None),
        ("PASS WON", 38.0, 65.96, 46.36, 34.6, None),
        ("PASS WON", 22.77, 20.67, 55.39, 10.71, None),
        ("PASS WON", 30.23, 37.67, 48.95, 14.39, None),
        ("PASS WON", 48.76, 71.62, 57.77, 77.96, None),
        ("PASS WON", 50.3, 58.48, 52.95, 50.54, None),
        ("PASS WON", 33.98, 62.4, 32.26, 67.18, None),
        ("PASS WON", 77.86, 44.28, 72.34, 78.56, None),
        ("PASS WON", 70.47, 62.25, 77.45, 80.0, None),
        ("PASS WON", 83.99, 46.92, 108.25, 45.07, None),
        ("PASS WON", 52.71, 25.91, 49.68, 23.86, None),
        ("PASS WON", 18.1, 58.93, 34.62, 69.79, None),
        ("PASS WON", 35.98, 41.34, 45.71, 35.05, None),
        ("PASS WON", 0.0, 37.91, 6.18, 25.93, None),
        ("PASS WON", 41.39, 29.05, 53.57, 47.34, None),
        ("PASS WON", 40.41, 60.7, 47.09, 51.77, None),
        ("PASS WON", 36.15, 21.86, 38.78, 19.81, None),
        ("PASS WON", 49.72, 57.83, 74.68, 67.76, None),
        ("PASS WON", 97.21, 20.99, 106.76, 24.77, None),
        ("PASS WON", 51.41, 66.66, 45.74, 80.0, None),
        ("PASS WON", 60.38, 21.29, 55.52, 33.02, None),
        ("PASS WON", 36.63, 40.98, 51.3, 17.29, None),
        ("PASS WON", 75.81, 54.65, 107.01, 66.99, None),
        ("PASS WON", 36.44, 21.38, 72.93, 31.41, None),
        ("PASS WON", 27.09, 57.72, 27.81, 63.05, None),
        ("PASS WON", 69.04, 42.98, 55.89, 27.24, None),
        ("PASS WON", 43.63, 57.3, 34.61, 72.75, None),
        ("PASS WON", 19.51, 42.73, 39.27, 36.01, None),
        ("PASS WON", 45.03, 48.7, 20.88, 49.74, None),
        ("PASS LOST", 60.45, 64.48, 82.22, 73.38, None),
        ("PASS LOST", 113.95, 39.2, 120.0, 19.71, None),
        ("PASS LOST", 116.21, 80.0, 120.0, 80.0, None),
        ("PASS LOST", 50.74, 65.76, 73.78, 47.36, None),
        ("PASS LOST", 91.15, 53.98, 98.45, 54.53, None),
        ("PASS LOST", 21.45, 21.81, 53.15, 38.36, None),
        ("PASS LOST", 19.38, 34.03, 22.99, 35.16, None),
        ("PASS LOST", 95.64, 46.39, 109.21, 67.74, None),
        ("PASS LOST", 100.1, 28.07, 108.59, 2.75, None),
        ("PASS LOST", 53.9, 25.43, 56.25, 19.41, None),
        ("PASS LOST", 68.96, 51.44, 86.11, 66.73, None),
    ],
    "Opponent 15 (04-05)": [
        ("PASS WON", 37.95, 46.06, 53.9, 39.68, None),
        ("PASS WON", 50.69, 40.64, 60.34, 43.11, None),
        ("PASS WON", 33.27, 36.18, 51.29, 17.0, None),
        ("PASS WON", 60.52, 62.07, 63.44, 73.73, None),
        ("PASS WON", 94.0, 54.68, 68.12, 63.73, None),
        ("PASS WON", 65.19, 15.65, 63.8, 32.65, None),
        ("PASS WON", 114.17, 55.03, 120.0, 38.93, None),
        ("PASS WON", 11.13, 24.38, 24.91, 49.8, None),
        ("PASS WON", 43.12, 42.88, 64.76, 52.61, None),
        ("PASS WON", 71.37, 36.04, 95.37, 54.25, None),
        ("PASS WON", 65.42, 80.0, 86.92, 80.0, None),
        ("PASS WON", 28.56, 45.16, 42.5, 60.28, None),
        ("PASS WON", 60.35, 10.2, 84.24, 8.74, None),
        ("PASS WON", 46.36, 66.03, 59.04, 71.37, None),
        ("PASS WON", 72.13, 41.91, 106.14, 77.22, None),
        ("PASS WON", 58.04, 19.86, 79.58, 10.09, None),
        ("PASS WON", 60.25, 76.93, 65.84, 80.0, None),
        ("PASS WON", 60.04, 28.89, 76.64, 10.28, None),
        ("PASS WON", 43.29, 51.22, 70.58, 80.0, None),
        ("PASS WON", 66.28, 36.13, 83.18, 45.5, None),
        ("PASS WON", 59.75, 62.6, 87.92, 78.24, None),
        ("PASS WON", 67.14, 37.03, 57.66, 35.67, None),
        ("PASS WON", 35.95, 0.0, 48.06, 0.0, None),
        ("PASS WON", 55.41, 28.38, 66.68, 14.43, None),
        ("PASS WON", 33.54, 40.41, 51.19, 25.81, None),
        ("PASS WON", 75.7, 42.3, 87.42, 41.24, None),
        ("PASS WON", 51.61, 28.46, 68.4, 40.45, None),
        ("PASS WON", 92.05, 60.69, 89.74, 30.45, None),
        ("PASS WON", 38.25, 57.97, 62.94, 16.5, None),
        ("PASS WON", 46.42, 57.4, 72.75, 41.65, None),
        ("PASS WON", 67.06, 39.02, 77.56, 34.32, None),
        ("PASS WON", 52.07, 34.08, 68.2, 20.56, None),
        ("PASS WON", 53.95, 27.97, 74.84, 29.79, None),
        ("PASS WON", 85.56, 40.09, 94.98, 43.07, None),
        ("PASS WON", 58.35, 40.51, 89.37, 52.9, None),
        ("PASS WON", 45.29, 61.89, 43.84, 80.0, None),
        ("PASS WON", 50.57, 32.05, 57.81, 51.76, None),
        ("PASS WON", 61.37, 42.56, 70.61, 33.47, None),
        ("PASS WON", 51.59, 29.94, 65.88, 11.69, None),
        ("PASS LOST", 52.19, 43.05, 70.14, 43.59, None),
        ("PASS LOST", 111.32, 43.1, 120.0, 16.01, None),
        ("PASS LOST", 85.09, 63.24, 91.51, 69.01, None),
        ("PASS LOST", 85.84, 32.67, 106.96, 18.54, None),
        ("PASS LOST", 57.38, 74.22, 92.86, 68.44, None),
        ("PASS LOST", 105.3, 23.78, 116.7, 15.73, None),
        ("PASS LOST", 63.18, 37.4, 86.82, 34.91, None),
        ("PASS LOST", 105.38, 54.25, 120.0, 60.6, None),
        ("PASS LOST", 55.57, 40.27, 66.2, 43.5, None),
        ("PASS LOST", 87.13, 32.47, 90.95, 29.96, None),
    ],
    "Opponent 16 (04-09)": [
        ("PASS WON", 80.19, 64.83, 83.64, 55.94, None),
        ("PASS WON", 54.8, 15.56, 57.75, 20.56, None),
        ("PASS WON", 57.89, 80.0, 74.33, 56.52, None),
        ("PASS WON", 62.3, 23.35, 66.12, 6.94, None),
        ("PASS WON", 51.48, 22.67, 62.11, 23.0, None),
        ("PASS WON", 75.68, 50.84, 92.76, 65.79, None),
        ("PASS WON", 66.91, 56.78, 74.58, 66.05, None),
        ("PASS WON", 50.62, 58.77, 44.81, 38.06, None),
        ("PASS WON", 53.84, 55.03, 57.88, 78.82, None),
        ("PASS WON", 32.62, 46.31, 38.48, 70.14, None),
        ("PASS WON", 81.56, 80.0, 99.72, 80.0, None),
        ("PASS WON", 72.44, 32.67, 72.36, 18.04, None),
        ("PASS WON", 18.06, 56.59, 14.58, 32.05, None),
        ("PASS WON", 59.38, 43.93, 41.62, 30.38, None),
        ("PASS WON", 98.83, 69.52, 106.87, 49.63, None),
        ("PASS WON", 42.16, 38.94, 52.63, 48.14, None),
        ("PASS WON", 50.59, 42.23, 73.47, 36.02, None),
        ("PASS WON", 42.5, 36.43, 61.71, 43.96, None),
        ("PASS WON", 116.41, 55.15, 120.0, 60.34, None),
        ("PASS WON", 17.53, 21.67, 37.42, 19.2, None),
        ("PASS WON", 52.26, 66.7, 46.38, 72.26, None),
        ("PASS WON", 28.94, 45.11, 23.05, 13.79, None),
        ("PASS WON", 31.33, 49.64, 36.08, 52.02, None),
        ("PASS WON", 93.13, 22.43, 106.21, 35.38, None),
        ("PASS WON", 48.74, 24.97, 56.61, 20.42, None),
        ("PASS WON", 29.5, 54.89, 49.33, 49.36, None),
        ("PASS WON", 54.48, 28.87, 58.42, 40.31, None),
        ("PASS WON", 40.98, 46.93, 40.31, 33.79, None),
        ("PASS WON", 58.02, 11.44, 49.96, 22.63, None),
        ("PASS WON", 49.41, 33.09, 80.43, 26.42, None),
        ("PASS LOST", 73.5, 53.92, 103.58, 68.61, None),
        ("PASS LOST", 74.15, 32.79, 94.0, 38.18, None),
        ("PASS LOST", 61.41, 72.54, 76.9, 80.0, None),
        ("PASS LOST", 72.13, 54.14, 74.53, 56.51, None),
        ("PASS LOST", 65.64, 29.12, 62.2, 0.0, None),
        ("PASS LOST", 57.84, 38.35, 76.35, 21.62, None),
    ],
    "Opponent 17 (04-13)": [
        ("PASS WON", 86.75, 35.43, 82.9, 51.47, None),
        ("PASS WON", 50.0, 15.17, 52.6, 30.49, None),
        ("PASS WON", 55.19, 34.5, 60.97, 36.83, None),
        ("PASS WON", 66.35, 41.29, 70.88, 33.01, None),
        ("PASS WON", 44.16, 53.19, 45.87, 46.94, None),
        ("PASS WON", 33.08, 51.93, 56.6, 39.13, None),
        ("PASS WON", 74.24, 53.58, 71.02, 30.94, None),
        ("PASS WON", 70.85, 51.83, 74.43, 48.81, None),
        ("PASS WON", 37.26, 43.77, 62.52, 34.15, None),
        ("PASS WON", 22.13, 24.11, 44.55, 16.84, None),
        ("PASS WON", 37.1, 3.48, 28.59, 7.85, None),
        ("PASS WON", 63.82, 37.74, 61.12, 55.57, None),
        ("PASS WON", 71.79, 46.01, 97.86, 64.3, None),
        ("PASS WON", 39.1, 80.0, 51.03, 78.36, None),
        ("PASS WON", 59.74, 33.98, 73.26, 12.88, None),
        ("PASS WON", 109.4, 26.24, 112.6, 46.61, None),
        ("PASS WON", 105.79, 6.64, 120.0, 0.0, None),
        ("PASS WON", 74.66, 39.98, 94.72, 49.82, None),
        ("PASS WON", 3.6, 51.5, 0.0, 47.84, None),
        ("PASS WON", 57.29, 54.47, 62.97, 51.71, None),
        ("PASS WON", 28.48, 49.42, 64.83, 22.32, None),
        ("PASS WON", 70.43, 16.1, 75.55, 16.69, None),
        ("PASS WON", 83.69, 38.04, 74.88, 26.87, None),
        ("PASS LOST", 55.72, 34.12, 82.94, 47.5, None),
        ("PASS LOST", 73.79, 68.43, 72.75, 76.94, None),
        ("PASS LOST", 26.76, 57.5, 38.01, 80.0, None),
        ("PASS LOST", 30.22, 27.95, 47.33, 0.0, None),
    ],
    "Opponent 18 (04-17)": [
        ("PASS WON", 64.17, 53.62, 54.56, 23.73, None),
        ("PASS WON", 13.91, 5.77, 22.95, 0.0, None),
        ("PASS WON", 72.73, 30.8, 85.81, 19.11, None),
        ("PASS WON", 48.6, 69.74, 60.49, 61.12, None),
        ("PASS WON", 95.64, 66.48, 111.07, 80.0, None),
        ("PASS WON", 40.99, 65.4, 66.67, 48.33, None),
        ("PASS WON", 113.83, 42.63, 120.0, 43.97, None),
        ("PASS WON", 47.54, 44.72, 77.15, 29.85, None),
        ("PASS WON", 49.32, 68.44, 55.93, 71.72, None),
        ("PASS WON", 46.65, 55.17, 51.13, 37.43, None),
        ("PASS WON", 26.75, 18.47, 7.37, 28.89, None),
        ("PASS WON", 30.05, 52.13, 33.69, 52.79, None),
        ("PASS WON", 53.75, 42.0, 68.4, 47.05, None),
        ("PASS WON", 56.74, 49.82, 73.96, 45.93, None),
        ("PASS WON", 19.67, 60.0, 43.88, 58.06, None),
        ("PASS WON", 62.68, 40.61, 83.51, 28.79, None),
        ("PASS WON", 14.15, 62.89, 42.46, 54.62, None),
        ("PASS WON", 79.98, 5.46, 98.32, 27.04, None),
        ("PASS WON", 30.45, 75.14, 48.38, 80.0, None),
        ("PASS WON", 33.05, 59.29, 60.68, 75.0, None),
        ("PASS WON", 49.43, 57.29, 36.27, 67.4, None),
        ("PASS WON", 118.3, 50.76, 120.0, 46.91, None),
        ("PASS WON", 61.7, 43.03, 50.69, 66.37, None),
        ("PASS WON", 86.43, 70.21, 71.33, 30.7, None),
        ("PASS WON", 81.99, 30.87, 100.83, 8.96, None),
        ("PASS WON", 36.14, 39.76, 52.16, 23.23, None),
        ("PASS WON", 54.88, 38.74, 63.78, 53.96, None),
        ("PASS LOST", 63.39, 50.22, 76.77, 57.84, None),
        ("PASS LOST", 56.52, 39.02, 83.54, 47.42, None),
        ("PASS LOST", 95.03, 43.85, 113.28, 38.46, None),
        ("PASS LOST", 95.71, 80.0, 120.0, 74.5, None),
        ("PASS LOST", 61.9, 55.83, 102.99, 53.46, None),
    ],
    "Opponent 19 (04-21)": [
        ("PASS WON", 31.0, 33.18, 66.13, 59.38, None),
        ("PASS WON", 63.89, 59.61, 37.6, 23.83, None),
        ("PASS WON", 33.64, 80.0, 49.63, 71.56, None),
        ("PASS WON", 59.95, 28.12, 54.48, 31.22, None),
        ("PASS WON", 96.85, 74.06, 82.97, 65.02, None),
        ("PASS WON", 85.63, 23.52, 73.92, 31.73, None),
        ("PASS WON", 56.31, 18.26, 69.61, 12.06, None),
        ("PASS WON", 28.86, 24.97, 43.95, 45.25, None),
        ("PASS WON", 48.45, 54.4, 71.42, 70.17, None),
        ("PASS WON", 59.96, 10.55, 56.26, 20.38, None),
        ("PASS WON", 74.87, 66.85, 83.35, 74.43, None),
        ("PASS WON", 45.31, 63.54, 65.05, 80.0, None),
        ("PASS WON", 61.72, 37.5, 82.37, 18.13, None),
        ("PASS WON", 89.06, 36.26, 118.7, 15.57, None),
        ("PASS WON", 37.92, 39.91, 36.37, 58.2, None),
        ("PASS WON", 49.34, 30.53, 68.92, 36.49, None),
        ("PASS WON", 90.94, 56.48, 99.45, 60.32, None),
        ("PASS WON", 19.41, 37.64, 0.31, 15.14, None),
        ("PASS WON", 44.96, 72.0, 28.3, 51.97, None),
        ("PASS WON", 83.63, 56.1, 93.48, 56.89, None),
        ("PASS WON", 95.28, 50.71, 94.24, 52.36, None),
        ("PASS LOST", 70.21, 17.17, 106.55, 0.0, None),
        ("PASS LOST", 72.63, 28.74, 106.38, 37.16, None),
        ("PASS LOST", 39.72, 79.27, 57.64, 72.68, None),
        ("PASS LOST", 89.01, 61.12, 109.55, 65.26, None),
        ("PASS LOST", 50.18, 29.26, 77.58, 29.42, None),
        ("PASS LOST", 88.2, 49.52, 98.46, 60.98, None),
        ("PASS LOST", 41.15, 48.58, 60.64, 64.56, None),
    ],
    "Opponent 20 (04-25)": [
        ("PASS WON", 55.5, 37.89, 68.95, 42.54, None),
        ("PASS WON", 32.54, 37.9, 45.29, 36.71, None),
        ("PASS WON", 89.63, 33.21, 103.36, 8.45, None),
        ("PASS WON", 42.64, 30.45, 55.44, 51.75, None),
        ("PASS WON", 64.98, 44.96, 79.69, 49.82, None),
        ("PASS WON", 18.61, 49.36, 33.01, 61.21, None),
        ("PASS WON", 42.83, 69.0, 36.93, 59.58, None),
        ("PASS WON", 56.14, 28.9, 78.1, 48.56, None),
        ("PASS WON", 32.88, 16.23, 56.13, 27.63, None),
        ("PASS WON", 88.06, 50.82, 94.5, 34.22, None),
        ("PASS WON", 54.39, 39.5, 48.75, 32.16, None),
        ("PASS WON", 30.04, 28.09, 16.58, 49.79, None),
        ("PASS WON", 78.76, 18.44, 93.75, 25.14, None),
        ("PASS WON", 51.46, 76.94, 64.34, 54.88, None),
        ("PASS WON", 86.53, 21.87, 85.29, 0.41, None),
        ("PASS WON", 36.42, 36.57, 59.06, 48.61, None),
        ("PASS WON", 40.45, 51.93, 63.94, 70.36, None),
        ("PASS WON", 30.28, 45.17, 59.52, 44.64, None),
        ("PASS WON", 44.3, 65.86, 64.01, 21.5, None),
        ("PASS WON", 62.31, 46.98, 78.69, 28.92, None),
        ("PASS WON", 69.89, 48.22, 67.33, 63.58, None),
        ("PASS WON", 40.33, 69.33, 89.38, 58.63, None),
        ("PASS LOST", 86.78, 50.96, 95.38, 43.06, None),
        ("PASS LOST", 82.24, 46.1, 87.52, 44.78, None),
        ("PASS LOST", 27.11, 72.5, 67.21, 78.89, None),
    ],
}
# DEFENSIVE ACTIONS
DEFENSIVE_MATCHES_DATA = {
    "Opponent 01 (03-05)": [
        ("DUEL_WON", 66.51, 18.11),
        ("DUEL_LOST", 56.16, 49.45),
        ("INTERCEPTION", 31.86, 20.39),
    ],
    "Opponent 02 (03-09)": [
        ("DUEL_WON", 54.82, 26.14),
        ("DUEL_WON", 53.95, 11.32),
        ("DUEL_WON", 2.09, 51.93),
        ("DUEL_WON", 38.57, 20.26),
        ("DUEL_WON", 24.02, 20.62),
        ("DUEL_WON", 30.17, 37.46),
        ("DUEL_WON", 47.04, 27.11),
        ("DUEL_WON", 11.89, 34.85),
        ("DUEL_LOST", 51.09, 46.58),
        ("DUEL_LOST", 32.74, 51.04),
        ("DUEL_LOST", 80.34, 80.0),
        ("DUEL_LOST", 50.94, 35.21),
        ("INTERCEPTION", 49.8, 63.4),
        ("INTERCEPTION", 35.68, 19.16),
        ("INTERCEPTION", 43.92, 10.07),
        ("INTERCEPTION", 47.82, 34.63),
        ("INTERCEPTION", 7.68, 36.95),
        ("INTERCEPTION", 57.27, 80.0),
        ("INTERCEPTION", 26.35, 77.9),
        ("INTERCEPTION", 10.34, 18.87),
        ("INTERCEPTION", 59.55, 36.12),
    ],
    "Opponent 03 (03-13)": [
        ("DUEL_LOST", 26.86, 36.96),
        ("DUEL_LOST", 37.34, 53.49),
        ("INTERCEPTION", 57.96, 33.1),
        ("INTERCEPTION", 57.98, 10.59),
    ],
    "Opponent 04 (03-17)": [
        ("DUEL_WON", 110.72, 34.05),
        ("DUEL_WON", 42.41, 30.13),
        ("DUEL_WON", 45.05, 31.71),
        ("DUEL_LOST", 52.36, 46.94),
        ("DUEL_LOST", 83.17, 40.11),
        ("INTERCEPTION", 40.95, 54.32),
    ],
    "Opponent 05 (03-21)": [
        ("DUEL_WON", 32.92, 48.91),
        ("DUEL_WON", 43.01, 64.89),
        ("DUEL_WON", 53.2, 28.5),
        ("DUEL_WON", 5.23, 18.53),
        ("DUEL_WON", 42.63, 20.3),
        ("DUEL_WON", 53.1, 23.72),
        ("DUEL_LOST", 16.83, 77.89),
        ("DUEL_LOST", 13.6, 25.94),
        ("DUEL_LOST", 38.65, 21.21),
        ("DUEL_LOST", 43.58, 54.2),
        ("DUEL_LOST", 36.46, 0.0),
        ("INTERCEPTION", 69.13, 7.61),
        ("INTERCEPTION", 51.1, 19.28),
        ("INTERCEPTION", 67.1, 45.87),
        ("INTERCEPTION", 40.38, 5.18),
        ("INTERCEPTION", 79.7, 21.96),
    ],
    "Opponent 06 (03-25)": [
        ("DUEL_WON", 42.23, 23.27),
        ("DUEL_WON", 44.07, 17.75),
        ("DUEL_WON", 11.38, 48.95),
        ("DUEL_WON", 18.99, 53.16),
        ("DUEL_WON", 61.07, 19.42),
        ("DUEL_LOST", 35.49, 39.12),
        ("INTERCEPTION", 26.88, 59.89),
        ("INTERCEPTION", 52.59, 36.58),
    ],
    "Opponent 07 (03-01)": [
        ("DUEL_LOST", 46.84, 31.8),
        ("INTERCEPTION", 43.42, 34.62),
    ],
    "Opponent 08 (03-05)": [
        ("DUEL_WON", 71.29, 74.35),
        ("DUEL_LOST", 65.54, 47.22),
    ],
    "Opponent 09 (03-09)": [
        ("DUEL_WON", 71.46, 24.68),
        ("DUEL_WON", 0.0, 23.27),
        ("DUEL_WON", 25.89, 34.78),
        ("DUEL_WON", 45.68, 23.61),
        ("INTERCEPTION", 69.52, 38.89),
    ],
    "Opponent 10 (03-13)": [
        ("DUEL_WON", 5.71, 52.91),
        ("DUEL_WON", 41.34, 6.88),
        ("DUEL_WON", 20.65, 80.0),
        ("DUEL_WON", 41.76, 50.55),
        ("DUEL_WON", 52.51, 18.2),
        ("DUEL_WON", 6.98, 22.82),
        ("DUEL_WON", 37.15, 26.07),
        ("DUEL_WON", 60.07, 55.25),
        ("DUEL_WON", 55.79, 61.91),
        ("DUEL_WON", 45.01, 0.0),
        ("DUEL_LOST", 77.95, 35.08),
        ("DUEL_LOST", 11.31, 80.0),
        ("DUEL_LOST", 9.71, 49.6),
        ("DUEL_LOST", 43.82, 63.33),
        ("INTERCEPTION", 73.63, 55.88),
        ("INTERCEPTION", 83.16, 70.42),
        ("INTERCEPTION", 67.22, 43.46),
    ],
    "Opponent 11 (04-17)": [
        ("DUEL_WON", 66.47, 54.23),
        ("DUEL_WON", 23.12, 43.4),
        ("DUEL_WON", 42.46, 24.3),
        ("DUEL_WON", 77.79, 15.72),
        ("DUEL_WON", 0.0, 58.54),
        ("DUEL_WON", 55.9, 50.75),
        ("DUEL_WON", 46.53, 66.24),
        ("DUEL_WON", 20.02, 8.32),
        ("DUEL_WON", 46.47, 17.43),
        ("DUEL_LOST", 64.09, 17.59),
        ("DUEL_LOST", 40.78, 26.66),
        ("DUEL_LOST", 21.05, 59.95),
        ("DUEL_LOST", 53.53, 16.41),
        ("DUEL_LOST", 40.28, 40.96),
        ("DUEL_LOST", 38.45, 80.0),
        ("INTERCEPTION", 42.03, 72.21),
        ("INTERCEPTION", 66.7, 39.89),
        ("INTERCEPTION", 64.84, 0.0),
        ("INTERCEPTION", 40.13, 46.93),
    ],
    "Opponent 12 (04-21)": [
        ("DUEL_WON", 49.89, 45.89),
        ("DUEL_WON", 53.71, 52.47),
        ("DUEL_WON", 77.96, 39.25),
        ("DUEL_LOST", 55.52, 80.0),
        ("DUEL_LOST", 21.14, 55.93),
        ("DUEL_LOST", 13.5, 13.21),
        ("INTERCEPTION", 81.6, 33.91),
        ("INTERCEPTION", 62.24, 20.41),
        ("INTERCEPTION", 56.67, 0.2),
    ],
    "Opponent 13 (04-25)": [
        ("DUEL_WON", 79.14, 49.08),
        ("DUEL_WON", 63.69, 60.37),
        ("DUEL_WON", 58.16, 26.7),
        ("DUEL_WON", 90.52, 67.58),
        ("DUEL_WON", 47.48, 13.56),
        ("DUEL_WON", 86.7, 60.2),
        ("DUEL_WON", 55.92, 28.01),
        ("DUEL_WON", 49.2, 59.6),
        ("DUEL_LOST", 44.06, 43.39),
        ("DUEL_LOST", 65.3, 20.03),
        ("DUEL_LOST", 33.53, 38.63),
        ("DUEL_LOST", 38.58, 22.61),
        ("DUEL_LOST", 55.15, 27.94),
        ("DUEL_LOST", 49.96, 72.44),
        ("DUEL_LOST", 40.16, 19.42),
        ("INTERCEPTION", 49.37, 53.22),
        ("INTERCEPTION", 14.16, 15.78),
        ("INTERCEPTION", 39.24, 67.58),
    ],
    "Opponent 14 (04-01)": [
        ("DUEL_WON", 52.04, 18.8),
        ("DUEL_LOST", 68.29, 56.07),
        ("DUEL_LOST", 61.68, 8.47),
        ("INTERCEPTION", 28.13, 30.08),
    ],
    "Opponent 15 (04-05)": [
        ("DUEL_WON", 35.05, 51.81),
        ("DUEL_WON", 66.0, 71.4),
        ("DUEL_LOST", 18.61, 38.04),
        ("DUEL_LOST", 43.79, 37.09),
        ("DUEL_LOST", 11.62, 0.71),
        ("DUEL_LOST", 53.77, 53.01),
        ("INTERCEPTION", 59.67, 65.15),
        ("INTERCEPTION", 93.7, 11.09),
        ("INTERCEPTION", 55.22, 23.97),
        ("INTERCEPTION", 61.03, 41.15),
    ],
    "Opponent 16 (04-09)": [
        ("DUEL_WON", 63.6, 42.34),
        ("DUEL_WON", 47.89, 41.39),
        ("DUEL_WON", 30.75, 34.2),
        ("DUEL_LOST", 47.19, 24.33),
        ("INTERCEPTION", 42.21, 80.0),
        ("INTERCEPTION", 41.28, 60.26),
        ("INTERCEPTION", 38.79, 11.67),
    ],
    "Opponent 17 (04-13)": [
        ("DUEL_WON", 52.47, 29.57),
        ("DUEL_WON", 31.82, 41.83),
        ("DUEL_WON", 60.13, 45.51),
        ("DUEL_LOST", 42.48, 38.46),
        ("DUEL_LOST", 46.87, 21.98),
        ("DUEL_LOST", 38.23, 44.39),
        ("DUEL_LOST", 33.83, 54.93),
        ("DUEL_LOST", 47.66, 70.98),
        ("INTERCEPTION", 61.51, 35.57),
        ("INTERCEPTION", 75.94, 80.0),
        ("INTERCEPTION", 29.46, 43.35),
        ("INTERCEPTION", 18.49, 71.53),
        ("INTERCEPTION", 41.68, 69.22),
        ("INTERCEPTION", 49.66, 22.62),
    ],
    "Opponent 18 (04-17)": [
        ("DUEL_WON", 64.15, 0.0),
        ("DUEL_LOST", 24.16, 80.0),
    ],
    "Opponent 19 (04-21)": [
        ("DUEL_WON", 29.85, 25.18),
        ("DUEL_WON", 36.75, 67.5),
        ("DUEL_LOST", 53.05, 35.91),
        ("INTERCEPTION", 51.23, 40.63),
    ],
    "Opponent 20 (04-25)": [
        ("DUEL_WON", 50.07, 25.34),
        ("DUEL_WON", 68.43, 40.82),
        ("DUEL_WON", 54.58, 6.4),
        ("DUEL_WON", 45.78, 22.04),
        ("DUEL_WON", 30.7, 67.71),
        ("DUEL_WON", 29.62, 47.02),
        ("DUEL_WON", 13.67, 32.46),
        ("DUEL_WON", 63.94, 71.46),
        ("DUEL_WON", 44.47, 61.84),
        ("DUEL_WON", 54.52, 32.77),
        ("DUEL_LOST", 40.7, 28.12),
        ("DUEL_LOST", 31.78, 47.66),
        ("DUEL_LOST", 15.28, 44.64),
        ("INTERCEPTION", 39.78, 50.82),
        ("INTERCEPTION", 70.43, 23.87),
        ("INTERCEPTION", 49.92, 49.77),
        ("INTERCEPTION", 29.64, 66.61),
        ("INTERCEPTION", 28.34, 43.86),
        ("INTERCEPTION", 62.89, 41.16),
    ],
}

# MATCH MINUTES
MATCH_MINUTES = {
    "Opponent 01 (03-05)": 90.0,
    "Opponent 02 (03-09)": 60.0,
    "Opponent 03 (03-13)": 45.0,
    "Opponent 04 (03-17)": 45.0,
    "Opponent 05 (03-21)": 90.0,
    "Opponent 06 (03-25)": 90.0,
    "Opponent 07 (03-01)": 45.0,
    "Opponent 08 (03-05)": 63.0,
    "Opponent 09 (03-09)": 45.0,
    "Opponent 10 (03-13)": 65.0,
    "Opponent 11 (04-17)": 65.0,
    "Opponent 12 (04-21)": 90.0,
    "Opponent 13 (04-25)": 90.0,
    "Opponent 14 (04-01)": 90.0,
    "Opponent 15 (04-05)": 90.0,
    "Opponent 16 (04-09)": 75.0,
    "Opponent 17 (04-13)": 90.0,
    "Opponent 18 (04-17)": 65.0,
    "Opponent 19 (04-21)": 90.0,
    "Opponent 20 (04-25)": 65.0,
}
# HELPERS
def apply_date_mapping(name: str) -> str:
    # Generic dataset: no name remapping needed.
    return name

def get_match_minutes(match_name: str) -> float:
    if match_name == "All Matches":
        total = 0.0
        for k in dfs_by_match:
            total += get_match_minutes(k)
        return total
    return float(MATCH_MINUTES.get(match_name, 90.0))

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
        re.IGNORECASE,
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

def load_docx_matches(docx_filename="Passes - Generic Player.docx") -> dict:
    p = Path(docx_filename)
    if not p.exists():
        return {}
    txt = read_docx_text(p)
    return parse_docx_events(txt)

# DATA LOADING
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

# BUILD DATAFRAMES & REORDER MATCHES
dfs_by_match = {}
for match_name, events in combined_matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["progressive"] = dfm.apply(
        lambda r: r["is_won"] and is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1
    )
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"]) ** 2 + (dfm["y_end"] - dfm["y_start"]) ** 2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"] = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"], dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    dfs_by_match[match_name] = dfm

# REORDER LOGIC
items = list(dfs_by_match.items())
if len(items) >= 18:
    part1 = items[:6]
    part2 = items[14:18]
    part3 = items[6:14]
    part4 = items[18:]
    dfs_by_match = dict(part1 + part2 + part3 + part4)

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)

# DEFENSIVE DATA LOADING
defensive_dfs_by_match = {}
for match_name, events in DEFENSIVE_MATCHES_DATA.items():
    df_def = pd.DataFrame(events, columns=["type", "x", "y"])
    df_def["match"] = match_name
    df_def["is_attacking_half"] = df_def["x"] >= FIELD_X / 2
    df_def["is_duel_won"] = df_def["type"] == "DUEL_WON"
    df_def["is_duel_lost"] = df_def["type"] == "DUEL_LOST"
    df_def["is_duel"] = df_def["is_duel_won"] | df_def["is_duel_lost"]
    df_def["is_interception"] = df_def["type"] == "INTERCEPTION"
    df_def["in_funnel"] = df_def.apply(lambda r: is_in_funnel_zone(r["x"], r["y"]), axis=1)
    defensive_dfs_by_match[match_name] = df_def

# STATS & SCORES
def compute_stats(df: pd.DataFrame, match_name: str) -> dict:
    total = len(df)
    mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    if total == 0:
        return {
            "total_passes": 0,
            "successful_passes": 0,
            "unsuccessful_passes": 0,
            "accuracy_pct": 0.0,
            "progressive_attempted": 0,
            "progressive_successful": 0,
            "progressive_accuracy_pct": 0.0,
            "to_final_third_total": 0,
            "to_final_third_success": 0,
            "to_final_third_accuracy_pct": 0.0,
            "fwd": 0,
            "fwd_pct": 0.0,
            "bwd": 0,
            "bwd_pct": 0.0,
            "lat": 0,
            "lat_pct": 0.0,
            "pos_count": 0,
            "pos_pct": 0.0,
            "high_xt_pct": 0.0,
            "sum_dxt": 0.0,
            "total_p90": 0.0,
            "prog_p90": 0.0,
            "f3_p90": 0.0,
            "xt_p90": 0.0,
            "neg_xt_p90": 0.0,
            "minutes": mins,
            "long_acc_pct": 0.0,
            "high_xt_p90": 0.0,
            "dz_p90": 0.0,
        }
    successful = int(df["is_won"].sum())
    unsuccessful = total - successful
    accuracy = successful / total * 100.0
    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int(
        (~df["is_won"] & df.apply(
            lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
            axis=1
        )).sum()
    )
    progressive_attempted = progressive_total + progressive_unsuccessful
    progressive_accuracy = (progressive_total / progressive_attempted * 100.0) if progressive_attempted else 0.0
    to_final_third = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    to_final_third_total = int(to_final_third.sum())
    to_final_third_success = int((to_final_third & df["is_won"]).sum())
    to_final_third_accuracy = (to_final_third_success / to_final_third_total * 100.0) if to_final_third_total else 0.0
    long_passes = df[df["pass_distance"] > 25.0]
    long_total = len(long_passes)
    long_success = int(long_passes["is_won"].sum())
    long_acc_pct = (long_success / long_total * 100.0) if long_total > 0 else 0.0
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
        "fwd": fwd,
        "fwd_pct": round(fwd / total * 100.0, 1),
        "bwd": bwd,
        "bwd_pct": round(bwd / total * 100.0, 1),
        "lat": lat,
        "lat_pct": round(lat / total * 100.0, 1),
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
        "dz_p90": round(dz_passes * p90_factor, 2),
    }

def compute_match_scores(dfs_dict, defensive_dfs_dict=None):
    records = []
    for m_name, df_m in dfs_dict.items():
        s = compute_stats(df_m, m_name)
        total_passes = s['total_passes']
        if total_passes == 0:
            continue
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
            'prog_acc_pct': s['progressive_accuracy_pct'],
        })
    df_scores = pd.DataFrame(records)
    if df_scores.empty:
        return df_scores
    def normalize_fixed(series, val_min, val_max):
        clipped_series = series.clip(lower=val_min, upper=val_max)
        if val_max == val_min:
            return pd.Series([70.0] * len(series))
        return 40 + ((clipped_series - val_min) / (val_max - val_min)) * 60
    df_scores['xt_norm'] = normalize_fixed(df_scores['xt_p90'], val_min=0.05, val_max=0.45)
    df_scores['prog_norm'] = normalize_fixed(df_scores['prog_p90'], val_min=1.0, val_max=15.0)
    df_scores['f3_norm'] = normalize_fixed(df_scores['f3_p90'], val_min=1.0, val_max=15.0)
    df_scores['pos_pct_norm'] = normalize_fixed(df_scores['pos_pct'], val_min=25.0, val_max=75.0)
    df_scores['total_p90_norm'] = normalize_fixed(df_scores['total_p90'], val_min=10.0, val_max=85.0)
    df_scores['neg_xt_norm'] = normalize_fixed(df_scores['neg_xt_p90'], val_min=-0.15, val_max=0.00)
    df_scores['Grade'] = (
        df_scores['xt_norm'] * 0.30 +
        df_scores['prog_norm'] * 0.20 +
        df_scores['f3_norm'] * 0.20 +
        df_scores['pos_pct_norm'] * 0.10 +
        df_scores['total_p90_norm'] * 0.10 +
        df_scores['neg_xt_norm'] * 0.10
    )
    if defensive_dfs_dict is not None:
        def_bonus_list = []
        duels_p90_list = []
        duels_won_pct_list = []
        interceptions_p90_list = []
        duels_won_p90_list = []
        int_xt_avg_list = []
        funnel_p90_list = []
        for _, row in df_scores.iterrows():
            team = row['match'].split('(')[0].strip()
            bonus = 0
            dp = 0; dwp = 0; ip = 0
            dwp90 = 0; int_xt_avg_rounded = 0
            funnel_p90_val = 0.0
            for def_name, def_df in defensive_dfs_dict.items():
                def_team = def_name.split('(')[0].strip()
                if def_team == team:
                    mins = get_match_minutes(def_name)
                    p90 = 90.0 / mins if mins > 0 else 1.0
                    duels_won = int(def_df["is_duel_won"].sum())
                    duels_lost = int(def_df["is_duel_lost"].sum())
                    total_duels = duels_won + duels_lost
                    dwp = (duels_won / total_duels * 100.0) if total_duels > 0 else 0
                    dp = round(total_duels * p90, 1)
                    dwp90 = round(duels_won * p90, 1)
                    interceptions = int(def_df["is_interception"].sum())
                    ip = round(interceptions * p90, 1)
                    funnel_count = int(def_df["in_funnel"].sum())
                    funnel_p90_val = round(funnel_count * p90, 1)
                    int_df = def_df[def_df["is_interception"]]
                    if len(int_df) > 0:
                        int_xt_vals = [xt_value(float(r["x"]), float(r["y"])) for _, r in int_df.iterrows()]
                        int_xt_avg = float(np.mean(int_xt_vals)) if int_xt_vals else 0
                    else:
                        int_xt_avg = 0
                    int_xt_avg_rounded = round(int_xt_avg, 3)
                    duel_bonus_val = (dwp / 100.0) * min(dp / 20.0, 1.0) * 5
                    int_bonus_val = min(ip / 10.0, 1.0) * (0.5 + int_xt_avg * 2) * 3
                    bonus = duel_bonus_val + int_bonus_val
                    break
            def_bonus_list.append(round(bonus, 2))
            duels_p90_list.append(dp)
            duels_won_pct_list.append(round(dwp, 1))
            interceptions_p90_list.append(ip)
            duels_won_p90_list.append(dwp90)
            int_xt_avg_list.append(int_xt_avg_rounded)
            funnel_p90_list.append(funnel_p90_val)
        df_scores['def_bonus'] = def_bonus_list
        df_scores['duels_p90'] = duels_p90_list
        df_scores['duels_won_pct'] = duels_won_pct_list
        df_scores['interceptions_p90'] = interceptions_p90_list
        df_scores['duels_won_p90'] = duels_won_p90_list
        df_scores['int_xt_avg'] = int_xt_avg_list
        df_scores['funnel_p90'] = funnel_p90_list
    else:
        df_scores['def_bonus'] = 0.0
        df_scores['duels_p90'] = 0.0
        df_scores['duels_won_pct'] = 0.0
        df_scores['interceptions_p90'] = 0.0
        df_scores['duels_won_p90'] = 0.0
        df_scores['int_xt_avg'] = 0.0
        df_scores['funnel_p90'] = 0.0
    df_scores['pass_grade'] = df_scores['Grade'].round(1).copy()
    def _norm_def(s, lo, hi):
        clipped = s.clip(lower=lo, upper=hi)
        if hi <= lo:
            return pd.Series([70.0] * len(s))
        return 40 + ((clipped - lo) / (hi - lo)) * 60
    df_scores['duels_won_pct_norm'] = _norm_def(df_scores['duels_won_pct'], 0, 100)
    df_scores['int_xt_norm'] = _norm_def(df_scores['int_xt_avg'], 0, 0.30)
    df_scores['duels_won_p90_norm'] = _norm_def(df_scores['duels_won_p90'], 0, 15)
    df_scores['interceptions_p90_norm'] = _norm_def(df_scores['interceptions_p90'], 0, 15)
    df_scores['funnel_p90_norm'] = _norm_def(df_scores['funnel_p90'], 0, 15)
    df_scores['def_grade'] = (
        df_scores['duels_won_pct_norm'] * 0.30 +
        df_scores['funnel_p90_norm'] * 0.15 +
        df_scores['int_xt_norm'] * 0.25 +
        df_scores['duels_won_p90_norm'] * 0.15 +
        df_scores['interceptions_p90_norm'] * 0.15
    ).round(1)
    df_scores['Grade'] = (df_scores['pass_grade'] * 0.75 + df_scores['def_grade'] * 0.25).round(1)
    return df_scores

def compute_defensive_stats(df: pd.DataFrame, match_name: str) -> dict:
    total_actions = len(df)
    if match_name == "All Matches":
        mins = sum(get_match_minutes(k) for k in defensive_dfs_by_match)
    else:
        mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    duels_won = int(df["is_duel_won"].sum())
    duels_lost = int(df["is_duel_lost"].sum())
    total_duels = duels_won + duels_lost
    duels_won_pct = (duels_won / total_duels * 100.0) if total_duels > 0 else 0.0
    interceptions = int(df["is_interception"].sum())
    attacking_half = df[df["is_attacking_half"]]
    actions_attacking = len(attacking_half)
    interceptions_attacking = int(attacking_half["is_interception"].sum())
    funnel_actions = int(df["in_funnel"].sum())
    return {
        "total_actions": total_actions,
        "total_actions_p90": round(total_actions * p90_factor, 1),
        "actions_attacking": actions_attacking,
        "actions_attacking_p90": round(actions_attacking * p90_factor, 1),
        "total_duels": total_duels,
        "duels_p90": round(total_duels * p90_factor, 1),
        "duels_won_pct": round(duels_won_pct, 1),
        "duels_won": duels_won,
        "interceptions": interceptions,
        "interceptions_p90": round(interceptions * p90_factor, 1),
        "interceptions_attacking": interceptions_attacking,
        "interceptions_attacking_p90": round(interceptions_attacking * p90_factor, 1),
        "funnel_actions": funnel_actions,
        "funnel_actions_p90": round(funnel_actions * p90_factor, 1),
    }

def compute_defensive_match_scores(dfs_dict):
    records = []
    for m_name, df_m in dfs_dict.items():
        mins = get_match_minutes(m_name)
        p90 = 90.0 / mins if mins > 0 else 1.0
        duels_won = int(df_m["is_duel_won"].sum())
        duels_lost = int(df_m["is_duel_lost"].sum())
        total_duels = duels_won + duels_lost
        interceptions = int(df_m["is_interception"].sum())
        funnel_count = int(df_m["in_funnel"].sum())
        records.append({
            'match': m_name,
            'duels_p90': round(total_duels * p90, 1),
            'interceptions_p90': round(interceptions * p90, 1),
            'funnel_p90': round(funnel_count * p90, 1),
        })
    return pd.DataFrame(records)

def compute_defensive_evolution_df(dfs_dict, df_scores=None):
    records = []
    for m_name, df_m in dfs_dict.items():
        mins = get_match_minutes(m_name)
        p90 = 90.0 / mins if mins > 0 else 1.0
        duels_won = int(df_m["is_duel_won"].sum())
        duels_lost = int(df_m["is_duel_lost"].sum())
        total_duels = duels_won + duels_lost
        duels_won_pct = (duels_won / total_duels * 100.0) if total_duels > 0 else 0.0
        interceptions = int(df_m["is_interception"].sum())
        attacking_half = df_m[df_m["is_attacking_half"]]
        actions_attacking = len(attacking_half)
        int_df = df_m[df_m["is_interception"]]
        if len(int_df) > 0:
            int_xt_vals = [xt_value(float(r["x"]), float(r["y"])) for _, r in int_df.iterrows()]
            int_xt_avg = float(np.mean(int_xt_vals)) if int_xt_vals else 0
        else:
            int_xt_avg = 0
        funnel_count = int(df_m["in_funnel"].sum())
        grade = None
        if df_scores is not None and 'Grade' in df_scores.columns and 'match' in df_scores.columns:
            team = m_name.split('(')[0].strip()
            for _, row in df_scores.iterrows():
                if row['match'].split('(')[0].strip() == team:
                    grade = row['Grade']
                    break
        records.append({
            'match': m_name,
            'def_actions_p90': round(len(df_m) * p90, 1),
            'duels_won_p90': round(duels_won * p90, 1),
            'interceptions_p90': round(interceptions * p90, 1),
            'duels_won_pct': round(duels_won_pct, 1),
            'actions_attacking_p90': round(actions_attacking * p90, 1),
            'int_xt_avg': round(int_xt_avg, 3),
            'funnel_actions_p90': round(funnel_count * p90, 1),
            'grade': round(grade, 1) if grade is not None else None,
        })
    return pd.DataFrame(records)

# UI HELPERS
def _safe_pct_diff(a: float, b: float) -> float:
    base = max(abs(b), 1.0)
    pct = (abs(a - b) / base) * 100.0
    return min(pct, 999.0)

def _arrow_html(val_game: float, val_avg: float) -> str:
    if np.isclose(val_game, val_avg, atol=1e-9):
        return ""
    if abs(val_game) < 1 and abs(val_avg) < 1:
        return ""
    if val_game > val_avg:
        pct = _safe_pct_diff(val_game, val_avg)
        return f' <span style="display:inline-block;font-size:11px;font-weight:700;color:#fff;background:#059669;padding:1px 7px;border-radius:10px;vertical-align:middle;line-height:1.6">+{pct:.0f}%</span>'
    else:
        pct = _safe_pct_diff(val_avg, val_game)
        return f' <span style="display:inline-block;font-size:11px;font-weight:700;color:#fff;background:#dc2626;padding:1px 7px;border-radius:10px;vertical-align:middle;line-height:1.6">-{pct:.0f}%</span>'

def section_card(title, border_color, items):
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:{bg};border:1px solid {bd};border-radius:10px;padding:14px;margin-bottom:8px">'
    html += f'<div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:10px;letter-spacing:0.3px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px">{title}</div>'
    html += f'<div style="opacity:0.88">'
    for idx, item in enumerate(items):
        label = item[0]
        value = item[1]
        sub = item[2] if len(item) > 2 else ""
        tooltip = item[3] if len(item) > 3 else ""
        is_last = idx == len(items) - 1
        sep = "" if is_last else 'style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:6px;margin-bottom:6px"'
        html += f'<div {sep}>'
        html += f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        if tooltip:
            label_html = f'<span style="font-size:15px;color:#ffffff;font-weight:600;cursor:help;border-bottom:1px dotted rgba(255,255,255,0.15)" title="{tooltip}">{label}</span>'
            label_html += f'<span style="display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;font-size:9px;font-weight:700;color:#888;background:rgba(0,0,0,0.25);margin-left:4px;cursor:help;vertical-align:middle" title="{tooltip}">?</span>'
            html += f'<div>{label_html}</div>'
        else:
            html += f'<div style="font-size:15px;color:#ffffff;font-weight:600">{label}</div>'
        html += f'<div style="text-align:right">'
        html += f'<div style="font-size:20px;font-weight:800;color:#ffffff;line-height:1.2">{value}</div>'
        if sub:
            html += f'<div style="font-size:12px;font-style:italic;color:#ffffff;opacity:0.55;margin-top:1px">{sub}</div>'
        html += '</div>'
        html += '</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def cmp_section_card(title, border_color, items):
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:{bg};border:1px solid {bd};border-radius:10px;padding:14px;margin-bottom:8px">'
    html += f'<div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:10px;letter-spacing:0.3px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px">{title}</div>'
    html += f'<div style="opacity:0.88">'
    for idx, item in enumerate(items):
        label = item[0]
        val_game = item[1]
        val_avg = item[2]
        disp_game = item[3] if len(item) > 3 else str(val_game)
        disp_avg = item[4] if len(item) > 4 else str(val_avg)
        tooltip = item[5] if len(item) > 5 else ""
        arrow = _arrow_html(float(val_game), float(val_avg))
        is_last = idx == len(items) - 1
        sep = "" if is_last else 'style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:6px;margin-bottom:6px"'
        html += f'<div {sep}>'
        html += f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        if tooltip:
            label_html = f'<span style="font-size:15px;color:#ffffff;font-weight:600;cursor:help;border-bottom:1px dotted rgba(255,255,255,0.15)" title="{tooltip}">{label}</span>'
            label_html += f'<span style="display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;font-size:9px;font-weight:700;color:#888;background:rgba(0,0,0,0.25);margin-left:4px;cursor:help;vertical-align:middle" title="{tooltip}">?</span>'
            html += f'<div>{label_html}</div>'
        else:
            html += f'<div style="font-size:15px;color:#ffffff;font-weight:600">{label}</div>'
        html += f'<div style="text-align:right">'
        html += f'<div style="font-size:20px;font-weight:800;color:#ffffff;line-height:1.2">{disp_game}{arrow}</div>'
        html += f'<div style="font-size:11px;color:#ffffff;opacity:0.55;margin-top:2px">AVG: {disp_avg}</div>'
        html += '</div>'
        html += '</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# DRAW HELPERS (PITCH)
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
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11,
        linewidth=1.6, color="#aaaaaa"
    ))
    fig.text(0.50 + ox, 0.012, "Attacking Direction", ha="center", va="bottom",
             transform=fig.transFigure, fontsize=7.5, color="#aaaaaa")

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
        if is_lost:
            color, alpha = COLOR_FAIL, 0.72
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.3, headwidth=2.0, headlength=2.0,
                     ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row["x_start"], row["y_start"], s=32, marker="o",
                      color=color, edgecolors="white", linewidths=0.6,
                      ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
            Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
            Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
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
                                   facecolor=cmap(norm(value)),
                                   edgecolor=(1, 1, 1, 0.12), lw=0.5, alpha=0.95, zorder=2))
            ax.text((x0_ + x1_) / 2, (y0 + y1) / 2, str(value),
                    ha="center", va="center",
                    color="#000000" if value <= threshold else "#ffffff",
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
        xa = x0 + (x1 - x0) * t0; ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1; yb = y0 + (y1 - y0) * t1
        alpha = 0.85 * (0.15 + 0.85 * t1)
        lw = 2.5 * (0.80 + 0.20 * t1)
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=4, solid_capstyle="round")
    ax.scatter(x0, y0, s=20, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, zorder=5, alpha=0.85)
    ax.scatter(x1, y1, s=32, marker="o", facecolors=color, edgecolors="white", linewidths=0.9, zorder=6, alpha=0.85)

def draw_top_xt_map(df, top_n=5):
    fig, ax, pitch = _base_pitch()
    top_passes = (df[(df["is_won"]) & (df["delta_xt_adj"] > 0)]
            .sort_values("delta_xt_adj", ascending=False).head(top_n).copy().reset_index(drop=True))
    if not top_passes.empty:
        for _, row in top_passes.iterrows():
            val = float(row["delta_xt_adj"])
            color = CMAP_TOP10(NORM_TOP10(np.clip(val, 0.05, 0.40)))
            _draw_comet_arrow(ax, float(row["x_start"]), float(row["y_start"]),
                              float(row["x_end"]), float(row["y_end"]), color)
        sm = plt.cm.ScalarMappable(cmap=CMAP_TOP10, norm=NORM_TOP10)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
        cbar.set_label("Pass Impact", color="#ffffff", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

# DEFENSIVE PITCH DRAW HELPERS
COLOR_DUEL_WON = "#10b981"
COLOR_DUEL_LOST = "#E07070"
COLOR_INTERCEPTION = "#2F80ED"

def draw_defensive_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        if row["is_duel_won"]:
            color, marker, s, alpha = COLOR_DUEL_WON, "o", 90, 0.85
        elif row["is_duel_lost"]:
            color, marker, s, alpha = COLOR_DUEL_LOST, "X", 100, 0.85
        else:
            color, marker, s, alpha = COLOR_INTERCEPTION, "^", 80, 0.85
        pitch.scatter(row["x"], row["y"], s=s, marker=marker, color=color,
                      edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DUEL_WON, markersize=7, label="Duel Won", alpha=0.90),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_DUEL_LOST, markersize=8, label="Duel Lost", alpha=0.90),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_INTERCEPTION, markersize=7, label="Interception", alpha=0.90),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_funnel_protection_map(df):
    """Map showing defensive actions: golden stars inside funnel zone, faded outside."""
    fig, ax, pitch = _base_pitch()
    funnel_rect = Rectangle(
        (0, PENALTY_AREA_Y_MIN), FUNNEL_X_EXTEND, PENALTY_AREA_Y_MAX - PENALTY_AREA_Y_MIN,
        facecolor="#ffd700", edgecolor="#ffd700", lw=1.5, linestyle="--", alpha=0.12, zorder=2
    )
    ax.add_patch(funnel_rect)
    for _, row in df.iterrows():
        x, y = float(row["x"]), float(row["y"])
        in_funnel = bool(row.get("in_funnel", is_in_funnel_zone(x, y)))
        if in_funnel:
            marker, s, color, edge = "*", 120, "#ffd700", "#b8860b"
        else:
            marker, s, color, edge = "o", 60, "#888888", "#555555"
        pitch.scatter(x, y, s=s, marker=marker, color=color,
                      edgecolors=edge, linewidths=0.5, ax=ax, zorder=6, alpha=0.85)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd700", markersize=9, label="Funnel Action", alpha=0.95),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=6, label="Other Action", alpha=0.50),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_defensive_heatmap(df):
    corridors = {
        "Left": (LANE_LEFT_MIN, FIELD_Y),
        "Center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "Right": (0.0, LANE_RIGHT_MAX),
    }
    corridor_data = {}
    for cname, (y0, y1) in corridors.items():
        mask = (df["y"] >= y0) & (df["y"] < y1)
        corr_df = df[mask]
        total = len(corr_df)
        duels_total = int(corr_df["is_duel"].sum())
        duels_won = int(corr_df["is_duel_won"].sum())
        corridor_data[cname] = {"count": total, "duels_won": duels_won, "duels_total": duels_total}
    all_counts = [d["count"] for d in corridor_data.values()]
    vmax = max(1, max(all_counts))
    cmap_def = LinearSegmentedColormap.from_list("def_corr", ["#ffffff", "#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a5f"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        d = corridor_data[cname]
        value = d["count"]
        ax.add_patch(Rectangle((0, y0), FIELD_X, y1 - y0,
                               facecolor=cmap_def(norm(value)),
                               edgecolor=(1, 1, 1, 0.15), lw=0.5, alpha=0.95, zorder=2))
        duel_pct = (d["duels_won"] / d["duels_total"] * 100) if d["duels_total"] > 0 else None
        if duel_pct is not None:
            label = f"{cname}\nTotal: {value}\nWon: {d['duels_won']}/{d['duels_total']} ({duel_pct:.0f}%)"
        else:
            label = f"{cname}\nTotal: {value}"
        ax.text(FIELD_X / 2, (y0 + y1) / 2, label,
                ha="center", va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=9, fontweight="600", zorder=4)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.20, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.20, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_defensive_xt_map(df):
    fig, ax, pitch = _base_pitch()
    vals = [xt_value(float(row["x"]), float(row["y"])) for _, row in df.iterrows()]
    vals = np.array(vals)
    if len(vals) > 0:
        vmin, vmax_vals = float(vals.min()), float(vals.max())
        norm_def = Normalize(vmin=vmin, vmax=vmax_vals) if vmax_vals > vmin else Normalize(vmin=0, vmax=1)
        cmap_def_xt = LinearSegmentedColormap.from_list("def_xt", ["#2d1b69", "#4a148c", "#7b1fa2", "#ab47bc", "#ce93d8"])
        for i, (_, row) in enumerate(df.iterrows()):
            marker, s = ("o", 85) if row["is_duel_won"] else ("X", 95) if row["is_duel_lost"] else ("^", 75)
            pitch.scatter(row["x"], row["y"], s=s, marker=marker,
                          color=cmap_def_xt(norm_def(vals[i])),
                          edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=0.85)
        sm = plt.cm.ScalarMappable(cmap=cmap_def_xt, norm=norm_def)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
        cbar.set_label("xT Threat", color="#ffffff", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

# PLOTLY CHARTS
def draw_total_passes_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["total_p90"]
    mean_val = y.mean()
    fig.add_trace(go.Scatter(
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color="#00d2ff", width=3, shape='spline'),
        marker=dict(size=8, color="#00d2ff"),
        fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.05)',
        name="Total Passes",
        hovertemplate="%{customdata}<br>Total Passes: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_val] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_val:.1f}", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=290, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Total Passes", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_grade_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y_grade = df_scores["Grade"]
    y_pass_grade = df_scores["pass_grade"]
    y_def_grade = df_scores["def_grade"]
    mean_grade = y_grade.mean()
    pass_metrics = {
        'Pass Impact Value': 'xt_p90',
        'Progressive': 'prog_p90',
        'Final Third': 'f3_p90',
        '% Positive Impact': 'pos_pct',
        'Total Passes': 'total_p90',
    }
    def_metrics = {
        '% Duels Won': 'duels_won_pct',
        'Funnel Actions': 'funnel_p90',
        'Interception xT': 'int_xt_avg',
        'Duels Won': 'duels_won_p90',
        'Interceptions': 'interceptions_p90',
    }
    pass_avgs = {name: df_scores[col].mean() for name, col in pass_metrics.items()}
    def_avgs = {name: df_scores[col].mean() for name, col in def_metrics.items()}
    hover_texts_pass = []
    hover_texts_def = []
    for _, row in df_scores.iterrows():
        pass_diffs = {}
        for name, col in pass_metrics.items():
            avg = pass_avgs[name]
            if abs(avg) > 1e-9:
                pass_diffs[name] = ((row[col] - avg) / abs(avg)) * 100
            else:
                pass_diffs[name] = 0.0
        def_diffs = {}
        for name, col in def_metrics.items():
            avg = def_avgs[name]
            if abs(avg) > 1e-9:
                def_diffs[name] = ((row[col] - avg) / abs(avg)) * 100
            else:
                def_diffs[name] = 0.0
        if row['Grade'] >= mean_grade:
            best_pass = max(pass_diffs, key=pass_diffs.get)
            p_val = pass_diffs[best_pass]
            p_color = '#34d399'
            p_sign = '+'
            best_def = max(def_diffs, key=def_diffs.get)
            d_val = def_diffs[best_def]
            d_color = '#34d399'
            d_sign = '+'
        else:
            best_pass = min(pass_diffs, key=pass_diffs.get)
            p_val = pass_diffs[best_pass]
            p_color = '#f87171'
            p_sign = ''
            best_def = min(def_diffs, key=def_diffs.get)
            d_val = def_diffs[best_def]
            d_color = '#f87171'
            d_sign = ''
        hover_texts_pass.append(f"Passe: {best_pass} <span style='color:{p_color}'>{p_sign}{p_val:.1f}%</span>")
        hover_texts_def.append(f"Defesa: {best_def} <span style='color:{d_color}'>{d_sign}{d_val:.1f}%</span>")
    customdata = np.stack((df_scores["match"], hover_texts_pass, hover_texts_def), axis=-1)
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_grade, customdata=customdata,
        mode='lines+markers',
        line=dict(color=C_BLUE_DARK, width=3, shape='spline'),
        marker=dict(size=8, color=C_BLUE_DARK),
        fill='tozeroy', fillcolor=f'rgba(26, 86, 219, 0.05)',
        name="Combined Grade",
        hovertemplate="<span style='color:#ffd700'>%{customdata[0]}</span><br>Grade: %{y:.1f}<br>%{customdata[1]}<br>%{customdata[2]}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_pass_grade,
        mode='lines+markers',
        line=dict(color="#10b981", width=1, dash='dot'),
        marker=dict(size=5, color="#10b981", symbol='circle-open'),
        opacity=0.15,
        name="Pass Grade (only)",
        hovertemplate="%{x}<br>Pass Grade: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_def_grade,
        mode='lines+markers',
        line=dict(color="#a78bfa", width=1, dash='dot'),
        marker=dict(size=5, color="#a78bfa", symbol='circle-open'),
        opacity=0.15,
        name="Defensive Grade",
        hovertemplate="%{x}<br>Defensive Grade: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_grade] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_grade:.1f}", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=370, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[40, 100], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Combined Grade", font=dict(size=14, color="#ffffff"))
    )
    return fig

def draw_progressive_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["prog_p90"]
    mean_prog = y.mean()
    fig.add_trace(go.Scatter(
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color="#10b981", width=3, shape='spline'),
        marker=dict(size=8, color="#10b981"),
        fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.05)',
        name="Progressive Passes",
        hovertemplate="%{customdata}<br>Progressive: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_prog] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_prog:.1f}", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=290, margin=dict(l=20, r=20, t=40, b=20),
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
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color="#8b5cf6", width=3, shape='spline'),
        marker=dict(size=8, color="#8b5cf6"),
        fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.05)',
        name="Final Third Passes",
        hovertemplate="%{customdata}<br>Final Third: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_f3] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_f3:.1f}", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=290, margin=dict(l=20, r=20, t=40, b=20),
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
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color="#f59e0b", width=3, shape='spline'),
        marker=dict(size=8, color="#f59e0b"),
        fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.05)',
        name="Pass Impact Value",
        hovertemplate="%{customdata}<br>Pass Impact Value: %{y:.2f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_xt] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_xt:.2f}", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=290, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Pass Impact Value", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_positive_impact_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["pos_pct"]
    mean_val = y.mean()
    fig.add_trace(go.Scatter(
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color="#f43f5e", width=3, shape='spline'),
        marker=dict(size=8, color="#f43f5e"),
        fill='tozeroy', fillcolor='rgba(244, 63, 94, 0.05)',
        name="% Positive Impact",
        hovertemplate="%{customdata}<br>% Positive Impact: %{y:.1f}%"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_val] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_val:.1f}%", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=290, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="% Positive Impact", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_defensive_duels_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["duels_p90"]
    mean_val = y.mean()
    fig.add_trace(go.Scatter(
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color="#f97316", width=3, shape='spline'),
        marker=dict(size=8, color="#f97316"),
        fill='tozeroy', fillcolor='rgba(249, 115, 22, 0.05)',
        name="Defensive Duels",
        hovertemplate="%{customdata}<br>Duels p90: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_val] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_val:.1f}", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=290, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Defensive Duels", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_defensive_interceptions_chart(df_scores):
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores["interceptions_p90"]
    mean_val = y.mean()
    fig.add_trace(go.Scatter(
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color="#8b5cf6", width=3, shape='spline'),
        marker=dict(size=8, color="#8b5cf6"),
        fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.05)',
        name="Interceptions",
        hovertemplate="%{customdata}<br>Interceptions p90: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=[mean_val] * len(x_labels),
        mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
        name=f"Avg: {mean_val:.1f}", hoverinfo='skip'
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=290, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Interceptions", font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_comparison_bar(title, val_first, val_last, suffix=""):
    color_last = "#10b981" if val_last >= val_first else "#E07070"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["First 9 Matches", "Last 9 Matches"],
        y=[val_first, val_last],
        marker_color=["#444466", color_last],
        text=[f"{val_first:.2f}{suffix}", f"{val_last:.2f}{suffix}"],
        textposition='auto',
        width=[0.35, 0.35],
        hovertemplate="%{x}<br>" + title + ": %{y:.2f}" + suffix
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=250, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, max(val_first, val_last) * 1.2],
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        title=dict(text=title, font=dict(size=14, color="#a0a0b5")),
        showlegend=False, bargap=0.2
    )
    return fig

# SIDEBAR
st.sidebar.markdown("""
<div style="text-align:center;padding:8px 0 4px 0">
    <div style="font-size:20px;font-weight:300;letter-spacing:2px;color:#a0a0b5;text-transform:uppercase">Pass Stats</div>
    <div style="font-size:13px;font-weight:600;color:#ffffff;margin-top:-2px">Dashboard</div>
</div>
<div style="border-bottom:1px solid #2a2a3e;margin:6px 0 12px 0"></div>
<div style="font-size:11px;font-weight:500;letter-spacing:1px;color:#6b6b80;text-transform:uppercase;margin:0 10px 6px 10px">2026 Season</div>
<div style="font-size:16px;font-weight:600;color:#e0e0f0;margin:0 10px 12px 10px">Generic Player</div>
""", unsafe_allow_html=True)

img_path = "player_photo.png"
if os.path.exists(img_path):
    st.sidebar.image(img_path, use_container_width=True)

st.sidebar.markdown("""
<div style="border-bottom:1px solid #2a2a3e;margin:16px 0 8px 0"></div>
""", unsafe_allow_html=True)

num_matches = len(dfs_by_match)
all_match_stats = [compute_stats(dfs_by_match[m], m) for m in dfs_by_match]

# TABS & LAYOUT
tab_graf, tab_dash, tab_evo = st.tabs(["Charts & Analysis", "Detailed Dashboard", "Development"])

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

        st.markdown("### Passes")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            section_card("📋 Overview", C_BLUE_PASTEL, [
                ("Passes p90", f"{avg_total_p90:.1f}", f"Total: {total_passes_all}"),
                ("Successful %", f"{avg_acc:.1f}%", f"Total: {total_succ_all}"),
            ])
        with col_s2:
            section_card("📊 Advanced", C_GREEN_PASTEL, [
                ("Progressive p90", f"{avg_prog_p90:.1f}", f"Total: {total_prog_all}"),
                ("Final Third p90", f"{avg_f3_p90:.1f}", f"Total: {total_f3_all}"),
            ])
        with col_s3:
            section_card("⚡ Impact", C_AMBER_PASTEL, [
                ("% Positive Impact", f"{avg_pos_pct:.1f}%", f"Total: {total_pos_all}",
                 "Passes that generated a positive impact based on where they ended on the field"),
                ("Pass Impact Value", f"{avg_xt_p90:.3f}", f"Total: {total_xt_all:.3f}",
                 "Calculation used to evaluate the offensive value added by a pass."),
            ])

        st.markdown("", unsafe_allow_html=True)

        defensive_num_matches = len(defensive_dfs_by_match)
        defensive_all_stats = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]
        if defensive_num_matches > 0:
            total_def_actions_all = sum(s['total_actions'] for s in defensive_all_stats)
            total_def_att_all = sum(s['actions_attacking'] for s in defensive_all_stats)
            total_duels_all = sum(s['total_duels'] for s in defensive_all_stats)
            total_duels_won_all = sum(s['duels_won'] for s in defensive_all_stats)
            total_interceptions_all = sum(s['interceptions'] for s in defensive_all_stats)
            total_int_att_all = sum(s['interceptions_attacking'] for s in defensive_all_stats)
            avg_def_actions_p90 = sum(s['total_actions_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_def_att_p90 = sum(s['actions_attacking_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_duels_p90 = sum(s['duels_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_duels_won_pct = sum(s['duels_won_pct'] for s in defensive_all_stats) / defensive_num_matches
            avg_interceptions_p90 = sum(s['interceptions_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_int_att_p90 = sum(s['interceptions_attacking_p90'] for s in defensive_all_stats) / defensive_num_matches

            st.markdown("### Defensive Actions")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                section_card("🛡️ General", C_BLUE_PASTEL, [
                    ("Defensive Actions p90", f"{avg_def_actions_p90:.1f}", f"Total: {total_def_actions_all}"),
                    ("Actions in Opp. Field p90", f"{avg_def_att_p90:.1f}", f"Total: {total_def_att_all}"),
                ])
            with col_d2:
                section_card("⚔️ Duels", C_GREEN_PASTEL, [
                    ("Defensive Duels p90", f"{avg_duels_p90:.1f}", f"Total: {total_duels_all}"),
                    ("% Duels Won", f"{avg_duels_won_pct:.1f}%", f"({total_duels_won_all}/{total_duels_all})"),
                ])
            with col_d3:
                section_card("❌ Interceptions", C_AMBER_PASTEL, [
                    ("Interceptions p90", f"{avg_interceptions_p90:.1f}", f"Total: {total_interceptions_all}"),
                    ("Interceptions in Opp Field p90", f"{avg_int_att_p90:.1f}", f"Total: {total_int_att_all}"),
                ])

        st.markdown(f'<div style="text-align:center;font-size:12px;color:#666666;margin-top:12px">{num_matches} matches collected</div>', unsafe_allow_html=True)
        st.markdown("", unsafe_allow_html=True)

        df_scores = compute_match_scores(dfs_by_match, defensive_dfs_by_match)
        if not df_scores.empty:
            st.markdown("### Grade per Match")
            fig_scores = draw_grade_chart(df_scores)
            st.plotly_chart(fig_scores, use_container_width=True)

            with st.expander("How is the Grade calculated?"):
                st.markdown("""
**Grade**

**Pass Grade**
- **Pass Impact:** Measures actual danger created by passes.
- **Progressive Passes:** Line-breaking ability.
- **Final Third Passes:** Attacking presence in dangerous zones.
- **% Positive Pass Impact:** Efficiency of threat generation.
- **Total Passes:** Overall involvement.
- **Negative Pass Impact:** Penalty for passes that lose threat.

**Defensive Grade**
- **Duels Won %:** Rewards efficiency in defensive duels.
- **Funnel Defensive Actions:** Rewards actions in the defensive funnel zone.
- **Interception xT:** Rewards interceptions in high-threat zones.
- **Duels Won Count:** Rewards volume of duels won.
- **Interceptions Count:** Rewards volume of interceptions.
""")

            st.markdown("", unsafe_allow_html=True)
            st.markdown("### Stats")
            fig_total = draw_total_passes_chart(df_scores)
            st.plotly_chart(fig_total, use_container_width=True)
            fig_prog = draw_progressive_chart(df_scores)
            st.plotly_chart(fig_prog, use_container_width=True)
            fig_f3 = draw_final_third_chart(df_scores)
            st.plotly_chart(fig_f3, use_container_width=True)
            fig_xt = draw_xt_chart(df_scores)
            st.plotly_chart(fig_xt, use_container_width=True)
            fig_pos = draw_positive_impact_chart(df_scores)
            st.plotly_chart(fig_pos, use_container_width=True)

            df_def_scores = compute_defensive_match_scores(defensive_dfs_by_match)
            if not df_def_scores.empty:
                st.markdown("### Defensive Actions")
                fig_duels = draw_defensive_duels_chart(df_def_scores)
                st.plotly_chart(fig_duels, use_container_width=True)
                fig_interceptions = draw_defensive_interceptions_chart(df_def_scores)
                st.plotly_chart(fig_interceptions, use_container_width=True)
            else:
                st.warning("Not enough data to generate charts.")

with tab_dash:
    sub_tab_passes, sub_tab_def = st.tabs(["Passes", "Defensive Actions"])

    with sub_tab_passes:
        st.markdown("### Match Filters")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            pass_match_options = ["All Matches"] + list(dfs_by_match.keys())
            selected_match = st.selectbox("Select Match", options=pass_match_options, index=0, key="pass_match")
        with col_f2:
            pass_filter = st.radio(
                "Pass Type",
                ["All", "Successful", "Unsuccessful", "Progressive", "Final Third"],
                index=0, horizontal=True, key="pass_filter"
            )

        if selected_match == "All Matches":
            df_game_filtered = pd.concat(dfs_by_match.values(), ignore_index=True)
            match_name_for_stats = "All Matches"
        else:
            df_game_filtered = dfs_by_match[selected_match].copy()
            match_name_for_stats = selected_match

        def apply_filter(df):
            if pass_filter == "Successful":
                return df[df["is_won"]].copy()
            if pass_filter == "Unsuccessful":
                return df[~df["is_won"]].copy()
            if pass_filter == "Progressive":
                return df[df["progressive"]].copy()
            if pass_filter == "Final Third":
                return df[(df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)].copy()
            return df.copy()

        df_game = apply_filter(df_game_filtered)
        s_game = compute_stats(df_game, match_name_for_stats)
        s_avg = {}
        if num_matches > 0:
            for k in all_match_stats[0].keys():
                if isinstance(all_match_stats[0][k], (int, float)):
                    s_avg[k] = sum(s[k] for s in all_match_stats) / num_matches
                else:
                    s_avg[k] = 0
        else:
            s_avg = s_game.copy()

        force_avg = selected_match == "All Matches"
        if force_avg:
            s_game = s_avg.copy()

        st.markdown("---")
        img_pm_game, fig_pm_game = draw_pass_map(df_game); plt.close(fig_pm_game)
        img_ht_game, fig_ht_game = draw_corridor_heatmap(df_game); plt.close(fig_ht_game)
        top_n_xt = 10 if force_avg else 5
        img_xt_game, fig_xt_game = draw_top_xt_map(df_game, top_n=top_n_xt); plt.close(fig_xt_game)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Pass Map</div>', unsafe_allow_html=True)
            st.image(img_pm_game, use_container_width=True)
        with col_m2:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Zone Heatmap</div>', unsafe_allow_html=True)
            st.image(img_ht_game, use_container_width=True)
        with col_m3:
            label = "Top 10" if force_avg else "Top 5"
            st.markdown(f'<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">{label} Pass Impact</div>', unsafe_allow_html=True)
            st.image(img_xt_game, use_container_width=True)

        st.markdown("", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        if force_avg:
            with col_s1:
                section_card("📋 Pass Overview", C_BLUE_PASTEL, [
                    ("Total Passes", f"{s_game['total_p90']:.2f}"),
                    ("Successful %", f"{s_game['accuracy_pct']:.2f}%"),
                ])
            with col_s2:
                section_card("📊 Advanced", C_GREEN_PASTEL, [
                    ("Progressive", f"{s_game['prog_p90']:.2f}"),
                    ("Final Third", f"{s_game['f3_p90']:.2f}"),
                ])
            with col_s3:
                section_card("⚡ Impact", C_AMBER_PASTEL, [
                    ("% Positive Impact", f"{s_game['pos_pct']:.2f}%"),
                    ("Pass Impact Value", f"{s_game['xt_p90']:.3f}"),
                ])
        else:
            with col_s1:
                cmp_section_card("📋 Pass Overview", C_BLUE_PASTEL, [
                    ("Total Passes", s_game["total_p90"], f"{s_avg['total_p90']:.1f}"),
                    ("Successful %", s_game["accuracy_pct"], s_avg["accuracy_pct"],
                     f"{s_game['accuracy_pct']:.1f}%", f"{s_avg['accuracy_pct']:.1f}%"),
                ])
            with col_s2:
                cmp_section_card("📊 Advanced", C_GREEN_PASTEL, [
                    ("Progressive", s_game["prog_p90"], f"{s_avg['prog_p90']:.1f}"),
                    ("Final Third", s_game["f3_p90"], f"{s_avg['f3_p90']:.1f}"),
                ])
            with col_s3:
                cmp_section_card("⚡ Impact", C_AMBER_PASTEL, [
                    ("% Positive Impact", s_game["pos_pct"], s_avg["pos_pct"],
                     f"{s_game['pos_pct']:.1f}%", f"{s_avg['pos_pct']:.1f}%",
                     "Passes that generated a positive impact based on where they ended on the field"),
                    ("Pass Impact Value", s_game["xt_p90"], s_avg["xt_p90"],
                     f"{s_game['xt_p90']:.3f}", f"{s_avg['xt_p90']:.3f}",
                     "Calculation used to define the value of pass impact based on expected threat (xT) progression"),
                ])

    with sub_tab_def:
        st.markdown("### Match Filter")
        col_df1, col_df2 = st.columns(2)
        with col_df1:
            def_match_options = ["All Matches"] + list(defensive_dfs_by_match.keys())
            selected_def_match = st.selectbox("Select Match", options=def_match_options, index=0, key="def_match")
        with col_df2:
            def_type_filter = st.radio("Filter Type", ["All", "Duels Only", "Interceptions Only"],
                                       horizontal=True, key="def_type_filter")

        if selected_def_match == "All Matches":
            df_def_game_raw = pd.concat(defensive_dfs_by_match.values(), ignore_index=True)
            def_match_name_for_stats = "All Matches"
        else:
            df_def_game_raw = defensive_dfs_by_match[selected_def_match].copy()
            def_match_name_for_stats = selected_def_match

        if def_type_filter == "Duels Only":
            df_def_game = df_def_game_raw[df_def_game_raw["is_duel"]].copy()
        elif def_type_filter == "Interceptions Only":
            df_def_game = df_def_game_raw[df_def_game_raw["is_interception"]].copy()
        else:
            df_def_game = df_def_game_raw.copy()

        d_game = compute_defensive_stats(df_def_game, def_match_name_for_stats)
        def_all = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]
        d_avg = {}
        if len(def_all) > 0:
            for k in def_all[0].keys():
                if isinstance(def_all[0][k], (int, float)):
                    d_avg[k] = sum(s[k] for s in def_all) / len(def_all)
                else:
                    d_avg[k] = 0
        else:
            d_avg = d_game.copy()

        force_avg_def = selected_def_match == "All Matches"
        if force_avg_def:
            d_game = d_avg.copy()

        st.markdown("---")
        img_def_map, fig_def_map = draw_defensive_map(df_def_game); plt.close(fig_def_map)
        img_def_hm, fig_def_hm = draw_defensive_heatmap(df_def_game); plt.close(fig_def_hm)
        img_funnel, fig_funnel = draw_funnel_protection_map(df_def_game); plt.close(fig_funnel)

        col_dm1, col_dm2, col_dm3 = st.columns(3)
        with col_dm1:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Defensive Actions Map</div>', unsafe_allow_html=True)
            st.image(img_def_map, use_container_width=True)
        with col_dm2:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Defensive Heatmap</div>', unsafe_allow_html=True)
            st.image(img_def_hm, use_container_width=True)
        with col_dm3:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Funnel Protection Actions</div>', unsafe_allow_html=True)
            st.image(img_funnel, use_container_width=True)

        st.markdown("", unsafe_allow_html=True)
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        if force_avg_def:
            with col_ds1:
                section_card("🛡️ General", C_BLUE_PASTEL, [
                    ("Defensive Actions", f"{d_game['total_actions_p90']:.2f}"),
                    ("Actions in Opp. Field", f"{d_game['actions_attacking_p90']:.2f}"),
                ])
            with col_ds2:
                section_card("⚔️ Duels", C_GREEN_PASTEL, [
                    ("Defensive Duels", f"{d_game['duels_p90']:.2f}"),
                    ("% Duels Won", f"{d_game['duels_won_pct']:.2f}%"),
                ])
            with col_ds3:
                section_card("👁️ Interceptions", C_AMBER_PASTEL, [
                    ("Interceptions", f"{d_game['interceptions_p90']:.2f}"),
                    ("Interceptions in Opp Field", f"{d_game['interceptions_attacking_p90']:.2f}"),
                ])
        else:
            with col_ds1:
                cmp_section_card("🛡️ General", C_BLUE_PASTEL, [
                    ("Defensive Actions", d_game["total_actions_p90"], f"{d_avg['total_actions_p90']:.1f}"),
                    ("Actions in Opp. Field", d_game["actions_attacking_p90"], f"{d_avg['actions_attacking_p90']:.1f}"),
                ])
            with col_ds2:
                cmp_section_card("⚔️ Duels", C_GREEN_PASTEL, [
                    ("Defensive Duels", d_game["duels_p90"], f"{d_avg['duels_p90']:.1f}"),
                    ("% Duels Won", d_game["duels_won_pct"], d_avg["duels_won_pct"],
                     f"{d_game['duels_won_pct']:.1f}%", f"{d_avg['duels_won_pct']:.1f}%"),
                ])
            with col_ds3:
                cmp_section_card("👁️ Interceptions", C_AMBER_PASTEL, [
                    ("Interceptions", d_game["interceptions_p90"], f"{d_avg['interceptions_p90']:.1f}"),
                    ("Interceptions in Opp Field", d_game["interceptions_attacking_p90"], f"{d_avg['interceptions_attacking_p90']:.1f}"),
                ])

with tab_evo:
    sub_tab_evo_passes, sub_tab_evo_def = st.tabs(["Passes", "Defensive Actions"])

    with sub_tab_evo_passes:
        st.markdown("### First 9 vs Last 9 Matches")
        st.markdown("Comparing the average passing performance between the first 9 and the last 9 matches to analyze player evolution.")
        df_scores = compute_match_scores(dfs_by_match, defensive_dfs_by_match)
        if len(df_scores) > 0:
            if len(df_scores) < 18:
                st.info(f"Note: Only {len(df_scores)} matches available. The comparison will overlap or use available data.")
            first_9 = df_scores.head(9)
            last_9 = df_scores.tail(9)

            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                fig_grade = draw_comparison_bar("Pass Grade", first_9["Grade"].mean(), last_9["Grade"].mean())
                st.plotly_chart(fig_grade, use_container_width=True)
            with r1c2:
                fig_xt_evo = draw_comparison_bar("Σ Pass Impact", first_9["xt_p90"].mean(), last_9["xt_p90"].mean())
                st.plotly_chart(fig_xt_evo, use_container_width=True)
            with r1c3:
                fig_high_xt_evo = draw_comparison_bar("High Impact Passes", first_9["high_xt_p90"].mean(), last_9["high_xt_p90"].mean())
                st.plotly_chart(fig_high_xt_evo, use_container_width=True)

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

    with sub_tab_evo_def:
        st.markdown("### First 9 vs Last 9 Matches")
        st.markdown("Comparing the average defensive performance between the first 9 and the last 9 matches.")
        df_scores = compute_match_scores(dfs_by_match, defensive_dfs_by_match)
        df_def_evo = compute_defensive_evolution_df(defensive_dfs_by_match, df_scores)
        if len(df_def_evo) > 0:
            if len(df_def_evo) < 18:
                st.info(f"Note: Only {len(df_def_evo)} matches available. The comparison will overlap or use available data.")
            first_9_def = df_def_evo.head(9)
            last_9_def = df_def_evo.tail(9)

            rd1c1, rd1c2, rd1c3 = st.columns(3)
            with rd1c1:
                g1 = first_9_def["grade"].dropna()
                g2 = last_9_def["grade"].dropna()
                v1 = g1.mean() if len(g1) > 0 else 0
                v2 = g2.mean() if len(g2) > 0 else 0
                fig_grade_def = draw_comparison_bar("Defensive Actions Grade", v1, v2)
                st.plotly_chart(fig_grade_def, use_container_width=True)
            with rd1c2:
                fig_duels_won = draw_comparison_bar("Duels Won p90", first_9_def["duels_won_p90"].mean(), last_9_def["duels_won_p90"].mean())
                st.plotly_chart(fig_duels_won, use_container_width=True)
            with rd1c3:
                fig_duels_pct = draw_comparison_bar("% Duels Won", first_9_def["duels_won_pct"].mean(), last_9_def["duels_won_pct"].mean(), suffix="%")
                st.plotly_chart(fig_duels_pct, use_container_width=True)

            rd2c1, rd2c2, rd2c3 = st.columns(3)
            with rd2c1:
                fig_def_act = draw_comparison_bar("Defensive Actions p90", first_9_def["def_actions_p90"].mean(), last_9_def["def_actions_p90"].mean())
                st.plotly_chart(fig_def_act, use_container_width=True)
            with rd2c2:
                fig_int_evo = draw_comparison_bar("Interceptions p90", first_9_def["interceptions_p90"].mean(), last_9_def["interceptions_p90"].mean())
                st.plotly_chart(fig_int_evo, use_container_width=True)
            with rd2c3:
                fig_funnel_evo = draw_comparison_bar("Funnel Actions p90", first_9_def["funnel_actions_p90"].mean(), last_9_def["funnel_actions_p90"].mean())
                st.plotly_chart(fig_funnel_evo, use_container_width=True)
        else:
            st.warning("Not enough data to generate defensive evolution charts.")
