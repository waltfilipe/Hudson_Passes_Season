import re
import math
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="Hudson Cicala — Passes Season Dashboard"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.row-label{
  font-size:13px;font-weight:700;color:#c8d6e5;letter-spacing:.3px;
  margin-bottom:4px;margin-top:8px;padding:5px 9px;
  background:rgba(255,255,255,.04);border-radius:4px;line-height:1.3;}
.row-label-blue  {border-left:3px solid #2F80ED;}
.row-label-green {border-left:3px solid #10b981;}
.row-label-amber {border-left:3px solid #f59e0b;}
.sec-hdr{
  font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.9px;
  padding:6px 10px;border-radius:5px;margin:0 0 8px 0;
  background:rgba(255,255,255,.04);}
.cmp-box{
  background:rgba(255,255,255,.04);border-radius:10px;
  padding:10px 12px;margin-bottom:8px;}
.cmp-label{
  font-size:11px;color:#94a3b8;text-transform:uppercase;
  letter-spacing:.6px;font-weight:600;margin-bottom:7px;}
.cmp-row{display:flex;justify-content:space-between;align-items:flex-start;gap:6px;}
.cmp-cell{flex:1;}
.cc-tag{font-size:10px;font-weight:700;margin-bottom:2px;}
.cc-val{font-size:21px;font-weight:700;color:#f1f5f9;line-height:1.15;}
.cc-sub{font-size:9px;color:#64748b;margin-top:3px;}
.cmp-sep{width:1px;background:rgba(255,255,255,.10);min-height:40px;flex-shrink:0;margin-top:18px;}
.row-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:8px 0 6px 0;}
</style>
""", unsafe_allow_html=True)

st.title("Hudson Cicala — Passes Season Dashboard")

# =========================================================
# CONSTANTS
# =========================================================
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67

FIG_W, FIG_H = 7.0, 4.7
FIG_DPI = 180

COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.08

CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)

C_BLUE = "#2F80ED"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"

# =========================================================
# RAW EVENTS TEXT
# COLE AQUI o texto completo dos eventos
# =========================================================
RAW_EVENTS_TEXT = r"""
COLE_AQUI_O_TEXTO_COMPLETO_DOS_EVENTOS
"""

# =========================================================
# PARSER
# =========================================================
def parse_raw_events(text: str) -> dict:
    """
    Espera blocos no formato:
      Vs Nome do Jogo (...)
      Sucesso
      Seta 1: (x1, y1) -> (x2, y2)
      ...
      Errado / Errados
      Seta ...
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    matches = {}
    current_match = None
    current_section = None  # "WON" ou "LOST"

    re_match = re.compile(r"^Vs\s+(.+)$", re.IGNORECASE)
    re_success = re.compile(r"^Sucesso$", re.IGNORECASE)
    re_fail = re.compile(r"^Errado[s]?$", re.IGNORECASE)
    re_arrow = re.compile(
        r"^Seta\s+\d+:\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*->\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$",
        re.IGNORECASE
    )

    for ln in lines:
        m_game = re_match.match(ln)
        if m_game:
            current_match = m_game.group(1).strip()
            if current_match not in matches:
                matches[current_match] = []
            current_section = None
            continue

        if re_success.match(ln):
            current_section = "PASS WON"
            continue

        if re_fail.match(ln):
            current_section = "PASS LOST"
            continue

        m_arrow = re_arrow.match(ln)
        if m_arrow and current_match and current_section:
            x1, y1, x2, y2 = map(float, m_arrow.groups())
            matches[current_match].append((current_section, x1, y1, x2, y2, None))
            continue

    # remove jogos vazios
    matches = {k: v for k, v in matches.items() if len(v) > 0}
    return matches

# =========================================================
# xT + HELPERS
# =========================================================
NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60

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

def progressive_pass(x_start: float, x_end: float) -> bool:
    dist_start = FIELD_X - x_start
    dist_end = FIELD_X - x_end
    closer_by = dist_start - dist_end
    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    if start_own and end_own:
        return closer_by >= 30.0
    if start_own != end_own:
        return closer_by >= 15.0
    return closer_by >= 10.0

def classify_pass_direction(x_start, y_start, x_end, y_end) -> str:
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx**2 + dy**2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:
        return "forward"
    if angle_deg >= 135.0:
        return "backward"
    if dist > 12.0:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"

# =========================================================
# BUILD DATAFRAMES
# =========================================================
matches_data = parse_raw_events(RAW_EVENTS_TEXT)

if len(matches_data) == 0:
    st.error("Nenhum evento foi parseado. Cole o texto completo dos eventos no RAW_EVENTS_TEXT.")
    st.stop()

dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["outcome"] = np.where(dfm["is_won"], "completed", "incomplete")
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r.x_start, r.y_start, r.x_end, r.y_end), axis=1
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["is_progressive"] = dfm.apply(lambda r: progressive_pass(r.x_start, r.x_end), axis=1)
    dfm["pass_distance"] = np.sqrt((dfm.x_end - dfm.x_start) ** 2 + (dfm.y_end - dfm.y_start) ** 2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r.x_end, r.y_end), axis=1)
    dfm["delta_xt"] = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"] = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"], dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    dfs_by_match[match_name] = dfm

df_season = pd.concat(dfs_by_match.values(), ignore_index=True)

# =========================================================
# STATS
# =========================================================
def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {
            "total": 0, "completed": 0, "incomplete": 0, "accuracy": 0.0,
            "prog_total": 0, "prog_completed": 0, "prog_pct": 0.0,
            "fwd": 0, "fwd_pct": 0.0, "bwd": 0, "bwd_pct": 0.0, "lat": 0, "lat_pct": 0.0,
            "pos_pct": 0.0, "high_xt_pct": 0.0, "sum_dxt": 0.0
        }

    completed = int(df["is_won"].sum())
    incomplete = total - completed
    prog_total = int(df["is_progressive"].sum())
    prog_completed = int((df["is_progressive"] & df["is_won"]).sum())
    fwd = int(df["is_forward"].sum())
    bwd = int(df["is_backward"].sum())
    lat = int(df["is_lateral"].sum())
    pos_count = int((df["is_won"] & (df["delta_xt_adj"] > 0)).sum())
    high_xt = int((df["delta_xt_adj"] > 0.1).sum())
    sum_dxt = float(df.loc[df["is_won"], "delta_xt_adj"].sum())

    return {
        "total": total,
        "completed": completed,
        "incomplete": incomplete,
        "accuracy": round(completed / total * 100, 1),
        "prog_total": prog_total,
        "prog_completed": prog_completed,
        "prog_pct": round(prog_total / total * 100, 1),
        "fwd": fwd, "fwd_pct": round(fwd / total * 100, 1),
        "bwd": bwd, "bwd_pct": round(bwd / total * 100, 1),
        "lat": lat, "lat_pct": round(lat / total * 100, 1),
        "pos_pct": round(pos_count / total * 100, 1),
        "high_xt_pct": round(high_xt / total * 100, 1),
        "sum_dxt": round(sum_dxt, 3),
    }

def _arrow_html(val_sel: float, val_season: float) -> tuple[str, str]:
    sel_a = season_a = ""
    if val_sel == val_season or val_sel == 0 or val_season == 0:
        return sel_a, season_a
    if val_sel > val_season:
        pct = (val_sel - val_season) / abs(val_season) * 100
        sel_a = f'<span style="font-size:11px;font-weight:700;color:#10b981;margin-left:4px;">↑ {pct:.0f}%</span>'
    else:
        pct = (val_season - val_sel) / abs(val_sel) * 100
        season_a = f'<span style="font-size:11px;font-weight:700;color:#10b981;margin-left:4px;">↑ {pct:.0f}%</span>'
    return sel_a, season_a

def cmp_box(label, val_sel, val_season, disp_sel=None, disp_season=None, sub_sel="", sub_season="", border="#3b82f6"):
    disp_sel = str(val_sel) if disp_sel is None else disp_sel
    disp_season = str(val_season) if disp_season is None else disp_season
    sel_a, season_a = _arrow_html(float(val_sel), float(val_season))
    sub_sel_html = f'<div class="cc-sub">{sub_sel}</div>' if sub_sel else ""
    sub_season_html = f'<div class="cc-sub">{sub_season}</div>' if sub_season else ""

    html = f"""
    <div class="cmp-box" style="border-left:3px solid {border};">
      <div class="cmp-label">{label}</div>
      <div class="cmp-row">
        <div class="cmp-cell">
          <div class="cc-tag" style="color:#f87171;">JOGO</div>
          <div class="cc-val">{disp_sel}{sel_a}</div>
          {sub_sel_html}
        </div>
        <div class="cmp-sep"></div>
        <div class="cmp-cell">
          <div class="cc-tag" style="color:#60a5fa;">TEMPORADA</div>
          <div class="cc-val">{disp_season}{season_a}</div>
          {sub_season_html}
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def sec_hdr(label, color="#3b82f6"):
    st.markdown(
        f'<div class="sec-hdr" style="border-left:3px solid {color};color:{color};">{label}</div>',
        unsafe_allow_html=True
    )

def row_label(text, cls="row-label-blue"):
    st.markdown(f'<div class="row-label {cls}">{text}</div>', unsafe_allow_html=True)

# =========================================================
# DRAW HELPERS
# =========================================================
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
        transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=11, linewidth=1.6, color="#aaaaaa"
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
        is_won = bool(row["is_won"])
        is_prog = bool(row["is_progressive"])
        if not is_won:
            color, alpha = COLOR_FAIL, 0.72
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS

        pitch.arrows(row.x_start, row.y_start, row.x_end, row.y_end,
                     color=color, width=1.3, headwidth=2.0, headlength=2.0,
                     ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row.x_start, row.y_start, s=32, marker="o", color=color,
                      edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=alpha)

    leg = ax.legend(handles=[
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
    ], loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True,
       facecolor="#1a1a2e", edgecolor="#444466", fontsize=6.5,
       labelspacing=0.35, borderpad=0.4)
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
            arr[i] = int(((df_s["x_end"] >= x0_) & (df_s["x_end"] < x1_)
                          & (df_s["y_end"] >= y0) & (df_s["y_end"] < y1)).sum())
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
            ax.text((x0_ + x1_) / 2, (y0 + y1) / 2, str(value), ha="center", va="center",
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
        xa = x0 + (x1 - x0) * t0
        ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1
        yb = y0 + (y1 - y0) * t1
        alpha = 0.85 * (0.15 + 0.85 * t1)
        lw = 2.5 * (0.80 + 0.20 * t1)
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=4, solid_capstyle="round")
    ax.scatter(x0, y0, s=20, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, zorder=5, alpha=0.85)
    ax.scatter(x1, y1, s=32, marker="o", facecolors=color, edgecolors="white", linewidths=0.9, zorder=6, alpha=0.85)

def draw_top10_xt_map(df):
    fig, ax, pitch = _base_pitch()
    top10 = (
        df[(df["is_won"]) & (df["delta_xt_adj"] > 0)]
        .sort_values("delta_xt_adj", ascending=False)
        .head(10).copy().reset_index(drop=True)
    )
    if not top10.empty:
        for _, row in top10.iterrows():
            val = float(row["delta_xt_adj"])
            color = CMAP_TOP10(NORM_TOP10(np.clip(val, 0.05, 0.40)))
            _draw_comet_arrow(ax, float(row.x_start), float(row.y_start), float(row.x_end), float(row.y_end), color)

    sm = plt.cm.ScalarMappable(cmap=CMAP_TOP10, norm=NORM_TOP10)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
    cbar.set_label("ΔxT", color="#ffffff", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")

    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

# =========================================================
# SIDEBAR / FILTERS
# =========================================================
st.sidebar.header("Filtros")
match_names = sorted(dfs_by_match.keys())
selected_match = st.sidebar.selectbox("Selecione o jogo", options=match_names, index=0)

pass_filter = st.sidebar.radio(
    "Tipo de passe",
    ["Todos", "Certos", "Errados", "Progressivos"],
    index=0
)

show_table = st.sidebar.checkbox("Mostrar tabela de eventos", value=False)

df_match_raw = dfs_by_match[selected_match].copy()
df_season_raw = df_season.copy()

def apply_filter(df):
    if pass_filter == "Certos":
        return df[df["is_won"]].copy()
    if pass_filter == "Errados":
        return df[~df["is_won"]].copy()
    if pass_filter == "Progressivos":
        return df[df["is_progressive"]].copy()
    return df.copy()

df_match = apply_filter(df_match_raw)
df_season_filtered = apply_filter(df_season_raw)

s_match = compute_stats(df_match)
s_season = compute_stats(df_season_filtered)

# =========================================================
# PRE-RENDER
# =========================================================
img_pm_match, fig_pm_match = draw_pass_map(df_match); plt.close(fig_pm_match)
img_pm_season, fig_pm_season = draw_pass_map(df_season_filtered); plt.close(fig_pm_season)
img_ht_match, fig_ht_match = draw_corridor_heatmap(df_match); plt.close(fig_ht_match)
img_ht_season, fig_ht_season = draw_corridor_heatmap(df_season_filtered); plt.close(fig_ht_season)
img_xt_match, fig_xt_match = draw_top10_xt_map(df_match); plt.close(fig_xt_match)
img_xt_season, fig_xt_season = draw_top10_xt_map(df_season_filtered); plt.close(fig_xt_season)

# =========================================================
# LAYOUT
# =========================================================
st.caption("Comparativo automático: Jogo Selecionado vs Total da Temporada (com o mesmo filtro aplicado).")

col_left, col_right, col_stats = st.columns([1, 1, 1], gap="medium")

# ROW 1 - PASS MAP
with col_left:
    row_label(f"🟦 Pass Map · {selected_match}", "row-label-blue")
    st.image(img_pm_match, use_container_width=True)

with col_right:
    row_label("🟦 Pass Map · TEMPORADA (Total)", "row-label-blue")
    st.image(img_pm_season, use_container_width=True)

with col_stats:
    sec_hdr("📋 Pass Overview", C_BLUE)
    cmp_box("Total Passes", s_match["total"], s_season["total"], border=C_BLUE)
    cmp_box(
        "Completed",
        s_match["completed"], s_season["completed"],
        disp_sel=f"{s_match['completed']} ({s_match['accuracy']:.0f}%)",
        disp_season=f"{s_season['completed']} ({s_season['accuracy']:.0f}%)",
        sub_sel=f"{s_match['incomplete']} incomplete",
        sub_season=f"{s_season['incomplete']} incomplete",
        border=C_BLUE
    )
    cmp_box(
        "Progressive Passes",
        s_match["prog_total"], s_season["prog_total"],
        disp_sel=f"{s_match['prog_total']} ({s_match['prog_pct']:.0f}%)",
        disp_season=f"{s_season['prog_total']} ({s_season['prog_pct']:.0f}%)",
        sub_sel=f"{s_match['prog_completed']} completed",
        sub_season=f"{s_season['prog_completed']} completed",
        border=C_BLUE
    )

st.markdown('<hr class="row-divider">', unsafe_allow_html=True)

# ROW 2 - HEATMAP
col_left2, col_right2, col_stats2 = st.columns([1, 1, 1], gap="medium")

with col_left2:
    row_label(f"🟩 Zone Heatmap · {selected_match}", "row-label-green")
    st.image(img_ht_match, use_container_width=True)

with col_right2:
    row_label("🟩 Zone Heatmap · TEMPORADA", "row-label-green")
    st.image(img_ht_season, use_container_width=True)

with col_stats2:
    sec_hdr("🧭 Pass Direction", C_GREEN)
    cmp_box(
        "⬆️ Forward",
        s_match["fwd_pct"], s_season["fwd_pct"],
        disp_sel=f"{s_match['fwd']} ({s_match['fwd_pct']:.0f}%)",
        disp_season=f"{s_season['fwd']} ({s_season['fwd_pct']:.0f}%)",
        border=C_GREEN
    )
    cmp_box(
        "⬇️ Backward",
        s_match["bwd_pct"], s_season["bwd_pct"],
        disp_sel=f"{s_match['bwd']} ({s_match['bwd_pct']:.0f}%)",
        disp_season=f"{s_season['bwd']} ({s_season['bwd_pct']:.0f}%)",
        border=C_GREEN
    )
    cmp_box(
        "↔️ Lateral",
        s_match["lat_pct"], s_season["lat_pct"],
        disp_sel=f"{s_match['lat']} ({s_match['lat_pct']:.0f}%)",
        disp_season=f"{s_season['lat']} ({s_season['lat_pct']:.0f}%)",
        border=C_GREEN
    )

st.markdown('<hr class="row-divider">', unsafe_allow_html=True)

# ROW 3 - TOP10 xT
col_left3, col_right3, col_stats3 = st.columns([1, 1, 1], gap="medium")

with col_left3:
    row_label(f"🟡 Top 10 ΔxT · {selected_match}", "row-label-amber")
    st.image(img_xt_match, use_container_width=True)

with col_right3:
    row_label("🟡 Top 10 ΔxT · TEMPORADA", "row-label-amber")
    st.image(img_xt_season, use_container_width=True)

with col_stats3:
    sec_hdr("⚡ xT Analysis", C_AMBER)
    cmp_box(
        "% Positive ΔxT",
        s_match["pos_pct"], s_season["pos_pct"],
        disp_sel=f"{s_match['pos_pct']:.1f}%",
        disp_season=f"{s_season['pos_pct']:.1f}%",
        sub_sel="passes that gained xT",
        sub_season="passes that gained xT",
        border=C_AMBER
    )
    cmp_box(
        "% ΔxT > 0.1",
        s_match["high_xt_pct"], s_season["high_xt_pct"],
        disp_sel=f"{s_match['high_xt_pct']:.1f}%",
        disp_season=f"{s_season['high_xt_pct']:.1f}%",
        sub_sel="high-threat passes",
        sub_season="high-threat passes",
        border=C_AMBER
    )
    cmp_box(
        "Σ ΔxT",
        s_match["sum_dxt"], s_season["sum_dxt"],
        disp_sel=f"{s_match['sum_dxt']:.3f}",
        disp_season=f"{s_season['sum_dxt']:.3f}",
        sub_sel="total xT generated",
        sub_season="total xT generated",
        border=C_AMBER
    )

st.caption(
    "Grey = Completed · Blue = Progressive · Red = Incomplete · "
    "Dashed line = Final Third · Comet = Top-10 ΔxT"
)

# =========================================================
# OPTIONAL TABLE
# =========================================================
if show_table:
    st.subheader(f"Eventos — {selected_match} ({pass_filter})")
    show_cols = ["number", "type", "x_start", "y_start", "x_end", "y_end", "is_progressive", "delta_xt_adj"]
    st.dataframe(df_match[show_cols].reset_index(drop=True), use_container_width=True)
