"""
AI-Based Demand Forecasting System for Inventory Optimization
=============================================================
Professional analytics dashboard — sidebar navigation architecture.
Pages: Dashboard | Data | EDA | Train Models | Model Comparison |
       Forecast | Inventory | Reports | Settings
"""

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

# ── Path Setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.eda import ExploratoryAnalysis
from src.inventory.optimization import InventoryOptimization
from src.models.arima_xgboost import HybridArimaXGBoost
from src.models.lstm_model import LSTMForecaster
from src.models.model_comparison import ModelComparison
from src.preprocessing import DataPreprocessor

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# ── Snowflake Connection ──────────────────────────────────────────────────────

def get_snowflake_connection():
    """Connect to Snowflake using Streamlit secrets."""
    if not SNOWFLAKE_AVAILABLE:
        raise ImportError(
            "Snowflake connector is missing. "
            "Add snowflake-connector-python to requirements.txt."
        )

    if "snowflake" not in st.secrets:
        raise RuntimeError(
            "Snowflake credentials are not configured in Streamlit Secrets."
        )

    cfg = st.secrets["snowflake"]

    return snowflake.connector.connect(
        account=cfg["account"],
        user=cfg["user"],
        password=cfg["password"],
        warehouse=cfg["warehouse"],
        database=cfg["database"],
        schema=cfg["schema"],
        role=cfg["role"],
    )


def test_snowflake_connection():
    """Check Snowflake access to the demand history table."""
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*)
                FROM DEMAND_FORECASTING_DB.PUBLIC.DEMAND_HISTORY
            """)
            return cursor.fetchone()[0]
        finally:
            cursor.close()
    finally:
        conn.close()

# ── Logging ───────────────────────────────────────────────────────────────────
# Emit per-model / per-stage timing logs for the forecasting pipeline to the
# terminal without changing global logging or other libraries' verbosity.
_src_logger = logging.getLogger('src')
_src_logger.setLevel(logging.INFO)
if not _src_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    _src_logger.addHandler(_handler)
    _src_logger.propagate = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Demand Forecasting",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
/* ─── Design Tokens ────────────────────────────────────────────────────── */
:root {
    --primary:         #1e40af;
    --primary-hover:   #1d3fa5;
    --primary-light:   #dbeafe;
    --primary-muted:   rgba(30,64,175,0.14);
    --accent:          #0ea5e9;
    --success:         #059669;
    --success-bg:      #d1fae5;
    --warning:         #b45309;
    --warning-bg:      #fef3c7;
    --danger:          #dc2626;
    --danger-bg:       #fee2e2;
    --text-primary:    #0f172a;
    --text-secondary:  #1e293b;
    --text-muted:      #475569;
    --text-subtle:     #64748b;
    --border:          #e2e8f0;
    --border-strong:   #cbd5e1;
    --surface:         #ffffff;
    --surface-2:       #f8fafc;
    --bg:              #f1f5f9;
    --shadow-sm:       0 1px 2px rgba(0,0,0,0.05);
    --shadow:          0 1px 3px rgba(0,0,0,0.07), 0 4px 12px rgba(0,0,0,0.04);
    --shadow-md:       0 4px 8px rgba(0,0,0,0.08), 0 12px 24px rgba(0,0,0,0.05);
    --radius-sm:       5px;
    --radius:          8px;
    --radius-lg:       12px;
}

/* ─── Global ────────────────────────────────────────────────────────────── */
.stApp {
    background: var(--bg) !important;
    font-size: 15px; line-height: 1.55;
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}
.block-container {
    padding: 1.8rem 2.2rem 2.2rem 2.2rem !important;
    max-width: 100% !important;
}
*, *::before, *::after {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
    box-sizing: border-box;
}

/* ─── Sidebar Shell ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
    min-width: 242px !important;
    max-width: 268px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 0 14px 24px 14px !important;
}

/* ─── Sidebar Branding ──────────────────────────────────────────────────── */
.sb-brand {
    padding: 20px 4px 14px 4px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 4px;
    display: flex;
    flex-direction: column;
    gap: 0;
}
.sb-brand-row { display: flex; align-items: center; gap: 9px; }
.sb-brand-mark {
    width: 26px; height: 26px; flex-shrink: 0;
    background: linear-gradient(135deg, #1e40af, #0ea5e9);
    border-radius: 6px;
}
.sb-brand-title {
    font-size: 17px; font-weight: 700; color: #f8fafc !important;
    letter-spacing: -0.2px;
}
.sb-brand-sub {
    font-size: 14px; color: #94a3b8 !important;
    margin: 6px 0 0 35px;
    letter-spacing: 0.2px;
}

/* ─── Sidebar Section Label ─────────────────────────────────────────────── */
.sb-sec {
    font-size: 13.5px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; color: #94a3b8 !important;
    padding: 16px 4px 6px 4px;
    display: block;
}

/* ─── Sidebar Divider ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid #1e293b !important;
    margin: 10px 0 !important;
}

/* ─── Sidebar Nav Radio ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 1px !important; flex-direction: column !important;
}
/* Each radio item */
section[data-testid="stSidebar"] [data-baseweb="radio"] {
    width: 100% !important;
    border-radius: 6px !important;
    transition: background 0.14s ease !important;
}
/* Hide the circle */
section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
    width: 0 !important; height: 0 !important;
}
/* Label */
section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    cursor: pointer !important;
    padding: 9px 12px !important;
    width: 100% !important;
    display: block !important;
    color: #cbd5e1 !important;
    font-size: 15.5px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: color 0.14s, background 0.14s !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    color: #cbd5e1 !important;
    background: rgba(255,255,255,0.05) !important;
}
/* Selected */
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] {
    background: rgba(30,64,175,0.18) !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] label {
    color: #93c5fd !important;
    font-weight: 600 !important;
}

/* ─── Sidebar Number Inputs ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] label p {
    font-size: 14px !important; color: #cbd5e1 !important;
    font-weight: 500 !important; letter-spacing: 0.1px !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #f1f5f9 !important; border-radius: 5px !important;
    font-size: 15px !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus {
    border-color: #3b82f6 !important; outline: none !important;
}
section[data-testid="stSidebar"] button {
    background: #1e293b !important; border-color: #334155 !important;
    color: #cbd5e1 !important;
}

/* ─── Sidebar Status ────────────────────────────────────────────────────── */
.sb-status {
    margin-top: 8px; padding: 10px 12px;
    background: #1a2744; border: 1px solid #1e293b;
    border-radius: 7px;
}
.sb-status-item {
    display: flex; align-items: center; gap: 8px;
    font-size: 14px; color: #94a3b8 !important;
    padding: 3px 0;
}
.sb-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.sb-dot-on  { background: #10b981; }
.sb-dot-off { background: #475569; }
.sb-dot-warn { background: #f59e0b; }
.sb-status-val { color: #e2e8f0 !important; font-weight: 600; }

/* ─── Sidebar Z-Score Box ─────────────────────────────────────────────────── */
.sb-z-box {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace !important;
    font-size: 16px; font-weight: 700; color: #93c5fd;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 5px; padding: 6px 10px; display: inline-block;
}

/* ─── Page Header ───────────────────────────────────────────────────────── */
.ph { padding-bottom: 16px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
.ph-title { font-size: 30px; font-weight: 800; color: var(--text-primary);
             letter-spacing: -0.6px; line-height: 1.2; margin: 0 0 5px 0; }
.ph-sub   { font-size: 15px; color: var(--text-muted); margin: 0; line-height: 1.4; }

/* ─── Section Label ─────────────────────────────────────────────────────── */
.sec-lbl {
    font-size: 19px; font-weight: 700; letter-spacing: -0.2px;
    color: var(--text-primary);
    margin: 0 0 14px 0; padding-left: 12px;
    border-left: 4px solid var(--primary);
}

/* ─── Cards ─────────────────────────────────────────────────────────────── */
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow);
}

/* ─── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow);
    min-height: 116px;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.kpi:hover { box-shadow: var(--shadow-md); border-color: var(--border-strong); }
.kpi-lbl  { font-size: 14px; font-weight: 600; letter-spacing: 0;
             color: var(--text-muted); margin: 0 0 8px 0; }
.kpi-val  { font-size: 30px; font-weight: 800; color: var(--text-primary);
             letter-spacing: -0.8px; margin: 0 0 5px 0; line-height: 1.1; }
.kpi-sub  { font-size: 14px; color: var(--text-muted); margin: 0; }
.kpi-blue .kpi-val { color: var(--primary); }
.kpi-green .kpi-val { color: var(--success); }

/* ─── Badges ────────────────────────────────────────────────────────────── */
.bdg {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 11px; border-radius: 999px;
    font-size: 13.5px; font-weight: 600; white-space: nowrap;
}
.bdg-ok   { background: var(--success-bg); color: var(--success); }
.bdg-warn { background: var(--warning-bg); color: var(--warning); }
.bdg-err  { background: var(--danger-bg);  color: var(--danger); }
.bdg-gray { background: #f1f5f9; color: var(--text-muted); }
.bdg-blue { background: var(--primary-light); color: var(--primary); }

/* ─── Quality Rows ──────────────────────────────────────────────────────── */
.qr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 0; border-bottom: 1px solid var(--border);
    font-size: 15px;
}
.qr:last-child { border-bottom: none; }
.qr-name { color: var(--text-secondary); }
.qr-detail { font-size: 14px; color: var(--text-muted); margin-left: 8px; }

/* ─── Model Cards ───────────────────────────────────────────────────────── */
.mc {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 22px 22px;
    box-shadow: var(--shadow); height: 100%;
}
.mc-title { font-size: 18px; font-weight: 700; color: var(--text-primary); margin: 0 0 3px 0; }
.mc-sub   { font-size: 14px; color: var(--text-muted); margin: 0 0 16px 0; }
.mc-arch  {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 14px;
    font-size: 14px; color: var(--text-secondary);
    font-family: "JetBrains Mono","Fira Code","Consolas",monospace !important;
    line-height: 1.8; margin: 0 0 14px 0;
}

/* ─── Empty State ───────────────────────────────────────────────────────── */
.es {
    text-align: center; padding: 52px 24px;
    background: var(--surface);
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--radius-lg);
}
.es-title { font-size: 20px; font-weight: 700; color: var(--text-secondary); margin: 0 0 8px 0; }
.es-sub   { font-size: 15px; color: var(--text-muted); margin: 0 0 22px 0;
             max-width: 420px; margin-left: auto; margin-right: auto; }

/* ─── Best Model Banner ─────────────────────────────────────────────────── */
.best-banner {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #6ee7b7; border-radius: var(--radius-lg);
    padding: 18px 22px; margin-bottom: 20px;
}
.best-banner-label { font-size: 14px; font-weight: 700; letter-spacing: 0.4px;
                      text-transform: uppercase; color: #059669; margin: 0 0 6px 0; }
.best-banner-model { font-size: 26px; font-weight: 800; color: #065f46; margin: 0 0 5px 0; letter-spacing: -0.4px; }
.best-banner-sub   { font-size: 15px; color: #047857; margin: 0; }

/* ─── Plotly Chart Wrapper ──────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden; box-shadow: var(--shadow-sm) !important;
    background: white;
}

/* ─── Metric (st.metric) ────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 18px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 14px !important; font-weight: 600 !important;
    letter-spacing: 0 !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-size: 28px !important; font-weight: 800 !important;
    color: var(--text-primary) !important; letter-spacing: -0.6px !important;
}

/* ─── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: var(--primary) !important; border: none !important;
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
    font-size: 15px !important; padding: 10px 24px !important; min-height: 44px !important;
    color: #fff !important; box-shadow: 0 1px 3px rgba(30,64,175,0.3) !important;
    transition: background 0.15s, transform 0.1s !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--primary-hover) !important; transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
    border: 1px solid var(--border-strong) !important; border-radius: var(--radius-sm) !important;
    font-size: 15px !important; font-weight: 500 !important; min-height: 44px !important;
    color: var(--text-secondary) !important; background: var(--surface) !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--primary) !important; color: var(--primary) !important;
}

/* ─── Download Button ───────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    border: 1px solid var(--border-strong) !important; border-radius: var(--radius-sm) !important;
    font-size: 15px !important; font-weight: 500 !important; min-height: 44px !important;
    background: var(--surface) !important; color: var(--text-secondary) !important;
    transition: border-color 0.15s, color 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--primary) !important; color: var(--primary) !important;
}

/* ─── Inputs ────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div {
    border-radius: var(--radius-sm) !important;
    font-size: 15px !important;
    transition: border-color 0.15s !important;
}
/* Form / widget labels (main workspace) */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    font-size: 15px !important; font-weight: 500 !important;
    color: var(--text-secondary) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(30,64,175,0.1) !important;
}

/* ─── Expander ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stExpander"] summary {
    font-size: 15px !important; font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* ─── Tabs (for sub-tabs in EDA) ────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 3px; gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 14.5px; font-weight: 500; color: var(--text-muted);
    border-radius: var(--radius-sm); padding: 7px 18px;
    border: none !important; transition: color 0.15s;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--primary); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--primary) !important; color: #fff !important; font-weight: 600 !important;
}

/* ─── Dataframe ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
    overflow: hidden; box-shadow: var(--shadow-sm) !important;
}

/* ─── Progress Bar ──────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: var(--primary) !important; border-radius: 999px !important;
}
[data-testid="stProgress"] > div { background: var(--border) !important; border-radius: 999px !important; }

/* ─── Alert ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: var(--radius-sm) !important; font-size: 15px !important; }

/* ─── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

/* ─── Footer ────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center; font-size: 14px; color: var(--text-muted);
    padding: 22px 0 8px 0; border-top: 1px solid var(--border); margin-top: 32px;
}

/* ─── Dataset Info Table ────────────────────────────────────────────────── */
.info-tbl { width: 100%; border-collapse: collapse; font-size: 15px; }
.info-tbl td { padding: 10px 4px; border-bottom: 1px solid var(--border); }
.info-tbl td:first-child { color: var(--text-muted); width: 44%; }
.info-tbl td:last-child  { color: var(--text-primary); font-weight: 500; }
.info-tbl tr:last-child td { border-bottom: none; }

/* ─── Inventory Highlight ────────────────────────────────────────────────── */
.inv-card {
    background: var(--surface); border-radius: var(--radius-lg);
    padding: 20px 22px; text-align: center;
    border: 1px solid var(--border); box-shadow: var(--shadow);
}
.inv-card-primary { border-color: #6ee7b7; background: linear-gradient(135deg,#ecfdf5,#f0fdf4); }
.inv-card-secondary { border-color: #a5b4fc; background: linear-gradient(135deg,#eef2ff,#f5f3ff); }
.inv-card-lbl { font-size: 14px; font-weight: 700; letter-spacing: 0.4px;
                 text-transform: uppercase; margin: 0 0 10px 0; }
.inv-card-primary .inv-card-lbl  { color: #059669; }
.inv-card-secondary .inv-card-lbl { color: #4f46e5; }
.inv-card-val { font-size: 32px; font-weight: 800; letter-spacing: -1px; margin: 0 0 5px 0; }
.inv-card-primary .inv-card-val  { color: #065f46; }
.inv-card-secondary .inv-card-val { color: #3730a3; }
.inv-card-sub { font-size: 14px; margin: 0; }
.inv-card-primary .inv-card-sub  { color: #047857; }
.inv-card-secondary .inv-card-sub { color: #4338ca; }

/* ─── Sidebar Navigation Contrast (consolidated) ────────────────────────── */
section[data-testid="stSidebar"] [data-baseweb="radio"] label,
section[data-testid="stSidebar"] [data-baseweb="radio"] label * {
    color: inherit !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] label,
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] label * {
    color: #93c5fd !important;
}
/* Sidebar form labels wrap in Streamlit's widget label wrapper */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    color: #cbd5e1 !important;
}

/* FIX: Force readable sidebar text */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] a {
    color: #f8fafc !important;
}

/* Keep sidebar form controls readable */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    color: #f8fafc !important;
}

/* ─── Responsive Typography ─────────────────────────────────────────────── */
@media (max-width: 1100px) {
    .ph-title { font-size: 26px; }
    .sec-lbl  { font-size: 18px; }
    .kpi-val  { font-size: 26px; }
}
@media (max-width: 900px) {
    .block-container { padding: 1.1rem 1rem 1.5rem 1rem !important; }
    .ph-title { font-size: 23px; }
    .kpi-val  { font-size: 23px; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
_KNOWN_DATE_ALIASES = [
    'date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP',
    'order_date', 'OrderDate', 'order_Date', 'week_start_date',
    'ds', 'time', 'Time', 'period', 'Period',
]
_KNOWN_DEMAND_ALIASES = [
    'sales', 'Sales', 'SALES',
    'unit_sales', 'units_sold', 'UnitSales', 'Units',
    'quantity', 'Quantity', 'QUANTITY', 'qty', 'Qty',
    'Sales_Quantity', 'sales_quantity',
    'demand', 'Demand', 'DEMAND',
    'Weekly_Sales', 'weekly_sales',
    'cnt', 'count', 'Count', 'sold',
    'revenue', 'Revenue',
]
_M5_MARKER_COLS = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}

_PAGES = [
    "Dashboard",
    "Data",
    "EDA",
    "Train Models",
    "Model Comparison",
    "Forecast",
    "Inventory",
    "Reports",
    "Settings",
]

_C_PRIMARY  = '#1e40af'
_C_LSTM     = '#059669'
_C_HYBRID   = '#b45309'
_C_HIST     = '#94a3b8'
_C_ACTUAL   = '#1e40af'
_C_ENS      = '#7c3aed'

# Z-score table: service level % -> safety factor
_Z_SCORES = {
    80:0.842, 85:1.036, 90:1.282, 91:1.341, 92:1.405,
    93:1.476, 94:1.555, 95:1.645, 96:1.751, 97:1.881,
    98:2.054, 99:2.326,
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    'data':               None,           # Legacy: single aggregated series
    'train_df':           None,           # train.csv with units_sold
    'test_df':            None,           # test.csv without units_sold
    'sample_submission':  None,           # sample_submission.csv template
    'preprocessor':       None,
    'models_trained':     False,
    'predictions':        {},
    'selected_date_col':  None,
    'selected_sales_col': None,
    'selected_store_col': None,
    'selected_sku_col':   None,
    'last_processed_sig': None,
    'forecast_generated': False,
    'forecast_lstm':      None,
    'forecast_hybrid':    None,
    'forecast_dates':     None,
    'test_predictions':   None,           # Predictions for test.csv
    'test_record_ids':    None,           # record_IDs from test.csv
    'best_model':         None,
    'comparison_df':      None,
    # UI param defaults
    'forecast_steps':     30,
    'seq_length':         30,
    'lead_time':          7,
    'service_level_pct':  95,
    'test_size':          0.2,            # Validation split inside train.csv
    # Advanced defaults (Settings page)
    'lstm_epochs':        50,
    'lstm_batch':         32,
    'arima_order_str':    '1,1,1',
    'use_duckdb':         DUCKDB_AVAILABLE,
    '_nav_idx':           0,   # tracks sidebar nav selection as an index
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _is_file_obj(src):
    return hasattr(src, 'read') and hasattr(src, 'seek')

def _src_name(src):
    return src.name if hasattr(src, 'name') else os.path.basename(str(src))

def _src_size(src):
    if hasattr(src, 'size') and src.size is not None:
        return int(src.size)
    try:
        return os.path.getsize(str(src))
    except OSError:
        return 0

def _list_csvs(folder):
    p = Path(folder)
    return sorted(str(f) for f in p.glob('*.csv')) if p.exists() and p.is_dir() else []

def _read_columns(src):
    if _is_file_obj(src):
        src.seek(0)
        cols = list(pd.read_csv(src, nrows=0).columns)
        src.seek(0)
        return cols
    return list(pd.read_csv(src, nrows=0).columns)

def _detect_wide_format(columns):
    col_set = {c.lower() for c in columns}
    day_cols = [c for c in columns if c.startswith('d_') and c[2:].isdigit()]
    return len(day_cols) > 100 and col_set & {c.lower() for c in _M5_MARKER_COLS}

def _auto_detect_cols(columns):
    lower_map = {c.lower(): c for c in columns}
    date_col   = next((lower_map[a.lower()] for a in _KNOWN_DATE_ALIASES   if a.lower() in lower_map), None)
    demand_col = next((lower_map[a.lower()] for a in _KNOWN_DEMAND_ALIASES if a.lower() in lower_map), None)
    return date_col, demand_col

def _build_sig(mode, sources, date_col, sales_col):
    parts = [mode, date_col or '', sales_col or '']
    for s in sources:
        parts.append(
            f'up::{_src_name(s)}::{_src_size(s)}' if _is_file_obj(s)
            else f'path::{os.path.abspath(str(s))}::{_src_size(s)}'
        )
    parts.sort()
    return '|'.join(parts)

@st.cache_data(show_spinner=False)
def _aggregate_chunked(file_bytes, filename, date_col, sales_col, chunksize=250_000):
    buf = io.BytesIO(file_bytes)
    agg = None
    for chunk in pd.read_csv(buf, usecols=[date_col, sales_col],
                              chunksize=chunksize, low_memory=False):
        chunk[date_col]   = pd.to_datetime(chunk[date_col], errors='coerce')
        chunk[sales_col]  = pd.to_numeric(chunk[sales_col], errors='coerce')
        chunk = chunk.dropna().query(f'{sales_col} >= 0')
        if chunk.empty:
            continue
        grp = chunk.groupby(date_col)[sales_col].sum()
        agg = grp if agg is None else agg.add(grp, fill_value=0)
    if agg is None:
        return pd.DataFrame(columns=[date_col, sales_col])
    return agg.reset_index().sort_values(date_col).reset_index(drop=True)

def _melt_m5_wide(df_wide):
    day_cols = [c for c in df_wide.columns if c.startswith('d_') and c[2:].isdigit()]
    return df_wide[day_cols].sum(axis=0)

def _fmt(n, decimals=1):
    """Format number with K/M suffix."""
    if n is None:
        return 'N/A'
    if abs(n) >= 1_000_000:
        return f'{n/1_000_000:.{decimals}f}M'
    if abs(n) >= 1_000:
        return f'{n/1_000:.{decimals}f}K'
    return f'{n:.{decimals}f}'

def _go_to(page):
    if page in _PAGES:
        st.session_state['_nav_idx'] = _PAGES.index(page)
    st.rerun()

def _get_z(sl_pct):
    """Return Z-score for a given service level percentage."""
    return _Z_SCORES.get(int(sl_pct), 1.645)

def _parse_arima_order():
    raw = st.session_state.get('arima_order_str', '1,1,1')
    try:
        p, d, q = [int(x.strip()) for x in raw.split(',')]
        return (p, d, q)
    except Exception:
        return (1, 1, 1)

def _download_csv(df, label, filename, key):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label=label, data=csv, file_name=filename,
                       mime='text/csv', key=key)

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _cl(**kw):
    """Return a professional Plotly layout dict."""
    base = dict(
        template='plotly_white',
        font=dict(family='Inter, Segoe UI, system-ui, sans-serif', size=14, color='#0f172a'),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=16, r=16, t=52, b=16),
        title=dict(font=dict(size=17), x=0, xanchor='left', y=0.97, yanchor='top'),
        hoverlabel=dict(bgcolor='white', bordercolor='#cbd5e1', font_size=13.5, font_family='Inter, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=13), bgcolor='rgba(255,255,255,0)'),
        xaxis=dict(showgrid=False, showline=True, linecolor='#cbd5e1', linewidth=1,
                   tickfont=dict(size=13), title_font=dict(size=14), zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#eef2f7', gridwidth=1,
                   showline=False, zeroline=False, tickfont=dict(size=13), title_font=dict(size=14)),
    )
    base.update(kw)
    return base

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def _ph(title, subtitle=None):
    """Render a page header."""
    sub = f'<p class="ph-sub">{subtitle}</p>' if subtitle else ''
    st.markdown(f'<div class="ph"><p class="ph-title">{title}</p>{sub}</div>',
                unsafe_allow_html=True)

def _sec(text):
    st.markdown(f'<p class="sec-lbl">{text}</p>', unsafe_allow_html=True)

def _badge(text, kind='gray'):
    return f'<span class="bdg bdg-{kind}">{text}</span>'

def _kpi(label, value, sub='', variant=''):
    return (f'<div class="kpi {variant}">'
            f'<p class="kpi-lbl">{label}</p>'
            f'<p class="kpi-val">{value}</p>'
            f'<p class="kpi-sub">{sub}</p>'
            f'</div>')

def _qrow(name, status, detail=''):
    kind = 'ok' if status == 'pass' else ('warn' if status == 'warn' else 'err')
    label = 'Passed' if status == 'pass' else ('Warning' if status == 'warn' else 'Failed')
    det = f'<span class="qr-detail">{detail}</span>' if detail else ''
    return (f'<div class="qr">'
            f'<span class="qr-name">{name}</span>'
            f'<div>{_badge(label, kind)}{det}</div>'
            f'</div>')

def _empty(title, subtitle, btn_label=None, btn_page=None):
    st.markdown(f'<div class="es"><p class="es-title">{title}</p>'
                f'<p class="es-sub">{subtitle}</p></div>',
                unsafe_allow_html=True)
    if btn_label and btn_page:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button(btn_label, type='primary', use_container_width=True):
                _go_to(btn_page)

def _guard_data(page='this page'):
    if st.session_state.data is None:
        _empty('No Dataset Loaded',
               f'Upload and preprocess a dataset before using {page}.',
               'Go to Data', 'Data')
        return True
    return False

def _guard_trained(page='this page'):
    if not st.session_state.models_trained:
        _empty('Models Not Trained',
               f'Train the forecasting models before using {page}.',
               'Go to Train Models', 'Train Models')
        return True
    return False

def _guard_forecast(page='this page'):
    if not st.session_state.forecast_generated:
        _empty('No Forecast Generated',
               f'Generate a demand forecast before using {page}.',
               'Go to Forecast', 'Forecast')
        return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-row">
            <div class="sb-brand-mark"></div>
            <span class="sb-brand-title">AI Demand Forecasting</span>
        </div>
        <div class="sb-brand-sub">Inventory Optimization</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<span class="sb-sec">Main Menu</span>', unsafe_allow_html=True)
    _sel = st.radio('nav', _PAGES, index=st.session_state.get('_nav_idx', 0),
                    label_visibility='collapsed')
    st.session_state['_nav_idx'] = _PAGES.index(_sel) if _sel in _PAGES else 0

    st.markdown('<hr>', unsafe_allow_html=True)

    # ── Inventory Parameters ──────────────────────────────────────────────────
    st.markdown('<span class="sb-sec">Inventory Parameters</span>', unsafe_allow_html=True)
    st.number_input('Lead Time (days)',   1, 90, key='lead_time',          step=1)
    st.number_input('Service Level (%)', 80, 99, key='service_level_pct', step=1)

    # Safety Factor (Z) — derived, read-only
    _z_val = _get_z(st.session_state.service_level_pct)
    st.markdown(
        '<p style="font-size:14px;font-weight:500;'
        'letter-spacing:0;color:#cbd5e1;margin:10px 0 5px 0;">Safety Factor (Z)</p>'
        f'<div class="sb-z-box">{_z_val:.3f}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr>', unsafe_allow_html=True)

    # ── Forecast Settings ─────────────────────────────────────────────────────
    st.markdown('<span class="sb-sec">Forecast Settings</span>', unsafe_allow_html=True)
    st.number_input('Horizon (days)',    7,  90, key='forecast_steps', step=1)
    st.number_input('Look-back (days)', 10, 60, key='seq_length',     step=1)

    st.markdown('<hr>', unsafe_allow_html=True)
    # ── Snowflake Connection Test ─────────────────────────────────────────
    if st.button("Test Snowflake Connection", use_container_width=True):
        try:
            row_count = test_snowflake_connection()
            st.success(
                f"Snowflake connected successfully! "
                f"DEMAND_HISTORY contains {row_count:,} rows."
            )
        except Exception as e:
            st.error(f"Snowflake connection failed: {e}")
    # ── Run Analysis button ───────────────────────────────────────────────────
    if st.button('Run Analysis', type='primary', use_container_width=True):
        if st.session_state.data is None:
            _go_to('Data')
        else:
            _go_to('Train Models')

    # ── System Status ─────────────────────────────────────────────────────────
    _has_data     = st.session_state.data is not None
    _has_trained  = st.session_state.models_trained
    _has_forecast = st.session_state.forecast_generated

    def _dot(cond): return 'sb-dot-on' if cond else 'sb-dot-off'

    st.markdown(f"""
    <div class="sb-status">
        <div class="sb-status-item">
            <span class="sb-dot {_dot(_has_data)}"></span>
            <span>Dataset <span class="sb-status-val">{'Loaded' if _has_data else 'None'}</span></span>
        </div>
        <div class="sb-status-item">
            <span class="sb-dot {_dot(_has_trained)}"></span>
            <span>Models <span class="sb-status-val">{'Trained' if _has_trained else 'Pending'}</span></span>
        </div>
        <div class="sb-status-item">
            <span class="sb-dot {_dot(_has_forecast)}"></span>
            <span>Forecast <span class="sb-status-val">{'Ready' if _has_forecast else 'Pending'}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Current page
_page = _PAGES[st.session_state.get('_nav_idx', 0)]



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def _page_dashboard():
    has_data = st.session_state.data is not None

    # ── Page Header ───────────────────────────────────────────────────────────
    hc1, hc2 = st.columns([5, 1])
    with hc1:
        st.markdown("""
        <p style="font-size:30px;font-weight:700;color:#0f172a;letter-spacing:-0.5px;margin:0 0 4px 0;">
            AI-Based Demand Forecasting System</p>
        <p style="font-size:14px;color:#64748b;margin:0;">
            LSTM &amp; Hybrid ARIMA+XGBoost
            <span style="color:#64748b;margin:0 8px;">&#183;</span>
            Inventory Optimization
            <span style="color:#64748b;margin:0 8px;">&#183;</span>
            Safety Stock &amp; Reorder Point
        </p>""", unsafe_allow_html=True)
    with hc2:
        if st.button("Upload New Data", use_container_width=True):
            _go_to("Data")
    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:14px 0 18px 0;">', unsafe_allow_html=True)

    if not has_data:
        st.markdown("""
        <div style="text-align:center;padding:52px 24px;background:white;border:1.5px dashed #cbd5e1;
             border-radius:12px;margin-top:8px;">
            <p style="font-size:18px;font-weight:600;color:#334155;margin:0 0 8px 0;">No Dataset Loaded</p>
            <p style="font-size:15px;color:#64748b;margin:0 0 20px 0;max-width:360px;margin-left:auto;margin-right:auto;">
                Upload a CSV dataset to begin demand forecasting and inventory analysis.</p>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Go to Data Upload", type="primary", use_container_width=True):
                _go_to("Data")
        return

    df = st.session_state.data
    dc = st.session_state.selected_date_col
    sc = st.session_state.selected_sales_col

    avg_d   = float(df[sc].mean()) if sc in df.columns else 0.0
    std_d   = float(df[sc].std())  if sc in df.columns else 0.0
    d_min   = df[dc].min().strftime('%Y-%m-%d') if dc in df.columns else 'N/A'
    d_max   = df[dc].max().strftime('%Y-%m-%d') if dc in df.columns else 'N/A'
    n_rec   = len(df)
    best_model  = st.session_state.best_model
    has_fc      = st.session_state.forecast_generated and st.session_state.forecast_lstm is not None
    fc_horizon  = len(st.session_state.forecast_lstm) if has_fc else None

    # ── Icon KPI Card helper ──────────────────────────────────────────────────
    def _ikpi(label, value, sub, icon, ibg, ifg, vclass='', fs='28px'):
        vst = (f'font-size:{fs};font-weight:800;color:{ifg if vclass=="green" else "#0f172a"};'
               f'letter-spacing:-0.6px;margin:0 0 5px 0;line-height:1.15;overflow-wrap:anywhere;')
        return f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;
             box-shadow:0 1px 3px rgba(0,0,0,0.07);display:flex;align-items:flex-start;
             justify-content:space-between;gap:10px;min-height:118px;">
          <div style="flex:1;min-width:0;">
            <p style="font-size:14px;font-weight:600;color:#475569;margin:0 0 8px 0;">{label}</p>
            <p style="{vst}">{value}</p>
            <p style="font-size:14px;color:#64748b;margin:0;">{sub}</p>
          </div>
          <div style="width:38px;height:38px;flex-shrink:0;border-radius:8px;background:{ibg};
               color:{ifg};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;">
               {icon}</div>
        </div>"""

    # ── Row 1: KPI Cards ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(_ikpi('Total Records',     _fmt(n_rec, 0),  'Historical records',   '&#9783;', '#dbeafe', '#1e40af'), unsafe_allow_html=True)
    k2.markdown(_ikpi('Date Range',        d_min,           f'to {d_max}',          '&#9781;', '#e0e7ff', '#4f46e5'), unsafe_allow_html=True)
    k3.markdown(_ikpi('Avg. Daily Demand', _fmt(avg_d, 0),  'Units per period',     '&#9636;', '#d1fae5', '#059669'), unsafe_allow_html=True)
    k4.markdown(_ikpi('Demand Std Dev',    _fmt(std_d, 0),  'Demand variability',   '&#963;',  '#fef3c7', '#b45309'), unsafe_allow_html=True)
    if best_model:
        cdf = st.session_state.comparison_df
        mstr = ''
        if cdf is not None and len(cdf) > 0:
            r = cdf[cdf['Model'] == best_model]
            if len(r): mstr = f"RMSE {r.iloc[0]['RMSE']:.3f}"
        k5.markdown(_ikpi('Best Model', best_model, mstr or 'Lowest RMSE', '&#9733;', '#d1fae5', '#059669', 'green', fs='22px'), unsafe_allow_html=True)
    else:
        k5.markdown(_ikpi('Best Model', 'Not Trained', 'Train models first', '&#9733;', '#f1f5f9', '#94a3b8'), unsafe_allow_html=True)
    if has_fc:
        k6.markdown(_ikpi('Forecast Horizon', f'{fc_horizon} days', f'From {d_max}', '&#8599;', '#ede9fe', '#7c3aed'), unsafe_allow_html=True)
    else:
        k6.markdown(_ikpi('Forecast Horizon', 'Pending', 'Generate forecast', '&#8599;', '#f1f5f9', '#94a3b8'), unsafe_allow_html=True)

    # ── Row 1b: Products & Stores (only when the dataset supports them) ────────
    sku_c_k = st.session_state.selected_sku_col
    store_c_k = st.session_state.selected_store_col
    n_sku   = int(df[sku_c_k].nunique()) if sku_c_k and sku_c_k in df.columns else None
    n_store = int(df[store_c_k].nunique()) if store_c_k and store_c_k in df.columns else None
    if n_sku is not None or n_store is not None:
        k_extras = []
        if n_store is not None:
            k_extras.append(_ikpi('Stores', _fmt(n_store, 0), 'Distinct store IDs', 'S', '#e0e7ff', '#4f46e5'))
        if n_sku is not None:
            k_extras.append(_ikpi('Products', _fmt(n_sku, 0), 'Distinct SKUs / items', 'P', '#ede9fe', '#7c3aed'))
        kx = st.columns(len(k_extras))
        for _col, _html in zip(kx, k_extras):
            _col.markdown(_html, unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 2: Dataset Overview | Data Quality ────────────────────────────────
    ov_col, dq_col = st.columns([1, 1])

    with ov_col:
        # Compute stats
        missing_pct = float(df[sc].isna().mean() * 100) if sc in df.columns else 0.0
        dup_count   = int(df.duplicated(subset=[dc]).sum()) if dc in df.columns else 0
        neg_count   = int((df[sc] < 0).sum()) if sc in df.columns else 0
        try:
            q1, q3  = df[sc].quantile([0.25, 0.75])
            iqr     = q3 - q1
            out_cnt = int(((df[sc] < q1 - 1.5*iqr) | (df[sc] > q3 + 1.5*iqr)).sum())
            out_pct = out_cnt / len(df) * 100
        except Exception:
            out_cnt, out_pct = 0, 0.0

        chk_passed   = sum([missing_pct < 5, dup_count == 0, neg_count == 0, out_pct < 5,
                            dc in df.columns, sc in df.columns])
        quality_score = round(chk_passed / 6 * 100)

        src_mode     = st.session_state.get('_src_mode', 'Upload Files')
        dataset_type = 'Multi-File Dataset' if src_mode == 'Local Folder Path' else 'Single File Dataset'
        tag_c        = '#dbeafe;color:#1e40af' if 'Multi' in dataset_type else '#d1fae5;color:#059669'
        miss_tag     = (f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:14px;'
                        f'font-weight:600;background:#fef3c7;color:#b45309;">{missing_pct:.2f}%</span>'
                        if missing_pct > 0 else
                        '<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:14px;'
                        'font-weight:600;background:#d1fae5;color:#059669;">0%</span>')

        deg        = round(quality_score * 3.6)
        ring_html  = (
            f'<div style="width:56px;height:56px;border-radius:50%;flex-shrink:0;'
            f'background:conic-gradient(#059669 {deg}deg,#e2e8f0 0deg);'
            f'display:flex;align-items:center;justify-content:center;position:relative;">'
            f'<div style="width:40px;height:40px;background:white;border-radius:50%;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:14px;font-weight:700;color:#059669;position:absolute;">{quality_score}%</div></div>'
        )

        def _dov_row(label, val_html):
            return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;'
                    f'align-items:center;padding:8px 0;border-bottom:1px solid #f1f5f9;">'
                    f'<span style="font-size:14px;color:#64748b;">{label}</span>'
                    f'<span>{val_html}</span></div>')

        tag_html = f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:14px;font-weight:600;background:{tag_c};">{dataset_type}</span>'
        gran_html= '<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:14px;font-weight:600;background:#d1fae5;color:#059669;">Daily</span>'

        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;height:100%;">
            <p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 14px 0;">Dataset Overview</p>
            {_dov_row('Dataset Type', tag_html)}
            {_dov_row('Granularity', gran_html)}
            {_dov_row('Total Missing Values', miss_tag)}
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:center;padding:8px 0;">
                <span style="font-size:14px;color:#64748b;">Data Quality Score</span>
                <div style="display:flex;align-items:center;gap:8px;">{ring_html}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with dq_col:
        def _dq(icon, ibg, ifg, name, status, ok=True):
            sc2 = '#059669' if ok else '#b45309'
            return (f'<div style="display:flex;align-items:flex-start;gap:9px;padding:10px;'
                    f'background:#f8fafc;border:1px solid #f1f5f9;border-radius:7px;">'
                    f'<div style="width:28px;height:28px;border-radius:6px;flex-shrink:0;background:{ibg};'
                    f'color:{ifg};display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:700;">'
                    f'{icon}</div>'
                    f'<div><p style="font-size:14px;font-weight:600;color:#334155;margin:0 0 1px 0;">{name}</p>'
                    f'<p style="font-size:14px;color:{sc2};margin:0;">{status}</p></div></div>')

        items = [
            _dq('&#9781;', '#dbeafe', '#1e40af', 'Date Range Check',
                'Passed' if dc in df.columns else 'Failed', dc in df.columns),
            _dq('T', '#e0e7ff', '#4f46e5', 'Data Type Check',
                'Passed' if sc in df.columns else 'Failed', sc in df.columns),
            _dq('&#8776;', '#d1fae5', '#059669', 'Missing Values Check',
                'Passed' if missing_pct == 0 else f'{missing_pct:.2f}% missing', missing_pct < 5),
            _dq('&#8722;', '#fef3c7', '#b45309', 'Negative Demand Check',
                'No negative values' if neg_count == 0 else f'{neg_count} rows', neg_count == 0),
            _dq('&#8801;', '#d1fae5', '#059669', 'Duplicate Records',
                'No duplicates found' if dup_count == 0 else f'{dup_count} found', dup_count == 0),
            _dq('&#9636;', '#ede9fe', '#7c3aed', 'Outlier Detection',
                f'{out_pct:.2f}% outliers detected' if out_cnt > 0 else 'No outliers', out_pct < 5),
        ]
        grid = ''.join(items)
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;height:100%;">
            <p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 14px 0;">Data Quality Checks</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">{grid}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 3: Demand Trend | Seasonality | Distribution ──────────────────────
    ch1, ch2, ch3 = st.columns([2, 1.5, 1.5])

    with ch1:
        st.markdown('<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">'
                    '<p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 6px 0;">Demand Trend</p>',
                    unsafe_allow_html=True)
        t_d, t_w, t_m = st.tabs(['Daily', 'Weekly', 'Monthly'])

        def _trend_fig(resample):
            try:
                if resample == 'D':
                    df_p = df[[dc, sc]].copy()
                else:
                    df_p = df.set_index(dc)[sc].resample(resample).sum().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p[dc], y=df_p[sc], mode='lines',
                    line=dict(color='#1e40af', width=1.5),
                    fill='tozeroy', fillcolor='rgba(30,64,175,0.06)'))
                fig.update_layout(**_cl(height=255, showlegend=False, yaxis_title='Demand',
                    margin=dict(l=10, r=10, t=8, b=10)))
                return fig
            except Exception:
                return go.Figure()

        with t_d: st.plotly_chart(_trend_fig('D'),  use_container_width=True)
        with t_w: st.plotly_chart(_trend_fig('W'),  use_container_width=True)
        with t_m: st.plotly_chart(_trend_fig('ME'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">'
                    '<p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Seasonality (Avg Demand)</p>',
                    unsafe_allow_html=True)
        try:
            df_s = df.copy()
            df_s['_m'] = df_s[dc].dt.month
            mon = df_s.groupby('_m')[sc].mean().reset_index()
            mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            mon['_mn'] = mon['_m'].apply(lambda x: mnames[x-1])
            fig_s = go.Figure(go.Bar(x=mon['_mn'], y=mon[sc],
                marker_color='#7c3aed', marker_line_width=0))
            fig_s.update_layout(**_cl(height=290, showlegend=False, bargap=0.25,
                margin=dict(l=10, r=10, t=8, b=10)))
            st.plotly_chart(fig_s, use_container_width=True)
        except Exception:
            st.info('Unavailable.')
        st.markdown('</div>', unsafe_allow_html=True)

    with ch3:
        st.markdown('<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">'
                    '<p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Demand Distribution</p>',
                    unsafe_allow_html=True)
        try:
            fig_d = go.Figure(go.Histogram(x=df[sc].dropna(), nbinsx=30,
                marker_color='#059669', marker_line_width=0, opacity=0.85))
            fig_d.update_layout(**_cl(height=290, showlegend=False,
                xaxis_title='Demand', yaxis_title='Count',
                margin=dict(l=10, r=10, t=8, b=10)))
            st.plotly_chart(fig_d, use_container_width=True)
        except Exception:
            st.info('Unavailable.')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 3b: Top Products | Top Stores ─────────────────────────────────────
    _rank_items = []
    if sku_c_k and sku_c_k in df.columns:
        _t = (df.groupby(sku_c_k)[sc].sum().reset_index()
                 .sort_values(sc, ascending=False).head(5).copy())
        _t['pct'] = _t[sc] / _t[sc].sum() * 100
        _rank_items.append(('Top Products', sku_c_k, _t))
    if store_c_k and store_c_k in df.columns:
        _t = (df.groupby(store_c_k)[sc].sum().reset_index()
                 .sort_values(sc, ascending=False).head(5).copy())
        _t['pct'] = _t[sc] / _t[sc].sum() * 100
        _rank_items.append(('Top Stores', store_c_k, _t))

    if _rank_items:
        _rp = st.columns(len(_rank_items))
        for _rcol, (_ttl, _name, _tbl) in zip(_rp, _rank_items):
            _maxv = float(_tbl[sc].max()) if len(_tbl) else 1.0
            _rows = ''.join(
                f'<div style="display:grid;grid-template-columns:1.6fr 1fr 0.8fr;gap:8px;'
                f'align-items:center;padding:7px 0;border-bottom:1px solid #f8fafc;font-size:14px;">'
                f'<span style="color:#334155;font-weight:500;font-size:14px;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{row[_name]}</span>'
                f'<span style="color:#0f172a;font-weight:600;">{_fmt(row[sc])}</span>'
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<span style="height:4px;border-radius:2px;display:inline-block;'
                f'width:{int(row[sc]/_maxv*60)}px;background:#059669;"></span>'
                f'<span style="font-size:14px;color:#64748b;">{row["pct"]:.1f}%</span></div></div>'
                for _, row in _tbl.iterrows()
            )
            _rcol.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px 16px;height:100%;">
                <p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">{_ttl}</p>
                <div style="display:grid;grid-template-columns:1.6fr 1fr 0.8fr;gap:8px;padding:5px 0 7px 0;border-bottom:1px solid #e2e8f0;">
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">{_name}</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">Total</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">% Share</span>
                </div>
                {_rows}
            </div>""", unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 4: Top Demand Periods | Model Performance | Quick Actions ─────────
    b1, b2, b3 = st.columns([1.5, 1.5, 1])

    with b1:
        # Top 5 Highest Demand Periods
        try:
            top5    = df[[dc, sc]].nlargest(5, sc).copy()
            top5['pct'] = top5[sc] / df[sc].sum() * 100
            max_val = float(top5[sc].max()) if len(top5) else 1.0

            rows_html = ''.join(
                f'<div style="display:grid;grid-template-columns:1.6fr 1fr 0.8fr;gap:8px;'
                f'align-items:center;padding:7px 0;border-bottom:1px solid #f8fafc;font-size:14px;">'
                f'<span style="color:#334155;font-weight:500;font-family:monospace;font-size:14px;">'
                f'{row[dc].strftime("%Y-%m-%d") if hasattr(row[dc], "strftime") else str(row[dc])}</span>'
                f'<span style="color:#0f172a;font-weight:600;">{_fmt(row[sc])}</span>'
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<span style="height:4px;border-radius:2px;display:inline-block;width:{int(row[sc]/max_val*60)}px;background:#1e40af;"></span>'
                f'<span style="font-size:14px;color:#64748b;">{row["pct"]:.1f}%</span></div></div>'
                for _, row in top5.iterrows()
            )
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">
                <p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Top Demand Periods</p>
                <div style="display:grid;grid-template-columns:1.6fr 1fr 0.8fr;gap:8px;padding:5px 0 7px 0;border-bottom:1px solid #e2e8f0;">
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">Date</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">Total Demand</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">% Share</span>
                </div>
                {rows_html}
            </div>""", unsafe_allow_html=True)
        except Exception:
            st.markdown('<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;"><p style="font-size:17px;font-weight:700;color:#0f172a;">Top Demand Periods</p><p style="font-size:14px;color:#64748b;padding:12px 0;">No data available</p></div>', unsafe_allow_html=True)

    with b2:
        if st.session_state.models_trained and st.session_state.comparison_df is not None:
            comp = st.session_state.comparison_df
            rows_html = ''.join(
                f'<div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:8px;'
                f'align-items:center;padding:7px 0;border-bottom:1px solid #f8fafc;font-size:14px;">'
                f'<span style="color:#334155;font-weight:700;font-size:14px;">{row["Model"]}</span>'
                f'<span style="color:#64748b;">{row["MAE"]:.3f}</span>'
                f'<span style="color:#64748b;">{row["RMSE"]:.3f}</span>'
                f'<span style="color:#64748b;">{row["MAPE (%)"]:.2f}%</span></div>'
                for _, row in comp.iterrows()
            )
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">
                <p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Model Performance</p>
                <div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:8px;padding:5px 0 7px 0;border-bottom:1px solid #e2e8f0;">
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">Model</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">MAE</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">RMSE</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">MAPE</span>
                </div>
                {rows_html}
            </div>""", unsafe_allow_html=True)
        else:
            # Demand statistics
            stats_rows = ''
            if sc in df.columns:
                p25, p50, p75 = df[sc].quantile([0.25, 0.5, 0.75])
                for nm, v in [('Mean', avg_d), ('Std Dev', std_d), ('Median', float(p50)),
                              ('25th Pct', float(p25)), ('75th Pct', float(p75)), ('Max', float(df[sc].max()))]:
                    stats_rows += (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;'
                                   f'padding:7px 0;border-bottom:1px solid #f8fafc;font-size:14px;">'
                                   f'<span style="color:#64748b;">{nm}</span>'
                                   f'<span style="color:#0f172a;font-weight:600;">{_fmt(v)}</span></div>')
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">
                <p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Demand Statistics</p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:5px 0 7px 0;border-bottom:1px solid #e2e8f0;">
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">Metric</span>
                    <span style="font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#64748b;">Value</span>
                </div>
                {stats_rows}
            </div>""", unsafe_allow_html=True)

    with b3:
        actions = [
            ('Upload Data',       'Data',         '&#8679;', '#dbeafe', '#1e40af', 'Import new dataset'),
            ('Run EDA',           'EDA',          '&#9636;', '#d1fae5', '#059669', 'Explore your data'),
            ('Train Models',      'Train Models', '&#9670;', '#e0e7ff', '#4f46e5', 'Train &amp; compare'),
            ('Forecast',          'Forecast',     '&#8599;', '#fef3c7', '#b45309', 'Predict future demand'),
            ('Inventory',         'Inventory',    '&#9783;', '#ede9fe', '#7c3aed', 'Safety stock &amp; ROP'),
            ('Download Report',   'Reports',      '&#8595;', '#d1fae5', '#059669', 'Export results'),
        ]
        cards = ''.join(
            f'<div style="background:#f8fafc;border:1px solid #f1f5f9;border-radius:8px;'
            f'padding:12px 8px 10px 8px;text-align:center;">'
            f'<div style="width:32px;height:32px;border-radius:8px;margin:0 auto 8px auto;'
            f'background:{bg};color:{fg};display:flex;align-items:center;justify-content:center;font-size:17px;">{icon}</div>'
            f'<p style="font-size:14px;font-weight:700;color:#334155;margin:0 0 2px 0;">{name}</p>'
            f'<p style="font-size:14px;color:#64748b;margin:0;">{sub}</p></div>'
            for name, _, icon, bg, fg, sub in actions
        )
        st.markdown(f'<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">'
                    f'<p style="font-size:17px;font-weight:700;color:#0f172a;margin:0 0 12px 0;">Quick Actions</p>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">{cards}</div></div>',
                    unsafe_allow_html=True)
        # Actual clickable buttons
        btn_cols = st.columns(3)
        for i, (name, target, *_) in enumerate(actions):
            with btn_cols[i % 3]:
                if st.button(name, key=f'qa_{i}', use_container_width=True):
                    _go_to(target)



def _page_data():
    _ph('Data', 'Upload train.csv, test.csv, and sample_submission.csv for demand forecasting')

    # Workflow Step Indicator
    steps = [
        ("1", "Upload Train Data", st.session_state.train_df is not None),
        ("2", "Upload Test Data", st.session_state.test_df is not None),
        ("3", "Upload Sample Submission", st.session_state.sample_submission is not None),
        ("4", "Validate & Preprocess", st.session_state.data is not None),
    ]
    
    step_html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">'
    for i, (num, label, done) in enumerate(steps):
        color = '#059669' if done else '#64748b'
        bg = '#d1fae5' if done else '#f1f5f9'
        step_html += (
            '<div style="flex:1;text-align:center;">'
            f'<div style="width:28px;height:28px;border-radius:50%;background:{bg};color:{color};'
            'display:flex;align-items:center;justify-content:center;margin:0 auto 6px;'
            'font-weight:700;font-size:14px;">{num}</div>'
            f'<div style="font-size:14px;color:{color};font-weight:600;">{label}</div>'
            '</div>'
        )
        if i < len(steps) - 1:
            step_html += '<div style="flex:1;border-top:2px solid #e2e8f0;margin-top:14px;"></div>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

    # Source Selection
    _sec('Upload Datasets')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('**train.csv** (with units_sold)')
        train_uploaded = st.file_uploader(
            'Upload train.csv',
            type=['csv'],
            accept_multiple_files=False,
            help='Training data with target column (units_sold). Approx 150,150 rows.',
            key='train_uploader',
            label_visibility='collapsed'
        )
        if train_uploaded:
            try:
                train_df = pd.read_csv(train_uploaded)
                st.session_state.train_df = train_df
                st.success(f'Loaded train.csv: {len(train_df):,} rows, {len(train_df.columns)} cols')
            except Exception as e:
                st.error(f'Failed to load train.csv: {e}')
    
    with col2:
        st.markdown('**test.csv** (no units_sold)')
        test_uploaded = st.file_uploader(
            'Upload test.csv',
            type=['csv'],
            accept_multiple_files=False,
            help='Test data without target column. Approx 13,860 rows.',
            key='test_uploader',
            label_visibility='collapsed'
        )
        if test_uploaded:
            try:
                test_df = pd.read_csv(test_uploaded)
                st.session_state.test_df = test_df
                st.success(f'Loaded test.csv: {len(test_df):,} rows, {len(test_df.columns)} cols')
            except Exception as e:
                st.error(f'Failed to load test.csv: {e}')
    
    with col3:
        st.markdown('**sample_submission.csv**')
        sub_uploaded = st.file_uploader(
            'Upload sample_submission.csv',
            type=['csv'],
            accept_multiple_files=False,
            help='Submission template with record_ID and units_sold columns.',
            key='sub_uploader',
            label_visibility='collapsed'
        )
        if sub_uploaded:
            try:
                sub_df = pd.read_csv(sub_uploaded)
                st.session_state.sample_submission = sub_df
                st.success(f'Loaded sample_submission.csv: {len(sub_df):,} rows')
            except Exception as e:
                st.error(f'Failed to load sample_submission.csv: {e}')

    # Show uploaded data info
    if st.session_state.train_df is not None or st.session_state.test_df is not None:
        st.markdown('<br>', unsafe_allow_html=True)
        _sec('Uploaded Data Summary')
        
        if st.session_state.train_df is not None:
            df = st.session_state.train_df
            st.markdown('**Train Data**')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Rows', f'{len(df):,}')
            c2.metric('Columns', len(df.columns))
            c3.metric('Date Range', f"{df['week'].min()} to {df['week'].max()}" if 'week' in df.columns else 'N/A')
            c4.metric('StoreSKU Groups', df.groupby(['store_id', 'sku_id']).ngroups if 'store_id' in df.columns and 'sku_id' in df.columns else 'N/A')
            st.dataframe(df.head(10), use_container_width=True, height=200)
        
        if st.session_state.test_df is not None:
            df = st.session_state.test_df
            st.markdown('**Test Data**')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Rows', f'{len(df):,}')
            c2.metric('Columns', len(df.columns))
            c3.metric('Date Range', f"{df['week'].min()} to {df['week'].max()}" if 'week' in df.columns else 'N/A')
            c4.metric('StoreSKU Groups', df.groupby(['store_id', 'sku_id']).ngroups if 'store_id' in df.columns and 'sku_id' in df.columns else 'N/A')
            st.dataframe(df.head(10), use_container_width=True, height=200)

    # Column Mapping
    if st.session_state.train_df is not None:
        st.markdown('<br>', unsafe_allow_html=True)
        _sec('Column Mapping')
        
        train_cols = list(st.session_state.train_df.columns)
        test_cols = list(st.session_state.test_df.columns) if st.session_state.test_df is not None else []
        
        # Auto-detect columns
        date_aliases = ['week', 'date', 'Date', 'timestamp', 'ds', 'period']
        sales_aliases = ['units_sold', 'sales', 'Sales', 'demand', 'quantity', 'qty']
        store_aliases = ['store_id', 'store', 'Store', 'storeId']
        sku_aliases = ['sku_id', 'sku', 'SKU', 'item_id', 'product_id']
        
        def auto_detect(cols, aliases):
            for a in aliases:
                if a in cols:
                    return a
            for a in aliases:
                for c in cols:
                    if a.lower() == c.lower():
                        return c
            return None
        
        auto_date = auto_detect(train_cols, date_aliases)
        auto_sales = auto_detect(train_cols, sales_aliases)
        auto_store = auto_detect(train_cols, store_aliases)
        auto_sku = auto_detect(train_cols, sku_aliases)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            date_col = st.selectbox('Date/Week column', train_cols, 
                                     index=train_cols.index(auto_date) if auto_date in train_cols else 0,
                                     key='_dc')
        with c2:
            sales_col = st.selectbox('Target column (units_sold)', train_cols,
                                      index=train_cols.index(auto_sales) if auto_sales in train_cols else 0,
                                      key='_sc')
        with c3:
            store_col = st.selectbox('Store ID column', ['(none)'] + train_cols,
                                      index=(train_cols.index(auto_store) + 1) if auto_store in train_cols else 0,
                                      key='_storec')
            if store_col == '(none)':
                store_col = None
        with c4:
            sku_col = st.selectbox('SKU ID column', ['(none)'] + train_cols,
                                    index=(train_cols.index(auto_sku) + 1) if auto_sku in train_cols else 0,
                                    key='_skuc')
            if sku_col == '(none)':
                sku_col = None
        
        st.session_state.selected_date_col = date_col
        st.session_state.selected_sales_col = sales_col
        st.session_state.selected_store_col = store_col
        st.session_state.selected_sku_col = sku_col

    # Validation Split Configuration
    if st.session_state.train_df is not None:
        st.markdown('<br>', unsafe_allow_html=True)
        _sec('Validation Split (inside train.csv)')
        st.caption('The test.csv has no ground truth. We create a validation set from the last 20% of train.csv for model evaluation.')
        
        test_size = st.slider('Validation split ratio', 0.1, 0.4, st.session_state.test_size, 0.05,
                               key='val_split',
                               help='Fraction of train.csv to hold out for validation (chronological split)')
        st.session_state.test_size = test_size

    # Load & Preprocess Button
    if st.session_state.train_df is not None:
        st.markdown('<br>', unsafe_allow_html=True)
        
        col_btn, col_rst = st.columns([2, 6])
        with col_btn:
            load_btn = st.button('Validate & Preprocess Train Data', type='primary', use_container_width=True)
        with col_rst:
            if st.button('Reset All', use_container_width=False):
                for k in ['data', 'train_df', 'test_df', 'sample_submission', 'preprocessor', 
                          'models_trained', 'predictions', 'selected_date_col', 'selected_sales_col',
                          'selected_store_col', 'selected_sku_col', 'last_processed_sig',
                          'forecast_generated', 'forecast_lstm', 'forecast_hybrid',
                          'forecast_dates', 'test_predictions', 'test_record_ids',
                          'best_model', 'comparison_df']:
                    st.session_state[k] = _DEFAULTS.get(k)
                st.rerun()

        if load_btn:
            prog = st.progress(0, text='Preparing training dataset...')
            try:
                preprocessor = DataPreprocessor()
                
                prog.progress(20, text='Cleaning train data...')
                df_clean = preprocessor.clean_data(
                    st.session_state.train_df, 
                    date_col=st.session_state.selected_date_col,
                    sales_col=st.session_state.selected_sales_col,
                    store_col=st.session_state.selected_store_col,
                    sku_col=st.session_state.selected_sku_col
                )
                
                prog.progress(50, text='Validating data...')
                is_valid, msgs = preprocessor.validate_data(
                    df_clean,
                    st.session_state.selected_date_col,
                    st.session_state.selected_sales_col,
                    st.session_state.selected_store_col,
                    st.session_state.selected_sku_col,
                    seq_length=st.session_state.seq_length
                )
                if not is_valid:
                    prog.empty()
                    st.error('Validation failed. See details below.')
                    for m in msgs:
                        st.error(m)
                    return
                for w in msgs:
                    st.warning(w)
                
                prog.progress(80, text='Finalizing...')
                st.session_state.data = df_clean
                st.session_state.preprocessor = preprocessor
                st.session_state.last_processed_sig = 'train_processed'
                # Reset downstream
                st.session_state.models_trained = False
                st.session_state.predictions = {}
                st.session_state.forecast_generated = False
                st.session_state.forecast_lstm = None
                st.session_state.forecast_hybrid = None
                st.session_state.forecast_dates = None
                st.session_state.test_predictions = None
                st.session_state.test_record_ids = None
                st.session_state.best_model = None
                st.session_state.comparison_df = None
                
                prog.progress(100, text='Done.')
                st.rerun()
            
            except Exception as exc:
                prog.empty()
                st.error('Failed to preprocess dataset.')
                with st.expander('Technical details'):
                    st.code(str(exc))
                return

    # Show Preprocessed Data Info
    if st.session_state.data is not None:
        st.markdown('<br>', unsafe_allow_html=True)
        _sec('Preprocessed Training Data')
        
        df = st.session_state.data
        dc = st.session_state.selected_date_col
        sc = st.session_state.selected_sales_col
        store_c = st.session_state.selected_store_col
        sku_c = st.session_state.selected_sku_col
        
        info_col, qual_col = st.columns([1, 1])
        
        with info_col:
            n_rows = len(df)
            d_min = df[dc].min().strftime('%Y-%m-%d') if dc in df else 'N/A'
            d_max = df[dc].max().strftime('%Y-%m-%d') if dc in df else 'N/A'
            avg_d = df[sc].mean() if sc in df else 0
            total_d = df[sc].sum() if sc in df else 0
            n_groups = df.groupby([store_c, sku_c]).ngroups if store_c and sku_c and store_c in df and sku_c in df else 1
            
            st.markdown(f"""
            <div class="card">
            <table class="info-tbl">
                <tr><td>Records</td><td>{n_rows:,}</td></tr>
                <tr><td>Date Range</td><td>{d_min} - {d_max}</td></tr>
                <tr><td>Average Demand</td><td>{avg_d:,.2f} units</td></tr>
                <tr><td>Total Demand</td><td>{_fmt(total_d)}</td></tr>
                <tr><td>StoreSKU Groups</td><td>{n_groups}</td></tr>
                <tr><td>Date Column</td><td>{dc}</td></tr>
                <tr><td>Target Column</td><td>{sc}</td></tr>
                <tr><td>Store Column</td><td>{store_c or 'N/A (single series)'}</td></tr>
                <tr><td>SKU Column</td><td>{sku_c or 'N/A (single series)'}</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        with qual_col:
            _sec('Data Quality')
            missing_pct = df[sc].isna().mean() * 100 if sc in df else 0
            dup_count = df.duplicated(subset=[dc]).sum() if dc in df else 0
            neg_count = (df[sc] < 0).sum() if sc in df else 0
            std_val = df[sc].std() if sc in df else 1
            n_rows_chk = len(df)
            
            qrows = ''
            qrows += _qrow('Date column parseable',   'pass' if dc in df else 'err')
            qrows += _qrow('Target column numeric',   'pass' if sc in df else 'err')
            qrows += _qrow('Missing values',
                            'pass' if missing_pct == 0 else ('warn' if missing_pct < 5 else 'err'),
                            f'{missing_pct:.1f}%')
            qrows += _qrow('Negative demand',
                            'pass' if neg_count == 0 else 'warn',
                            f'{int(neg_count)} rows' if neg_count > 0 else '')
            qrows += _qrow('Demand variance',
                            'pass' if std_val > 0 else 'err',
                            'Constant series' if std_val == 0 else '')
            qrows += _qrow('Minimum row count (100)',
                            'pass' if n_rows_chk >= 100 else 'warn',
                            f'{n_rows_chk} rows')
            
            st.markdown(f'<div class="card">{qrows}</div>', unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        _sec('Data Preview')
        st.dataframe(df.head(50), use_container_width=True, height=220)

# ========================================================================
# PAGE: EDA
# ========================================================================
def _page_eda():
    _ph('Exploratory Analysis', 'Understand demand patterns, seasonality, and distribution')
    if _guard_data('EDA'):
        return

    df = st.session_state.data
    dc = st.session_state.selected_date_col
    sc = st.session_state.selected_sales_col
    eda = ExploratoryAnalysis()

    # ── Summary Stats ─────────────────────────────────────────────────────────
    stats = eda.generate_summary_statistics(df, sc)
    _sec('Summary Statistics')
    stat_cols = st.columns(5)
    stat_items = [
        ('Mean', f'{stats["Mean"]:,.2f}'),
        ('Std Dev', f'{stats["Std Dev"]:,.2f}'),
        ('Median', f'{stats["Median"]:,.2f}'),
        ('Min', f'{stats["Min"]:,.2f}'),
        ('Max', f'{stats["Max"]:,.2f}'),
    ]
    for col, (lbl, val) in zip(stat_cols, stat_items):
        col.markdown(_kpi(lbl, val, '', ''), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 1: Trend + Seasonality ────────────────────────────────────────────
    _sec('Demand Trend & Seasonality')
    r1c1, r1c2 = st.columns([3, 2])

    with r1c1:
        # Trend with MA
        fig_trend = eda.analyze_trend(df, dc, sc, window=30)
        fig_trend.update_layout(**_cl(title='Demand Trend with Moving Averages', height=280))
        st.plotly_chart(fig_trend, use_container_width=True)

    with r1c2:
        # Seasonality tabs
        t_month, t_dow = st.tabs(['Monthly', 'Day of Week'])
        with t_month:
            fig_m = eda.plot_seasonal_pattern(df, dc, sc, 'Month')
            fig_m.update_layout(**_cl(title='Average Demand by Month', height=240))
            st.plotly_chart(fig_m, use_container_width=True)
        with t_dow:
            fig_d = eda.plot_seasonal_pattern(df, dc, sc, 'Day')
            fig_d.update_layout(**_cl(title='Average Demand by Day of Week', height=240))
            st.plotly_chart(fig_d, use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 2: Distribution + Outliers ───────────────────────────────────────
    _sec('Distribution & Outliers')
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_dist = eda.plot_distribution(df, sc, 'Demand Distribution')
        fig_dist.update_layout(**_cl(title='Demand Distribution', height=270))
        st.plotly_chart(fig_dist, use_container_width=True)

    with r2c2:
        df_out = eda.detect_outliers(df, sc, method='iqr')
        out_pts = df_out[df_out['Outlier']]
        norm_pts = df_out[~df_out['Outlier']]
        fig_out = go.Figure()
        fig_out.add_trace(go.Scatter(
            x=norm_pts[dc], y=norm_pts[sc],
            mode='markers', name='Normal',
            marker=dict(color=_C_PRIMARY, size=4, opacity=0.5),
        ))
        fig_out.add_trace(go.Scatter(
            x=out_pts[dc], y=out_pts[sc],
            mode='markers', name=f'Outliers ({len(out_pts)})',
            marker=dict(color='#dc2626', size=6, symbol='diamond'),
        ))
        fig_out.update_layout(**_cl(title='Outlier Detection (IQR 1.5x)', height=270))
        st.plotly_chart(fig_out, use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 3: Box Plot + Correlation ────────────────────────────────────────
    _sec('Monthly Pattern & Feature Correlation')
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        fig_box = eda.plot_monthly_boxplot(df, dc, sc)
        fig_box.update_layout(**_cl(title='Monthly Demand Spread', height=270))
        st.plotly_chart(fig_box, use_container_width=True)

    with r3c2:
        # Build features for correlation
        df_feat = eda.detect_outliers(df[[dc, sc]].copy(), sc)
        df_feat = df_feat.drop(columns=['Outlier'])
        # Add time features
        df_feat['Month']     = df_feat[dc].dt.month
        df_feat['DayOfWeek'] = df_feat[dc].dt.dayofweek
        df_feat['Quarter']   = df_feat[dc].dt.quarter
        # Lag
        df_feat[f'{sc}_Lag7']  = df_feat[sc].shift(7)
        df_feat[f'{sc}_Lag30'] = df_feat[sc].shift(30)
        df_feat = df_feat.dropna()
        num_cols = [c for c in df_feat.columns if c != dc]
        fig_corr = eda.plot_correlation_heatmap(df_feat, num_cols)
        if fig_corr:
            fig_corr.update_layout(**_cl(title='Feature Correlation Heatmap', height=270))
            st.plotly_chart(fig_corr, use_container_width=True)

    # ── ACF / PACF ────────────────────────────────────────────────────────────
    with st.expander('Autocorrelation (ACF / PACF)'):
        try:
            import matplotlib
            matplotlib.use('Agg')
            fig_acf = eda.plot_acf_pacf(df, sc)
            st.pyplot(fig_acf, use_container_width=True)
        except Exception as e:
            st.info(f'ACF/PACF unavailable: {e}')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
def _page_train():
    _ph('Train Models', 'Train LSTM and Hybrid ARIMA+XGBoost on panel data (store_id, sku_id)')
    if _guard_data('Train Models'):
        return

    df = st.session_state.data
    dc = st.session_state.selected_date_col
    sc = st.session_state.selected_sales_col
    store_c = st.session_state.selected_store_col
    sku_c = st.session_state.selected_sku_col
    seq_length = st.session_state.seq_length
    test_size = st.session_state.test_size
    arima_tuple = _parse_arima_order()
    lstm_epochs  = st.session_state.lstm_epochs
    lstm_batch   = st.session_state.lstm_batch
    lstm_status_t = 'Trained' if st.session_state.models_trained else 'Not Trained'
    hyb_status_t = 'Trained' if st.session_state.models_trained else 'Not Trained'

    is_panel = store_c is not None and sku_c is not None and store_c in df.columns and sku_c in df.columns

    # Training Config Summary
    _sec('Training Configuration')
    cfg_cols = st.columns(4)
    cfg_cols[0].markdown(_kpi('Records', f'{len(df):,}', 'training data', ''), unsafe_allow_html=True)
    cfg_cols[1].markdown(_kpi('Look-back', f'{seq_length} days', 'LSTM sequence length', ''), unsafe_allow_html=True)
    cfg_cols[2].markdown(_kpi('Val Split', f'{int((1-test_size)*100)}%', f'{int(test_size*100)}% held for validation', ''), unsafe_allow_html=True)
    cfg_cols[3].markdown(_kpi('ARIMA Order', f'{arima_tuple}', 'p,d,q — with fallback', ''), unsafe_allow_html=True)
    if is_panel:
        n_groups = df.groupby([store_c, sku_c]).ngroups
        st.caption(f'Panel data: {n_groups} unique (store_id, sku_id) combinations')

    st.markdown('<br>', unsafe_allow_html=True)

    # Minimum row check
    MIN_ROWS = seq_length * 2 + 20
    if len(df) < MIN_ROWS:
        st.error(f'Insufficient data for training. Need at least {MIN_ROWS} rows '
                 f'for look-back={seq_length}. Current: {len(df)} rows.')
        return

    # Model Cards
    _sec('Models')
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown(f"""
        <div class="mc">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
                <p class="mc-title">LSTM</p>
                {_badge(lstm_status_t, 'ok' if st.session_state.models_trained else 'gray')}
            </div>
            <p class="mc-sub">Deep Learning Time-Series Forecasting</p>
            <div class="mc-arch">
LSTM({st.session_state.lstm_batch * 2}) -> Dropout(0.2)
LSTM({st.session_state.lstm_batch}) -> Dropout(0.2)
Dense(16, ReLU) -> Dense(1)
            </div>
            <table class="info-tbl">
                <tr><td>Optimizer</td><td>Adam (lr=0.001)</td></tr>
                <tr><td>Loss</td><td>Mean Squared Error</td></tr>
                <tr><td>Early Stopping</td><td>patience=8</td></tr>
                <tr><td>Max Epochs</td><td>{lstm_epochs}</td></tr>
                <tr><td>Batch Size</td><td>{lstm_batch}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="mc">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
                <p class="mc-title">Hybrid ARIMA + XGBoost</p>
                {_badge(hyb_status_t, 'ok' if st.session_state.models_trained else 'gray')}
            </div>
            <p class="mc-sub">Statistical + Gradient Boosting Ensemble</p>
            <div class="mc-arch">
Step 1 - ARIMA{arima_tuple} on training data
Step 2 - Residuals = Actual - ARIMA fit
Step 3 - XGBoost on lagged residuals (10 lags)
Step 4 - Forecast = ARIMA + XGBoost correction
            </div>
            <table class="info-tbl">
                <tr><td>ARIMA Order</td><td>{arima_tuple} (with fallback)</td></tr>
                <tr><td>XGBoost</td><td>n_est=100, lr=0.05, depth=5</td></tr>
                <tr><td>Residual Lags</td><td>10</td></tr>
                <tr><td>Train Split</td><td>{int((1-test_size)*100)}% chronological</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Train Button
    c1, c2, c3 = st.columns([2, 2, 4])
    with c1:
        train_btn = st.button('Train Both Models', type='primary', use_container_width=True)
    with c2:
        if st.session_state.models_trained:
            if st.button('View Comparison', use_container_width=True):
                _go_to('Model Comparison')

    if not train_btn:
        if st.session_state.models_trained:
            st.markdown('<br>', unsafe_allow_html=True)
            _sec('Last Training Results')
            if st.session_state.comparison_df is not None:
                st.dataframe(st.session_state.comparison_df, use_container_width=True, hide_index=True)
        return

    # Training
    prog = st.progress(0)
    status_box = st.empty()

    try:
        preprocessor = st.session_state.preprocessor
        from src.services.forecasting_service import ForecastingService

        if is_panel:
            # Use panel data training
            status_box.info('Initializing Forecasting Service for panel data...')
            prog.progress(5)

            service = ForecastingService(
                seq_length=seq_length,
                lstm_epochs=lstm_epochs,
                lstm_batch_size=lstm_batch,
                arima_order=arima_tuple,
            )

            status_box.info('Training models on panel data (per store/SKU)...')
            prog.progress(15, text='Training LSTM and hybrid models on panel data...')

            def _panel_progress(fraction, message):
                # Map the service's 0..1 progress into the 15..68% band so the
                # later evaluation steps (70/85/100) still advance forward.
                try:
                    pct = int(15 + 53 * max(0.0, min(1.0, fraction)))
                    prog.progress(pct, text=message)
                    status_box.info(message)
                except Exception:
                    # Never let a UI/socket error abort a long training run.
                    pass

            results = service.train_all_models_panel(
                train_df=df,
                sales_col=sc,
                test_size=test_size,
                store_col=store_c,
                sku_col=sku_c,
                date_col=dc,
                preprocessor=preprocessor,
                progress_callback=_panel_progress,
            )
            print("DEBUG 1: Panel training returned", flush=True)
            print("DEBUG 2: Retrieving test predictions", flush=True)
            status_box.info('Evaluating models on validation set...')
            prog.progress(70, text='Evaluating models on validation set...')

            # Get test predictions from service
            test_preds = service.get_test_predictions()
            
            print("DEBUG 3: Test predictions retrieved", flush=True)
            print(
                "DEBUG 4: Prediction lengths:",
                len(test_preds.get('LSTM', [])),
                len(test_preds.get('Hybrid ARIMA+XGBoost', [])),
                len(test_preds.get('y_test', [])),
                flush=True,
            )

            lstm_preds = test_preds.get('LSTM', np.array([]))
            hybrid_preds = test_preds.get('Hybrid ARIMA+XGBoost', np.array([]))
            y_test_actual = test_preds.get('y_test', np.array([]))

            # Comparison
            status_box.info('Comparing models...')
            prog.progress(85)
            comparator = ModelComparison()
            comp_df = comparator.compare_models(
                y_test_actual,
                {'LSTM': lstm_preds, 'Hybrid ARIMA+XGBoost': hybrid_preds}
            )
            best_info = comparator.get_best_model(comp_df, 'RMSE')

            # Store for later test.csv prediction
            st.session_state._forecasting_service = service

        else:
            # Legacy single-series training
            status_box.info('Preparing LSTM sequences...')
            prog.progress(5)
            X_tr, X_te, y_tr, y_te, lstm_scaler = preprocessor.prepare_lstm_data(
                df, sc, seq_length, test_size=test_size
            )

            X_tr = np.asarray(X_tr, dtype=np.float32)
            X_te = np.asarray(X_te, dtype=np.float32)

            if X_tr.ndim == 2:
                X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1], 1))
            if X_te.ndim == 2:
                X_te = X_te.reshape((X_te.shape[0], X_te.shape[1], 1))

            lstm_test_start_idx = preprocessor.train_size

            status_box.info('Training LSTM model...')
            prog.progress(15, text='Training LSTM model...')
            lstm = LSTMForecaster(seq_length=seq_length, epochs=lstm_epochs, batch_size=lstm_batch)
            lstm.build_model((X_tr.shape[1], X_tr.shape[2]))

            def _lstm_progress(epoch, total_epochs):
                try:
                    pct = int(15 + 30 * (epoch / max(total_epochs, 1)))
                    prog.progress(pct, text=f'Training LSTM epoch {epoch}/{total_epochs}...')
                    status_box.info(f'Training LSTM epoch {epoch}/{total_epochs}...')
                except Exception:
                    pass

            lstm.train(X_tr, y_tr, X_te, y_te, verbose=0, progress_callback=_lstm_progress)

            status_box.info('Evaluating LSTM...')
            prog.progress(45, text='Evaluating LSTM...')
            lstm_preds_scaled = lstm.predict(X_te)
            lstm_preds = lstm_scaler.inverse_transform(lstm_preds_scaled).flatten()
            y_test_actual = lstm_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

            status_box.info('Fitting ARIMA...')
            prog.progress(55)
            values = df[sc].values.astype(float)
            train_series = values[:lstm_test_start_idx]
            test_series = values[lstm_test_start_idx:]
            hybrid = HybridArimaXGBoost(arima_order=arima_tuple)
            hybrid.fit(train_series)

            status_box.info('Training XGBoost on residuals...')
            prog.progress(70)
            hybrid_preds = hybrid.evaluate_on_test(train_series, test_series)

            min_len = min(len(lstm_preds), len(hybrid_preds), len(y_test_actual))
            if min_len < len(lstm_preds):
                lstm_preds = lstm_preds[:min_len]
            if min_len < len(hybrid_preds):
                hybrid_preds = hybrid_preds[:min_len]
            if min_len < len(y_test_actual):
                y_test_actual = y_test_actual[:min_len]

            status_box.info('Comparing models...')
            prog.progress(85)
            comparator = ModelComparison()
            comp_df = comparator.compare_models(
                y_test_actual,
                {'LSTM': lstm_preds, 'Hybrid ARIMA+XGBoost': hybrid_preds}
            )
            best_info = comparator.get_best_model(comp_df, 'RMSE')

            status_box.info('Fitting Hybrid model on full dataset for forecasting...')
            prog.progress(90)
            hybrid_full = HybridArimaXGBoost(arima_order=arima_tuple)
            hybrid_full.fit(values)

        prog.progress(100)
        status_box.success('Training complete.')

        # Persist
        st.session_state.models_trained = True
        st.session_state.predictions = {
            'lstm': lstm_preds,
            'hybrid': hybrid_preds,
            'y_test': y_test_actual,
            'comp_df': comp_df,
        }
        if is_panel:
            st.session_state.predictions['lstm_obj'] = service.lstm_model
            st.session_state.predictions['hybrid_obj'] = service.hybrid_model
        else:
            st.session_state.predictions['lstm_obj'] = lstm
            st.session_state.predictions['hybrid_obj'] = hybrid
            st.session_state.predictions['hybrid_obj_full'] = hybrid_full

        st.session_state.comparison_df = comp_df
        st.session_state.best_model = best_info['best_model']
        st.session_state.forecast_generated = False
        st.session_state.forecast_lstm = None
        st.session_state.forecast_hybrid = None
        st.session_state.forecast_dates = None

        st.rerun()

    except Exception as exc:
        prog.empty()
        status_box.empty()
        st.error('Training failed.')
        with st.expander('Technical details'):
            st.code(str(exc))
            import traceback
            st.code(traceback.format_exc())

# PAGE: MODEL COMPARISON
def _page_comparison():
    _ph('Model Comparison', 'Evaluate and compare LSTM vs Hybrid ARIMA+XGBoost performance')
    if _guard_data('Model Comparison'):
        return
    if _guard_trained('Model Comparison'):
        return

    preds      = st.session_state.predictions
    comp_df    = st.session_state.comparison_df
    best_model = st.session_state.best_model
    y_test     = preds['y_test']
    lstm_preds = preds['lstm']
    hyb_preds  = preds['hybrid']

    # ── Best Model Banner ─────────────────────────────────────────────────────
    best_row = comp_df[comp_df['Model'] == best_model].iloc[0] if len(comp_df) > 0 else None
    if best_row is not None:
        rmse_val = best_row['RMSE']
        mae_val  = best_row['MAE']
        mape_val = best_row['MAPE (%)']
        st.markdown(f"""
        <div class="best-banner">
            <p class="best-banner-label">Best Performing Model</p>
            <p class="best-banner-model">{best_model}</p>
            <p class="best-banner-sub">MAE {mae_val:.4f} &nbsp;·&nbsp; RMSE {rmse_val:.4f} &nbsp;·&nbsp; MAPE {mape_val:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics Table ─────────────────────────────────────────────────────────
    _sec('Performance Metrics')
    # Highlight the minimum values
    styled = comp_df.style.highlight_min(
        subset=['MAE', 'RMSE', 'MAPE (%)'],
        color='#d1fae5'
    ).format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'MAPE (%)': '{:.2f}%'})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Metric Bar Chart ──────────────────────────────────────────────────────
    _sec('Visual Comparison')
    r1, r2 = st.columns([1, 2])

    with r1:
        metrics_list = ['MAE', 'RMSE', 'MAPE (%)']
        fig_bar = go.Figure()
        colors = [_C_LSTM, _C_HYBRID]
        for i, row in comp_df.iterrows():
            fig_bar.add_trace(go.Bar(
                name=row['Model'],
                x=metrics_list,
                y=[row['MAE'], row['RMSE'], row['MAPE (%)']],
                marker_color=colors[i % 2],
                text=[f'{row["MAE"]:.3f}', f'{row["RMSE"]:.3f}', f'{row["MAPE (%)"]:.2f}%'],
                textposition='outside',
                textfont=dict(size=12),
            ))
        fig_bar.update_layout(**_cl(
            title='Model Error Metrics', height=300,
            barmode='group', showlegend=True,
            yaxis_title='Error Value',
        ))
        st.plotly_chart(fig_bar, use_container_width=True)

    with r2:
        # Actual vs Predicted
        x_ax = list(range(len(y_test)))
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=x_ax, y=y_test, mode='lines', name='Actual',
            line=dict(color=_C_ACTUAL, width=2),
        ))
        fig_avp.add_trace(go.Scatter(
            x=x_ax, y=lstm_preds, mode='lines', name='LSTM',
            line=dict(color=_C_LSTM, width=1.5, dash='dash'),
        ))
        fig_avp.add_trace(go.Scatter(
            x=x_ax, y=hyb_preds, mode='lines', name='Hybrid',
            line=dict(color=_C_HYBRID, width=1.5, dash='dot'),
        ))
        fig_avp.update_layout(**_cl(
            title='Actual vs Predicted (Test Set)', height=300,
            xaxis_title='Test Period', yaxis_title='Demand',
        ))
        st.plotly_chart(fig_avp, use_container_width=True)

    # ── LSTM Training Loss ────────────────────────────────────────────────────
    lstm_obj = preds.get('lstm_obj')
    if lstm_obj and hasattr(lstm_obj, 'get_training_history'):
        hist = lstm_obj.get_training_history()
        if hist and 'loss' in hist:
            with st.expander('LSTM Training Loss Curve'):
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=hist['loss'], mode='lines', name='Train Loss',
                    line=dict(color=_C_LSTM, width=1.5),
                ))
                if 'val_loss' in hist:
                    fig_loss.add_trace(go.Scatter(
                        y=hist['val_loss'], mode='lines', name='Val Loss',
                        line=dict(color=_C_HYBRID, width=1.5, dash='dash'),
                    ))
                fig_loss.update_layout(**_cl(
                    title='Training Loss (MSE)', xaxis_title='Epoch', yaxis_title='Loss', height=260,
                ))
                st.plotly_chart(fig_loss, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    _download_csv(comp_df, 'Download comparison CSV', 'model_comparison.csv', 'dl_comp')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST
# ══════════════════════════════════════════════════════════════════════════════
def _page_forecast():
    _ph('Forecast', 'Generate predictions for test.csv and future demand forecasts')
    if _guard_data('Forecast'):
        return
    if _guard_trained('Forecast'):
        return

    df = st.session_state.data
    dc = st.session_state.selected_date_col
    sc = st.session_state.selected_sales_col
    store_c = st.session_state.selected_store_col
    sku_c = st.session_state.selected_sku_col
    forecast_steps = st.session_state.forecast_steps
    seq_length = st.session_state.seq_length
    best_model = st.session_state.best_model
    test_df = st.session_state.test_df

    is_panel = store_c is not None and sku_c is not None and store_c in df.columns and sku_c in df.columns

    # Forecast Mode Selection
    _sec('Forecast Mode')
    mode = st.radio(
        'Select forecast type',
        ['Predict test.csv (for submission)', 'Future forecast (beyond train data)'],
        horizontal=True,
        key='forecast_mode'
    )

    if mode == 'Predict test.csv (for submission)':
        if test_df is None:
            st.warning('Please upload test.csv in the Data page first.')
            return

        _sec('Test.csv Prediction')
        st.caption('Generates predictions for test.csv using historical context from train.csv. Output matches sample_submission.csv format.')

        c1, c2 = st.columns([2, 6])
        with c1:
            pred_btn = st.button('Generate Test Predictions', type='primary', use_container_width=True)

        if pred_btn:
            prog = st.progress(0)
            status = st.empty()
            try:
                from src.services.forecasting_service import ForecastingService

                # Get the forecasting service
                service = st.session_state.get('_forecasting_service')
                if service is None:
                    service = ForecastingService(
                        seq_length=seq_length,
                        lstm_epochs=st.session_state.lstm_epochs,
                        lstm_batch_size=st.session_state.lstm_batch,
                        arima_order=_parse_arima_order(),
                    )
                    # Load trained models from session state
                    service.lstm_model = st.session_state.predictions.get('lstm_obj')
                    service.hybrid_model = st.session_state.predictions.get('hybrid_obj')
                    service.preprocessor = st.session_state.preprocessor
                    service._panel_scalers = st.session_state.preprocessor._panel_scalers
                    # Restore the validation predictions so best-model selection
                    # matches what the dashboard shows.
                    service._test_predictions = {
                        'LSTM': st.session_state.predictions.get('lstm'),
                        'Hybrid ARIMA+XGBoost': st.session_state.predictions.get('hybrid'),
                        'y_test': st.session_state.predictions.get('y_test'),
                    }
                    service._is_trained = True

                status.info('Generating predictions for test.csv...')
                prog.progress(20)

                # Generate test predictions.
                # Use the CLEANED training data the models were actually trained on
                # (st.session_state.data == preprocessor.clean_data(train_df)), not
                # the raw uploaded CSV: the raw frame can carry NaN/non-numeric sales
                # and a different row count, which mis-scales the context and shifts
                # the hybrid forecast horizon.
                submission_df = service.generate_test_predictions(
                    train_df=df,
                    test_df=test_df,
                    sales_col=sc,
                    store_col=store_c,
                    sku_col=sku_c,
                    date_col=dc,
                    record_id_col='record_ID',
                )

                prog.progress(90)

                st.session_state.test_predictions = submission_df
                st.session_state.test_record_ids = submission_df['record_ID'].values

                prog.progress(100)
                status.success('Test predictions generated successfully.')
                st.rerun()

            except Exception as exc:
                prog.empty()
                status.empty()
                st.error('Test prediction failed.')
                with st.expander('Technical details'):
                    st.code(str(exc))
                    import traceback
                    st.code(traceback.format_exc())
            return

        # Show test predictions if available
        if st.session_state.test_predictions is not None:
            st.markdown('<br>', unsafe_allow_html=True)
            _sec('Test Predictions (Submission Format)')
            
            sub_df = st.session_state.test_predictions
            st.dataframe(sub_df.head(20), use_container_width=True, height=300)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric('Total Predictions', len(sub_df))
            with c2:
                st.metric('Avg Predicted Units', f'{sub_df["units_sold"].mean():.2f}')

            # Download button
            _download_csv(sub_df, 'Download predictions.csv', 'predictions.csv', 'dl_test_pred')

            # Validate against sample submission if available
            if st.session_state.sample_submission is not None:
                sample = st.session_state.sample_submission
                if len(sub_df) == len(sample):
                    st.success(f'Prediction count matches sample submission: {len(sample)} rows')
                else:
                    st.warning(f'Row count mismatch: predictions={len(sub_df)}, sample={len(sample)}')

    else:
        # Future forecast mode (existing functionality)
        _sec('Future Forecast Configuration')
        cfg_cols = st.columns(3)
        cfg_cols[0].markdown(_kpi('Model', best_model or 'Both', 'Selected best model', ''), unsafe_allow_html=True)
        cfg_cols[1].markdown(_kpi('Horizon', f'{forecast_steps} days', 'Future periods', ''), unsafe_allow_html=True)
        cfg_cols[2].markdown(_kpi('Last Date', df[dc].max().strftime('%d %b %Y') if dc in df else 'N/A',
                                  'Forecast starts after this', ''), unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 6])
        with c1:
            gen_btn = st.button('Generate Future Forecast', type='primary', use_container_width=True)

        if gen_btn:
            prog = st.progress(0)
            status = st.empty()
            try:
                preprocessor = st.session_state.preprocessor
                lstm_obj = st.session_state.predictions.get('lstm_obj')
                hybrid_obj = st.session_state.predictions.get('hybrid_obj_full')

                if hybrid_obj is None:
                    hybrid_obj = st.session_state.predictions.get('hybrid_obj')

                status.info('Preparing LSTM input sequence...')
                prog.progress(10)

                if is_panel:
                    # Panel future forecast
                    from src.services.forecasting_service import ForecastingService
                    service = st.session_state.get('_forecasting_service')
                    if service is None:
                        service = ForecastingService(
                            seq_length=seq_length,
                            lstm_epochs=st.session_state.lstm_epochs,
                            lstm_batch_size=st.session_state.lstm_batch,
                            arima_order=_parse_arima_order(),
                        )
                        service.lstm_model = lstm_obj
                        service.hybrid_model = hybrid_obj
                        service.preprocessor = preprocessor
                        service._panel_scalers = preprocessor._panel_scalers
                        service._is_trained = True

                    panel_forecasts = service.generate_future_forecast_panel(
                        train_df=df,
                        sales_col=sc,
                        forecast_steps=forecast_steps,
                        store_col=store_c,
                        sku_col=sku_c,
                    )

                    # For display, aggregate across groups (mean)
                    all_lstm = []
                    all_hybrid = []
                    for preds in panel_forecasts.values():
                        all_lstm.append(preds['LSTM'])
                        all_hybrid.append(preds['Hybrid ARIMA+XGBoost'])

                    if all_lstm:
                        lstm_fc = np.mean(all_lstm, axis=0)
                        hybrid_fc = np.mean(all_hybrid, axis=0)
                    else:
                        lstm_fc = np.zeros(forecast_steps)
                        hybrid_fc = np.zeros(forecast_steps)
                else:
                    # Single series
                    scaled_all = preprocessor.scale_data(df[sc].values.reshape(-1, 1), fit=False)
                    last_seq = scaled_all[-seq_length:].flatten()
                    lstm_fc = lstm_obj.forecast_future(last_seq, steps=forecast_steps, scaler=preprocessor.scaler)
                    hybrid_fc = hybrid_obj.forecast_future(steps=forecast_steps)

                prog.progress(60)

                last_date = df[dc].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                             periods=forecast_steps, freq='D')

                st.session_state.forecast_generated = True
                st.session_state.forecast_lstm = lstm_fc
                st.session_state.forecast_hybrid = hybrid_fc
                st.session_state.forecast_dates = future_dates

                prog.progress(100)
                status.success('Future forecast generated successfully.')
                st.rerun()

            except Exception as exc:
                prog.empty()
                status.empty()
                st.error('Forecast generation failed.')
                with st.expander('Technical details'):
                    st.code(str(exc))
                    import traceback
                    st.code(traceback.format_exc())
            return

    # Show future forecast if generated
    if mode == 'Future forecast (beyond train data)' and st.session_state.forecast_generated:
        lstm_fc = st.session_state.forecast_lstm
        hybrid_fc = st.session_state.forecast_hybrid
        ens = (lstm_fc + hybrid_fc) / 2
        dates = st.session_state.forecast_dates

        st.markdown('<br>', unsafe_allow_html=True)
        _sec('Forecast Summary')
        kpi_cols = st.columns(5)
        kpi_cols[0].markdown(_kpi('Horizon', f'{forecast_steps} days', 'Forecast periods', ''), unsafe_allow_html=True)
        kpi_cols[1].markdown(_kpi('Avg (Ensemble)', _fmt(ens.mean()), 'units', 'kpi-blue'), unsafe_allow_html=True)
        kpi_cols[2].markdown(_kpi('Peak Forecast', _fmt(ens.max()), 'units', ''), unsafe_allow_html=True)
        kpi_cols[3].markdown(_kpi('Min Forecast', _fmt(ens.min()), 'units', ''), unsafe_allow_html=True)
        kpi_cols[4].markdown(_kpi('Best Model', best_model or 'N/A', 'By RMSE', 'kpi-green'), unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        _sec('Forecast Chart')
        tail = min(90, len(df))
        hist_df = df.tail(tail)
        y_test = st.session_state.predictions.get('y_test', [])

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=hist_df[dc], y=hist_df[sc],
            mode='lines', name='Historical',
            line=dict(color=_C_HIST, width=1.5),
        ))

        # Test actual (last portion of df)
        test_len = len(y_test)
        if test_len > 0:
            test_dates = df[dc].values[-test_len:]
            fig.add_trace(go.Scatter(
                x=test_dates, y=y_test,
                mode='lines', name='Validation Actual',
                line=dict(color=_C_ACTUAL, width=2),
            ))

        # Future forecasts
        fig.add_trace(go.Scatter(
            x=dates, y=lstm_fc,
            mode='lines', name='LSTM Forecast',
            line=dict(color=_C_LSTM, width=2, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=hybrid_fc,
            mode='lines', name='Hybrid Forecast',
            line=dict(color=_C_HYBRID, width=2, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=ens,
            mode='lines', name='Ensemble',
            line=dict(color=_C_ENS, width=2.5),
        ))

        # Divider line
        last_d = df[dc].max()
        fig.add_vline(x=last_d, line_dash='dot', line_color='#94a3b8', line_width=1)
        fig.add_annotation(x=last_d, y=1, yref='paper', text='Forecast start',
                           showarrow=False, font=dict(size=12, color='#64748b'),
                           xanchor='left', xshift=4)

        fig.update_layout(**_cl(title='Demand Forecast', height=370,
                                xaxis_title='Date', yaxis_title='Demand'))
        st.plotly_chart(fig, use_container_width=True)

        # Forecast Table
        _sec('Forecast Data')
        fc_df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'LSTM': np.round(lstm_fc, 2),
            'Hybrid': np.round(hybrid_fc, 2),
            'Ensemble': np.round(ens, 2),
        })
        st.dataframe(fc_df, use_container_width=True, height=220, hide_index=True)

        st.markdown('<br>', unsafe_allow_html=True)
        _download_csv(fc_df, 'Download forecast CSV', 'forecast_results.csv', 'dl_fc')

# PAGE: INVENTORY
def _page_inventory():
    _ph('Inventory Optimization', 'Convert demand forecasts into replenishment decisions')
    if _guard_data('Inventory'):
        return

    df = st.session_state.data
    sc = st.session_state.selected_sales_col
    store_c = st.session_state.selected_store_col
    sku_c = st.session_state.selected_sku_col

    is_panel = store_c is not None and sku_c is not None and store_c in df.columns and sku_c in df.columns

    # Use test predictions if available, otherwise use future forecast
    if st.session_state.test_predictions is not None:
        # Aggregate test predictions for inventory planning
        test_preds = st.session_state.test_predictions['units_sold'].values
        forecast_data = test_preds
        forecast_source = 'test.csv predictions'
    elif st.session_state.forecast_generated:
        lstm_fc = st.session_state.forecast_lstm
        hybrid_fc = st.session_state.forecast_hybrid
        ens_fc = (lstm_fc + hybrid_fc) / 2
        forecast_data = ens_fc
        forecast_source = 'future forecast'
    else:
        st.warning('Please generate test predictions (Forecast page) or future forecast first.')
        return

    lead_time = st.session_state.lead_time
    service_level = st.session_state.service_level_pct / 100.0

    # Inventory Inputs
    _sec('Configuration')
    inp_cols = st.columns([1, 1, 1, 3])
    with inp_cols[0]:
        current_stock = st.number_input('Current Stock (units)', min_value=0, value=1000, step=50,
                                        key='_curr_stock')
    with inp_cols[1]:
        st.markdown(_kpi('Lead Time', f'{lead_time} days', 'From sidebar', ''), unsafe_allow_html=True)
    with inp_cols[2]:
        st.markdown(_kpi('Service Level', f'{st.session_state.service_level_pct}%', 'From sidebar', ''), unsafe_allow_html=True)
    with inp_cols[3]:
        st.markdown(_kpi('Forecast Source', forecast_source, '', ''), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # For panel data, show per-store/SKU inventory recommendations
    if is_panel:
        st.info('Panel data detected. Showing aggregated inventory recommendations. For per-store/SKU details, use the Inventory Service API.')

    # Calculations
    try:
        inv_opt = InventoryOptimization(service_level=service_level)
        recs = inv_opt.generate_inventory_recommendations(
            demand_data=df[sc].values,
            lead_time=lead_time,
            forecast_data=forecast_data,
        )

        safety_stock = recs['safety_stock']
        rop = recs['reorder_point']
        avg_demand = recs['average_daily_demand']
        demand_lt = recs['demand_during_lead_time']
        z_score = recs['z_score']

    except Exception as exc:
        st.error('Failed to compute inventory parameters.')
        with st.expander('Technical details'):
            st.code(str(exc))
        return

    # Key Metrics
    _sec('Inventory Parameters')
    inv_c1, inv_c2, inv_c3, inv_c4 = st.columns(4)

    inv_c1.markdown(f"""
    <div class="inv-card inv-card-primary">
        <p class="inv-card-lbl">Safety Stock</p>
        <p class="inv-card-val">{safety_stock:,.0f}</p>
        <p class="inv-card-sub">units</p>
    </div>
    """, unsafe_allow_html=True)

    inv_c2.markdown(f"""
    <div class="inv-card inv-card-secondary">
        <p class="inv-card-lbl">Reorder Point</p>
        <p class="inv-card-val">{rop:,.0f}</p>
        <p class="inv-card-sub">units</p>
    </div>
    """, unsafe_allow_html=True)

    inv_c3.markdown(_kpi('Avg Daily Demand', f'{avg_demand:,.2f}', 'units/day', ''), unsafe_allow_html=True)
    inv_c4.markdown(_kpi('Demand During LT', f'{demand_lt:,.1f}', f'over {lead_time} days', ''), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Formula Expander
    with st.expander('How is this calculated?'):
        st.markdown(f"""
        **Safety Stock** — additional inventory held to buffer demand uncertainty during lead time.
        ```
        Safety Stock = Z × σ_demand × √(Lead Time)
                     = {z_score:.4f} × {recs['demand_std_dev']:.2f} × √{lead_time}
                     = {safety_stock:.1f} units
        ```
        *(Based on {recs['safety_stock_basis']})*

        **Reorder Point** — trigger a new order when stock reaches this level.
        ```
        Reorder Point = (Avg Daily Demand × Lead Time) + Safety Stock
                      = ({avg_demand:.2f} × {lead_time}) + {safety_stock:.1f}
                      = {rop:.1f} units
        ```

        **Service Level** — {st.session_state.service_level_pct}% → Z-score = {z_score:.4f}

        The Safety Stock uses **forecast variability** (std dev of the forecast)
        as the demand uncertainty estimate, which is more forward-looking than historical std dev.
        """)

    # Inventory Projection
    _sec('Projected Inventory Levels')
    try:
        inv_proj = inv_opt.forecast_inventory_levels(
            current_stock=current_stock,
            forecast_demand=forecast_data,
            reorder_point=rop,
            lead_time=lead_time,
            safety_stock=safety_stock,
        )

        fig_inv = go.Figure()

        # Inventory level
        fig_inv.add_trace(go.Scatter(
            x=inv_proj['Period'], y=inv_proj['Inventory_Level'],
            mode='lines', name='Inventory Level',
            line=dict(color=_C_PRIMARY, width=2),
            fill='tozeroy', fillcolor='rgba(30,64,175,0.06)',
        ))

        # Reorder point line
        fig_inv.add_hline(y=rop, line_dash='dash', line_color='#dc2626', line_width=1.5,
                          annotation_text=f'Reorder Point ({rop:,.0f})',
                          annotation_position='top right',
                          annotation_font_size=12)

        # Safety stock line
        fig_inv.add_hline(y=safety_stock, line_dash='dot', line_color='#d97706', line_width=1,
                          annotation_text=f'Safety Stock ({safety_stock:,.0f})',
                          annotation_position='bottom right',
                          annotation_font_size=12)

        # Order events
        order_days = inv_proj[inv_proj['Order_Placed']]['Period'].tolist()
        if order_days:
            order_levels = inv_proj.loc[inv_proj['Order_Placed'], 'Inventory_Level'].tolist()
            fig_inv.add_trace(go.Scatter(
                x=order_days, y=order_levels,
                mode='markers', name='Order Triggered',
                marker=dict(symbol='triangle-up', size=9, color='#dc2626'),
            ))

        fig_inv.update_layout(**_cl(
            title='Inventory Projection Over Forecast Horizon',
            xaxis_title='Day', yaxis_title='Units',
            height=320,
        ))
        st.plotly_chart(fig_inv, use_container_width=True)

    except Exception as exc:
        st.warning('Could not generate inventory projection chart.')
        with st.expander('Details'):
            st.code(str(exc))

    # Downloads
    st.markdown('<br>', unsafe_allow_html=True)
    _sec('Downloads')
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 4])

    recs_df = pd.DataFrame([{k: v for k, v in recs.items() if not isinstance(v, str) or k == 'safety_stock_basis'}])
    with dl_col1:
        _download_csv(recs_df, 'Inventory Parameters', 'inventory_parameters.csv', 'dl_inv_params')
    with dl_col2:
        try:
            _download_csv(inv_proj, 'Projection Data', 'inventory_projection.csv', 'dl_inv_proj')
        except NameError:
            pass

# PAGE: REPORTS
def _page_reports():
    _ph('Reports', 'Download analysis results and forecasting outputs')

    has_data = st.session_state.data is not None
    has_trained = st.session_state.models_trained
    has_forecast = st.session_state.forecast_generated
    has_test_preds = st.session_state.test_predictions is not None

    if not has_data:
        _empty('No Reports Available',
               'Load a dataset and run your analysis to generate downloadable reports.',
               'Go to Data', 'Data')
        return

    _sec('Available Downloads')

    reports = [
        {
            'title': 'Dataset Preview',
            'desc': 'Processed and cleaned dataset used for training.',
            'available': has_data,
            'df': st.session_state.data,
            'filename': 'dataset_processed.csv',
            'key': 'dl_rpt_data',
        },
        {
            'title': 'Model Comparison',
            'desc': 'MAE, RMSE, and MAPE for all trained models.',
            'available': has_trained and st.session_state.comparison_df is not None,
            'df': st.session_state.comparison_df,
            'filename': 'model_comparison.csv',
            'key': 'dl_rpt_comp',
        },
        {
            'title': 'Test Predictions (Submission)',
            'desc': 'Predictions for test.csv in sample_submission.csv format.',
            'available': has_test_preds,
            'df': st.session_state.test_predictions,
            'filename': 'predictions.csv',
            'key': 'dl_rpt_test_pred',
        },
        {
            'title': 'Forecast Results',
            'desc': 'LSTM, Hybrid, and Ensemble forecasts with dates.',
            'available': has_forecast,
            'df': None,
            'filename': 'forecast_results.csv',
            'key': 'dl_rpt_fc',
            'forecast': True,
        },
    ]

    for rpt in reports:
        avail = rpt['available']
        border_color = '#e2e8f0' if avail else '#f1f5f9'
        text_color = 'var(--text-primary)' if avail else 'var(--text-subtle)'

        col_info, col_dl = st.columns([5, 2])
        with col_info:
            badge_html = _badge('Available', 'ok') if avail else _badge('Not generated', 'gray')
            st.markdown(
                f'<div style="border:1px solid {border_color};border-radius:8px;'
                f'padding:14px 18px;background:white;margin-bottom:10px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
                f'<span style="font-size:15px;font-weight:600;color:{text_color};">{rpt["title"]}</span>'
                f'{badge_html}</div>'
                f'<span style="font-size:14px;color:var(--text-muted);">{rpt["desc"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_dl:
            st.markdown('<div style="margin-top:14px;">', unsafe_allow_html=True)
            if avail:
                if rpt.get('forecast'):
                    lstm_fc = st.session_state.forecast_lstm
                    hybrid_fc = st.session_state.forecast_hybrid
                    ens = (lstm_fc + hybrid_fc) / 2
                    dates = st.session_state.forecast_dates
                    fc_df = pd.DataFrame({
                        'Date': dates.strftime('%Y-%m-%d'),
                        'LSTM': np.round(lstm_fc, 2),
                        'Hybrid': np.round(hybrid_fc, 2),
                        'Ensemble': np.round(ens, 2),
                    })
                    _download_csv(fc_df, 'Download CSV', rpt['filename'], rpt['key'])
                elif rpt['df'] is not None:
                    _download_csv(rpt['df'], 'Download CSV', rpt['filename'], rpt['key'])
            else:
                st.button('Not available', disabled=True, key=f'dis_{rpt["key"]}', use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# PAGE: SETTINGS
def _page_settings():
    _ph('Settings', 'Configure advanced model parameters and application options')

    _sec('LSTM Settings')
    c1, c2 = st.columns(2)
    with c1:
        st.number_input('Max Training Epochs', min_value=10, max_value=300,
                        value=st.session_state.get('lstm_epochs', 50), step=5, key='lstm_epochs',
                        help='Maximum epochs before EarlyStopping. Usually stops earlier.')
    with c2:
        st.number_input('Batch Size', min_value=8, max_value=128,
                        value=st.session_state.get('lstm_batch', 32), step=8, key='lstm_batch',
                        help='Mini-batch size for LSTM training. Smaller = more updates per epoch.')

    st.markdown('<br>', unsafe_allow_html=True)
    _sec('ARIMA Settings')
    st.text_input('ARIMA Order (p,d,q)', value=st.session_state.get('arima_order_str', '1,1,1'),
                  key='arima_order_str', help='Comma-separated p,d,q values. Fallback orders applied automatically.',
                  placeholder='1,1,1')
    parsed = _parse_arima_order()
    st.caption(f'Parsed as: ARIMA{parsed}')

    st.markdown('<br>', unsafe_allow_html=True)
    _sec('Data Loading')
    st.checkbox(
        'Use DuckDB for large files (>80 MB)',
        value=st.session_state.get('use_duckdb', DUCKDB_AVAILABLE),
        key='use_duckdb',
        disabled=not DUCKDB_AVAILABLE,
        help='DuckDB is not installed.' if not DUCKDB_AVAILABLE else 'Faster SQL-based loading for large CSV files.',
    )
    if not DUCKDB_AVAILABLE:
        st.caption('DuckDB is not installed. Chunked pandas loading will be used for large files.')

    st.markdown('<br>', unsafe_allow_html=True)
    _sec('About')
    st.markdown("""
    <div class="card">
        <table class="info-tbl">
            <tr><td>Project</td><td>AI-Based Demand Forecasting System for Inventory Optimization</td></tr>
            <tr><td>Models</td><td>LSTM (TensorFlow/Keras) + Hybrid ARIMA + XGBoost</td></tr>
            <tr><td>Inventory</td><td>Safety Stock, Reorder Point, EOQ, Projection Simulation</td></tr>
            <tr><td>Degree</td><td>B.E. Artificial Intelligence & Machine Learning</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE DISPATCH
# ══════════════════════════════════════════════════════════════════════════════
if _page == 'Dashboard':
    _page_dashboard()
elif _page == 'Data':
    _page_data()
elif _page == 'EDA':
    _page_eda()
elif _page == 'Train Models':
    _page_train()
elif _page == 'Model Comparison':
    _page_comparison()
elif _page == 'Forecast':
    _page_forecast()
elif _page == 'Inventory':
    _page_inventory()
elif _page == 'Reports':
    _page_reports()
elif _page == 'Settings':
    _page_settings()
else:
    _page_dashboard()

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="app-footer">'
    'AI-Based Demand Forecasting System &nbsp;·&nbsp; '
    'LSTM + Hybrid ARIMA+XGBoost &nbsp;·&nbsp; '
    'Safety Stock &amp; Reorder Point'
    '</div>',
    unsafe_allow_html=True
)

