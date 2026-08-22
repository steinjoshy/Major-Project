"""
AI-Based Demand Forecasting System for Inventory Optimization
=============================================================
Professional analytics dashboard — sidebar navigation architecture.
Pages: Dashboard | Data | EDA | Train Models | Model Comparison |
       Forecast | Inventory | Reports | Settings
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    --text-secondary:  #334155;
    --text-muted:      #64748b;
    --text-subtle:     #94a3b8;
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
.stApp { background: var(--bg) !important; }
.block-container {
    padding: 1.6rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}
*, *::before, *::after {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
    box-sizing: border-box;
}

/* ─── Sidebar Shell ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
    border-right: 1px solid #1e293b !important;
    min-width: 226px !important;
    max-width: 246px !important;
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
    font-size: 13.5px; font-weight: 700; color: #f1f5f9;
    letter-spacing: -0.2px;
}
.sb-brand-sub {
    font-size: 10px; color: #475569;
    margin: 6px 0 0 35px;
    letter-spacing: 0.3px;
}

/* ─── Sidebar Section Label ─────────────────────────────────────────────── */
.sb-sec {
    font-size: 9.5px; font-weight: 700; letter-spacing: 1.1px;
    text-transform: uppercase; color: #334155;
    padding: 14px 4px 5px 4px;
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
    padding: 8px 10px !important;
    width: 100% !important;
    display: block !important;
    color: #64748b !important;
    font-size: 13px !important;
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
    font-size: 10.5px !important; color: #475569 !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #e2e8f0 !important; border-radius: 5px !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus {
    border-color: #1e40af !important; outline: none !important;
}
section[data-testid="stSidebar"] button {
    background: #1e293b !important; border-color: #334155 !important;
    color: #64748b !important;
}

/* ─── Sidebar Status ────────────────────────────────────────────────────── */
.sb-status {
    margin-top: 8px; padding: 10px 12px;
    background: #1a2744; border: 1px solid #1e293b;
    border-radius: 7px;
}
.sb-status-item {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: #475569;
    padding: 2px 0;
}
.sb-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.sb-dot-on  { background: #059669; box-shadow: 0 0 4px rgba(5,150,105,0.5); }
.sb-dot-off { background: #334155; }
.sb-dot-warn { background: #d97706; }
.sb-status-val { color: #cbd5e1; font-weight: 600; }

/* ─── Sidebar Z-Score Box ─────────────────────────────────────────────────── */
.sb-z-box {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace !important;
    font-size: 13px; font-weight: 700; color: #93c5fd;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 5px; padding: 4px 8px; display: inline-block;
}

/* ─── Page Header ───────────────────────────────────────────────────────── */
.ph { padding-bottom: 14px; border-bottom: 1px solid var(--border); margin-bottom: 22px; }
.ph-title { font-size: 20px; font-weight: 700; color: var(--text-primary);
             letter-spacing: -0.4px; margin: 0 0 3px 0; }
.ph-sub   { font-size: 13px; color: var(--text-muted); margin: 0; }

/* ─── Section Label ─────────────────────────────────────────────────────── */
.sec-lbl {
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: var(--text-muted);
    margin: 0 0 10px 0; padding-left: 8px;
    border-left: 3px solid var(--primary);
}

/* ─── Cards ─────────────────────────────────────────────────────────────── */
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow);
}

/* ─── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 15px 18px; box-shadow: var(--shadow);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.kpi:hover { box-shadow: var(--shadow-md); border-color: var(--border-strong); }
.kpi-lbl  { font-size: 9.5px; font-weight: 700; letter-spacing: 0.8px;
             text-transform: uppercase; color: var(--text-muted); margin: 0 0 7px 0; }
.kpi-val  { font-size: 22px; font-weight: 700; color: var(--text-primary);
             letter-spacing: -0.6px; margin: 0 0 4px 0; line-height: 1.1; }
.kpi-sub  { font-size: 11px; color: var(--text-subtle); margin: 0; }
.kpi-blue .kpi-val { color: var(--primary); }
.kpi-green .kpi-val { color: var(--success); }

/* ─── Badges ────────────────────────────────────────────────────────────── */
.bdg {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 9px; border-radius: 999px;
    font-size: 11px; font-weight: 600; white-space: nowrap;
}
.bdg-ok   { background: var(--success-bg); color: var(--success); }
.bdg-warn { background: var(--warning-bg); color: var(--warning); }
.bdg-err  { background: var(--danger-bg);  color: var(--danger); }
.bdg-gray { background: #f1f5f9; color: var(--text-muted); }
.bdg-blue { background: var(--primary-light); color: var(--primary); }

/* ─── Quality Rows ──────────────────────────────────────────────────────── */
.qr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 0; border-bottom: 1px solid var(--border);
    font-size: 13px;
}
.qr:last-child { border-bottom: none; }
.qr-name { color: var(--text-secondary); }
.qr-detail { font-size: 11px; color: var(--text-subtle); margin-left: 8px; }

/* ─── Model Cards ───────────────────────────────────────────────────────── */
.mc {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 22px 22px;
    box-shadow: var(--shadow); height: 100%;
}
.mc-title { font-size: 15px; font-weight: 700; color: var(--text-primary); margin: 0 0 3px 0; }
.mc-sub   { font-size: 12px; color: var(--text-muted); margin: 0 0 16px 0; }
.mc-arch  {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 10px 12px;
    font-size: 11px; color: var(--text-muted);
    font-family: "JetBrains Mono","Fira Code","Consolas",monospace !important;
    line-height: 1.75; margin: 0 0 14px 0;
}

/* ─── Empty State ───────────────────────────────────────────────────────── */
.es {
    text-align: center; padding: 52px 24px;
    background: var(--surface);
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--radius-lg);
}
.es-title { font-size: 16px; font-weight: 600; color: var(--text-secondary); margin: 0 0 8px 0; }
.es-sub   { font-size: 13px; color: var(--text-muted); margin: 0 0 22px 0;
             max-width: 360px; margin-left: auto; margin-right: auto; }

/* ─── Best Model Banner ─────────────────────────────────────────────────── */
.best-banner {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #6ee7b7; border-radius: var(--radius-lg);
    padding: 18px 22px; margin-bottom: 20px;
}
.best-banner-label { font-size: 9.5px; font-weight: 700; letter-spacing: 1px;
                      text-transform: uppercase; color: #059669; margin: 0 0 5px 0; }
.best-banner-model { font-size: 19px; font-weight: 700; color: #065f46; margin: 0 0 4px 0; }
.best-banner-sub   { font-size: 12px; color: #047857; margin: 0; }

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
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 0.7px !important; text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-size: 20px !important; font-weight: 700 !important;
    color: var(--text-primary) !important; letter-spacing: -0.4px !important;
}

/* ─── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: var(--primary) !important; border: none !important;
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
    font-size: 13.5px !important; padding: 8px 22px !important;
    color: #fff !important; box-shadow: 0 1px 3px rgba(30,64,175,0.3) !important;
    transition: background 0.15s, transform 0.1s !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--primary-hover) !important; transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
    border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: var(--text-secondary) !important; background: var(--surface) !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--primary) !important; color: var(--primary) !important;
}

/* ─── Download Button ───────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
    font-size: 13px !important; font-weight: 500 !important;
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
    font-size: 13px !important;
    transition: border-color 0.15s !important;
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
    font-size: 13px !important; font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* ─── Tabs (for sub-tabs in EDA) ────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 3px; gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 12.5px; font-weight: 500; color: var(--text-muted);
    border-radius: var(--radius-sm); padding: 5px 16px;
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
[data-testid="stAlert"] { border-radius: var(--radius-sm) !important; font-size: 13px !important; }

/* ─── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

/* ─── Footer ────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center; font-size: 11px; color: var(--text-subtle);
    padding: 20px 0 6px 0; border-top: 1px solid var(--border); margin-top: 32px;
}

/* ─── Dataset Info Table ────────────────────────────────────────────────── */
.info-tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
.info-tbl td { padding: 8px 4px; border-bottom: 1px solid var(--border); }
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
.inv-card-lbl { font-size: 10px; font-weight: 700; letter-spacing: 0.9px;
                 text-transform: uppercase; margin: 0 0 8px 0; }
.inv-card-primary .inv-card-lbl  { color: #059669; }
.inv-card-secondary .inv-card-lbl { color: #4f46e5; }
.inv-card-val { font-size: 30px; font-weight: 700; letter-spacing: -1px; margin: 0 0 4px 0; }
.inv-card-primary .inv-card-val  { color: #065f46; }
.inv-card-secondary .inv-card-val { color: #3730a3; }
.inv-card-sub { font-size: 11.5px; margin: 0; }
.inv-card-primary .inv-card-sub  { color: #047857; }
.inv-card-secondary .inv-card-sub { color: #4338ca; }

/* Dashboard reference layout */
.dash-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px;
             padding: 0 0 16px; margin-bottom:18px; border-bottom:1px solid var(--border); }
.dash-title { font-size:25px; line-height:1.2; font-weight:750; letter-spacing:-.65px;
              color:var(--text-primary); margin:0 0 5px; }
.dash-sub { font-size:12.5px; color:var(--text-muted); margin:0; }
.dash-panel { background:var(--surface); border:1px solid var(--border); border-radius:10px;
              padding:15px; box-shadow:var(--shadow); height:100%; }
.dash-panel-title { font-size:12px; font-weight:700; color:var(--text-primary); margin:0 0 13px; }
.dash-kpi { position:relative; min-height:100px; padding:15px 16px; }
.dash-kpi .kpi-lbl { text-transform:none; letter-spacing:0; font-size:10.5px; margin-bottom:8px; }
.dash-kpi .kpi-val { font-size:20px; }
.dash-kpi-icon { position:absolute; right:15px; top:16px; display:grid; place-items:center; width:34px; height:34px;
                 border-radius:8px; font-size:18px; background:#eff6ff; color:#2563eb; }
.dash-kpi-icon.green { background:#ecfdf5; color:#059669; }
.dash-kpi-icon.violet { background:#f5f3ff; color:#7c3aed; }
.dash-kpi-icon.orange { background:#fff7ed; color:#ea580c; }
.quality-item { display:flex; gap:10px; align-items:center; min-height:44px; padding:8px 10px;
                border:1px solid #e1f0e8; border-radius:8px; background:linear-gradient(90deg,#f0fdf4,#f8fafc); }
.quality-icon { color:#059669; font-weight:700; font-size:17px; width:20px; text-align:center; }
.quality-name { font-size:10.5px; font-weight:650; line-height:1.2; color:var(--text-primary); }
.quality-detail { font-size:9.5px; color:var(--text-muted); margin-top:2px; }
.dash-table { width:100%; border-collapse:collapse; font-size:10.5px; }
.dash-table th { background:#f8fafc; color:var(--text-secondary); font-size:9px; text-align:left; padding:7px 6px;
                 border-bottom:1px solid var(--border); }
.dash-table td { padding:8px 6px; color:var(--text-secondary); border-bottom:1px solid #f1f5f9; }
.dash-table tr:last-child td { border-bottom:0; }
.dash-table .val { text-align:right; font-weight:650; color:var(--text-primary); }
.dash-bar { display:inline-block; width:46px; height:4px; margin-right:4px; vertical-align:middle; border-radius:8px;
            background:linear-gradient(90deg,#2563eb var(--value),#e2e8f0 var(--value)); }
.action-card { min-height:74px; padding:12px; border:1px solid #dbeafe; border-radius:8px;
               background:linear-gradient(135deg,#f8fbff,#f1f7ff); }
.action-card.green { border-color:#d9f1e3; background:linear-gradient(135deg,#f7fdf9,#f0fdf4); }
.action-card.violet { border-color:#e9ddff; background:linear-gradient(135deg,#fcfaff,#f7f3ff); }
.action-icon { float:left; width:31px; font-size:21px; color:#2563eb; line-height:37px; }
.action-card.green .action-icon { color:#059669; }
.action-card.violet .action-icon { color:#7c3aed; }
.action-title { font-size:10.5px; font-weight:700; color:var(--text-primary); margin:1px 0 4px 38px; }
.action-desc { font-size:9.5px; color:var(--text-muted); margin-left:38px; }
@media (max-width: 900px) { .dash-title { font-size:21px; } .dash-head { display:block; } }
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
    'data':               None,
    'preprocessor':       None,
    'models_trained':     False,
    'predictions':        {},
    'selected_date_col':  None,
    'selected_sales_col': None,
    'last_processed_sig': None,
    'forecast_generated': False,
    'forecast_lstm':      None,
    'forecast_hybrid':    None,
    'forecast_dates':     None,
    'best_model':         None,
    'comparison_df':      None,
    # UI param defaults
    'forecast_steps':     30,
    'seq_length':         30,
    'lead_time':          7,
    'service_level_pct':  95,
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
        font=dict(family='Inter, Segoe UI, system-ui, sans-serif', size=12, color='#0f172a'),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=14, r=14, t=38, b=14),
        hoverlabel=dict(bgcolor='white', bordercolor='#e2e8f0', font_size=12, font_family='Inter, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=11), bgcolor='rgba(255,255,255,0)'),
        xaxis=dict(showgrid=False, showline=True, linecolor='#e2e8f0', linewidth=1,
                   tickfont=dict(size=11), zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', gridwidth=1,
                   showline=False, zeroline=False, tickfont=dict(size=11)),
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
        '<p style="font-size:10.5px;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.5px;color:#475569;margin:8px 0 4px 0;">Safety Factor (Z)</p>'
        f'<div class="sb-z-box">{_z_val:.3f}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr>', unsafe_allow_html=True)

    # ── Forecast Settings ─────────────────────────────────────────────────────
    st.markdown('<span class="sb-sec">Forecast Settings</span>', unsafe_allow_html=True)
    st.number_input('Horizon (days)',    7,  90, key='forecast_steps', step=1)
    st.number_input('Look-back (days)', 10, 60, key='seq_length',     step=1)

    st.markdown('<hr>', unsafe_allow_html=True)

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
        <p style="font-size:22px;font-weight:700;color:#0f172a;letter-spacing:-0.5px;margin:0 0 4px 0;">
            AI-Based Demand Forecasting System</p>
        <p style="font-size:12px;color:#64748b;margin:0;">
            LSTM &amp; Hybrid ARIMA+XGBoost
            <span style="color:#cbd5e1;margin:0 8px;">&#183;</span>
            Inventory Optimization
            <span style="color:#cbd5e1;margin:0 8px;">&#183;</span>
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
            <p style="font-size:16px;font-weight:600;color:#334155;margin:0 0 8px 0;">No Dataset Loaded</p>
            <p style="font-size:13px;color:#64748b;margin:0 0 20px 0;max-width:360px;margin-left:auto;margin-right:auto;">
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
    def _ikpi(label, value, sub, icon, ibg, ifg, vclass=''):
        vst = f'font-size:19px;font-weight:700;color:{ifg if vclass=="green" else "#0f172a"};letter-spacing:-0.5px;margin:0 0 3px 0;line-height:1.1;'
        return f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;
             box-shadow:0 1px 3px rgba(0,0,0,0.07);display:flex;align-items:flex-start;
             justify-content:space-between;gap:10px;">
          <div style="flex:1;min-width:0;">
            <p style="font-size:9.5px;font-weight:700;letter-spacing:0.7px;text-transform:uppercase;
               color:#64748b;margin:0 0 6px 0;">{label}</p>
            <p style="{vst}">{value}</p>
            <p style="font-size:10.5px;color:#94a3b8;margin:0;">{sub}</p>
          </div>
          <div style="width:36px;height:36px;flex-shrink:0;border-radius:8px;background:{ibg};
               color:{ifg};display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;">
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
        k5.markdown(_ikpi('Best Model', best_model, mstr or 'Lowest RMSE', '&#9733;', '#d1fae5', '#059669', 'green'), unsafe_allow_html=True)
    else:
        k5.markdown(_ikpi('Best Model', 'Not Trained', 'Train models first', '&#9733;', '#f1f5f9', '#94a3b8'), unsafe_allow_html=True)
    if has_fc:
        k6.markdown(_ikpi('Forecast Horizon', f'{fc_horizon} days', f'From {d_max}', '&#8599;', '#ede9fe', '#7c3aed'), unsafe_allow_html=True)
    else:
        k6.markdown(_ikpi('Forecast Horizon', 'Pending', 'Generate forecast', '&#8599;', '#f1f5f9', '#94a3b8'), unsafe_allow_html=True)

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
        miss_tag     = (f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;'
                        f'font-weight:600;background:#fef3c7;color:#b45309;">{missing_pct:.2f}%</span>'
                        if missing_pct > 0 else
                        '<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;'
                        'font-weight:600;background:#d1fae5;color:#059669;">0%</span>')

        deg        = round(quality_score * 3.6)
        ring_html  = (
            f'<div style="width:56px;height:56px;border-radius:50%;flex-shrink:0;'
            f'background:conic-gradient(#059669 {deg}deg,#e2e8f0 0deg);'
            f'display:flex;align-items:center;justify-content:center;position:relative;">'
            f'<div style="width:40px;height:40px;background:white;border-radius:50%;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:12px;font-weight:700;color:#059669;position:absolute;">{quality_score}%</div></div>'
        )

        def _dov_row(label, val_html):
            return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;'
                    f'align-items:center;padding:8px 0;border-bottom:1px solid #f1f5f9;">'
                    f'<span style="font-size:12px;color:#64748b;">{label}</span>'
                    f'<span>{val_html}</span></div>')

        tag_html = f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:{tag_c};">{dataset_type}</span>'
        gran_html= '<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:#d1fae5;color:#059669;">Daily</span>'

        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;height:100%;">
            <p style="font-size:13.5px;font-weight:700;color:#0f172a;margin:0 0 14px 0;">Dataset Overview</p>
            {_dov_row('Dataset Type', tag_html)}
            {_dov_row('Granularity', gran_html)}
            {_dov_row('Total Missing Values', miss_tag)}
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:center;padding:8px 0;">
                <span style="font-size:12px;color:#64748b;">Data Quality Score</span>
                <div style="display:flex;align-items:center;gap:8px;">{ring_html}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with dq_col:
        def _dq(icon, ibg, ifg, name, status, ok=True):
            sc2 = '#059669' if ok else '#b45309'
            return (f'<div style="display:flex;align-items:flex-start;gap:9px;padding:10px;'
                    f'background:#f8fafc;border:1px solid #f1f5f9;border-radius:7px;">'
                    f'<div style="width:28px;height:28px;border-radius:6px;flex-shrink:0;background:{ibg};'
                    f'color:{ifg};display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;">'
                    f'{icon}</div>'
                    f'<div><p style="font-size:11.5px;font-weight:600;color:#334155;margin:0 0 1px 0;">{name}</p>'
                    f'<p style="font-size:10.5px;color:{sc2};margin:0;">{status}</p></div></div>')

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
            <p style="font-size:13.5px;font-weight:700;color:#0f172a;margin:0 0 14px 0;">Data Quality Checks</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">{grid}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Row 3: Demand Trend | Seasonality | Distribution ──────────────────────
    ch1, ch2, ch3 = st.columns([2, 1.5, 1.5])

    with ch1:
        st.markdown('<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">'
                    '<p style="font-size:13px;font-weight:700;color:#0f172a;margin:0 0 6px 0;">Demand Trend</p>',
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
                    '<p style="font-size:13px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Seasonality (Avg Demand)</p>',
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
                    '<p style="font-size:13px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Demand Distribution</p>',
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
                f'align-items:center;padding:7px 0;border-bottom:1px solid #f8fafc;font-size:12px;">'
                f'<span style="color:#334155;font-weight:500;font-family:monospace;font-size:11px;">'
                f'{row[dc].strftime("%Y-%m-%d") if hasattr(row[dc], "strftime") else str(row[dc])}</span>'
                f'<span style="color:#0f172a;font-weight:600;">{_fmt(row[sc])}</span>'
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<span style="height:4px;border-radius:2px;display:inline-block;width:{int(row[sc]/max_val*60)}px;background:#1e40af;"></span>'
                f'<span style="font-size:10.5px;color:#64748b;">{row["pct"]:.1f}%</span></div></div>'
                for _, row in top5.iterrows()
            )
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">
                <p style="font-size:13px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Top Demand Periods</p>
                <div style="display:grid;grid-template-columns:1.6fr 1fr 0.8fr;gap:8px;padding:5px 0 7px 0;border-bottom:1px solid #e2e8f0;">
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">Date</span>
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">Total Demand</span>
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">% Share</span>
                </div>
                {rows_html}
            </div>""", unsafe_allow_html=True)
        except Exception:
            st.markdown('<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;"><p style="font-size:13px;font-weight:700;color:#0f172a;">Top Demand Periods</p><p style="font-size:12px;color:#94a3b8;padding:12px 0;">No data available</p></div>', unsafe_allow_html=True)

    with b2:
        if st.session_state.models_trained and st.session_state.comparison_df is not None:
            comp = st.session_state.comparison_df
            rows_html = ''.join(
                f'<div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:8px;'
                f'align-items:center;padding:7px 0;border-bottom:1px solid #f8fafc;font-size:12px;">'
                f'<span style="color:#334155;font-weight:700;font-size:11px;">{row["Model"]}</span>'
                f'<span style="color:#64748b;">{row["MAE"]:.3f}</span>'
                f'<span style="color:#64748b;">{row["RMSE"]:.3f}</span>'
                f'<span style="color:#64748b;">{row["MAPE (%)"]:.2f}%</span></div>'
                for _, row in comp.iterrows()
            )
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">
                <p style="font-size:13px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Model Performance</p>
                <div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:8px;padding:5px 0 7px 0;border-bottom:1px solid #e2e8f0;">
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">Model</span>
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">MAE</span>
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">RMSE</span>
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">MAPE</span>
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
                                   f'padding:7px 0;border-bottom:1px solid #f8fafc;font-size:12px;">'
                                   f'<span style="color:#64748b;">{nm}</span>'
                                   f'<span style="color:#0f172a;font-weight:600;">{_fmt(v)}</span></div>')
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">
                <p style="font-size:13px;font-weight:700;color:#0f172a;margin:0 0 10px 0;">Demand Statistics</p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:5px 0 7px 0;border-bottom:1px solid #e2e8f0;">
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">Metric</span>
                    <span style="font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;">Value</span>
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
            f'background:{bg};color:{fg};display:flex;align-items:center;justify-content:center;font-size:15px;">{icon}</div>'
            f'<p style="font-size:11px;font-weight:700;color:#334155;margin:0 0 2px 0;">{name}</p>'
            f'<p style="font-size:9.5px;color:#94a3b8;margin:0;">{sub}</p></div>'
            for name, _, icon, bg, fg, sub in actions
        )
        st.markdown(f'<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;">'
                    f'<p style="font-size:13px;font-weight:700;color:#0f172a;margin:0 0 12px 0;">Quick Actions</p>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">{cards}</div></div>',
                    unsafe_allow_html=True)
        # Actual clickable buttons
        btn_cols = st.columns(3)
        for i, (name, target, *_) in enumerate(actions):
            with btn_cols[i % 3]:
                if st.button(name, key=f'qa_{i}', use_container_width=True):
                    _go_to(target)



def _page_data():
    _ph('Data', 'Upload, validate, and preprocess your demand dataset')

    # ── Source Selection ──────────────────────────────────────────────────────
    _sec('Data Source')
    src_mode = st.radio('Source mode', ['Upload Files', 'Local Folder Path'],
                        horizontal=True, label_visibility='collapsed', key='_src_mode')

    selected_sources = []

    if src_mode == 'Upload Files':
        uploaded = st.file_uploader(
            'Upload CSV file(s)',
            type=['csv'],
            accept_multiple_files=True,
            help='Upload one or more CSV files. Multi-file datasets (e.g., Favorita, M5, Walmart) are supported.',
            label_visibility='collapsed',
        )
        selected_sources = list(uploaded) if uploaded else []

        if selected_sources:
            _sec('Uploaded Files')
            for f in selected_sources:
                size_kb = _src_size(f) / 1024
                size_str = f'{size_kb:.1f} KB' if size_kb < 1024 else f'{size_kb/1024:.2f} MB'
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
                    f'border-bottom:1px solid #e2e8f0;font-size:13px;">'
                    f'<span style="color:#1e40af;font-weight:600;">{f.name}</span>'
                    f'<span style="color:#94a3b8;">{size_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        folder_path = st.text_input('Folder path (absolute)', placeholder='/path/to/data/',
                                    key='_folder_path', label_visibility='collapsed')
        if folder_path:
            csv_files = _list_csvs(folder_path)
            if csv_files:
                selected_sources = csv_files
                _sec(f'Found {len(csv_files)} CSV File(s)')
                for fp in csv_files:
                    size_kb = _src_size(fp) / 1024
                    size_str = f'{size_kb:.1f} KB' if size_kb < 1024 else f'{size_kb/1024:.2f} MB'
                    st.markdown(
                        f'<div style="font-size:13px;padding:6px 0;border-bottom:1px solid #e2e8f0;">'
                        f'<span style="color:#334155;font-weight:500;">{Path(fp).name}</span>'
                        f' <span style="color:#94a3b8;">— {size_str}</span></div>',
                        unsafe_allow_html=True
                    )
            elif folder_path:
                st.warning('No CSV files found in the specified folder.')

    if not selected_sources:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:18px 20px;">
            <p style="font-size:12.5px;font-weight:600;color:#334155;margin:0 0 10px 0;">Supported Dataset Formats</p>
            <ul style="font-size:12px;color:#64748b;margin:0;padding-left:20px;line-height:2;">
                <li><b>Single CSV</b> — any file with a date column and a sales/demand column</li>
                <li><b>Kaggle Favorita</b> — train.csv + items.csv + stores.csv (auto-detected)</li>
                <li><b>M5 Competition</b> — wide-format sales_train_evaluation.csv (auto-melted)</li>
                <li><b>Walmart</b> — multi-file store/feature/sales datasets</li>
                <li><b>Large files &gt;100 MB</b> — DuckDB SQL engine used automatically</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Column Detection ──────────────────────────────────────────────────────
    primary_src  = selected_sources[0]
    all_cols     = _read_columns(primary_src)
    is_wide      = _detect_wide_format(all_cols)
    auto_dc, auto_sc = _auto_detect_cols(all_cols)

    st.markdown('<br>', unsafe_allow_html=True)
    _sec('Column Mapping')

    if is_wide:
        st.info('M5 wide-format detected. Day columns will be melted and aggregated automatically.')
        date_col  = 'date'
        sales_col = 'sales'
        non_day   = [c for c in all_cols if not (c.startswith('d_') and c[2:].isdigit())]
        c1, c2 = st.columns(2)
        with c1:
            date_col = st.selectbox('Date column', ['(auto-generated)'] + non_day, key='_dc')
            if date_col == '(auto-generated)':
                date_col = 'date'
        with c2:
            sales_col = st.selectbox('Demand column', ['(aggregated d_* cols)'] + non_day, key='_sc')
            if sales_col == '(aggregated d_* cols)':
                sales_col = 'sales'
    else:
        c1, c2 = st.columns(2)
        with c1:
            idx = all_cols.index(auto_dc) if auto_dc in all_cols else 0
            date_col = st.selectbox('Date column', all_cols, index=idx, key='_dc')
            if auto_dc:
                st.caption(f'Auto-detected: {auto_dc}')
            else:
                st.caption('Could not auto-detect. Please select manually.')
        with c2:
            idx = all_cols.index(auto_sc) if auto_sc in all_cols else 0
            sales_col = st.selectbox('Demand column', all_cols, index=idx, key='_sc')
            if auto_sc:
                st.caption(f'Auto-detected: {auto_sc}')
            else:
                st.caption('Could not auto-detect. Please select manually.')

    # ── Load & Preprocess ─────────────────────────────────────────────────────
    sig = _build_sig(src_mode, selected_sources, date_col, sales_col)
    already_done = (st.session_state.last_processed_sig == sig
                    and st.session_state.data is not None)

    if already_done:
        st.success('Dataset already loaded and preprocessed.')

    col_btn, col_rst = st.columns([2, 6])
    with col_btn:
        load_btn = st.button('Load & Preprocess', type='primary', use_container_width=True)
    with col_rst:
        if st.button('Reset', use_container_width=False):
            for k in ['data', 'preprocessor', 'models_trained', 'predictions',
                      'selected_date_col', 'selected_sales_col', 'last_processed_sig',
                      'forecast_generated', 'forecast_lstm', 'forecast_hybrid',
                      'forecast_dates', 'best_model', 'comparison_df']:
                st.session_state[k] = _DEFAULTS.get(k)
            st.rerun()

    if not load_btn and already_done:
        pass
    elif load_btn or not already_done:
        if not load_btn and already_done:
            pass
        elif load_btn:
            # ── Actual data loading ───────────────────────────────────────
            prog = st.progress(0, text='Preparing dataset...')
            try:
                preprocessor = DataPreprocessor()
                use_duck = st.session_state.use_duckdb and DUCKDB_AVAILABLE

                if is_wide:
                    prog.progress(10, text='Melting wide format (M5)...')
                    dfs = []
                    for s_src in selected_sources:
                        if _is_file_obj(s_src):
                            s_src.seek(0)
                            df_w = pd.read_csv(s_src, low_memory=False)
                        else:
                            df_w = pd.read_csv(s_src, low_memory=False)
                        day_cols = [c for c in df_w.columns if c.startswith('d_') and c[2:].isdigit()]
                        if day_cols:
                            melted = df_w[day_cols].sum(axis=0)
                            melted.index = range(len(melted))
                            dfs.append(pd.DataFrame({'date': melted.index, 'sales': melted.values}))
                    merged_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                    date_col = 'date'
                    sales_col = 'sales'

                elif len(selected_sources) > 1:
                    prog.progress(10, text='Merging multiple files...')
                    dfs = []
                    for s_src in selected_sources:
                        try:
                            if _is_file_obj(s_src):
                                s_src.seek(0)
                                df_tmp = pd.read_csv(s_src, low_memory=False)
                            else:
                                df_tmp = pd.read_csv(s_src, low_memory=False)
                            if date_col in df_tmp.columns and sales_col in df_tmp.columns:
                                dfs.append(df_tmp[[date_col, sales_col]])
                        except Exception:
                            continue
                    merged_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=[date_col, sales_col])

                else:
                    prog.progress(10, text='Reading dataset...')
                    s_src = selected_sources[0]
                    s_size = _src_size(s_src)
                    DUCK_THRESHOLD = 80 * 1024 * 1024

                    if use_duck and s_size > DUCK_THRESHOLD:
                        prog.progress(20, text='Loading via DuckDB...')
                        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                            if _is_file_obj(s_src):
                                s_src.seek(0)
                                tmp.write(s_src.read())
                                s_src.seek(0)
                            else:
                                with open(str(s_src), 'rb') as ff:
                                    tmp.write(ff.read())
                            tmp_path = tmp.name
                        try:
                            con = duckdb.connect()
                            merged_df = con.execute(
                                f"SELECT {date_col}, SUM(CAST({sales_col} AS DOUBLE)) AS {sales_col} "
                                f"FROM read_csv_auto('{tmp_path}') "
                                f"WHERE {sales_col} IS NOT NULL AND CAST({sales_col} AS DOUBLE) >= 0 "
                                f"GROUP BY {date_col} ORDER BY {date_col}"
                            ).df()
                            con.close()
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                    elif s_size > DUCK_THRESHOLD:
                        prog.progress(20, text='Loading large file (chunked)...')
                        if _is_file_obj(s_src):
                            s_src.seek(0)
                            file_bytes = s_src.read()
                            s_src.seek(0)
                        else:
                            with open(str(s_src), 'rb') as ff:
                                file_bytes = ff.read()
                        merged_df = _aggregate_chunked(file_bytes, _src_name(s_src), date_col, sales_col)
                    else:
                        if _is_file_obj(s_src):
                            s_src.seek(0)
                            merged_df = pd.read_csv(s_src, low_memory=False)
                            s_src.seek(0)
                        else:
                            merged_df = pd.read_csv(str(s_src), low_memory=False)

                if merged_df.empty or date_col not in merged_df.columns or sales_col not in merged_df.columns:
                    st.error('Could not load data. Check that the selected columns exist in the file.')
                    prog.empty()
                    return

                prog.progress(50, text='Validating data...')
                is_valid, msgs = preprocessor.validate_data(merged_df, date_col, sales_col)
                if not is_valid:
                    prog.empty()
                    st.error('Validation failed. See details below.')
                    for m in msgs:
                        st.error(m)
                    return
                for w in msgs:
                    st.warning(w)

                prog.progress(70, text='Cleaning and feature engineering...')
                df_clean = preprocessor.clean_data(merged_df, date_col, sales_col)

                prog.progress(90, text='Finalizing...')
                st.session_state.data               = df_clean
                st.session_state.preprocessor       = preprocessor
                st.session_state.selected_date_col  = date_col
                st.session_state.selected_sales_col = sales_col
                st.session_state.last_processed_sig = sig
                # Reset downstream
                st.session_state.models_trained    = False
                st.session_state.predictions       = {}
                st.session_state.forecast_generated = False
                st.session_state.forecast_lstm     = None
                st.session_state.forecast_hybrid   = None
                st.session_state.forecast_dates    = None
                st.session_state.best_model        = None
                st.session_state.comparison_df     = None
                prog.progress(100, text='Done.')
                st.rerun()

            except Exception as exc:
                prog.empty()
                st.error('Failed to load dataset.')
                with st.expander('Technical details'):
                    st.code(str(exc))
                return

    # ── Show Data Info if loaded ──────────────────────────────────────────────
    if st.session_state.data is None:
        return

    df  = st.session_state.data
    dc  = st.session_state.selected_date_col
    sc  = st.session_state.selected_sales_col

    st.markdown('<br>', unsafe_allow_html=True)

    info_col, qual_col = st.columns([1, 1])

    with info_col:
        _sec('Dataset Information')
        n_rows = len(df)
        d_min  = df[dc].min().strftime('%Y-%m-%d') if dc in df else 'N/A'
        d_max  = df[dc].max().strftime('%Y-%m-%d') if dc in df else 'N/A'
        avg_d  = df[sc].mean() if sc in df else 0
        total_d = df[sc].sum() if sc in df else 0
        n_files = len(selected_sources)

        st.markdown(f"""
        <div class="card">
        <table class="info-tbl">
            <tr><td>Records</td><td>{n_rows:,}</td></tr>
            <tr><td>Date Range</td><td>{d_min} — {d_max}</td></tr>
            <tr><td>Average Demand</td><td>{avg_d:,.2f} units</td></tr>
            <tr><td>Total Demand</td><td>{_fmt(total_d)}</td></tr>
            <tr><td>Date Column</td><td>{dc}</td></tr>
            <tr><td>Demand Column</td><td>{sc}</td></tr>
            <tr><td>Files Loaded</td><td>{n_files}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with qual_col:
        _sec('Data Quality')
        # Run quality checks on actual data
        missing_pct = df[sc].isna().mean() * 100 if sc in df else 0
        dup_count   = df.duplicated(subset=[dc]).sum() if dc in df else 0
        neg_count   = (df[sc] < 0).sum() if sc in df else 0
        std_val     = df[sc].std() if sc in df else 1
        n_rows_chk  = len(df)

        qrows = ''
        qrows += _qrow('Date column parseable',   'pass' if dc in df else 'err')
        qrows += _qrow('Demand column numeric',    'pass' if sc in df else 'err')
        qrows += _qrow('Missing values',
                        'pass' if missing_pct == 0 else ('warn' if missing_pct < 5 else 'err'),
                        f'{missing_pct:.1f}%')
        qrows += _qrow('Duplicate dates',
                        'pass' if dup_count == 0 else 'warn',
                        f'{int(dup_count)} found' if dup_count > 0 else '')
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

    # ── Preview ───────────────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    _sec('Data Preview')
    st.dataframe(df.head(50), use_container_width=True, height=220)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
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
    _ph('Train Models', 'Train LSTM and Hybrid ARIMA+XGBoost forecasting models')
    if _guard_data('Train Models'):
        return

    df = st.session_state.data
    dc = st.session_state.selected_date_col
    sc = st.session_state.selected_sales_col
    seq_length = st.session_state.seq_length
    arima_tuple = _parse_arima_order()
    lstm_epochs  = st.session_state.lstm_epochs
    lstm_batch   = st.session_state.lstm_batch
    lstm_status_t = 'Trained' if st.session_state.models_trained else 'Not Trained'

    # ── Training Config Summary ───────────────────────────────────────────────
    _sec('Training Configuration')
    cfg_cols = st.columns(4)
    cfg_cols[0].markdown(_kpi('Records', f'{len(df):,}', 'training data', ''), unsafe_allow_html=True)
    cfg_cols[1].markdown(_kpi('Look-back', f'{seq_length} days', 'LSTM sequence length', ''), unsafe_allow_html=True)
    cfg_cols[2].markdown(_kpi('Train Split', '80%', '20% held for test', ''), unsafe_allow_html=True)
    cfg_cols[3].markdown(_kpi('ARIMA Order', f'{arima_tuple}', 'p,d,q — with fallback', ''), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Minimum row check ─────────────────────────────────────────────────────
    MIN_ROWS = seq_length * 2 + 20
    if len(df) < MIN_ROWS:
        st.error(f'Insufficient data for training. Need at least {MIN_ROWS} rows '
                 f'for look-back={seq_length}. Current: {len(df)} rows.\n\n'
                 'Recommended action: Reduce the Look-back window in the sidebar, '
                 'or upload more historical data.')
        return

    # ── Model Cards ───────────────────────────────────────────────────────────
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
LSTM({st.session_state.lstm_batch * 2}) → Dropout(0.2)
LSTM({st.session_state.lstm_batch}) → Dropout(0.2)
Dense(16, ReLU) → Dense(1)
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
Step 1 — ARIMA{arima_tuple} on training data
Step 2 — Residuals = Actual − ARIMA fit
Step 3 — XGBoost on lagged residuals (10 lags)
Step 4 — Forecast = ARIMA + XGBoost correction
            </div>
            <table class="info-tbl">
                <tr><td>ARIMA Order</td><td>{arima_tuple} (with fallback)</td></tr>
                <tr><td>XGBoost</td><td>n_est=100, lr=0.05, depth=5</td></tr>
                <tr><td>Residual Lags</td><td>10</td></tr>
                <tr><td>Train Split</td><td>80% chronological</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Train Button ──────────────────────────────────────────────────────────
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

    # ── Training ─────────────────────────────────────────────────────────────
    prog = st.progress(0)
    status_box = st.empty()

    try:
        preprocessor = st.session_state.preprocessor

        # LSTM data prep (NO LEAKAGE - split first, then scale)
        status_box.info('Preparing LSTM sequences...')
        prog.progress(5)
        X_tr, X_te, y_tr, y_te, lstm_scaler = preprocessor.prepare_lstm_data(
            df, sc, seq_length, test_size=0.2
        )
        lstm_test_start_idx = preprocessor.train_size + seq_length  # First test date index

        status_box.info('Training LSTM model...')
        prog.progress(15)
        lstm = LSTMForecaster(seq_length=seq_length, epochs=lstm_epochs, batch_size=lstm_batch)
        lstm.build_model((X_tr.shape[1], X_tr.shape[2]))
        lstm.train(X_tr, y_tr, X_te, y_te, verbose=0)

        status_box.info('Evaluating LSTM...')
        prog.progress(45)
        lstm_preds_scaled = lstm.predict(X_te)
        lstm_preds = lstm_scaler.inverse_transform(lstm_preds_scaled).flatten()
        y_test_actual = lstm_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

        # Hybrid data prep - ALIGN test window with LSTM (Bug B3 fix)
        status_box.info('Fitting ARIMA...')
        prog.progress(55)
        values = df[sc].values.astype(float)
        # Hybrid train ends where LSTM test begins (aligned evaluation window)
        train_series = values[:lstm_test_start_idx]
        test_series = values[lstm_test_start_idx:]
        hybrid = HybridArimaXGBoost(arima_order=arima_tuple)
        hybrid.fit(train_series)

        status_box.info('Training XGBoost on residuals...')
        prog.progress(70)
        hybrid_preds = hybrid.evaluate_on_test(train_series, test_series)

        # Lengths should now match naturally (no min_len truncation needed)
        # But guard against small off-by-one
        min_len = min(len(lstm_preds), len(hybrid_preds), len(y_test_actual))
        if min_len < len(lstm_preds):
            lstm_preds = lstm_preds[:min_len]
        if min_len < len(hybrid_preds):
            hybrid_preds = hybrid_preds[:min_len]
        if min_len < len(y_test_actual):
            y_test_actual = y_test_actual[:min_len]

        # Comparison
        status_box.info('Comparing models...')
        prog.progress(85)
        comparator = ModelComparison()
        comp_df = comparator.compare_models(
            y_test_actual,
            {'LSTM': lstm_preds, 'Hybrid ARIMA+XGBoost': hybrid_preds}
        )
        best_info = comparator.get_best_model(comp_df, 'RMSE')

        prog.progress(100)
        status_box.success('Training complete.')

        # Persist
        st.session_state.models_trained   = True
        st.session_state.predictions      = {
            'lstm':        lstm_preds,
            'hybrid':      hybrid_preds,
            'y_test':      y_test_actual,
            'lstm_obj':    lstm,
            'hybrid_obj':  hybrid,
            'comp_df':     comp_df,
        }
        st.session_state.comparison_df = comp_df
        st.session_state.best_model    = best_info['best_model']
        # Reset forecast
        st.session_state.forecast_generated = False
        st.session_state.forecast_lstm      = None
        st.session_state.forecast_hybrid    = None
        st.session_state.forecast_dates     = None

        st.rerun()

    except Exception as exc:
        prog.empty()
        status_box.empty()
        st.error('Training failed.')
        with st.expander('Technical details'):
            st.code(str(exc))
            import traceback
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
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
                textfont=dict(size=10),
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
    _ph('Forecast', 'Generate future demand forecasts from trained models')
    if _guard_data('Forecast'):
        return
    if _guard_trained('Forecast'):
        return

    df            = st.session_state.data
    dc            = st.session_state.selected_date_col
    sc            = st.session_state.selected_sales_col
    forecast_steps = st.session_state.forecast_steps
    seq_length     = st.session_state.seq_length
    best_model     = st.session_state.best_model

    # ── Config Summary ────────────────────────────────────────────────────────
    _sec('Forecast Configuration')
    cfg_cols = st.columns(3)
    cfg_cols[0].markdown(_kpi('Model', best_model or 'Both', 'Selected best model', ''), unsafe_allow_html=True)
    cfg_cols[1].markdown(_kpi('Horizon', f'{forecast_steps} days', 'Future periods', ''), unsafe_allow_html=True)
    cfg_cols[2].markdown(_kpi('Last Date', df[dc].max().strftime('%d %b %Y') if dc in df else 'N/A',
                              'Forecast starts after this', ''), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 6])
    with c1:
        gen_btn = st.button('Generate Forecast', type='primary', use_container_width=True)

    if gen_btn:
        prog = st.progress(0)
        status = st.empty()
        try:
            preprocessor = st.session_state.preprocessor
            lstm_obj  = st.session_state.predictions.get('lstm_obj')
            hybrid_obj = st.session_state.predictions.get('hybrid_obj')

            status.info('Preparing LSTM input sequence...')
            prog.progress(10)
            # Use the scaler that was fit on TRAINING DATA ONLY (no leakage)
            # The preprocessor.scaler is now the training-fitted scaler
            scaled_all = preprocessor.scale_data(df[sc].values.reshape(-1, 1), fit=False)
            last_seq   = scaled_all[-seq_length:].flatten()

            status.info('Generating LSTM forecast...')
            prog.progress(30)
            lstm_fc = lstm_obj.forecast_future(last_seq, steps=forecast_steps, scaler=preprocessor.scaler)

            status.info('Generating Hybrid ARIMA+XGBoost forecast...')
            prog.progress(60)
            # Use the ALREADY-TRAINED hybrid model from evaluation (no silent refit)
            # This ensures the forecast model matches the evaluated model (Bug B4 fix)
            hybrid_fc = hybrid_obj.forecast_future(steps=forecast_steps)

            prog.progress(90)
            # Future dates
            last_date = df[dc].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                         periods=forecast_steps, freq='D')

            st.session_state.forecast_generated = True
            st.session_state.forecast_lstm      = lstm_fc
            st.session_state.forecast_hybrid    = hybrid_fc
            st.session_state.forecast_dates     = future_dates

            prog.progress(100)
            status.success('Forecast generated successfully.')
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

    if not st.session_state.forecast_generated:
        return

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    _sec('Forecast Summary')
    lstm_fc   = st.session_state.forecast_lstm
    hybrid_fc = st.session_state.forecast_hybrid
    ens       = (lstm_fc + hybrid_fc) / 2
    dates     = st.session_state.forecast_dates

    kpi_cols = st.columns(5)
    kpi_cols[0].markdown(_kpi('Horizon',         f'{forecast_steps} days', 'Forecast periods', ''), unsafe_allow_html=True)
    kpi_cols[1].markdown(_kpi('Avg (Ensemble)',   _fmt(ens.mean()), 'units', 'kpi-blue'), unsafe_allow_html=True)
    kpi_cols[2].markdown(_kpi('Peak Forecast',    _fmt(ens.max()), 'units', ''), unsafe_allow_html=True)
    kpi_cols[3].markdown(_kpi('Min Forecast',     _fmt(ens.min()), 'units', ''), unsafe_allow_html=True)
    kpi_cols[4].markdown(_kpi('Best Model',       best_model or 'N/A', 'By RMSE', 'kpi-green'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Forecast Chart ────────────────────────────────────────────────────────
    _sec('Forecast Chart')
    tail = min(90, len(df))
    hist_df = df.tail(tail)
    y_test  = st.session_state.predictions.get('y_test', [])
    preds_lstm_test = st.session_state.predictions.get('lstm', [])

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
            mode='lines', name='Test Actual',
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
                       showarrow=False, font=dict(size=10, color='#94a3b8'),
                       xanchor='left', xshift=4)

    fig.update_layout(**_cl(title='Demand Forecast', height=370,
                            xaxis_title='Date', yaxis_title='Demand'))
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast Table ────────────────────────────────────────────────────────
    _sec('Forecast Data')
    fc_df = pd.DataFrame({
        'Date':     dates.strftime('%Y-%m-%d'),
        'LSTM':     np.round(lstm_fc, 2),
        'Hybrid':   np.round(hybrid_fc, 2),
        'Ensemble': np.round(ens, 2),
    })
    st.dataframe(fc_df, use_container_width=True, height=220, hide_index=True)

    st.markdown('<br>', unsafe_allow_html=True)
    _download_csv(fc_df, 'Download forecast CSV', 'forecast_results.csv', 'dl_fc')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
def _page_inventory():
    _ph('Inventory Optimization', 'Convert demand forecasts into replenishment decisions')
    if _guard_data('Inventory'):
        return
    if _guard_forecast('Inventory'):
        return

    df        = st.session_state.data
    sc        = st.session_state.selected_sales_col
    lstm_fc   = st.session_state.forecast_lstm
    hybrid_fc = st.session_state.forecast_hybrid
    ens_fc    = (lstm_fc + hybrid_fc) / 2

    lead_time       = st.session_state.lead_time
    service_level   = st.session_state.service_level_pct / 100.0

    # ── Inventory Inputs ──────────────────────────────────────────────────────
    _sec('Configuration')
    inp_cols = st.columns([1, 1, 1, 3])
    with inp_cols[0]:
        current_stock = st.number_input('Current Stock (units)', min_value=0, value=1000, step=50,
                                        key='_curr_stock')
    with inp_cols[1]:
        st.markdown(_kpi('Lead Time', f'{lead_time} days', 'From sidebar', ''), unsafe_allow_html=True)
    with inp_cols[2]:
        st.markdown(_kpi('Service Level', f'{st.session_state.service_level_pct}%', 'From sidebar', ''), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Calculations ──────────────────────────────────────────────────────────
    try:
        inv_opt = InventoryOptimization(service_level=service_level)
        recs    = inv_opt.generate_inventory_recommendations(
            demand_data   = df[sc].values,
            lead_time     = lead_time,
            forecast_data = ens_fc,
        )

        safety_stock = recs['safety_stock']
        rop          = recs['reorder_point']
        avg_demand   = recs['average_daily_demand']
        demand_lt    = recs['demand_during_lead_time']
        z_score      = recs['z_score']

    except Exception as exc:
        st.error('Failed to compute inventory parameters.')
        with st.expander('Technical details'):
            st.code(str(exc))
        return

    # ── Key Metrics ───────────────────────────────────────────────────────────
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

    # ── Formula Expander ──────────────────────────────────────────────────────
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

        The Safety Stock uses **forecast variability** (std dev of the ensemble forecast)
        as the demand uncertainty estimate, which is more forward-looking than historical std dev.
        """)

    # ── Inventory Projection ──────────────────────────────────────────────────
    _sec('Projected Inventory Levels')
    try:
        inv_proj = inv_opt.forecast_inventory_levels(
            current_stock  = current_stock,
            forecast_demand = ens_fc,
            reorder_point  = rop,
            lead_time      = lead_time,
            safety_stock   = safety_stock,
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
                          annotation_font_size=10)

        # Safety stock line
        fig_inv.add_hline(y=safety_stock, line_dash='dot', line_color='#d97706', line_width=1,
                          annotation_text=f'Safety Stock ({safety_stock:,.0f})',
                          annotation_position='bottom right',
                          annotation_font_size=10)

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

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    _sec('Downloads')
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 4])

    recs_df = pd.DataFrame([{k: v for k, v in recs.items() if not isinstance(v, str) or k == 'safety_stock_basis'}])
    with dl_col1:
        _download_csv(recs_df, 'Inventory Parameters', 'inventory_parameters.csv', 'dl_inv_params')
    with dl_col2:
        # inv_proj is defined in the try block above; use a flag
        try:
            _download_csv(inv_proj, 'Projection Data', 'inventory_projection.csv', 'dl_inv_proj')
        except NameError:
            pass  # inv_proj not available (projection failed)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ══════════════════════════════════════════════════════════════════════════════
def _page_reports():
    _ph('Reports', 'Download analysis results and forecasting outputs')

    has_data     = st.session_state.data is not None
    has_trained  = st.session_state.models_trained
    has_forecast = st.session_state.forecast_generated

    if not has_data:
        _empty('No Reports Available',
               'Load a dataset and run your analysis to generate downloadable reports.',
               'Go to Data', 'Data')
        return

    _sec('Available Downloads')

    reports = [
        {
            'title':     'Dataset Preview',
            'desc':      'Processed and cleaned dataset used for training.',
            'available': has_data,
            'df':        st.session_state.data,
            'filename':  'dataset_processed.csv',
            'key':       'dl_rpt_data',
        },
        {
            'title':     'Model Comparison',
            'desc':      'MAE, RMSE, and MAPE for all trained models.',
            'available': has_trained and st.session_state.comparison_df is not None,
            'df':        st.session_state.comparison_df,
            'filename':  'model_comparison.csv',
            'key':       'dl_rpt_comp',
        },
        {
            'title':     'Forecast Results',
            'desc':      'LSTM, Hybrid, and Ensemble forecasts with dates.',
            'available': has_forecast,
            'df':        None,
            'filename':  'forecast_results.csv',
            'key':       'dl_rpt_fc',
            'forecast':  True,
        },
    ]

    for rpt in reports:
        avail = rpt['available']
        border_color = '#e2e8f0' if avail else '#f1f5f9'
        text_color   = 'var(--text-primary)' if avail else 'var(--text-subtle)'

        col_info, col_dl = st.columns([5, 2])
        with col_info:
            badge_html = _badge('Available', 'ok') if avail else _badge('Not generated', 'gray')
            st.markdown(
                f'<div style="border:1px solid {border_color};border-radius:8px;'
                f'padding:14px 18px;background:white;margin-bottom:10px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
                f'<span style="font-size:13.5px;font-weight:600;color:{text_color};">{rpt["title"]}</span>'
                f'{badge_html}</div>'
                f'<span style="font-size:12px;color:var(--text-muted);">{rpt["desc"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_dl:
            st.markdown('<div style="margin-top:14px;">', unsafe_allow_html=True)
            if avail:
                if rpt.get('forecast'):
                    lstm_fc   = st.session_state.forecast_lstm
                    hybrid_fc = st.session_state.forecast_hybrid
                    ens       = (lstm_fc + hybrid_fc) / 2
                    dates     = st.session_state.forecast_dates
                    fc_df = pd.DataFrame({
                        'Date':     dates.strftime('%Y-%m-%d'),
                        'LSTM':     np.round(lstm_fc, 2),
                        'Hybrid':   np.round(hybrid_fc, 2),
                        'Ensemble': np.round(ens, 2),
                    })
                    _download_csv(fc_df, 'Download CSV', rpt['filename'], rpt['key'])
                elif rpt['df'] is not None:
                    _download_csv(rpt['df'], 'Download CSV', rpt['filename'], rpt['key'])
            else:
                st.button('Not available', disabled=True, key=f'dis_{rpt["key"]}', use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
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

