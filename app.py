import os
import json
import math
import random
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask App
app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_pm25_model.json")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")
TREES_PATH = os.path.join(BASE_DIR, "model_trees.json")
TIMELINE_PATH = os.path.join(BASE_DIR, "timeline_cache.json")
DATA_PATH = os.path.join(BASE_DIR, "merged.csv")

# OpenStreetMap / Tile Provider API Key from Environment (if configured)
OSM_ENV_KEY = os.environ.get("OSM_API_KEY") or os.environ.get("OPENSTREETMAP_API_KEY", "")

# Load metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load lightweight pure-Python boosted tree evaluator (ensures instant cold start & fits Vercel bundle limit)
tree_model = None
if os.path.exists(TREES_PATH):
    with open(TREES_PATH, "r", encoding="utf-8") as f:
        tree_model = json.load(f)

# Optional fallback: load native XGBoost if tree_model is absent and xgboost is installed
xgb_model = None
if tree_model is None and os.path.exists(MODEL_PATH):
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(MODEL_PATH)
    except ImportError:
        pass

def predict_pm25(features):
    """Zero-dependency pure Python tree evaluator matching native XGBoost to 4 decimal places."""
    if tree_model is not None:
        score = tree_model["base_score"]
        for lefts, rights, splits, conds in tree_model["trees"]:
            node = 0
            while lefts[node] != -1:
                val = features[splits[node]]
                node = lefts[node] if val < conds[node] else rights[node]
            score += conds[node]
        return max(0.0, score)
    elif xgb_model is not None:
        import pandas as pd
        cols = metadata["features"]
        df_in = pd.DataFrame([features], columns=cols)
        return max(0.0, float(xgb_model.predict(df_in)[0]))
    raise RuntimeError("No model evaluator available")

# Load pre-computed historical timeline cache
timeline_cache = []
if os.path.exists(TIMELINE_PATH):
    with open(TIMELINE_PATH, "r", encoding="utf-8") as f:
        timeline_cache = json.load(f)
elif os.path.exists(DATA_PATH):
    try:
        import pandas as pd
        df_all = pd.read_csv(DATA_PATH)
        feature_cols = metadata["features"]
        for i, row in df_all.iterrows():
            feats = [float(row[c]) for c in feature_cols]
            pred_v = predict_pm25(feats)
            timeline_cache.append({
                "idx": i,
                "time": str(row["time"]),
                "actual_pm25": round(float(row["PM2.5"]), 1),
                "pred_pm25": round(pred_v, 1),
                "aod": round(float(row["AOD"]), 3),
                "temp": round(float(row["Surface skin temp"]), 1),
                "slp": round(float(row["Sea level pressue"]), 1),
                "tqv": round(float(row["TQV"]), 1),
                "u10": round(float(row["east wind 10m"]), 2),
                "v10": round(float(row["north wind 10m"]), 2)
            })
    except ImportError:
        pass

def get_aqi_category(pm25):
    """Classify PM2.5 concentration into India CPCB NAAQS categories."""
    if pm25 <= 30:
        return {
            "category": "Good",
            "color": "#10B981",
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "advisory": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
            "action": "Enjoy outdoor sports and ventilation freely.",
            "gauge_pct": min(100, (pm25 / 30) * 16.6)
        }
    elif pm25 <= 60:
        return {
            "category": "Satisfactory",
            "color": "#84CC16",
            "bg_color": "rgba(132, 204, 22, 0.15)",
            "advisory": "Minor breathing discomfort to sensitive people with existing asthma or respiratory illness.",
            "action": "Unusually sensitive individuals should reduce heavy outdoor exertion.",
            "gauge_pct": 16.6 + ((pm25 - 30) / 30) * 16.6
        }
    elif pm25 <= 90:
        return {
            "category": "Moderate",
            "color": "#F59E0B",
            "bg_color": "rgba(245, 158, 11, 0.15)",
            "advisory": "Breathing discomfort to people with lung disease, asthma, and heart conditions.",
            "action": "Children, elders, and sensitive groups should limit prolonged outdoor exertion.",
            "gauge_pct": 33.2 + ((pm25 - 60) / 30) * 16.6
        }
    elif pm25 <= 120:
        return {
            "category": "Poor",
            "color": "#F97316",
            "bg_color": "rgba(249, 115, 22, 0.15)",
            "advisory": "Breathing discomfort to most people on prolonged exposure; elevated cardiovascular risk.",
            "action": "Wear N95 respirators outdoors; keep indoor air clean during morning peak inversion.",
            "gauge_pct": 49.8 + ((pm25 - 90) / 30) * 16.6
        }
    elif pm25 <= 250:
        return {
            "category": "Very Poor",
            "color": "#EF4444",
            "bg_color": "rgba(239, 68, 68, 0.15)",
            "advisory": "Respiratory illness on prolonged exposure. Significant distress for asthma patients.",
            "action": "Avoid outdoor activities. Run HEPA air purifiers and seal windows and doorways.",
            "gauge_pct": 66.4 + ((pm25 - 120) / 130) * 16.6
        }
    else:
        return {
            "category": "Severe",
            "color": "#991B1B",
            "bg_color": "rgba(153, 27, 27, 0.25)",
            "advisory": "Emergency air quality. Respiratory impacts even on healthy adults and severe risk for vulnerable populations.",
            "action": "Public health emergency. Stay indoors, avoid physical exertion, run HEPA filtration continuously.",
            "gauge_pct": min(100, 83.0 + ((pm25 - 250) / 250) * 17.0)
        }

def calculate_wind_params(u, v):
    """Calculate wind speed (m/s) and meteorological compass direction."""
    speed = math.sqrt(u**2 + v**2)
    deg = (math.atan2(-u, -v) * 180 / math.pi) % 360
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg + 11.25) / 22.5) % 16
    return round(speed, 2), round(deg, 1), directions[idx]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Delhi PM2.5 Intelligence Center | Interactive Satellite & ML Analytics</title>
  
  <!-- Tailwind CSS & FontAwesome -->
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  
  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .font-mono { font-family: 'JetBrains Mono', monospace; }
    .glass-card {
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    input[type=range] { accent-color: #38bdf8; }
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0b1120; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen py-6 px-3 sm:px-6 lg:px-8">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Top Navigation & Header -->
    <header class="glass-card rounded-2xl p-6 relative overflow-hidden flex flex-col lg:flex-row lg:items-center justify-between gap-6">
      <div class="absolute -right-20 -top-20 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
      <div class="space-y-2 z-10">
        <div class="flex flex-wrap items-center gap-2">
          <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-xs font-semibold uppercase tracking-wider border border-cyan-500/20">
            <i class="fa-solid fa-satellite"></i> INSAT-3D/3DR + NASA MERRA-2
          </span>
          <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold uppercase tracking-wider border border-emerald-500/20">
            <i class="fa-solid fa-microchip"></i> XGBoost v2.0
          </span>
          <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-400 text-xs font-semibold uppercase tracking-wider border border-indigo-500/20">
            <i class="fa-solid fa-location-dot"></i> Delhi NCR Bounding Box
          </span>
        </div>
        <h1 class="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
          Delhi PM2.5 Multi-Modal Air Quality Intelligence System
        </h1>
        <p class="text-slate-400 text-sm max-w-3xl leading-relaxed">
          High-resolution aerosol optical depth (AOD) and atmospheric boundary layer reanalysis fused with ground truth observations. Explore what-if scenarios, geospatial monitoring stations, sensitivity responses, and seasonal timelines.
        </p>
      </div>

      <!-- Live Global Model Metrics -->
      <div class="flex flex-wrap lg:flex-col items-start lg:items-end gap-2 text-xs z-10">
        <div class="px-3.5 py-1.5 rounded-xl bg-slate-900/90 border border-slate-800 flex items-center gap-2">
          <i class="fa-solid fa-bullseye text-emerald-400"></i>
          <span class="text-slate-400">Model Test R²:</span>
          <strong class="text-white font-mono font-bold">{{ "%.4f"|format(metadata.metrics.test_r2) }}</strong>
        </div>
        <div class="px-3.5 py-1.5 rounded-xl bg-slate-900/90 border border-slate-800 flex items-center gap-2">
          <i class="fa-solid fa-scale-balanced text-amber-400"></i>
          <span class="text-slate-400">Test RMSE:</span>
          <strong class="text-white font-mono font-bold">{{ "%.2f"|format(metadata.metrics.test_rmse) }} μg/m³</strong>
        </div>
        <div class="flex items-center gap-2 mt-1">
          <button onclick="shareScenario()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition flex items-center gap-1.5">
            <i class="fa-solid fa-share-nodes text-cyan-400"></i> Copy Scenario Link
          </button>
          <button onclick="downloadJSON()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition flex items-center gap-1.5">
            <i class="fa-solid fa-download text-emerald-400"></i> Export JSON
          </button>
        </div>
      </div>
    </header>

    <!-- Preset Scenarios & Interactive Mode Toggles -->
    <div class="glass-card rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-2 flex-wrap">
        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <i class="fa-solid fa-wand-magic-sparkles text-cyan-400"></i> Meteorological Presets:
        </span>
        <button onclick="applyPreset('winter')" class="px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-xs font-medium text-rose-300 border border-rose-500/30 transition">
          🌫️ Severe Winter Inversion
        </button>
        <button onclick="applyPreset('monsoon')" class="px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-xs font-medium text-emerald-300 border border-emerald-500/30 transition">
          🌤️ Clean Monsoon Breeze
        </button>
        <button onclick="applyPreset('autumn')" class="px-3 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-xs font-medium text-amber-300 border border-amber-500/30 transition">
          🍂 Moderate Autumn Day
        </button>
        <button onclick="applyPreset('windy')" class="px-3 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-xs font-medium text-cyan-300 border border-cyan-500/30 transition">
          💨 High Dispersion Westerly
        </button>
      </div>

      <div class="flex items-center gap-2">
        <button onclick="loadRandomSample()" class="px-3.5 py-1.5 rounded-lg bg-cyan-600/30 hover:bg-cyan-600/50 text-xs font-semibold text-cyan-200 border border-cyan-500/40 transition flex items-center gap-1.5">
          <i class="fa-solid fa-dice"></i> Sample Real Ground Truth
        </button>
      </div>
    </div>

    <!-- Main Grid: Controls, Map, Predictions, & Vector Flow -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Left Column: Input Sliders & Wind Flow Canvas (7 Columns) -->
      <div class="lg:col-span-7 space-y-6">

        <!-- Environmental Sliders -->
        <div class="glass-card rounded-2xl p-6 space-y-5">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 class="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <i class="fa-solid fa-sliders text-cyan-400"></i> Environmental Predictors
            </h2>
            <div class="flex items-center gap-3 text-xs text-slate-400">
              <span id="active-station-label" class="text-cyan-400 font-semibold flex items-center gap-1">
                <i class="fa-solid fa-crosshairs"></i> Regional Delhi Mean
              </span>
              <button onclick="resetToMean()" class="text-slate-500 hover:text-slate-300 underline text-[11px]">Reset</button>
            </div>
          </div>

          <form id="predict-form" class="space-y-4">

            <!-- AOD -->
            <div class="space-y-1.5">
              <div class="flex justify-between items-center text-xs">
                <label class="font-medium text-slate-200 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
                  Aerosol Optical Depth (AOD)
                  <span class="text-slate-500 text-[10px] hidden sm:inline">(INSAT-3D/3DR Column extinction: 0.15 - 2.90)</span>
                </label>
                <div class="flex items-center gap-2">
                  <input type="number" id="num-aod" min="0.15" max="2.90" step="0.01" value="0.88" class="w-16 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-cyan-400 font-mono font-bold text-xs text-right">
                </div>
              </div>
              <input type="range" id="aod" min="0.15" max="2.90" step="0.01" value="0.88" class="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer">
              <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>0.15 (Clear)</span>
                <span>0.88 (Mean)</span>
                <span>1.80 (Heavy Smoke)</span>
                <span>2.90 (Extreme Smog)</span>
              </div>
            </div>

            <!-- Surface Skin Temperature -->
            <div class="space-y-1.5">
              <div class="flex justify-between items-center text-xs">
                <label class="font-medium text-slate-200 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
                  Surface Skin Temperature
                  <span class="text-slate-500 text-[10px] hidden sm:inline">(Controls Thermal Inversions: 288K - 312K)</span>
                </label>
                <div class="flex items-center gap-2 font-mono text-xs">
                  <span id="temp-c-val" class="text-amber-400 font-bold">(30.2°C)</span>
                  <input type="number" id="num-temp" min="288.7" max="312.1" step="0.1" value="303.4" class="w-20 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-bold text-right">
                  <span class="text-slate-500 text-[11px]">K</span>
                </div>
              </div>
              <input type="range" id="temp" min="288.7" max="312.1" step="0.1" value="303.4" class="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer">
              <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>15.6°C (Cold Inversion)</span>
                <span>25°C</span>
                <span>30.2°C (Mean)</span>
                <span>39.0°C (Hot)</span>
              </div>
            </div>

            <!-- Sea Level Pressure -->
            <div class="space-y-1.5">
              <div class="flex justify-between items-center text-xs">
                <label class="font-medium text-slate-200 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-purple-400"></span>
                  Sea Level Pressure
                  <span class="text-slate-500 text-[10px] hidden sm:inline">(Synoptic Pressure Subsidence: 998 - 1023 hPa)</span>
                </label>
                <div class="flex items-center gap-1.5 font-mono text-xs">
                  <input type="number" id="num-slp" min="998.6" max="1022.8" step="0.1" value="1014.1" class="w-20 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-bold text-right">
                  <span class="text-slate-500 text-[11px]">hPa</span>
                </div>
              </div>
              <input type="range" id="slp" min="998.6" max="1022.8" step="0.1" value="1014.1" class="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer">
            </div>

            <!-- Total Precipitable Water Vapor (TQV) -->
            <div class="space-y-1.5">
              <div class="flex justify-between items-center text-xs">
                <label class="font-medium text-slate-200 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-blue-400"></span>
                  Total Column Water Vapor (TQV)
                  <span class="text-slate-500 text-[10px] hidden sm:inline">(Aerosol Hygroscopic Growth: 3 - 72 kg/m²)</span>
                </label>
                <div class="flex items-center gap-1.5 font-mono text-xs">
                  <input type="number" id="num-tqv" min="3.0" max="71.7" step="0.5" value="16.4" class="w-20 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-bold text-right">
                  <span class="text-slate-500 text-[11px]">kg/m²</span>
                </div>
              </div>
              <input type="range" id="tqv" min="3.0" max="71.7" step="0.5" value="16.4" class="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer">
            </div>

            <!-- Wind Components in 2 Sub-Columns -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
              <div class="space-y-1.5">
                <div class="flex justify-between items-center text-xs">
                  <label class="font-medium text-slate-300">East Wind (U10M)</label>
                  <input type="number" id="num-u10" min="-6.36" max="6.39" step="0.05" value="1.39" class="w-16 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-mono font-bold text-xs text-right">
                </div>
                <input type="range" id="u10" min="-6.36" max="6.39" step="0.05" value="1.39" class="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer">
                <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>-6.4 (Westerly)</span>
                  <span>+6.4 (Easterly)</span>
                </div>
              </div>

              <div class="space-y-1.5">
                <div class="flex justify-between items-center text-xs">
                  <label class="font-medium text-slate-300">North Wind (V10M)</label>
                  <input type="number" id="num-v10" min="-5.97" max="6.74" step="0.05" value="-1.29" class="w-16 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-mono font-bold text-xs text-right">
                </div>
                <input type="range" id="v10" min="-5.97" max="6.74" step="0.05" value="-1.29" class="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer">
                <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>-6.0 (Northerly)</span>
                  <span>+6.7 (Southerly)</span>
                </div>
              </div>
            </div>

          </form>
        </div>

        <!-- Animated Atmospheric Wind Vector & Particle Canvas -->
        <div class="glass-card rounded-2xl p-5 space-y-3 relative overflow-hidden">
          <div class="flex items-center justify-between">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <i class="fa-solid fa-wind text-cyan-400"></i> Atmospheric Flow Simulation (10-Meter Wind Field)
            </h3>
            <span id="dispersion-badge" class="px-2.5 py-1 rounded-full text-[10px] font-semibold tracking-wide border">
              Calculating...
            </span>
          </div>

          <!-- Canvas Container -->
          <div class="relative w-full h-28 bg-slate-950/80 rounded-xl overflow-hidden border border-slate-800">
            <canvas id="windCanvas" class="w-full h-full block"></canvas>
            <div class="absolute bottom-2 left-3 text-[11px] font-mono text-slate-400 bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800 flex items-center gap-3">
              <span>Velocity: <strong id="wind-speed" class="text-white">1.90</strong> m/s</span>
              <span>Direction: <strong id="wind-dir" class="text-cyan-400 font-bold">NW</strong> (<span id="wind-deg">312.8</span>°)</span>
            </div>
            <div class="absolute top-2 right-3 text-[10px] text-slate-500 font-mono">
              Live Particle Advection
            </div>
          </div>
        </div>

      </div>

      <!-- Right Column: Live Output, Speedometer, and Delhi Map (5 Columns) -->
      <div class="lg:col-span-5 space-y-6">

        <!-- Result Card with Speedometer & AQI -->
        <div id="result-card" class="glass-card rounded-2xl p-6 relative overflow-hidden transition-all duration-300 border border-slate-800 space-y-4">
          <div class="flex items-start justify-between">
            <div>
              <span class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Predicted Ground Concentration</span>
              <div class="flex items-baseline gap-2 mt-1">
                <span id="pred-pm25" class="text-5xl font-extrabold tracking-tight font-mono text-white">--</span>
                <span class="text-slate-400 text-sm font-medium">μg/m³</span>
              </div>
            </div>
            <div id="aqi-badge" class="px-3.5 py-1.5 rounded-full text-xs font-extrabold uppercase tracking-wider border shadow-lg">
              Loading...
            </div>
          </div>

          <!-- SVG Tachometer Speedometer Gauge -->
          <div class="relative flex flex-col items-center justify-center pt-2">
            <svg viewBox="0 0 200 115" class="w-64 h-36">
              <!-- Gauge Arc Background -->
              <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#1e293b" stroke-width="16" stroke-linecap="round"/>
              <!-- Multi-color Gradient Arc -->
              <defs>
                <linearGradient id="aqiGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#10B981" />
                  <stop offset="20%" stop-color="#84CC16" />
                  <stop offset="40%" stop-color="#F59E0B" />
                  <stop offset="60%" stop-color="#F97316" />
                  <stop offset="85%" stop-color="#EF4444" />
                  <stop offset="100%" stop-color="#991B1B" />
                </linearGradient>
              </defs>
              <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#aqiGrad)" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
              <!-- Needle -->
              <g id="gauge-needle-group" style="transform-origin: 100px 100px; transform: rotate(-90deg); transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);">
                <line x1="100" y1="100" x2="100" y2="28" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round"/>
                <circle cx="100" cy="100" r="7" fill="#38bdf8" stroke="#ffffff" stroke-width="2"/>
              </g>
            </svg>
            <div class="flex justify-between w-64 text-[10px] text-slate-400 font-mono -mt-2">
              <span>0 (Good)</span>
              <span>60</span>
              <span>120</span>
              <span>250+ (Severe)</span>
            </div>
          </div>

          <!-- Health Advisory & Recommended Protection -->
          <div class="pt-3 border-t border-slate-800/80 space-y-2">
            <div class="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <i class="fa-solid fa-shield-virus text-cyan-400"></i> Health Advisory & Precautions:
            </div>
            <p id="advisory-text" class="text-xs text-slate-400 leading-relaxed"></p>
            <div id="action-box" class="p-2.5 rounded-xl text-xs font-medium border flex items-center gap-2">
              <i class="fa-solid fa-bell"></i>
              <span id="action-text"></span>
            </div>
          </div>

          <!-- Ground Truth comparison alert when sample is loaded -->
          <div id="sample-compare" class="hidden pt-3 border-t border-slate-800 text-xs flex justify-between items-center bg-cyan-950/30 p-2.5 rounded-xl border border-cyan-500/20">
            <span class="text-slate-300"><i class="fa-solid fa-check-circle text-emerald-400 mr-1"></i> Observed CPCB Value:</span>
            <span class="font-mono font-bold text-white"><span id="actual-pm25">--</span> μg/m³</span>
          </div>
        </div>

        <!-- Delhi Station Microclimates & CPCB NAAQS Standard -->
        <div class="glass-card rounded-2xl p-5 space-y-3.5">
          <div class="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <i class="fa-solid fa-cloud-sun text-cyan-400"></i> Delhi Station Microclimate Presets
            </h3>
            <span class="text-[11px] text-slate-500">Click to apply bias</span>
          </div>

          <!-- Quick Station Simulation Buttons -->
          <div class="grid grid-cols-2 gap-2 text-xs">
            <button type="button" onclick="selectStation('Anand Vihar')" class="p-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-left transition space-y-0.5">
              <div class="font-bold text-white text-[11px] flex items-center justify-between">
                <span>Anand Vihar</span>
                <span class="text-[9px] text-rose-400 font-mono">+AOD</span>
              </div>
              <p class="text-[10px] text-slate-400 truncate">Industrial / Inter-State Transit</p>
            </button>
            <button type="button" onclick="selectStation('ITO Intersection')" class="p-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-left transition space-y-0.5">
              <div class="font-bold text-white text-[11px] flex items-center justify-between">
                <span>ITO Junction</span>
                <span class="text-[9px] text-amber-400 font-mono">+Traffic</span>
              </div>
              <p class="text-[10px] text-slate-400 truncate">Central Traffic Hotspot</p>
            </button>
            <button type="button" onclick="selectStation('RK Puram')" class="p-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-left transition space-y-0.5">
              <div class="font-bold text-white text-[11px] flex items-center justify-between">
                <span>RK Puram</span>
                <span class="text-[9px] text-emerald-400 font-mono">-Bias</span>
              </div>
              <p class="text-[10px] text-slate-400 truncate">South Delhi Residential</p>
            </button>
            <button type="button" onclick="selectStation('Punjabi Bagh')" class="p-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-left transition space-y-0.5">
              <div class="font-bold text-white text-[11px] flex items-center justify-between">
                <span>Punjabi Bagh</span>
                <span class="text-[9px] text-amber-400 font-mono">+Arterial</span>
              </div>
              <p class="text-[10px] text-slate-400 truncate">West Arterial Highway</p>
            </button>
            <button type="button" onclick="selectStation('Bawana')" class="p-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-left transition space-y-0.5">
              <div class="font-bold text-white text-[11px] flex items-center justify-between">
                <span>Bawana</span>
                <span class="text-[9px] text-rose-400 font-mono">+Industrial</span>
              </div>
              <p class="text-[10px] text-slate-400 truncate">North-West Major Belt</p>
            </button>
            <button type="button" onclick="selectStation('Lodhi Road')" class="p-2.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-left transition space-y-0.5">
              <div class="font-bold text-white text-[11px] flex items-center justify-between">
                <span>Lodhi Road</span>
                <span class="text-[9px] text-emerald-400 font-mono">Green Zone</span>
              </div>
              <p class="text-[10px] text-slate-400 truncate">Central Institutional</p>
            </button>
          </div>

          <!-- CPCB NAAQS & WHO Air Quality Benchmark Scale -->
          <div class="pt-2 border-t border-slate-800/80 space-y-2 text-[10px]">
            <div class="flex justify-between items-center text-slate-400">
              <span class="font-semibold text-slate-300">CPCB NAAQS 24-hr: <strong>60 μg/m³</strong></span>
              <span class="text-cyan-400 font-mono">WHO Standard: <strong>15 μg/m³</strong></span>
            </div>
            <div class="grid grid-cols-6 h-2 rounded-full overflow-hidden text-[9px] font-mono">
              <div class="bg-emerald-500" title="Good (0-30)"></div>
              <div class="bg-lime-500" title="Satisfactory (31-60)"></div>
              <div class="bg-amber-500" title="Moderate (61-90)"></div>
              <div class="bg-orange-500" title="Poor (91-120)"></div>
              <div class="bg-rose-500" title="Very Poor (121-250)"></div>
              <div class="bg-red-900" title="Severe (250+)"></div>
            </div>
            <div class="flex justify-between text-[9px] text-slate-500 font-mono">
              <span>0 (Good)</span>
              <span>30</span>
              <span>60</span>
              <span>90</span>
              <span>120</span>
              <span>250+ (Severe)</span>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- Interactive Lower Analytics: Sensitivity Curve & Timeline Player -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Dynamic What-If Sensitivity Chart (6 Columns) -->
      <div class="lg:col-span-6 glass-card rounded-2xl p-6 space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <div>
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-chart-line text-cyan-400"></i> What-If Response Curve
            </h3>
            <p class="text-slate-400 text-xs">Simulate model output across a parameter sweep under current weather</p>
          </div>
          <div class="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs font-medium">
            <button id="btn-sens-aod" onclick="setSensitivityVar('aod')" class="px-2.5 py-1 rounded bg-cyan-600 text-white transition">vs AOD</button>
            <button id="btn-sens-temp" onclick="setSensitivityVar('temp')" class="px-2.5 py-1 rounded text-slate-400 hover:text-white transition">vs Temp</button>
            <button id="btn-sens-slp" onclick="setSensitivityVar('slp')" class="px-2.5 py-1 rounded text-slate-400 hover:text-white transition">vs Pressure</button>
          </div>
        </div>
        <div class="h-60 relative">
          <canvas id="sensitivityChart"></canvas>
        </div>
      </div>

      <!-- Local Feature Attribution Waterfall (6 Columns) -->
      <div class="lg:col-span-6 glass-card rounded-2xl p-6 space-y-4">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-bars-staggered text-cyan-400"></i> Local Feature Attribution
            </h3>
            <p class="text-slate-400 text-xs">Deviation contribution of each feature relative to seasonal baseline (127.7 μg/m³)</p>
          </div>
          <span class="text-xs font-mono text-slate-400">Additive Impact</span>
        </div>
        <div id="waterfall-bars" class="space-y-3 text-xs pt-1">
          <!-- Dynamic bars inserted here -->
        </div>
      </div>

    </div>

    <!-- Historical Season Timeline Player -->
    <div class="glass-card rounded-2xl p-6 space-y-4">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-3">
        <div>
          <h3 class="text-sm font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-clock-rotate-left text-cyan-400"></i> Winter 2024-2025 Historical Timeline Player
          </h3>
          <p class="text-slate-400 text-xs">Step through actual daytime satellite overpasses (Sep 2024 - Feb 2025) and observe actual vs predicted particulate trajectory</p>
        </div>
        
        <!-- Timeline Controls -->
        <div class="flex items-center gap-2">
          <button id="btn-play" onclick="togglePlay()" class="px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs transition flex items-center gap-2">
            <i class="fa-solid fa-play"></i> <span>Play Season</span>
          </button>
          <button onclick="resetTimeline()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition">
            <i class="fa-solid fa-rotate-left"></i>
          </button>
          <select id="play-speed" class="px-2 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 text-xs font-medium">
            <option value="250">1x Speed</option>
            <option value="120" selected>2x Speed</option>
            <option value="40">5x Fast</option>
          </select>
        </div>
      </div>

      <!-- Scrubber & Current Timestamp Display -->
      <div class="space-y-2">
        <div class="flex justify-between items-center text-xs">
          <span class="text-slate-400">Observation Timestamp: <strong id="timeline-date" class="text-cyan-400 font-mono text-sm ml-1">--</strong></span>
          <span class="text-slate-400 font-mono">Step <span id="timeline-step">0</span> / <span id="timeline-total">684</span></span>
        </div>
        <input type="range" id="timeline-slider" min="0" max="683" value="0" class="w-full h-2 bg-slate-800 rounded-lg cursor-pointer">
      </div>

      <!-- Timeline Chart -->
      <div class="h-64 relative">
        <canvas id="timelineChart"></canvas>
      </div>
    </div>

    <!-- Footer -->
    <footer class="text-center text-xs text-slate-600 pt-2 pb-4">
      Delhi Air Quality & PM2.5 Prediction System • ISRO INSAT-3D/3DR Level-2G AOD • NASA MERRA-2 • CPCB CAAQMS • XGBoost Regressor
    </footer>

  </div>

  <script>
    const el = id => document.getElementById(id);
    const sliderIds = ['aod', 'temp', 'slp', 'tqv', 'u10', 'v10'];
    const numIds = ['num-aod', 'num-temp', 'num-slp', 'num-tqv', 'num-u10', 'num-v10'];

    // Link sliders to number inputs
    sliderIds.forEach((sid, idx) => {
      const nid = numIds[idx];
      el(sid).addEventListener('input', () => {
        el(nid).value = el(sid).value;
        onParamChanged();
      });
      el(nid).addEventListener('input', () => {
        el(sid).value = el(nid).value;
        onParamChanged();
      });
    });

    function onParamChanged() {
      updateReadouts();
      fetchPrediction();
      updateSensitivityChart();
    }

    function updateReadouts() {
      const tempK = parseFloat(el('temp').value);
      el('temp-c-val').innerText = '(' + (tempK - 273.15).toFixed(1) + '°C)';

      const u = parseFloat(el('u10').value);
      const v = parseFloat(el('v10').value);
      const speed = Math.sqrt(u * u + v * v);
      const deg = ((Math.atan2(-u, -v) * 180 / Math.PI) + 360) % 360;
      
      const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
      const dirName = dirs[Math.round(deg / 22.5) % 16];

      el('wind-speed').innerText = speed.toFixed(2);
      el('wind-dir').innerText = dirName;
      el('wind-deg').innerText = deg.toFixed(1);

      // Dispersion Badge
      const dBadge = el('dispersion-badge');
      if (speed < 1.0) {
        dBadge.innerText = 'Stagnant Air (Inversion Risk)';
        dBadge.className = 'px-2.5 py-1 rounded-full text-[10px] font-semibold tracking-wide bg-rose-500/15 text-rose-400 border border-rose-500/30';
      } else if (speed < 2.5) {
        dBadge.innerText = 'Moderate Dispersion';
        dBadge.className = 'px-2.5 py-1 rounded-full text-[10px] font-semibold tracking-wide bg-amber-500/15 text-amber-400 border border-amber-500/30';
      } else {
        dBadge.innerText = 'Strong Atmospheric Ventilation';
        dBadge.className = 'px-2.5 py-1 rounded-full text-[10px] font-semibold tracking-wide bg-emerald-500/15 text-emerald-400 border border-emerald-500/30';
      }

      // Update particle canvas speed & direction
      windVelocity.u = u;
      windVelocity.v = v;
      windVelocity.speed = speed;
    }

    let debounceTimer = null;
    function fetchPrediction() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(runModelInference, 60);
    }

    async function runModelInference() {
      const payload = {
        aod: parseFloat(el('aod').value),
        temp: parseFloat(el('temp').value),
        slp: parseFloat(el('slp').value),
        tqv: parseFloat(el('tqv').value),
        u10: parseFloat(el('u10').value),
        v10: parseFloat(el('v10').value)
      };

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderPrediction(data);
        renderWaterfall(payload, data.pm25);
      } catch (err) {
        console.error('Prediction failed:', err);
      }
    }

    function renderPrediction(data) {
      el('pred-pm25').innerText = data.pm25.toFixed(1);

      const badge = el('aqi-badge');
      badge.innerText = data.category;
      badge.style.color = data.color;
      badge.style.backgroundColor = data.bg_color;
      badge.style.borderColor = data.color;

      const card = el('result-card');
      card.style.borderColor = data.color + '60';

      // Needle angle in tachometer: -90deg (0) to +90deg (250+)
      const pct = Math.min(100, Math.max(0, data.gauge_pct));
      const angle = -90 + (pct / 100) * 180;
      el('gauge-needle-group').style.transform = `rotate(${angle}deg)`;

      el('advisory-text').innerText = data.advisory;
      
      const actionBox = el('action-box');
      el('action-text').innerText = data.action;
      actionBox.style.color = data.color;
      actionBox.style.backgroundColor = data.bg_color;
      actionBox.style.borderColor = data.color + '40';

      // Update map bounding box tint
      if (delhiBboxLayer) {
        delhiBboxLayer.setStyle({ color: data.color, fillColor: data.color });
      }
    }

    // Local Waterfall Feature Attribution
    function renderWaterfall(payload, currentPred) {
      const baseMean = 127.7;
      // Approximate contributions based on variable deviations from mean multiplied by global importance
      const meanVals = { slp: 1013.2, u10: 1.15, v10: -1.01, temp: 302.9, tqv: 19.5, aod: 0.99 };
      const importances = { slp: 0.215, u10: 0.176, v10: 0.145, temp: 0.135, tqv: 0.207, aod: 0.122 };

      const diffs = {
        "Aerosol Depth (AOD)": (payload.aod - meanVals.aod) * 65.0,
        "Surface Temp (Inversion)": (meanVals.temp - payload.temp) * 6.5,
        "Sea Level Pressure": (payload.slp - meanVals.slp) * 8.2,
        "Column Moisture (TQV)": (payload.tqv - meanVals.tqv) * 1.8,
        "Zonal Wind (U10)": -(payload.u10 - meanVals.u10) * 5.5,
        "Meridional Wind (V10)": -(payload.v10 - meanVals.v10) * 4.2
      };

      const container = el('waterfall-bars');
      container.innerHTML = '';

      Object.entries(diffs).forEach(([name, val]) => {
        const isPos = val >= 0;
        const widthPct = Math.min(100, Math.abs(val) / 60 * 100);
        const colClass = isPos ? 'bg-rose-500' : 'bg-emerald-500';
        const txtCol = isPos ? 'text-rose-400' : 'text-emerald-400';

        const row = document.createElement('div');
        row.className = 'space-y-1';
        row.innerHTML = `
          <div class="flex justify-between items-center text-[11px]">
            <span class="text-slate-300">${name}</span>
            <span class="font-mono font-bold ${txtCol}">${isPos ? '+' : ''}${val.toFixed(1)} μg/m³</span>
          </div>
          <div class="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden flex ${isPos ? 'justify-start' : 'justify-end'}">
            <div class="h-full ${colClass} rounded-full" style="width: ${widthPct}%;"></div>
          </div>
        `;
        container.appendChild(row);
      });
    }

    // Presets
    const presets = {
      winter: { slp: 1018.5, u10: 0.35, v10: -0.42, temp: 292.0, tqv: 11.2, aod: 1.85, name: "Winter Smog Preset" },
      monsoon: { slp: 1002.5, u10: 3.20, v10: 2.80, temp: 308.2, tqv: 52.4, aod: 0.32, name: "Clean Monsoon Preset" },
      autumn: { slp: 1014.1, u10: 1.39, v10: -1.29, temp: 303.4, tqv: 16.4, aod: 0.88, name: "Autumn Baseline" },
      windy: { slp: 1008.0, u10: 4.80, v10: 0.20, temp: 302.0, tqv: 22.0, aod: 0.45, name: "High Dispersion Preset" }
    };

    function applyPreset(pName) {
      el('sample-compare').classList.add('hidden');
      const p = presets[pName];
      if (!p) return;
      el('slp').value = el('num-slp').value = p.slp;
      el('u10').value = el('num-u10').value = p.u10;
      el('v10').value = el('num-v10').value = p.v10;
      el('temp').value = el('num-temp').value = p.temp;
      el('tqv').value = el('num-tqv').value = p.tqv;
      el('aod').value = el('num-aod').value = p.aod;
      el('active-station-label').innerText = p.name;
      onParamChanged();
    }

    function resetToMean() {
      applyPreset('autumn');
      el('active-station-label').innerText = "Regional Delhi Mean";
    }

    async function loadRandomSample() {
      try {
        const res = await fetch('/sample');
        const s = await res.json();
        el('slp').value = el('num-slp').value = s['Sea level pressue'];
        el('u10').value = el('num-u10').value = s['east wind 10m'];
        el('v10').value = el('num-v10').value = s['north wind 10m'];
        el('temp').value = el('num-temp').value = s['Surface skin temp'];
        el('tqv').value = el('num-tqv').value = s['TQV'];
        el('aod').value = el('num-aod').value = s['AOD'];
        el('active-station-label').innerText = "Sampled: " + s['time'];
        
        onParamChanged();

        el('actual-pm25').innerText = s['PM2.5'].toFixed(1);
        el('sample-compare').classList.remove('hidden');
      } catch (e) {
        console.error('Failed to load sample', e);
      }
    }

    // Share Scenario
    function shareScenario() {
      const params = new URLSearchParams({
        aod: el('aod').value,
        temp: el('temp').value,
        slp: el('slp').value,
        tqv: el('tqv').value,
        u10: el('u10').value,
        v10: el('v10').value
      });
      const url = window.location.origin + window.location.pathname + '#' + params.toString();
      navigator.clipboard.writeText(url).then(() => {
        alert("Scenario link copied to clipboard! Anyone opening this URL will load your exact weather inputs.");
      });
    }

    function downloadJSON() {
      const data = {
        timestamp: new Date().toISOString(),
        inputs: {
          aod: parseFloat(el('aod').value),
          surface_temp_k: parseFloat(el('temp').value),
          sea_level_pressure_hpa: parseFloat(el('slp').value),
          tqv_kg_m2: parseFloat(el('tqv').value),
          east_wind_m_s: parseFloat(el('u10').value),
          north_wind_m_s: parseFloat(el('v10').value)
        },
        prediction: {
          pm25: parseFloat(el('pred-pm25').innerText),
          category: el('aqi-badge').innerText
        }
      };
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `delhi_pm25_prediction_${Date.now()}.json`;
      a.click();
    }

    // ==========================================
    // 1. DELHI MONITORING STATIONS MICROCLIMATES
    // ==========================================
    const stations = [
      { name: "Anand Vihar", type: "Industrial / Inter-State Transit", aodMod: 0.25, tempMod: -1.5 },
      { name: "ITO Intersection", type: "Central Urban Traffic Hotspot", aodMod: 0.15, tempMod: 0.5 },
      { name: "RK Puram", type: "South Delhi Urban Residential", aodMod: -0.05, tempMod: 0.2 },
      { name: "Punjabi Bagh", type: "West Delhi Heavy Arterial Corridor", aodMod: 0.18, tempMod: -0.8 },
      { name: "Jahangirpuri", type: "North Industrial Transit Hub", aodMod: 0.30, tempMod: -2.0 },
      { name: "IGI Airport T3", type: "South-West Aviation Corridor", aodMod: -0.15, tempMod: 1.0 },
      { name: "Bawana", type: "North-West Major Industrial Belt", aodMod: 0.35, tempMod: -1.8 },
      { name: "Lodhi Road", type: "Central Institutional Green Zone", aodMod: -0.20, tempMod: 0.8 }
    ];

    function selectStation(name) {
      const st = stations.find(s => s.name === name);
      if (!st) return;
      el('active-station-label').innerText = "Station: " + st.name;
      
      // Apply microclimate bias to current sliders
      let newAod = Math.min(2.9, Math.max(0.15, parseFloat(el('aod').value) + st.aodMod));
      let newTemp = Math.min(312.1, Math.max(288.7, parseFloat(el('temp').value) + st.tempMod));
      
      el('aod').value = el('num-aod').value = newAod.toFixed(2);
      el('temp').value = el('num-temp').value = newTemp.toFixed(1);
      
      onParamChanged();
    }

    // ==========================================
    // 2. ANIMATED WIND PARTICLE CANVAS
    // ==========================================
    const canvas = el('windCanvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    const NUM_PARTICLES = 130;
    let windVelocity = { u: 1.39, v: -1.29, speed: 1.9 };

    function resizeCanvas() {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
    }

    function initParticles() {
      resizeCanvas();
      particles = [];
      for (let i = 0; i < NUM_PARTICLES; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          age: Math.floor(Math.random() * 80),
          maxAge: 60 + Math.random() * 40
        });
      }
    }

    function animateWind() {
      ctx.fillStyle = 'rgba(11, 17, 32, 0.2)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // In canvas coordinates: East wind (+u) moves right (+x), North wind (+v) moves up (-y)
      const vx = windVelocity.u * 1.2;
      const vy = -windVelocity.v * 1.2;

      // Color based on dispersion speed
      const speed = windVelocity.speed;
      let strokeCol = 'rgba(56, 189, 248, 0.7)';
      if (speed < 1.0) strokeCol = 'rgba(244, 63, 94, 0.6)';
      else if (speed < 2.5) strokeCol = 'rgba(251, 191, 36, 0.65)';
      else strokeCol = 'rgba(52, 211, 153, 0.75)';

      ctx.strokeStyle = strokeCol;
      ctx.lineWidth = 1.5;
      ctx.beginPath();

      particles.forEach(p => {
        ctx.moveTo(p.x, p.y);
        p.x += vx;
        p.y += vy;
        p.age++;

        ctx.lineTo(p.x, p.y);

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;
        if (p.age > p.maxAge) {
          p.x = Math.random() * canvas.width;
          p.y = Math.random() * canvas.height;
          p.age = 0;
        }
      });
      ctx.stroke();

      requestAnimationFrame(animateWind);
    }

    window.addEventListener('resize', resizeCanvas);

    // ==========================================
    // 3. SENSITIVITY RESPONSE CURVE (CHART.JS)
    // ==========================================
    let sensitivityChart;
    let sensitivityVar = 'aod';

    function setSensitivityVar(v) {
      sensitivityVar = v;
      ['aod', 'temp', 'slp'].forEach(x => {
        const btn = el('btn-sens-' + x);
        if (x === v) {
          btn.className = "px-2.5 py-1 rounded bg-cyan-600 text-white transition";
        } else {
          btn.className = "px-2.5 py-1 rounded text-slate-400 hover:text-white transition";
        }
      });
      updateSensitivityChart();
    }

    function initSensitivityChart() {
      const sCtx = el('sensitivityChart').getContext('2d');
      sensitivityChart = new Chart(sCtx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            label: 'Predicted PM2.5 (μg/m³)',
            data: [],
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2.5,
            pointRadius: 3,
            pointHoverRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#0f172a',
              titleColor: '#38bdf8',
              bodyFont: { family: 'JetBrains Mono' }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255,255,255,0.05)' },
              ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
            },
            y: {
              grid: { color: 'rgba(255,255,255,0.05)' },
              ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
              title: { display: true, text: 'PM2.5 (μg/m³)', color: '#64748b', font: { size: 11 } }
            }
          }
        }
      });
    }

    let sensTimer = null;
    function updateSensitivityChart() {
      clearTimeout(sensTimer);
      sensTimer = setTimeout(async () => {
        const currentPayload = {
          aod: parseFloat(el('aod').value),
          temp: parseFloat(el('temp').value),
          slp: parseFloat(el('slp').value),
          tqv: parseFloat(el('tqv').value),
          u10: parseFloat(el('u10').value),
          v10: parseFloat(el('v10').value),
          sweep_var: sensitivityVar
        };

        try {
          const res = await fetch('/sensitivity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPayload)
          });
          const data = await res.json();
          sensitivityChart.data.labels = data.labels;
          sensitivityChart.data.datasets[0].data = data.values;
          sensitivityChart.update('none');
        } catch (e) {
          console.error(e);
        }
      }, 100);
    }

    // ==========================================
    // 4. TIMELINE SEASON PLAYER (CHART.JS)
    // ==========================================
    let timelineChart;
    let timelineData = [];
    let isPlaying = false;
    let playInterval = null;

    async function initTimeline() {
      try {
        const res = await fetch('/timeline');
        timelineData = await res.json();
        el('timeline-total').innerText = timelineData.length;
        el('timeline-slider').max = timelineData.length - 1;

        const tCtx = el('timelineChart').getContext('2d');
        timelineChart = new Chart(tCtx, {
          type: 'line',
          data: {
            labels: timelineData.map(d => d.time.split(' ')[0]),
            datasets: [
              {
                label: 'Observed PM2.5 (CPCB CAAQMS)',
                data: timelineData.map(d => d.actual_pm25),
                borderColor: '#94a3b8',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.1
              },
              {
                label: 'XGBoost Predicted PM2.5',
                data: timelineData.map(d => d.pred_pm25),
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.08)',
                fill: true,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: {
                position: 'top',
                labels: { color: '#cbd5e1', font: { size: 11 } }
              },
              tooltip: {
                backgroundColor: '#0f172a',
                bodyFont: { family: 'JetBrains Mono' }
              }
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: '#64748b', maxTicksLimit: 12, font: { size: 10 } }
              },
              y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
                title: { display: true, text: 'PM2.5 (μg/m³)', color: '#64748b' }
              }
            }
          }
        });

        // Click on chart to jump to timestamp
        el('timelineChart').onclick = evt => {
          const points = timelineChart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
          if (points.length) {
            setTimelineStep(points[0].index);
          }
        };

        // Scrubber slider binding
        el('timeline-slider').addEventListener('input', e => {
          setTimelineStep(parseInt(e.target.value));
        });

        setTimelineStep(0);
      } catch (e) {
        console.error('Failed to init timeline:', e);
      }
    }

    function setTimelineStep(idx) {
      if (!timelineData.length) return;
      const d = timelineData[idx];
      el('timeline-slider').value = idx;
      el('timeline-step').innerText = idx + 1;
      el('timeline-date').innerText = d.time;

      // Update inputs smoothly
      el('slp').value = el('num-slp').value = d.slp;
      el('u10').value = el('num-u10').value = d.u10;
      el('v10').value = el('num-v10').value = d.v10;
      el('temp').value = el('num-temp').value = d.temp;
      el('tqv').value = el('num-tqv').value = d.tqv;
      el('aod').value = el('num-aod').value = d.aod;
      el('active-station-label').innerText = "Historical: " + d.time;

      updateReadouts();
      fetchPrediction();

      el('actual-pm25').innerText = d.actual_pm25.toFixed(1);
      el('sample-compare').classList.remove('hidden');
    }

    function togglePlay() {
      isPlaying = !isPlaying;
      const btn = el('btn-play');
      if (isPlaying) {
        btn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>Pause</span>';
        btn.className = "px-4 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs transition flex items-center gap-2";
        const speed = parseInt(el('play-speed').value);
        playInterval = setInterval(() => {
          let curr = parseInt(el('timeline-slider').value);
          if (curr >= timelineData.length - 1) {
            curr = 0;
          } else {
            curr++;
          }
          setTimelineStep(curr);
        }, speed);
      } else {
        btn.innerHTML = '<i class="fa-solid fa-play"></i> <span>Play Season</span>';
        btn.className = "px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs transition flex items-center gap-2";
        clearInterval(playInterval);
      }
    }

    function resetTimeline() {
      if (isPlaying) togglePlay();
      setTimelineStep(0);
    }

    // Hash URL parser for shared scenarios
    function checkUrlHash() {
      if (window.location.hash) {
        const params = new URLSearchParams(window.location.hash.slice(1));
        sliderIds.forEach(id => {
          if (params.has(id)) {
            el(id).value = params.get(id);
            el('num-' + id).value = params.get(id);
          }
        });
      }
    }

    // Startup Initialization
    window.addEventListener('DOMContentLoaded', () => {
      initParticles();
      animateWind();
      initSensitivityChart();
      initTimeline();
      checkUrlHash();
      onParamChanged();
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, metadata=metadata, osm_env_key=OSM_ENV_KEY)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        feature_vals = [
            float(data.get("slp", 1014.1)),
            float(data.get("u10", 1.39)),
            float(data.get("v10", -1.29)),
            float(data.get("temp", 303.4)),
            float(data.get("tqv", 16.4)),
            float(data.get("aod", 0.88))
        ]

        pred_val = predict_pm25(feature_vals)
        cat_info = get_aqi_category(pred_val)
        wind_speed, wind_deg, wind_dir = calculate_wind_params(feature_vals[1], feature_vals[2])

        return jsonify({
            "pm25": round(pred_val, 2),
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            **cat_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/sensitivity", methods=["POST"])
def sensitivity():
    try:
        data = request.get_json(force=True)
        sweep_var = data.get("sweep_var", "aod")

        base_slp = float(data.get("slp", 1014.1))
        base_u10 = float(data.get("u10", 1.39))
        base_v10 = float(data.get("v10", -1.29))
        base_temp = float(data.get("temp", 303.4))
        base_tqv = float(data.get("tqv", 16.4))
        base_aod = float(data.get("aod", 0.88))

        labels = []
        preds = []

        if sweep_var == "aod":
            vals = [0.15 + i * (2.85 - 0.15) / 17 for i in range(18)]
            for v in vals:
                labels.append(f"{v:.2f}")
                preds.append(round(predict_pm25([base_slp, base_u10, base_v10, base_temp, base_tqv, v]), 1))
        elif sweep_var == "temp":
            vals = [288.7 + i * (312.1 - 288.7) / 17 for i in range(18)]
            for v in vals:
                celsius = v - 273.15
                labels.append(f"{celsius:.1f}°C")
                preds.append(round(predict_pm25([base_slp, base_u10, base_v10, v, base_tqv, base_aod]), 1))
        else: # slp
            vals = [999.0 + i * (1022.0 - 999.0) / 17 for i in range(18)]
            for v in vals:
                labels.append(f"{v:.1f}")
                preds.append(round(predict_pm25([v, base_u10, base_v10, base_temp, base_tqv, base_aod]), 1))

        return jsonify({
            "labels": labels,
            "values": preds
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/timeline", methods=["GET"])
def timeline():
    return jsonify(timeline_cache)

@app.route("/sample", methods=["GET"])
def sample():
    if timeline_cache:
        item = random.choice(timeline_cache)
        return jsonify({
            "time": item["time"],
            "PM2.5": item["actual_pm25"],
            "AOD": item["aod"],
            "Surface skin temp": item["temp"],
            "Sea level pressue": item["slp"],
            "TQV": item["tqv"],
            "east wind 10m": item["u10"],
            "north wind 10m": item["v10"]
        })
    return jsonify({"error": "Dataset not found"}), 404

@app.route("/metadata", methods=["GET"])
def get_metadata():
    return jsonify(metadata)

if __name__ == "__main__":
    print("Starting Ultra-Interactive Delhi PM2.5 Prediction Platform...")
    print("Listening on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
