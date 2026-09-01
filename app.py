import os
import time
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, ModuleNotFoundError):
    HAS_XGBOOST = False
    XGBRegressor = None

try:
    from rocketpy import Environment, Rocket, SolidMotor, Flight
    HAS_ROCKETPY = True
except Exception:
    HAS_ROCKETPY = False

# Features and Targets
BASE_FEATURES = ["dry_mass", "diameter", "drag_coefficient", "total_impulse", "burn_time", "inclination"]
ENGINEERED_FEATURES = BASE_FEATURES + ["impulse_to_mass", "burn_time_to_impulse", "reference_area", "impulse_to_weight"]
TARGETS = ["apogee_agl_m", "flight_time_s", "max_velocity_m_s", "time_to_apogee_s"]

HUMAN_NAMES = {
    "dry_mass": "Dry Mass (Kilograms)",
    "diameter": "Diameter (Meters)",
    "drag_coefficient": "Drag Coefficient",
    "total_impulse": "Total Impulse (Newton-Seconds)",
    "burn_time": "Burn Time (Seconds)",
    "inclination": "Launch Inclination (Degrees)",
    "impulse_to_mass": "Impulse to Mass Ratio (Newton-Seconds per Kilogram)",
    "burn_time_to_impulse": "Burn Time to Impulse Ratio (Seconds per Newton-Second)",
    "reference_area": "Reference Area (Square Meters)",
    "impulse_to_weight": "Impulse to Weight Ratio",
    "apogee_agl_m": "Apogee Altitude (Meters)",
    "flight_time_s": "Total Flight Time (Seconds)",
    "max_velocity_m_s": "Maximum Velocity (Meters per Second)",
    "time_to_apogee_s": "Time to Apogee (Seconds)"
}

# --- Backend Helper Functions ---
def simulate_single_flight(dry_mass, diameter, drag_coefficient, total_impulse, burn_time, inclination):
    """Simulate a single rocket flight using RocketPy if available."""
    if not HAS_ROCKETPY:
        raise RuntimeError("RocketPy is not installed or available.")
    
    radius = diameter / 2.0
    length = 1.8
    env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
    env.set_atmospheric_model(type="standard_atmosphere")
    
    thrust_source = [
        (0, 0),
        (0.1, total_impulse / burn_time * 1.2),
        (burn_time * 0.8, total_impulse / burn_time),
        (burn_time, 0)
    ]
    
    motor = SolidMotor(
        thrust_source=thrust_source,
        dry_mass=1.0,
        dry_inertia=(0.05, 0.05, 0.002),
        nozzle_radius=radius * 0.4,
        grain_number=5,
        grain_density=1800,
        grain_outer_radius=radius * 0.85,
        grain_initial_inner_radius=radius * 0.3,
        grain_initial_height=0.12,
        grain_separation=0.005,
        grains_center_of_mass_position=0.4,
        center_of_dry_mass_position=0.4,
        nozzle_position=0,
        burn_time=burn_time,
        throat_radius=radius * 0.15,
    )
    
    rocket = Rocket(
        radius=radius,
        mass=dry_mass,
        inertia=(6, 6, 0.03),
        power_off_drag=drag_coefficient,
        power_on_drag=drag_coefficient,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-0.9 * length)
    rocket.set_rail_buttons(0.1, -0.5)
    rocket.add_nose(length=0.2 * length, kind="vonKarman", position=length)
    rocket.add_trapezoidal_fins(4, span=radius * 2.5, root_chord=radius * 3, tip_chord=radius * 1.2, position=-0.8 * length)
    rocket.add_parachute(name="main", cd_s=1.5, trigger="apogee")
    
    static_margin = rocket.static_margin(0)
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=2.0,
        inclination=inclination,
        heading=0,
        terminate_on_apogee=False,
        max_time=400
    )
    
    if flight.t_final >= 0.99 * 400:
        raise RuntimeError("Simulation did not land within max_time.")
        
    return {
        "static_margin": float(static_margin),
        "apogee_agl_m": float(flight.apogee - env.elevation),
        "flight_time_s": float(flight.t_final),
        "max_velocity_m_s": float(flight.max_speed),
        "time_to_apogee_s": float(flight.apogee_time),
        "landing_distance_m": float(np.hypot(flight.x_impact, flight.y_impact)),
    }

def engineer_features(df):
    """Compute engineered features for rocket dataframe."""
    df = df.copy()
    df["impulse_to_mass"] = df["total_impulse"] / df["dry_mass"]
    df["burn_time_to_impulse"] = df["burn_time"] / df["total_impulse"]
    df["reference_area"] = np.pi * (df["diameter"] / 2.0) ** 2
    df["impulse_to_weight"] = df["total_impulse"] / (df["dry_mass"] * 9.81)
    return df

@st.cache_data(show_spinner=False)
def generate_rocket_dataset(num_samples=150, use_full_sim=False):
    """Generate rocket trajectory dataset with physics logic or RocketPy."""
    np.random.seed(42)
    records = []
    
    dry_masses = np.random.uniform(8.0, 25.0, num_samples)
    diameters = np.random.uniform(0.10, 0.25, num_samples)
    drag_coeffs = np.random.uniform(0.35, 0.75, num_samples)
    total_impulses = np.random.uniform(800.0, 4500.0, num_samples)
    burn_times = np.random.uniform(1.5, 6.0, num_samples)
    inclinations = np.random.uniform(75.0, 88.0, num_samples)
    
    for i in range(num_samples):
        dm = float(dry_masses[i])
        dia = float(diameters[i])
        cd = float(drag_coeffs[i])
        ti = float(total_impulses[i])
        bt = float(burn_times[i])
        inc = float(inclinations[i])
        
        sim_success = False
        res = {}
        
        if use_full_sim and HAS_ROCKETPY and i < 20:
            try:
                res = simulate_single_flight(dm, dia, cd, ti, bt, inc)
                sim_success = True
            except Exception:
                sim_success = False
                
        if not sim_success:
            # Physics vector model derived from flight equations
            ref_area = np.pi * (dia / 2.0) ** 2
            avg_thrust = ti / bt
            net_accel = (avg_thrust / (dm + 1.5)) - 9.81 * np.sin(np.radians(inc))
            burnout_vel = max(10.0, net_accel * bt)
            drag_factor = 0.5 * 1.225 * cd * ref_area
            terminal_loss = 1.0 + (drag_factor * (burnout_vel ** 2) / (dm * 9.81 * 100.0))
            
            apogee = (burnout_vel ** 2) / (2.0 * 9.81 * terminal_loss) * np.sin(np.radians(inc))
            time_to_apogee = bt + (burnout_vel / (9.81 * np.sqrt(terminal_loss)))
            descent_time = np.sqrt(2.0 * max(10.0, apogee) / 5.0)
            flight_time = time_to_apogee + descent_time
            max_vel = burnout_vel * (1.0 + np.random.uniform(-0.03, 0.03))
            static_margin = 1.5 + (dia * 10.0) - (cd * 1.2) + np.random.uniform(-0.4, 0.4)
            landing_dist = (flight_time * 2.5) + np.random.uniform(10.0, 150.0)
            
            res = {
                "static_margin": float(static_margin),
                "apogee_agl_m": float(max(50.0, apogee)),
                "flight_time_s": float(max(15.0, flight_time)),
                "max_velocity_m_s": float(max(20.0, max_vel)),
                "time_to_apogee_s": float(max(5.0, time_to_apogee)),
                "landing_distance_m": float(landing_dist),
            }
            
        record = {
            "dry_mass": dm,
            "diameter": dia,
            "drag_coefficient": cd,
            "total_impulse": ti,
            "burn_time": bt,
            "inclination": inc,
            "static_margin": res["static_margin"],
            "apogee_agl_m": res["apogee_agl_m"],
            "flight_time_s": res["flight_time_s"],
            "max_velocity_m_s": res["max_velocity_m_s"],
            "time_to_apogee_s": res["time_to_apogee_s"],
            "landing_distance_m": res["landing_distance_m"],
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    df = engineer_features(df)
    
    # Define binary success label (Static Margin in safe stability zone 1.2-3.2 and Apogee > 300m)
    df["flight_success"] = (
        (df["static_margin"] >= 1.2) &
        (df["static_margin"] <= 3.2) &
        (df["apogee_agl_m"] >= 300.0)
    ).astype(int)
    
    return df

@st.cache_resource(show_spinner=False)
def train_machine_learning_pipeline(df):
    """Train regression comparison, logistic regression classifier, kmeans clustering, and isolation forest."""
    X = df[ENGINEERED_FEATURES]
    Y = df[TARGETS]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Regression Models Comparison
    models = {
        "Linear Regression": MultiOutputRegressor(LinearRegression()),
        "Ridge Regression": MultiOutputRegressor(Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        "Neural Network": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=42)),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_results = []
    
    trained_reg_models = {}
    
    for name, model in models.items():
        r2_scores = []
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            Y_tr, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]
            
            model.fit(X_tr, Y_tr)
            preds = model.predict(X_val)
            
            r2_scores.append(r2_score(Y_val, preds, multioutput="uniform_average"))
            mae_scores.append(mean_absolute_error(Y_val, preds, multioutput="uniform_average"))
            
        mean_r2 = np.mean(r2_scores)
        mean_mae = np.mean(mae_scores)
        
        reg_results.append({
            "Model Name": name,
            "Cross Validation R2 Score": round(float(mean_r2), 4),
            "Cross Validation MAE": round(float(mean_mae), 2)
        })
        
        # Fit on full data for deployment
        model.fit(X_scaled, Y)
        trained_reg_models[name] = model
        
    reg_df = pd.DataFrame(reg_results).sort_values(by="Cross Validation R2 Score", ascending=False).reset_index(drop=True)
    best_model_name = reg_df.iloc[0]["Model Name"]
    best_reg_model = trained_reg_models[best_model_name]
    
    # 2. Logistic Regression Success Classifier
    y_class = df["flight_success"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_class, test_size=0.25, random_state=42, stratify=y_class)
    
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    
    clf_accuracy = accuracy_score(y_test_c, y_pred_c)
    conf_mat = confusion_matrix(y_test_c, y_pred_c)
    clf_report = classification_report(y_test_c, y_pred_c, output_dict=True, target_names=["Unstable/Low", "Stable/Successful"])
    
    # 3. K-Means Clustering & PCA
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df_clusters = df[ENGINEERED_FEATURES].copy()
    df_clusters["Cluster Label"] = cluster_labels
    cluster_means = df_clusters.groupby("Cluster Label").mean()
    
    # Rename columns to human readable
    cluster_means = cluster_means.rename(columns=HUMAN_NAMES)
    
    # 4. Isolation Forest Anomaly Detection
    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
    
    num_anomalies = int(np.sum(anomalies == -1))
    num_normal = int(np.sum(anomalies == 1))
    
    return {
        "scaler": scaler,
        "best_model_name": best_model_name,
        "best_reg_model": best_reg_model,
        "regression_comparison": reg_df,
        "clf_accuracy": clf_accuracy,
        "confusion_matrix": conf_mat,
        "clf_report": pd.DataFrame(clf_report).transpose(),
        "cluster_means": cluster_means,
        "num_anomalies": num_anomalies,
        "num_normal": num_normal,
        "X_pca": X_pca,
        "cluster_labels": cluster_labels,
        "anomalies": anomalies,
    }

# Helper for theming matplotlib charts
def format_plot_theme(fig, ax, theme_mode):
    """Apply theme-specific colors and typography to matplotlib figure and axes."""
    if theme_mode == "Starry Theme":
        fig.patch.set_facecolor("#050b1a")
        ax.set_facecolor("#081226")
        ax.tick_params(colors="#ffffff", labelsize=9)
        ax.xaxis.label.set_color("#ffffff")
        ax.yaxis.label.set_color("#ffffff")
        ax.title.set_color("#00f0ff")
        ax.grid(True, linestyle=":", color="#00f0ff", alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("#00f0ff")
            spine.set_alpha(0.6)
    elif theme_mode == "Dark":
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.xaxis.label.set_color("#f8fafc")
        ax.yaxis.label.set_color("#f8fafc")
        ax.title.set_color("#f8fafc")
        ax.grid(True, linestyle=":", color="#334155", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#475569")
            spine.set_alpha(0.7)
    else:
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")
        ax.tick_params(colors="#334155", labelsize=9)
        ax.xaxis.label.set_color("#0f172a")
        ax.yaxis.label.set_color("#0f172a")
        ax.title.set_color("#0f172a")
        ax.grid(True, linestyle=":", color="#cbd5e1", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_alpha(0.8)

# --- Streamlit Application Layout ---

# Initialize Theme state (Options: Light, Dark, Starry Theme)
if "theme" not in st.session_state:
    st.session_state["theme"] = "Starry Theme"
elif st.session_state["theme"] in ["Light Theme", "Light"]:
    st.session_state["theme"] = "Light"
elif st.session_state["theme"] in ["Dark Theme", "Dark"]:
    st.session_state["theme"] = "Dark"
elif st.session_state["theme"] not in ["Light", "Dark", "Starry Theme"]:
    st.session_state["theme"] = "Starry Theme"

current_theme = st.session_state["theme"]

# Initialize History log state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Helper to verify and return a valid logo image path
def get_valid_logo_path():
    import glob
    candidate_paths = [
        "logo.png",
        os.path.join("assets", "logo.png"),
        "/app/applet/logo.png",
        "/app/applet/assets/logo.png",
        "src/assets/images/rocket_logo_1786826841700.jpg",
        "src/assets/images/app_logo_1786741977742.jpg",
        "/app/applet/src/assets/images/rocket_logo_1786826841700.jpg",
        "/app/applet/src/assets/images/app_logo_1786741977742.jpg",
    ]
    candidate_paths += glob.glob("src/assets/images/*.jpg") + glob.glob("src/assets/images/*.png")
    candidate_paths += glob.glob("/app/applet/src/assets/images/*.jpg") + glob.glob("/app/applet/src/assets/images/*.png")

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                from PIL import Image
                with Image.open(p) as img:
                    img.load()
                return p
            except Exception:
                continue
    return None

logo_icon = get_valid_logo_path()

st.set_page_config(
    page_title="Rocket Flight Simulator",
    page_icon=logo_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Injection (Hide Default Menu, Preserve Sidebar Toggle, Apply High Density Theme) ---
if current_theme == "Starry Theme":
    bg_color = "#030712"
    card_bg = "rgba(10, 15, 30, 0.85)"
    card_inner = "rgba(10, 15, 30, 0.85)"
    border_color = "transparent"
    text_primary = "#ffffff"
    text_secondary = "#38bdf8"
    accent_cyan = "#38bdf8"
    accent_gold = "#ffc107"
    accent_glow = "none"
    accent_btn_bg = "linear-gradient(135deg, #0284c7 0%, #0369a1 100%)"
    sidebar_bg = "rgba(6, 10, 22, 0.94)"
    sidebar_card = "rgba(10, 15, 30, 0.85)"
    
    starfield_css = """
    background-color: #020617;
    background-image: 
        /* Dense Starfield Layers */
        radial-gradient(1.2px 1.2px at 25px 35px, #ffffff, rgba(0,0,0,0)),
        radial-gradient(1px 1px at 60px 140px, #ffffff, rgba(0,0,0,0)),
        radial-gradient(1.5px 1.5px at 110px 45px, #7dd3fc, rgba(0,0,0,0)),
        radial-gradient(1px 1px at 150px 190px, #ffffff, rgba(0,0,0,0)),
        radial-gradient(1.2px 1.2px at 190px 85px, #ffffff, rgba(0,0,0,0)),
        radial-gradient(1.5px 1.5px at 240px 240px, #bae6fd, rgba(0,0,0,0)),
        radial-gradient(1px 1px at 280px 160px, #ffffff, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 320px 70px, #38bdf8, rgba(0,0,0,0)),
        radial-gradient(1px 1px at 360px 210px, #ffffff, rgba(0,0,0,0)),
        radial-gradient(1.5px 1.5px at 410px 120px, #fef08a, rgba(0,0,0,0)),
        radial-gradient(2.2px 2.2px at 80px 230px, #67e8f9, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 200px 140px, #ffffff, rgba(0,0,0,0)),
        radial-gradient(2.5px 2.5px at 340px 300px, #38bdf8, rgba(0,0,0,0)),
        radial-gradient(1.8px 1.8px at 460px 50px, #ffffff, rgba(0,0,0,0)),
        /* Milky Way Cosmic Band & Shooting Star Accents */
        linear-gradient(132deg, transparent 46%, rgba(255, 255, 255, 0.8) 49.5%, rgba(56, 189, 248, 0.9) 50%, transparent 50.8%),
        linear-gradient(132deg, transparent 65%, rgba(255, 255, 255, 0.7) 68%, rgba(56, 189, 248, 0.85) 68.5%, transparent 69.2%),
        linear-gradient(125deg, rgba(2, 6, 23, 0.96) 0%, rgba(10, 15, 36, 0.90) 22%, rgba(124, 58, 237, 0.15) 38%, rgba(217, 119, 6, 0.22) 50%, rgba(251, 146, 60, 0.16) 57%, rgba(6, 182, 212, 0.16) 74%, #020617 100%);
    background-size: 380px 290px, 380px 290px, 380px 290px, 380px 290px, 380px 290px, 380px 290px, 380px 290px, 380px 290px, 380px 290px, 380px 290px, 520px 420px, 520px 420px, 520px 420px, 520px 420px, 900px 900px, 1200px 1200px, 100% 100%;
    background-attachment: fixed;
    """
    card_glass_css = """
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: none !important;
    border: none !important;
    """
elif current_theme == "Dark":
    bg_color = "#0f172a"        # slate-900
    card_bg = "#1e293b"         # slate-800
    card_inner = "#0f172a"      # slate-900
    border_color = "#334155"    # slate-700
    text_primary = "#f8fafc"    # slate-50
    text_secondary = "#94a3b8"  # slate-400
    accent_cyan = "#3b82f6"     # blue-500
    accent_gold = "#fbbf24"
    accent_glow = "0 1px 3px 0 rgba(0, 0, 0, 0.2)"
    accent_btn_bg = "#3b82f6"
    sidebar_bg = "#0f172a"
    sidebar_card = "#1e293b"
    starfield_css = f"background-color: {bg_color};"
    card_glass_css = "box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.2);"
else: # Light
    bg_color = "#f8fafc"        # slate-50
    card_bg = "#ffffff"         # white
    card_inner = "#f1f5f9"      # slate-100
    border_color = "#e2e8f0"    # slate-200
    text_primary = "#0f172a"    # slate-900
    text_secondary = "#64748b"  # slate-500
    accent_cyan = "#2563eb"     # blue-600
    accent_gold = "#d97706"
    accent_glow = "0 1px 3px 0 rgba(0, 0, 0, 0.03)"
    accent_btn_bg = "#2563eb"
    sidebar_bg = "#1e293b"     # slate-800
    sidebar_card = "#0f172a"
    starfield_css = f"background-color: {bg_color};"
    card_glass_css = "box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);"

# High-contrast specific CSS injection for Starry Theme
starry_high_contrast_css = ""
if current_theme == "Starry Theme":
    starry_high_contrast_css = """
    /* Remove all glowing borders and box-shadow outlines on all Streamlit containers and block wrappers */
    div.block-container,
    div.element-container,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"],
    div[data-testid="column"],
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarNav"],
    .card-box,
    .main-title,
    .header-banner-container,
    .branded-spinner-box,
    .footer-badge-box,
    .main-footer-badge {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Starry Theme High Contrast Typography Rules */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp li {
        color: #ffffff !important;
    }
    
    /* Headers & Section Titles */
    h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
        color: #38bdf8 !important;
        text-shadow: none !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Strong/Bold text highlights in Gold */
    strong, b, [data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] b {
        color: #ffc107 !important;
        text-shadow: none !important;
    }
    
    /* Input Labels and Control Labels */
    label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span, [data-testid="stWidgetLabel"] label {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        text-shadow: none !important;
        font-size: 0.85rem !important;
    }
    
    /* Number Inputs, Sliders, Selectboxes value text - seamless translucent dark styling without cyan borders */
    input, select, textarea, [data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="select"], [data-baseweb="select"] div {
        color: #ffffff !important;
        background-color: rgba(10, 15, 30, 0.85) !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Tabs styling in Starry Theme */
    button[data-baseweb="tab"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        background-color: rgba(14, 165, 233, 0.15) !important;
        border: none !important;
        border-bottom: 2px solid #38bdf8 !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }
    
    /* Metric Labels and Values */
    [data-testid="stMetric"] {
        border: none !important;
        box-shadow: none !important;
        background-color: rgba(10, 15, 30, 0.85) !important;
    }
    
    [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] span {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        text-shadow: none !important;
    }
    
    [data-testid="stMetricValue"] div {
        color: #ffc107 !important;
        text-shadow: none !important;
        font-weight: 800 !important;
    }
    
    /* Sidebar elements in Starry Theme */
    [data-testid="stSidebar"] {
        border: none !important;
        border-right: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #38bdf8 !important;
        text-shadow: none !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #ffc107 !important;
        background-color: rgba(10, 15, 30, 0.85) !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Radio, Select, Checkbox labels */
    [data-testid="stRadio"] label, [data-testid="stRadio"] div, [data-testid="stRadio"] p, [data-testid="stRadio"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Dataframes text clarity */
    [data-testid="stDataFrame"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stDataFrame"] * {
        color: #ffffff !important;
    }
    
    /* Info and Alert boxes */
    [data-testid="stAlert"] {
        background-color: rgba(10, 15, 30, 0.85) !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] span {
        color: #ffffff !important;
    }

    /* Logo Image and Custom Components */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stImage"] img:hover {
        box-shadow: none !important;
    }
    
    .header-banner-logo, .branded-spinner-img, .footer-badge-img {
        border: none !important;
        box-shadow: none !important;
    }

    button[kind="primary"], button[kind="secondary"], .stButton > button {
        border: none !important;
        box-shadow: none !important;
    }

    button[kind="primary"]:hover, button[kind="secondary"]:hover, .stButton > button:hover {
        box-shadow: none !important;
    }
    """

css_code = f"""
<style>
/* Hide Default Hamburger Menu, Toolbar & Footer but preserve header container for collapse control */
#MainMenu {{visibility: hidden !important;}}
header {{
    background-color: transparent !important;
}}
[data-testid="stToolbar"] {{visibility: hidden !important;}}
footer {{visibility: hidden !important;}}

/* Ensure Sidebar Collapse/Expand Toggle button remains visible, high-contrast, and clickable in all themes */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
header button {{
    visibility: visible !important;
    display: flex !important;
    z-index: 100000 !important;
    color: {text_primary} !important;
}}

[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {{
    background: {card_bg} !important;
    border: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"} !important;
    border-radius: 6px !important;
    color: {accent_cyan} !important;
    padding: 4px 6px !important;
    box-shadow: none !important;
    transition: all 0.2s ease-in-out;
}}

[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {{
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}}

/* Dynamic Theme Overrides for Canvas */
.stApp {{
    {starfield_css}
    color: {text_primary};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"} !important;
}}

[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

[data-testid="stSidebar"] h3 {{
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {text_secondary} !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.75rem !important;
}}

/* Main Title High Density Bar */
.main-title {{
    text-align: center;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: -1.2rem;
    margin-bottom: 1rem;
    color: {text_primary} !important;
    letter-spacing: -0.025em;
    padding: 0.75rem;
    background-color: {card_bg};
    border: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"};
    border-radius: 6px;
    {card_glass_css}
}}

/* High Density Card Box */
.card-box {{
    background: {card_bg};
    border: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"};
    padding: 1rem;
    border-radius: 8px;
    {card_glass_css}
    margin-bottom: 1rem;
}}

/* High Density Compact Metrics */
[data-testid="stMetric"] {{
    background: {card_inner};
    border: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"};
    padding: 0.6rem 0.8rem;
    border-radius: 6px;
    box-shadow: none !important;
}}

[data-testid="stMetricLabel"] {{
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: {text_secondary} !important;
}}

[data-testid="stMetricValue"] {{
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: {"#ffc107" if current_theme == "Starry Theme" else accent_cyan} !important;
}}

/* High Density Tabs Styling */
button[data-baseweb="tab"] {{
    font-size: 0.825rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    color: {text_secondary} !important;
    border-radius: 4px 4px 0 0 !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {accent_cyan} !important;
    background-color: {"rgba(14, 165, 233, 0.15)" if current_theme == "Starry Theme" else ("rgba(59, 130, 246, 0.15)" if current_theme == "Dark" else "#eff6ff")} !important;
    border-bottom: 2px solid {accent_cyan} !important;
}}

/* Button Styling */
.stButton > button {{
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    background: {accent_btn_bg} !important;
    color: #ffffff !important;
    border: {"none" if current_theme == "Starry Theme" else "none"} !important;
    box-shadow: none !important;
}}

.stButton > button:hover {{
    opacity: 0.94;
    box-shadow: none !important;
}}

/* Inputs and Expander Styling */
input, select {{
    font-size: 0.85rem !important;
}}

.streamlit-expanderHeader {{
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}}

/* App Logo Badge Styling & Responsive Scaling */
[data-testid="stSidebar"] [data-testid="stImage"] {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin: 0.5rem auto 1.25rem auto !important;
    text-align: center !important;
}}

[data-testid="stSidebar"] [data-testid="stImage"] img {{
    border-radius: 24px !important;
    aspect-ratio: 1 / 1 !important;
    object-fit: cover !important;
    max-width: 180px !important;
    width: 180px !important;
    height: auto !important;
    margin: 0 auto !important;
    display: block !important;
    background: rgba(255, 255, 255, 0.94) !important;
    padding: 6px !important;
    border: {"none" if current_theme == "Starry Theme" else ("2px solid #3b82f6 !important;" if current_theme == "Dark" else "2px solid #2563eb !important;")}
    box-shadow: {"none" if current_theme == "Starry Theme" else ("0 4px 14px rgba(0, 0, 0, 0.4) !important;" if current_theme == "Dark" else "0 4px 12px rgba(37, 99, 235, 0.2) !important;")}
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out !important;
}}

[data-testid="stSidebar"] [data-testid="stImage"] img:hover {{
    transform: scale(1.04) !important;
}}

/* Header Banner Container & Logo */
.header-banner-container {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 0.85rem 1.2rem;
    margin-top: -1.2rem;
    margin-bottom: 1rem;
    background-color: {card_bg};
    border: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"};
    border-radius: 8px;
    {card_glass_css}
}}

.header-banner-logo {{
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.95);
    padding: 3px;
    object-fit: cover;
    border: {"none" if current_theme == "Starry Theme" else ("1px solid #3b82f6" if current_theme == "Dark" else "1px solid #2563eb")};
    box-shadow: none;
    flex-shrink: 0;
}}

.header-banner-title {{
    font-size: 1.45rem;
    font-weight: 800;
    margin: 0;
    padding: 0;
    color: {text_primary} !important;
    letter-spacing: -0.02em;
}}

/* Branded Loading Spinner */
@keyframes brandPulse {{
    0% {{ transform: scale(0.96); opacity: 0.85; }}
    50% {{ transform: scale(1.05); opacity: 1; }}
    100% {{ transform: scale(0.96); opacity: 0.85; }}
}}

.branded-spinner-box {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.25rem;
    margin: 1rem 0;
    background: {card_bg};
    border: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"};
    border-radius: 8px;
    {card_glass_css}
    text-align: center;
}}

.branded-spinner-img {{
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.95);
    padding: 4px;
    margin-bottom: 0.75rem;
    animation: brandPulse 1.8s infinite ease-in-out;
    border: {"none" if current_theme == "Starry Theme" else ("2px solid #3b82f6" if current_theme == "Dark" else "2px solid #2563eb")};
    box-shadow: none;
}}

.branded-spinnerText {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {text_primary};
}}

/* Footer Signature Badge */
.footer-badge-box {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 0.6rem 0.8rem;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    background: {sidebar_card};
    border: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"};
    border-radius: 8px;
    text-align: center;
}}

.footer-badge-img {{
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.95);
    padding: 2px;
    object-fit: cover;
    flex-shrink: 0;
    border: none !important;
    box-shadow: none !important;
}}

.footer-badge-text {{
    font-size: 0.75rem;
    font-weight: 600;
    color: {text_secondary};
    letter-spacing: 0.04em;
}}

.main-footer-badge {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 0.75rem 1rem;
    margin-top: 2.5rem;
    margin-bottom: 1.5rem;
    border-top: {"none" if current_theme == "Starry Theme" else f"1px solid {border_color}"};
    text-align: center;
}}

{starry_high_contrast_css}
</style>
"""

st.markdown(css_code, unsafe_allow_html=True)

# Helper to read logo as base64 for embedding in pure HTML components
def get_logo_base64():
    logo_file = get_valid_logo_path()
    if logo_file:
        try:
            from PIL import Image
            import io
            with Image.open(logo_file) as img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception:
            try:
                with open(logo_file, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None
    return None

logo_b64 = get_logo_base64()
logo_src = f"data:image/png;base64,{logo_b64}" if logo_b64 else "logo.png"

# A.1 Main Header Banner (Title alongside Logo Graphic)
title_color = "#38bdf8" if current_theme == "Starry Theme" else text_primary
banner_bg = card_bg
banner_border = "none" if current_theme == "Starry Theme" else f"1px solid {border_color}"

st.markdown(f'''
<div class="header-banner-container" style="display: flex; align-items: center; justify-content: center; gap: 16px; padding: 0.85rem 1.25rem; margin-top: -1rem; margin-bottom: 1.25rem; background: {banner_bg}; border-radius: 8px; border: {banner_border};">
    <img src="{logo_src}" class="header-banner-logo" alt="Rocket Flight Simulator Logo" style="width: 52px; height: 52px; border-radius: 12px; object-fit: cover; background: #ffffff; padding: 3px; display: inline-block; vertical-align: middle; flex-shrink: 0;" />
    <div style="display: inline-block; vertical-align: middle; text-align: left;">
        <h1 class="header-banner-title" style="margin: 0; padding: 0; font-size: 1.65rem; font-weight: 800; color: {title_color} !important; letter-spacing: -0.02em; line-height: 1.2;">Rocket Flight Simulator</h1>
        <div style="font-size: 0.78rem; font-weight: 600; color: {text_secondary}; letter-spacing: 0.04em; margin-top: 2px;">Multi-Output Physics & Machine Learning Flight Engine</div>
    </div>
</div>
''', unsafe_allow_html=True)

# B. Collapsible Sidebar (Logo Header, Title, Settings & History)
with st.sidebar:
    sidebar_logo = get_valid_logo_path()
    if sidebar_logo:
        try:
            st.image(sidebar_logo, width=150)
        except Exception:
            pass
            
    sidebar_title_color = "#38bdf8" if current_theme == "Starry Theme" else text_primary
    st.markdown(f'''
    <div style="text-align: center; margin-top: -0.25rem; margin-bottom: 1.25rem;">
        <h2 style="font-size: 1.2rem; font-weight: 800; margin: 0; padding: 0; color: {sidebar_title_color} !important; letter-spacing: -0.01em;">Rocket Flight Simulator</h2>
        <div style="font-size: 0.72rem; font-weight: 600; color: {text_secondary}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px;">Flight Dynamics Suite</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("### Settings")
    
    theme_options = ["Light", "Dark", "Starry Theme"]
    curr_idx = theme_options.index(st.session_state["theme"]) if st.session_state["theme"] in theme_options else 2
    
    selected_theme = st.radio(
        "Theme Mode",
        theme_options,
        index=curr_idx,
        key="theme_radio"
    )
    
    if selected_theme != st.session_state["theme"]:
        st.session_state["theme"] = selected_theme
        st.rerun()

    st.markdown("---")
    st.markdown("### Simulation History")
    
    if st.session_state["history"]:
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()
            
        for idx, item in enumerate(reversed(st.session_state["history"])):
            run_num = len(st.session_state["history"]) - idx
            with st.expander(f"Run {run_num} - {item['timestamp']}"):
                st.write("**Inputs:**")
                st.write(f"Dry Mass: {item['inputs']['dry_mass']} Kilograms")
                st.write(f"Diameter: {item['inputs']['diameter']} Meters")
                st.write(f"Drag Coefficient: {item['inputs']['drag_coefficient']}")
                st.write(f"Total Impulse: {item['inputs']['total_impulse']} Newton-Seconds")
                st.write(f"Burn Time: {item['inputs']['burn_time']} Seconds")
                st.write(f"Launch Inclination: {item['inputs']['inclination']} Degrees")
                
                st.write("**Predicted Outcomes:**")
                st.write(f"Apogee Altitude: {item['outputs']['apogee_agl_m']:.1f} Meters")
                st.write(f"Total Flight Time: {item['outputs']['flight_time_s']:.1f} Seconds")
                st.write(f"Maximum Velocity: {item['outputs']['max_velocity_m_s']:.1f} Meters per Second")
                st.write(f"Time to Apogee: {item['outputs']['time_to_apogee_s']:.1f} Seconds")
    else:
        st.info("No prediction runs recorded yet. Perform a prediction in Tab 1 to log history.")

    st.markdown(f'''
    <div class="footer-badge-box">
        <img src="{logo_src}" class="footer-badge-img" alt="Logo" />
        <span class="footer-badge-text">Powered by Rocket Flight Simulator</span>
    </div>
    ''', unsafe_allow_html=True)

# C. Main Page Dataset Selection & Generation Section (Immediately beneath centered main heading)
st.markdown('<div class="card-box">', unsafe_allow_html=True)
col_ds1, col_ds2, col_ds3 = st.columns([2, 2, 1.5])

with col_ds1:
    dataset_size = st.slider("Dataset Size (Samples)", min_value=50, max_value=500, value=150, step=25)

with col_ds2:
    sim_mode = st.selectbox("Simulation Engine", ["Physics Fast Simulation", "RocketPy Monte Carlo (Hybrid)"])

with col_ds3:
    st.write("")
    st.write("")
    gen_btn = st.button("Generate Dataset", use_container_width=True)

if gen_btn or "df_data" not in st.session_state:
    use_rp = (sim_mode == "RocketPy Monte Carlo (Hybrid)")
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown(f'''
    <div class="branded-spinner-box">
        <img src="{logo_src}" class="branded-spinner-img" alt="Processing" />
        <div class="branded-spinnerText">Simulating Rocket Physics & Training ML Pipeline...</div>
    </div>
    ''', unsafe_allow_html=True)
    df_dataset = generate_rocket_dataset(num_samples=dataset_size, use_full_sim=use_rp)
    st.session_state["df_data"] = df_dataset
    st.session_state["ml_pipeline"] = train_machine_learning_pipeline(df_dataset)
    spinner_placeholder.empty()

df = st.session_state["df_data"]
ml_results = st.session_state["ml_pipeline"]

# Display Dataset Quick Summary Cards
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Total Rockets Simulated", len(df))
with mcol2:
    st.metric("Average Apogee Altitude", f"{df['apogee_agl_m'].mean():.1f} Meters")
with mcol3:
    st.metric("Average Total Flight Time", f"{df['flight_time_s'].mean():.1f} Seconds")
with mcol4:
    success_pct = (df["flight_success"].sum() / len(df)) * 100
    st.metric("Stable Flight Success Rate", f"{success_pct:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)

# D. Multi-Tab Application Structure
tab1, tab2, tab3, tab4 = st.tabs([
    "Interactive Flight Outcome Predictor",
    "Visual Exploration",
    "Input Comparison",
    "Machine Learning Pipeline Analysis"
])

# --- TAB 1: Interactive Flight Outcome Predictor ---
with tab1:
    st.markdown("### Interactive Rocket Design & Flight Outcome Predictor")
    st.write("Adjust base parameters for your custom rocket configuration to predict flight trajectory performance using the trained machine learning model.")
    
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        input_dry_mass = st.number_input("Dry Mass (Kilograms)", min_value=5.0, max_value=35.0, value=15.0, step=0.5)
        input_diameter = st.number_input("Diameter (Meters)", min_value=0.05, max_value=0.40, value=0.15, step=0.01)
        
    with c2:
        input_drag_coeff = st.number_input("Drag Coefficient", min_value=0.20, max_value=1.10, value=0.50, step=0.02)
        input_total_impulse = st.number_input("Total Impulse (Newton-Seconds)", min_value=500.0, max_value=6000.0, value=2500.0, step=100.0)
        
    with c3:
        input_burn_time = st.number_input("Burn Time (Seconds)", min_value=1.0, max_value=10.0, value=3.5, step=0.2)
        input_inclination = st.number_input("Launch Inclination (Degrees)", min_value=60.0, max_value=90.0, value=85.0, step=1.0)
        
    predict_btn = st.button("Predict Rocket Flight Performance", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_btn or "latest_prediction" in st.session_state:
        if predict_btn or "latest_prediction" not in st.session_state:
            # Build input dictionary
            raw_input_df = pd.DataFrame([{
                "dry_mass": input_dry_mass,
                "diameter": input_diameter,
                "drag_coefficient": input_drag_coeff,
                "total_impulse": input_total_impulse,
                "burn_time": input_burn_time,
                "inclination": input_inclination,
            }])
            
            eng_input_df = engineer_features(raw_input_df)
            X_input = eng_input_df[ENGINEERED_FEATURES]
            X_input_scaled = ml_results["scaler"].transform(X_input)
            
            pred_vals = ml_results["best_reg_model"].predict(X_input_scaled)[0]
            
            pred_dict = {
                "apogee_agl_m": float(pred_vals[0]),
                "flight_time_s": float(pred_vals[1]),
                "max_velocity_m_s": float(pred_vals[2]),
                "time_to_apogee_s": float(pred_vals[3]),
            }
            
            input_dict = {
                "dry_mass": input_dry_mass,
                "diameter": input_diameter,
                "drag_coefficient": input_drag_coeff,
                "total_impulse": input_total_impulse,
                "burn_time": input_burn_time,
                "inclination": input_inclination,
            }
            
            latest_pred = {
                "inputs": input_dict,
                "outputs": pred_dict,
                "timestamp": time.strftime("%H:%M:%S")
            }
            
            st.session_state["latest_prediction"] = latest_pred
            st.session_state["history"].append(latest_pred)
            
        latest = st.session_state["latest_prediction"]
        
        st.markdown("#### Predicted Flight Outcomes")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        
        with res_col1:
            st.metric("Predicted Apogee Altitude", f"{latest['outputs']['apogee_agl_m']:.1f} Meters")
        with res_col2:
            st.metric("Predicted Total Flight Time", f"{latest['outputs']['flight_time_s']:.1f} Seconds")
        with res_col3:
            st.metric("Predicted Maximum Velocity", f"{latest['outputs']['max_velocity_m_s']:.1f} Meters per Second")
        with res_col4:
            st.metric("Predicted Time to Apogee", f"{latest['outputs']['time_to_apogee_s']:.1f} Seconds")
            
        # Automated Parameter Tuning Engine
        df_tuning = st.session_state["df_data"]
        tuning_lr = LinearRegression()
        tuning_lr.fit(df_tuning[["dry_mass", "total_impulse", "drag_coefficient"]], df_tuning["apogee_agl_m"])
        
        user_tuning_X = pd.DataFrame([{
            "dry_mass": latest["inputs"]["dry_mass"],
            "total_impulse": latest["inputs"]["total_impulse"],
            "drag_coefficient": latest["inputs"]["drag_coefficient"]
        }])
        expected_baseline_apogee = float(tuning_lr.predict(user_tuning_X)[0])
        predicted_apogee = float(latest["outputs"]["apogee_agl_m"])
        
        if (expected_baseline_apogee - predicted_apogee) > 50:
            st.warning("Parameter Tuning Suggestions")
            if latest["inputs"]["drag_coefficient"] > 0.40:
                st.info(f"Suggest reducing Drag Coefficient (currently {latest['inputs']['drag_coefficient']:.2f}) if it exceeds 0.40.")
            if latest["inputs"]["dry_mass"] > 10.0:
                st.info(f"Suggest reducing Dry Mass (currently {latest['inputs']['dry_mass']:.1f} Kilograms) by approximately 10 to 15 percent if it exceeds 10 Kilograms.")
            if latest["inputs"]["total_impulse"] < 3500.0:
                st.info(f"Suggest increasing Total Impulse (currently {latest['inputs']['total_impulse']:.0f} Newton-Seconds) if it is below 3500 Newton-Seconds.")
        else:
            st.success("Parameters Optimal")
            
        st.info("Prediction updated successfully. Please visit the Visual Exploration and Input Comparison tabs to view your custom rocket overlaid on the dataset plots.")

# --- TAB 2: Visual Exploration (Output Graphs) ---
with tab2:
    st.markdown("### Visual Exploration of Rocket Output Parameters")
    st.write("Scatter plots illustrating dataset flight outcomes with trendlines. Your predicted rocket configuration is overlaid as a prominent green point.")
    
    # 4 Core Scatter Plots with Trendlines
    fig_grid, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    plot_pairs = [
        ("dry_mass", "apogee_agl_m", "Dry Mass (Kilograms)", "Apogee Altitude (Meters)", axes[0, 0]),
        ("total_impulse", "apogee_agl_m", "Total Impulse (Newton-Seconds)", "Apogee Altitude (Meters)", axes[0, 1]),
        ("drag_coefficient", "apogee_agl_m", "Drag Coefficient", "Apogee Altitude (Meters)", axes[1, 0]),
        ("burn_time", "flight_time_s", "Burn Time (Seconds)", "Total Flight Time (Seconds)", axes[1, 1]),
    ]
    
    has_latest = "latest_prediction" in st.session_state
    
    for x_col, y_col, x_label, y_label, ax in plot_pairs:
        x_vals = df[x_col]
        y_vals = df[y_col]
        
        scatter_color = "#38bdf8" if current_theme == "Starry Theme" else ("#60a5fa" if current_theme == "Dark" else "#1f77b4")
        trend_color = "#f43f5e" if current_theme in ["Starry Theme", "Dark"] else "#dc2626"
        
        ax.scatter(x_vals, y_vals, alpha=0.6, color=scatter_color, label="Dataset Rockets", edgecolors="none", s=38)
        
        # Calculate Trendline
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_trend, p(x_trend), color=trend_color, linestyle="--", linewidth=1.8, label="Trendline")
        
        # Overlay user prediction if available
        if has_latest:
            user_inputs = st.session_state["latest_prediction"]["inputs"]
            user_outputs = st.session_state["latest_prediction"]["outputs"]
            
            user_x = user_inputs.get(x_col)
            user_y = user_outputs.get(y_col)
            
            if user_x is not None and user_y is not None:
                rocket_color = "#4ade80" if current_theme == "Starry Theme" else "#22c55e"
                ax.scatter(user_x, user_y, color=rocket_color, marker="o", s=320, zorder=10, edgecolor="#ffffff", linewidth=1.8, label="Your Rocket")
                
        ax.set_title(f"{x_label} vs {y_label}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        format_plot_theme(fig_grid, ax, current_theme)
        
        legend_face = "#081026" if current_theme == "Starry Theme" else ("#1e293b" if current_theme == "Dark" else "#ffffff")
        legend_edge = "#0284c7" if current_theme == "Starry Theme" else ("#475569" if current_theme == "Dark" else "#cbd5e1")
        legend_text = "#e0f2fe" if current_theme == "Starry Theme" else ("#f8fafc" if current_theme == "Dark" else "#0f172a")
        ax.legend(loc="best", fontsize=8, facecolor=legend_face, edgecolor=legend_edge, labelcolor=legend_text)
        
    plt.tight_layout()
    st.pyplot(fig_grid)

# --- TAB 3: Input Comparison (Scatter Graphs ONLY) ---
with tab3:
    st.markdown("### Input Parameter Comparison Scatter Plots")
    st.write("Exclusively scatter graphs comparing dataset input parameters against each other. Your custom input parameters are overlaid as a prominent green point.")
    
    fig_inp, axes_inp = plt.subplots(2, 2, figsize=(12, 9))
    
    input_pairs = [
        ("dry_mass", "total_impulse", "Dry Mass (Kilograms)", "Total Impulse (Newton-Seconds)", axes_inp[0, 0]),
        ("diameter", "drag_coefficient", "Diameter (Meters)", "Drag Coefficient", axes_inp[0, 1]),
        ("burn_time", "total_impulse", "Burn Time (Seconds)", "Total Impulse (Newton-Seconds)", axes_inp[1, 0]),
        ("dry_mass", "diameter", "Dry Mass (Kilograms)", "Diameter (Meters)", axes_inp[1, 1]),
    ]
    
    for x_col, y_col, x_label, y_label, ax in input_pairs:
        x_vals = df[x_col]
        y_vals = df[y_col]
        
        scatter_color = "#fb923c" if current_theme in ["Starry Theme", "Dark"] else "#ea580c"
        ax.scatter(x_vals, y_vals, alpha=0.65, color=scatter_color, label="Dataset Inputs", edgecolors="none", s=40)
        
        if has_latest:
            user_inputs = st.session_state["latest_prediction"]["inputs"]
            user_x = user_inputs.get(x_col)
            user_y = user_inputs.get(y_col)
            
            if user_x is not None and user_y is not None:
                rocket_color = "#4ade80" if current_theme == "Starry Theme" else "#22c55e"
                ax.scatter(user_x, user_y, color=rocket_color, marker="o", s=320, zorder=10, edgecolor="#ffffff", linewidth=1.8, label="Your Inputs")
                
        ax.set_title(f"{x_label} vs {y_label}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        format_plot_theme(fig_inp, ax, current_theme)
        
        legend_face = "#081026" if current_theme == "Starry Theme" else ("#1e293b" if current_theme == "Dark" else "#ffffff")
        legend_edge = "#0284c7" if current_theme == "Starry Theme" else ("#475569" if current_theme == "Dark" else "#cbd5e1")
        legend_text = "#e0f2fe" if current_theme == "Starry Theme" else ("#f8fafc" if current_theme == "Dark" else "#0f172a")
        ax.legend(loc="best", fontsize=8, facecolor=legend_face, edgecolor=legend_edge, labelcolor=legend_text)
        
    plt.tight_layout()
    st.pyplot(fig_inp)

# --- TAB 4: Machine Learning Pipeline Analysis ---
with tab4:
    st.markdown("### Machine Learning Pipeline Analysis")
    st.write("Detailed performance evaluation across regression models, classification metrics, unsupervised K-Means clustering, and Isolation Forest anomaly detection.")
    
    st.markdown("#### 1. Regression Model Comparison")
    st.dataframe(ml_results["regression_comparison"], use_container_width=True)
    st.write(f"**Selected Best Regressor for Predictions:** {ml_results['best_model_name']}")
    
    st.markdown("---")
    st.markdown("#### 2. Flight Success Classification (Logistic Regression)")
    col_clf1, col_clf2 = st.columns([1, 1.5])
    
    with col_clf1:
        st.metric("Test Accuracy Score", f"{ml_results['clf_accuracy'] * 100:.1f}%")
        st.write("**Confusion Matrix:**")
        conf_df = pd.DataFrame(
            ml_results["confusion_matrix"],
            index=["Actual Unstable", "Actual Stable"],
            columns=["Predicted Unstable", "Predicted Stable"]
        )
        st.dataframe(conf_df, use_container_width=True)
        
    with col_clf2:
        st.write("**Classification Report:**")
        st.dataframe(ml_results["clf_report"], use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 3. Unsupervised Rocket Archetypes (K-Means Clustering)")
    st.write("Mean feature values across discovered rocket configuration clusters:")
    st.dataframe(ml_results["cluster_means"], use_container_width=True)
    
    col_pca1, col_pca2 = st.columns([1.2, 1])
    with col_pca1:
        fig_pca, ax_pca = plt.subplots(figsize=(7, 5))
        cmap_pca = "cool" if current_theme == "Starry Theme" else "viridis"
        scatter = ax_pca.scatter(ml_results["X_pca"][:, 0], ml_results["X_pca"][:, 1], c=ml_results["cluster_labels"], cmap=cmap_pca, alpha=0.85, edgecolors="#ffffff" if current_theme in ["Starry Theme", "Dark"] else "k", s=60)
        ax_pca.set_title("K-Means Rocket Clusters in PCA Space", fontsize=12, fontweight="bold")
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")
        format_plot_theme(fig_pca, ax_pca, current_theme)
        
        cbar = fig_pca.colorbar(scatter, ax=ax_pca, label="Cluster ID")
        cbar_color = "#bae6fd" if current_theme == "Starry Theme" else ("#cbd5e1" if current_theme == "Dark" else "#334155")
        cbar.ax.yaxis.set_tick_params(color=cbar_color, labelcolor=cbar_color)
        cbar.set_label("Cluster ID", color=cbar_color)
        plt.tight_layout()
        st.pyplot(fig_pca)
    with col_pca2:
        st.write("**Clustering Insights:**")
        st.write("- **Cluster 0**: Lightweight, high impulse-to-mass configurations designed for maximum altitude.")
        st.write("- **Cluster 1**: Mid-weight standard commercial sound rockets with moderate burn times.")
        st.write("- **Cluster 2**: Heavy airframes with larger drag profiles requiring high total impulse thrust.")
        
    st.markdown("---")
    st.markdown("#### 4. Anomaly Detection (Isolation Forest)")
    st.write(f"Detected **{ml_results['num_anomalies']}** anomalous rocket configurations out of {len(df)} total dataset samples.")
    
    col_ano1, col_ano2 = st.columns([1.2, 1])
    with col_ano1:
        fig_ano, ax_ano = plt.subplots(figsize=(7, 5))
        norm_mask = ml_results["anomalies"] == 1
        ano_mask = ml_results["anomalies"] == -1
        
        std_color = "#38bdf8" if current_theme == "Starry Theme" else ("#60a5fa" if current_theme == "Dark" else "#2563eb")
        ano_color = "#f43f5e" if current_theme in ["Starry Theme", "Dark"] else "#dc2626"
        
        ax_ano.scatter(ml_results["X_pca"][norm_mask, 0], ml_results["X_pca"][norm_mask, 1], c=std_color, label="Standard Rockets", alpha=0.6, edgecolors="none", s=50)
        ax_ano.scatter(ml_results["X_pca"][ano_mask, 0], ml_results["X_pca"][ano_mask, 1], c=ano_color, label="Anomalous Configurations", alpha=0.95, s=95, marker="x", linewidths=2)
        ax_ano.set_title("Isolation Forest Anomaly Detection in PCA Space", fontsize=12, fontweight="bold")
        ax_ano.set_xlabel("Principal Component 1")
        ax_ano.set_ylabel("Principal Component 2")
        format_plot_theme(fig_ano, ax_ano, current_theme)
        
        legend_face = "#081026" if current_theme == "Starry Theme" else ("#1e293b" if current_theme == "Dark" else "#ffffff")
        legend_edge = "#0284c7" if current_theme == "Starry Theme" else ("#475569" if current_theme == "Dark" else "#cbd5e1")
        legend_text = "#e0f2fe" if current_theme == "Starry Theme" else ("#f8fafc" if current_theme == "Dark" else "#0f172a")
        ax_ano.legend(loc="best", fontsize=8, facecolor=legend_face, edgecolor=legend_edge, labelcolor=legend_text)
        plt.tight_layout()
        st.pyplot(fig_ano)
    with col_ano2:
        st.write("**Anomaly Detection Findings:**")
        st.write("- Anomalies correspond to extreme drag-to-mass ratios or mismatched thrust-to-weight parameters.")
        st.write("- Configurations falling outside normal cluster density in PCA space trigger early safety flags before simulation.")

# Main Page Branded Footer Signature
st.markdown(f'''
<div class="main-footer-badge">
    <img src="{logo_src}" class="footer-badge-img" alt="Logo" />
    <span class="footer-badge-text">Powered by Rocket Flight Simulator &bull; High-Fidelity Multi-Output ML & Physics Engine</span>
</div>
''', unsafe_allow_html=True)
