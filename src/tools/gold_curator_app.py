import streamlit as st
import os
import sys
import json
import pandas as pd
from PIL import Image
import random


# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import NuScenesLoader

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Semantic-Drive Gold Curator")

# --- PATHS ---
SOURCE_FILE = "output/consensus_final.jsonl"
MASTER_FILE = "output/gold_annotations_master.json"
NUSCENES_ROOT = "nuscenes_data"

# --- SCHEMA DEFINITIONS ---
SCHEMA = {
    "weather": ["clear", "overcast", "rain", "heavy_rain", "snow", "fog"],
    "time_of_day": ["day", "night", "dawn_dusk"],
    "lighting_condition": ["nominal", "glare_high", "shadow_contrast", "pitch_black", "streetlights_only"],
    "road_surface_friction": ["dry", "wet", "icy", "snowy", "muddy", "gravel"],
    "sensor_integrity": ["nominal", "lens_flare", "droplets_on_lens", "dirt_on_lens", "motion_blur", "sun_glare"],
    
    "scene_type": ["urban_street", "highway", "intersection", "highway_ramp", "parking_lot", "construction_zone", "rural_road"],
    "lane_configuration": ["straight", "curve", "merge_left", "merge_right", "roundabout", "intersection_4way", "intersection_t_junction"],
    "drivable_area_status": ["nominal", "restricted_by_static_obstacle", "blocked_by_dynamic_object"],
    "traffic_controls": ["green_light", "red_light", "yellow_light", "stop_sign", "yield_sign", "police_manual", "none"],
    
    "vru_status": ["none", "legal_crossing", "jaywalking_fast", "jaywalking_hesitant", "roadside_static", "cyclist_in_lane"],
    "lead_vehicle_behavior": ["none", "nominal", "braking_suddenly", "stalled", "turning"],
    "adjacent_vehicle_behavior": ["none", "nominal", "cutting_in_aggressive", "drifting", "tailgating"],
    "special_agent_class": ["none", "police_car", "ambulance", "fire_truck", "school_bus", "construction_machinery"],
    
    "primary_challenge": ["none", "occlusion_risk", "prediction_uncertainty", "violation_of_map_topology", "perception_degradation", "rule_violation"],
    "ego_required_action": ["lane_keep", "slow_down", "stop", "nudge_around_static_obstacle", "yield", "emergency_brake", "lane_change", "unprotected_turn"],
    "blocking_factor": ["none", "construction_barrier", "pedestrian", "vehicle", "debris", "flood"],
    
    "wod_e2e_tags": ["construction", "intersection_complex", "vru_hazard", "fod_debris", "weather_adverse", "special_vehicle", "lane_diversion", "sensor_failure"]
}

FILTERS = {
    "All Candidates": None,
    "High Risk (Score > 7)": lambda df: df[df['risk_score'] >= 7],
    "Construction Zones": lambda df: df[df['search_blob'].str.contains("construction|cone|drum|barrier")],
    "Adverse Weather": lambda df: df[df['search_blob'].str.contains("rain|wet|glare|fog|night")],
    "VRU Hazards": lambda df: df[df['search_blob'].str.contains("vru|pedestrian|cyclist|jaywalking")],
    "Special Vehicles": lambda df: df[df['search_blob'].str.contains("police|ambulance|bus|truck")],
    "Nominal/Clear": lambda df: df[df['risk_score'] < 3]
}

# --- CACHED RESOURCES ---
@st.cache_resource
def get_loader():
    return NuScenesLoader(dataroot=NUSCENES_ROOT, version="v1.0-trainval")

@st.cache_data
def load_candidates():
    data = []
    if not os.path.exists(SOURCE_FILE): return pd.DataFrame()
    
    with open(SOURCE_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                search_blob = json.dumps(item).lower()
                data.append({
                    "token": item['token'],
                    "search_blob": search_blob,
                    "risk_score": int(item.get('scenario_criticality', {}).get('risk_score', 0)),
                    "full_json": item
                })
            except: pass
    return pd.DataFrame(data)

def load_master():
    if os.path.exists(MASTER_FILE):
        with open(MASTER_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_master(data):
    with open(MASTER_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# --- STATE MANAGEMENT ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'gold_set' not in st.session_state: st.session_state.gold_set = load_master()
if 'df' not in st.session_state: st.session_state.df = load_candidates()
if 'current_subset' not in st.session_state: st.session_state.current_subset = []
if 'last_filter' not in st.session_state: st.session_state.last_filter = ""

import random

# --- SIDEBAR: NAVIGATION ---
st.sidebar.header("🔍 Dataset Navigation")

# Filter dropdown (preset filters)
filter_name = st.sidebar.selectbox("Filter Strategy", list(FILTERS.keys()))

# Custom search box (persistent in session_state)
if 'custom_query' not in st.session_state:
    st.session_state.custom_query = ""
custom_query = st.sidebar.text_input("Custom search", value=st.session_state.custom_query)

# Determine and update subset priority:
# - If custom_query is non-empty, use it (live search)
# - Else use selected preset filter (only update subset when filter_name changes)
if custom_query and custom_query.strip() != "":
    # update session value
    st.session_state.custom_query = custom_query
    # reset index when query changes
    if st.session_state.get('last_custom_query', None) != custom_query:
        st.session_state.idx = 0
    st.session_state.last_custom_query = custom_query

    q = custom_query.lower()
    # guard against missing df
    if 'df' in st.session_state and not st.session_state.df.empty:
        matched = st.session_state.df[st.session_state.df['search_blob'].str.contains(q, na=False)]
        st.session_state.current_subset = matched.to_dict('records')
    else:
        st.session_state.current_subset = []
    st.session_state.last_filter = f"custom:{q}"
else:
    # Clear any stored custom query state if empty
    st.session_state.custom_query = ""
    st.session_state.last_custom_query = ""

    # Only update when the preset filter actually changes (prevents resetting index every rerun)
    if filter_name != st.session_state.last_filter:
        st.session_state.idx = 0
        if FILTERS[filter_name]:
            if 'df' in st.session_state and not st.session_state.df.empty:
                try:
                    st.session_state.current_subset = FILTERS[filter_name](st.session_state.df).to_dict('records')
                except Exception:
                    st.session_state.current_subset = []
            else:
                st.session_state.current_subset = []
        else:
            st.session_state.current_subset = st.session_state.df.to_dict('records') if 'df' in st.session_state else []
        st.session_state.last_filter = filter_name

# Expose for main app
subset = st.session_state.get('current_subset', [])
total = len(subset)

# Navigation buttons (prev / index display / next)
col_nav1, col_nav2, col_nav3 = st.sidebar.columns([1, 2, 1])

if total > 0:
    if col_nav1.button("⬅️"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
    col_nav2.markdown(f"**{st.session_state.idx + 1} / {total}**")
    if col_nav3.button("➡️"):
        st.session_state.idx = min(total - 1, st.session_state.idx + 1)
else:
    # show 0/0 when empty
    col_nav2.markdown("**0 / 0**")

st.sidebar.markdown("---")

# Random jump
if st.sidebar.button("🎲 Random"):
    if total > 0:
        st.session_state.idx = random.randint(0, total - 1)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Verified Frames", len(st.session_state.gold_set))


# --- MAIN UI ---
if total == 0:
    st.warning("No candidates found for this filter.")
    st.stop()

item = subset[st.session_state.idx]
token = item['token']
pred = item['full_json']

# --- CHECK EXISTING WORK ---
is_verified = token in st.session_state.gold_set
if is_verified:
    st.success(f"✅ VERIFIED FRAME: {token}")
    record = st.session_state.gold_set[token]
else:
    st.info(f"⚠️ PENDING REVIEW: {token}")
    # Default to prediction
    record = {
        'odd_attributes': pred.get('odd_attributes', {}),
        'road_topology': pred.get('road_topology', {}),
        'key_interacting_agents': pred.get('key_interacting_agents', {}),
        'scenario_criticality': pred.get('scenario_criticality', {}),
        'wod_e2e_tags': pred.get('wod_e2e_tags', [])
    }

# --- VISUALS ---
loader = get_loader()
try:
    paths = loader.get_camera_paths(token)
    imgs = []
    for cam in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]:
        if cam in paths: imgs.append(Image.open(paths[cam]))
    
    st.image(imgs, caption=["Front Left", "Front Center", "Front Right"], width=350)
except Exception as e:
    st.error(f"Image Load Error: {e}")

# --- REASONING CONTEXT (NEW SECTION) ---
st.markdown("### 🧠 Model Reasoning & Verification")
col_desc, col_log = st.columns([2, 1])

with col_desc:
    st.caption("Judge's Semantic Description")
    st.info(f"\"{pred.get('description', 'No description provided.')}\"")

with col_log:
    st.caption("Symbolic Verification Log (Reward System)")
    logs = pred.get('judge_log', [])
    score = pred.get('judge_score', 0)
    
    st.metric("Confidence Score", f"{score}/10")
    
    if logs:
        for log in logs:
            if "✅" in log:
                st.markdown(f":green[{log}]")
            elif "❌" in log:
                st.markdown(f":red[{log}]")
            else:
                st.markdown(log)
    else:
        st.markdown("*No verification logs available.*")

# --- ANNOTATION FORM ---
with st.form("annotation_form"):
    st.subheader("📝 Verify Scenario DNA")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["A. ODD", "B. Topology", "C. Agents", "D. Causal", "E. Tags"])
    
    with tab1:
        c1, c2 = st.columns(2)
        weather = c1.selectbox("Weather", SCHEMA['weather'], index=SCHEMA['weather'].index(record['odd_attributes'].get('weather', 'clear')) if record['odd_attributes'].get('weather') in SCHEMA['weather'] else 0)
        time = c2.selectbox("Time", SCHEMA['time_of_day'], index=SCHEMA['time_of_day'].index(record['odd_attributes'].get('time_of_day', 'day')) if record['odd_attributes'].get('time_of_day') in SCHEMA['time_of_day'] else 0)
        light = c1.selectbox("Lighting", SCHEMA['lighting_condition'], index=SCHEMA['lighting_condition'].index(record['odd_attributes'].get('lighting_condition', 'nominal')) if record['odd_attributes'].get('lighting_condition') in SCHEMA['lighting_condition'] else 0)
        surface = c2.selectbox("Surface", SCHEMA['road_surface_friction'], index=SCHEMA['road_surface_friction'].index(record['odd_attributes'].get('road_surface_friction', 'dry')) if record['odd_attributes'].get('road_surface_friction') in SCHEMA['road_surface_friction'] else 0)
        sensor = st.selectbox("Sensor Integrity", SCHEMA['sensor_integrity'], index=SCHEMA['sensor_integrity'].index(record['odd_attributes'].get('sensor_integrity', 'nominal')) if record['odd_attributes'].get('sensor_integrity') in SCHEMA['sensor_integrity'] else 0)

    with tab2:
        c1, c2 = st.columns(2)
        scene = c1.selectbox("Scene Type", SCHEMA['scene_type'], index=SCHEMA['scene_type'].index(record['road_topology'].get('scene_type', 'urban_street')) if record['road_topology'].get('scene_type') in SCHEMA['scene_type'] else 0)
        lane = c2.selectbox("Lane Config", SCHEMA['lane_configuration'], index=SCHEMA['lane_configuration'].index(record['road_topology'].get('lane_configuration', 'straight')) if record['road_topology'].get('lane_configuration') in SCHEMA['lane_configuration'] else 0)
        drivable = st.selectbox("Drivable Status", SCHEMA['drivable_area_status'], index=SCHEMA['drivable_area_status'].index(record['road_topology'].get('drivable_area_status', 'nominal')) if record['road_topology'].get('drivable_area_status') in SCHEMA['drivable_area_status'] else 0)
        
        # Handle Multiselect defaults safely
        def_traffic = record['road_topology'].get('traffic_controls', [])
        if not isinstance(def_traffic, list): def_traffic = []
        valid_def_traffic = [t for t in def_traffic if t in SCHEMA['traffic_controls']]
        traffic = st.multiselect("Traffic Controls", SCHEMA['traffic_controls'], default=valid_def_traffic)

    with tab3:
        c1, c2 = st.columns(2)
        vru = c1.selectbox("VRU Status", SCHEMA['vru_status'], index=SCHEMA['vru_status'].index(record['key_interacting_agents'].get('vru_status', 'none')) if record['key_interacting_agents'].get('vru_status') in SCHEMA['vru_status'] else 0)
        lead = c2.selectbox("Lead Vehicle", SCHEMA['lead_vehicle_behavior'], index=SCHEMA['lead_vehicle_behavior'].index(record['key_interacting_agents'].get('lead_vehicle_behavior', 'none')) if record['key_interacting_agents'].get('lead_vehicle_behavior') in SCHEMA['lead_vehicle_behavior'] else 0)
        adj = c1.selectbox("Adj Vehicle", SCHEMA['adjacent_vehicle_behavior'], index=SCHEMA['adjacent_vehicle_behavior'].index(record['key_interacting_agents'].get('adjacent_vehicle_behavior', 'none')) if record['key_interacting_agents'].get('adjacent_vehicle_behavior') in SCHEMA['adjacent_vehicle_behavior'] else 0)
        special = c2.selectbox("Special Agent", SCHEMA['special_agent_class'], index=SCHEMA['special_agent_class'].index(record['key_interacting_agents'].get('special_agent_class', 'none')) if record['key_interacting_agents'].get('special_agent_class') in SCHEMA['special_agent_class'] else 0)

    with tab4:
        c1, c2 = st.columns(2)
        challenge = c1.selectbox("Challenge", SCHEMA['primary_challenge'], index=SCHEMA['primary_challenge'].index(record['scenario_criticality'].get('primary_challenge', 'none')) if record['scenario_criticality'].get('primary_challenge') in SCHEMA['primary_challenge'] else 0)
        action = c2.selectbox("Action", SCHEMA['ego_required_action'], index=SCHEMA['ego_required_action'].index(record['scenario_criticality'].get('ego_required_action', 'lane_keep')) if record['scenario_criticality'].get('ego_required_action') in SCHEMA['ego_required_action'] else 0)
        blocker = st.selectbox("Blocker", SCHEMA['blocking_factor'], index=SCHEMA['blocking_factor'].index(record['scenario_criticality'].get('blocking_factor', 'none')) if record['scenario_criticality'].get('blocking_factor') in SCHEMA['blocking_factor'] else 0)
        risk = st.slider("Risk Score", 0, 10, int(record['scenario_criticality'].get('risk_score', 0)))

    with tab5:
        def_tags = record.get('wod_e2e_tags', [])
        valid_def_tags = [t for t in def_tags if t in SCHEMA['wod_e2e_tags']]
        tags = st.multiselect("WOD-E2E Tags", SCHEMA['wod_e2e_tags'], default=valid_def_tags)

    # SUBMIT
    submitted = st.form_submit_button("💾 CONFIRM & SAVE", type="primary")
    
    if submitted:
        # Construct Record
        new_entry = {
            "token": token,
            "odd_attributes": {
                "weather": weather, "time_of_day": time, 
                "lighting_condition": light, "road_surface_friction": surface,
                "sensor_integrity": sensor
            },
            "road_topology": {
                "scene_type": scene, "lane_configuration": lane,
                "drivable_area_status": drivable, "traffic_controls": traffic
            },
            "key_interacting_agents": {
                "vru_status": vru, "lead_vehicle_behavior": lead,
                "adjacent_vehicle_behavior": adj, "special_agent_class": special
            },
            "scenario_criticality": {
                "primary_challenge": challenge, "ego_required_action": action,
                "blocking_factor": blocker, "risk_score": risk
            },
            "wod_e2e_tags": tags
        }
        
        # Save to Memory
        st.session_state.gold_set[token] = new_entry
        # Save to Disk
        save_master(st.session_state.gold_set)
        st.success("Saved! Moving to next...")
        
        # Auto Advance
        if st.session_state.idx < total - 1:
            st.session_state.idx += 1
            st.rerun()

# To run: streamlit run src/tools/gold_curator_app.py