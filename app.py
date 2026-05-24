# app.py - Pollution Hunter Streamlit Dashboard
# Works for BOTH private (src/) and public (root) repository structures

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Dynamically import PollutionHunter (works for both structures)
try:
    # Try private repo structure (src/inference_pipeline.py)
    sys.path.append(str(Path(__file__).parent))
    from src.inference_pipeline import PollutionHunter
except ImportError:
    # Try public repo structure (inference.py in root)
    from inference import PollutionHunter

# Page configuration
st.set_page_config(
    page_title="Pollution Hunter - HPIS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize model with caching
@st.cache_resource
def load_model():
    """Load the Pollution Hunter model (cached for performance)"""
    return PollutionHunter(verbose=False)

# Title
st.title("🌍 Pollution Hunter")
st.markdown("### Hyperlocal Air Quality Intelligence System (HPIS)")
st.markdown("---")

# Load model
with st.spinner("🔮 Loading AI Model..."):
    try:
        ph = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Sidebar inputs
with st.sidebar:
    st.header("📍 Location & Weather")
    
    # City selection
    cities = ph.get_available_cities()
    city = st.selectbox("🏙️ Select City", cities, index=cities.index('Islamabad') if 'Islamabad' in cities else 0)
    
    st.markdown("---")
    st.header("🌡️ Current Weather")
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (°C)", -5, 50, 25, help="Current temperature in Celsius")
        humidity = st.slider("Humidity (%)", 0, 100, 50, help="Relative humidity percentage")
    with col2:
        wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 5, help="Wind speed in meters per second")
        pressure = st.slider("Pressure (hPa)", 980, 1040, 1015, help="Atmospheric pressure in hectopascals")
    
    st.markdown("---")
    st.header("⏰ Time Information")
    
    current_hour = datetime.now().hour
    hour = st.slider("Hour (0-23)", 0, 23, current_hour, help="Time of day (24-hour format)")
    month = st.slider("Month (1-12)", 1, 12, 1, help="Month of the year")
    is_weekend = st.checkbox("Weekend", value=False, help="Is it weekend?")
    
    st.markdown("---")
    st.header("📊 Historical PM2.5 Values")
    st.caption("Previous pollution readings (μg/m³)")
    
    col1, col2 = st.columns(2)
    with col1:
        lag1 = st.number_input("1 hour ago", value=50.0, step=5.0, format="%.1f")
        lag2 = st.number_input("2 hours ago", value=48.0, step=5.0, format="%.1f")
        lag3 = st.number_input("3 hours ago", value=52.0, step=5.0, format="%.1f")
    with col2:
        lag6 = st.number_input("6 hours ago", value=55.0, step=5.0, format="%.1f")
        lag12 = st.number_input("12 hours ago", value=60.0, step=5.0, format="%.1f")
        lag24 = st.number_input("24 hours ago", value=65.0, step=5.0, format="%.1f")
    
    st.markdown("---")
    st.header("📈 Rolling Statistics")
    
    rolling_mean_3 = st.number_input("3-hour average", value=lag1, step=5.0, format="%.1f")
    rolling_mean_6 = st.number_input("6-hour average", value=lag2, step=5.0, format="%.1f")
    rolling_mean_12 = st.number_input("12-hour average", value=lag3, step=5.0, format="%.1f")
    rolling_std_6 = st.number_input("6-hour std deviation", value=5.0, step=1.0, format="%.1f")

# Main content - Predict button centered
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 PREDICT AIR QUALITY", use_container_width=True):
        with st.spinner("Analyzing environmental data..."):
            try:
                # Make prediction
                result = ph.predict(
                    city=city,
                    temperature=temperature,
                    humidity=humidity,
                    wind_speed=wind_speed,
                    pressure=pressure,
                    hour=hour,
                    month=month,
                    is_weekend=1 if is_weekend else 0,
                    lag1=lag1, lag2=lag2, lag3=lag3, lag6=lag6,
                    lag12=lag12, lag24=lag24,
                    rolling_mean_3=rolling_mean_3,
                    rolling_mean_6=rolling_mean_6,
                    rolling_mean_12=rolling_mean_12,
                    rolling_std_6=rolling_std_6
                )
                
                # Extract results
                pm25 = result['pm2.5']
                category = result['aqi_category']
                advice = result['health_advice']
                
                # Set color based on AQI category
                if category == "Good":
                    color = "#00e400"
                    text_color = "black"
                    emoji = "🟢"
                elif category == "Moderate":
                    color = "#ffff00"
                    text_color = "black"
                    emoji = "🟡"
                elif category == "Unhealthy for Sensitive":
                    color = "#ff7e00"
                    text_color = "white"
                    emoji = "🟠"
                elif category == "Unhealthy":
                    color = "#ff0000"
                    text_color = "white"
                    emoji = "🔴"
                else:
                    color = "#8f3f97"
                    text_color = "white"
                    emoji = "⚫"
                
                # Display AQI Gauge
                st.markdown("---")
                st.markdown(f"""
                <div style="background-color: {color}; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;">
                    <h1 style="margin: 0; color: {text_color};">{emoji} {category}</h1>
                    <h2 style="margin: 10px 0; color: {text_color};">{pm25} μg/m³</h2>
                    <p style="margin: 5px 0; color: {text_color}; font-size: 16px;">PM2.5 Concentration</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display health advice
                st.info(f"💡 **Health Recommendation:** {advice}")
                
                # Display input summary
                with st.expander("📋 View Input Parameters"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown("**📍 Location**")
                        st.write(f"City: {city}")
                        st.write(f"Weekend: {'Yes' if is_weekend else 'No'}")
                        st.write(f"Hour: {hour}:00")
                    
                    with col_b:
                        st.markdown("**🌡️ Weather**")
                        st.write(f"Temperature: {temperature}°C")
                        st.write(f"Humidity: {humidity}%")
                        st.write(f"Wind Speed: {wind_speed} m/s")
                        st.write(f"Pressure: {pressure} hPa")
                    
                    with col_c:
                        st.markdown("**📊 Historical Data**")
                        st.write(f"Last hour: {lag1} μg/m³")
                        st.write(f"6-hour avg: {rolling_mean_6} μg/m³")
                        st.write(f"24-hour trend: {lag24} μg/m³")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: gray;">
        🌍 <strong>Pollution Hunter</strong> | Powered by XGBoost (R² = {ph.get_model_info()['r2_score']:.4f})<br>
        Hyperlocal Pollution Intelligence System (HPIS) | 10 Pakistani Cities
    </div>
    """, 
    unsafe_allow_html=True
)

# Model info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ About")
    info = ph.get_model_info()
    st.markdown(f"""
    - **Model:** {info['model_type']}
    - **R² Score:** {info['r2_score']:.4f}
    - **Features:** {info['features_count']}
    - **Cities:** {info['cities_count']}
    """)
    
    st.markdown("---")
    st.markdown("### 📖 AQI Reference")
    st.markdown("""
    - 🟢 **0-15:** Good
    - 🟡 **15-35:** Moderate  
    - 🟠 **35-55:** Unhealthy for Sensitive
    - 🔴 **55-150:** Unhealthy
    - ⚫ **150+:** Hazardous
    """)