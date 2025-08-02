import streamlit as st
import joblib
from utils import preprocess_input

# Load model
model = joblib.load("models/best_model_xgboost.pkl")

# Page configuration
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        color: #333;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        color: #333;
    }
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    /* Fix text color in white backgrounds */
    .metric-card h4, .metric-card p {
        color: #333 !important;
    }
    .info-box h3, .info-box h4, .info-box p, .info-box li {
        color: #333 !important;
    }
    /* Ensure form labels are visible */
    .stForm label {
        color: #333 !important;
        font-weight: 500;
    }
    /* Fix any other light text issues */
    .stMarkdown p {
        color: #333 !important;
    }
    /* Fix sidebar text visibility */
    .sidebar .stMarkdown {
        color: #000 !important;
    }
    .sidebar .stMarkdown h3, .sidebar .stMarkdown h4, .sidebar .stMarkdown p {
        color: #000 !important;
    }
    /* Ensure radio button labels are visible */
    .stRadio > label {
        color: #333 !important;
        font-weight: 500;
    }
    .stRadio > div > label {
        color: #333 !important;
    }
    /* Make radio button option labels much more visible */
    .stRadio > div > div > label {
        color: #000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    /* Ensure radio button text is black and bold */
    .stRadio div[data-baseweb="radio"] label {
        color: #000 !important;
        font-weight: 600 !important;
    }
    /* Target the specific radio button text */
    .stRadio span {
        color: #000 !important;
        font-weight: 600 !important;
    }
    /* Additional radio button styling for maximum visibility */
    .stRadio div[role="radiogroup"] label {
        color: #000 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    /* Make sure all radio button text is black and bold */
    .stRadio * {
        color: #000 !important;
    }
    .stRadio label {
        color: #000 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Health Insurance Cost Predictor</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">Predict your medical insurance charges with AI-powered accuracy</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional info
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        <h3>📊 Model Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; margin: 1rem 0; color: #000;">
        <h4 style="color: #000;">🎯 Model Performance</h4>
        <p style="color: #000;">• R² Score: 87.4%</p>
        <p style="color: #000;">• RMSE: $4,427</p>
        <p style="color: #000;">• Training Data: 1,338 records</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; margin: 1rem 0; color: #000;">
        <h4 style="color: #000;">📋 Model Details</h4>
        <p style="color: #000;">• Version: v1.0.0</p>
        <p style="color: #000;">• Algorithm: XGBoost</p>
        <p style="color: #000;">• Features: 6</p>
        <p style="color: #000;">• Last Updated: 2024-01-15</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; margin: 1rem 0; color: #000;">
        <h4 style="color: #000;">🔍 Key Factors</h4>
        <p style="color: #000;">• Smoking: Biggest impact</p>
        <p style="color: #000;">• Age: Moderate effect</p>
        <p style="color: #000;">• BMI: Health indicator</p>
        <p style="color: #000;">• Region: Geographic variation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; margin: 1rem 0; color: #000;">
        <h4 style="color: #000;">💡 Quick Tips</h4>
        <p style="color: #000;">• Smoking increases costs by 200-400%</p>
        <p style="color: #000;">• Age affects costs by 2-5% per year</p>
        <p style="color: #000;">• Higher BMI may increase premiums</p>
        <p style="color: #000;">• More children = higher costs</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Your Information")
    
    with st.form("prediction_form"):
        # Use session state to track form submissions
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        
        # Create two columns for form layout
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            st.markdown("#### 👤 Personal Details")
            age = st.slider("Age", 18, 65, 30, help="Select your age between 18 and 65")
            st.markdown("**Sex:**")
            sex = st.radio("", options=["male", "female"], horizontal=True, help="Select your gender", label_visibility="collapsed")
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1, 
                                help="Enter your BMI (10-60)")
        
        with form_col2:
            st.markdown("#### 🏥 Health & Location")
            children = st.selectbox("Number of Children", list(range(6)), 
                                  help="Select number of dependents (0-5)")
            st.markdown("**Smoker:**")
            smoker = st.radio("", options=["yes", "no"], horizontal=True, help="Are you a smoker?", label_visibility="collapsed")
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"],
                                help="Select your geographic region")
        
        # Submit button
        submit = st.form_submit_button("🚀 Predict Insurance Cost", use_container_width=True)

    # Prediction results
    if submit:
        input_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }

        try:
            processed_input = preprocess_input(input_data)
            prediction = model.predict(processed_input)[0]
            
            # Store prediction in session state
            st.session_state.last_prediction = {
                'input': input_data,
                'prediction': prediction
            }
            
            # Display result with enhanced styling
            st.markdown(f"""
            <div class="prediction-box">
                <h2>💰 Predicted Insurance Cost</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">${prediction:,.2f}</h1>
                <p style="font-size: 1.1rem;">Based on your provided information</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show comparison if we have a previous prediction
            if st.session_state.last_prediction and st.session_state.last_prediction['input'] != input_data:
                prev_prediction = st.session_state.last_prediction['prediction']
                diff = prediction - prev_prediction
                st.markdown(f"""
                <div class="info-box">
                    <h4>📊 Change from Last Prediction</h4>
                    <p style="font-size: 1.2rem; font-weight: bold; color: {'green' if diff > 0 else 'red'}">
                        ${diff:+,.2f} ({diff/prev_prediction*100:+.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

with col2:
    st.markdown("### 📊 Quick Insights")
    
    # Risk level indicator
    if submit and 'last_prediction' in st.session_state:
        prediction = st.session_state.last_prediction['prediction']
        
        if prediction < 5000:
            risk_level = "🟢 Low Risk"
            risk_color = "#28a745"
        elif prediction < 10000:
            risk_level = "🟡 Medium Risk"
            risk_color = "#ffc107"
        else:
            risk_level = "🔴 High Risk"
            risk_color = "#dc3545"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Risk Level</h4>
            <p style="color: {risk_color}; font-weight: bold; font-size: 1.2rem;">{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cost comparison
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #333;">💡 Cost Comparison</h4>
            <p style="color: #333;">• Low Risk: < $5,000</p>
            <p style="color: #333;">• Medium Risk: $5,000 - $10,000</p>
            <p style="color: #333;">• High Risk: > $10,000</p>
        </div>
        """, unsafe_allow_html=True)

# Footer with additional information
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h3 style="color: #333;">📚 About the Model</h3>
    <p style="color: #333;">This predictor uses an <strong>XGBoost machine learning model</strong> trained on comprehensive health insurance data to estimate medical charges based on:</p>
    <ul style="color: #333;">
        <li style="color: #333;"><strong>Age</strong>: 18-65 years (older age typically increases costs)</li>
        <li style="color: #333;"><strong>Sex</strong>: Male or Female (minimal impact on costs)</li>
        <li style="color: #333;"><strong>BMI</strong>: Body Mass Index 10-60 (higher BMI may increase costs)</li>
        <li style="color: #333;"><strong>Children</strong>: Number of dependents 0-5 (more children = higher costs)</li>
        <li style="color: #333;"><strong>Smoker</strong>: Yes or No (<strong>biggest impact</strong> on costs)</li>
        <li style="color: #333;"><strong>Region</strong>: Geographic location (regional cost variations)</li>
    </ul>
    <p style="color: #333;"><strong>💡 Tip:</strong> Smoking status has the most significant impact on insurance costs, typically increasing premiums by 200-400%.</p>
</div>
""", unsafe_allow_html=True)

