"""
Bank Telemarketing Success Predictor
A Streamlit application for real-time prediction of term deposit subscriptions.
Handles model loading, data preprocessing, and live predictions with confidence metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
import pickle

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Bank Telemarketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --danger-color: #d62728;
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    .main-header {
        font-size: 3em;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    
    .subtitle-text {
        font-size: 1.1em;
        color: #666;
        margin-bottom: 2em;
    }
    
    .prediction-success {
        background: linear-gradient(135deg, rgba(44, 160, 44, 0.1) 0%, rgba(44, 160, 44, 0.05) 100%);
        border-left: 4px solid #2ca02c;
        padding: 1.5em;
        border-radius: 8px;
        margin: 1em 0;
    }
    
    .prediction-danger {
        background: linear-gradient(135deg, rgba(214, 39, 40, 0.1) 0%, rgba(214, 39, 40, 0.05) 100%);
        border-left: 4px solid #d62728;
        padding: 1.5em;
        border-radius: 8px;
        margin: 1em 0;
    }
    
    .confidence-bar {
        margin: 1em 0;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.2em;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5em 0;
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 1em;
        border-radius: 4px;
        margin: 1em 0;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model with error handling."""
    try:
        model_path = Path(__file__).parent / 'random_forest_model.joblib'
        
        # Try the uploaded directory as fallback
        if not model_path.exists():
            model_path = Path('/mnt/user-data/uploads/random_forest_model.joblib')
        
        if not model_path.exists():
            return None, "Model file not found in expected locations."
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = joblib.load(model_path)
        
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_resource
def load_preprocessor():
    """Load the preprocessor if available."""
    try:
        preprocessor_path = Path(__file__).parent / 'preprocessor.joblib'
        
        if not preprocessor_path.exists():
            preprocessor_path = Path('/mnt/user-data/uploads/preprocessor.joblib')
        
        if not preprocessor_path.exists():
            return None
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            preprocessor = joblib.load(preprocessor_path)
        
        return preprocessor
    except Exception:
        return None

# ============================================================================
# DATA PREPARATION & FEATURE MAPPINGS
# ============================================================================

# Define mappings for ordinal features
EDUCATION_MAPPING = {
    'unknown': 0,
    'primary': 1,
    'secondary': 2,
    'tertiary': 3
}

MONTH_MAPPING = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

# Define possible values for categorical features
JOB_OPTIONS = [
    'admin.', 'technician', 'services', 'management', 'retired',
    'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
    'unknown', 'self-employed', 'student'
]

MARITAL_OPTIONS = ['married', 'single', 'divorced']
DEFAULT_OPTIONS = ['no', 'yes']
HOUSING_OPTIONS = ['yes', 'no']
LOAN_OPTIONS = ['no', 'yes']
CONTACT_OPTIONS = ['unknown', 'cellular', 'telephone']
POUTCOME_OPTIONS = ['unknown', 'failure', 'other', 'success']
MONTH_STR_OPTIONS = list(MONTH_MAPPING.keys())
EDUCATION_STR_OPTIONS = list(EDUCATION_MAPPING.keys())

# ============================================================================
# UI LAYOUT
# ============================================================================

# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">🏦 Bank Telemarketing Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Real-time prediction of term deposit subscription likelihood</p>', unsafe_allow_html=True)

# Initialize session state for real-time predictions
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

# ============================================================================
# SIDEBAR: INPUT COLLECTION
# ============================================================================

with st.sidebar:
    st.markdown("## 👤 Client Profile")
    
    # Numerical inputs
    age = st.slider("Age", 18, 95, 45, help="Client's age")
    balance = st.number_input(
        "Account Balance (€)",
        value=1500,
        min_value=-10000,
        max_value=100000,
        step=100,
        help="Annual account balance"
    )
    
    st.markdown("### 📞 Contact Details")
    day = st.slider("Day of Month", 1, 31, 15, help="Day of last contact")
    duration = st.number_input(
        "Last Contact Duration (seconds)",
        value=300,
        min_value=0,
        max_value=5000,
        step=30,
        help="Duration of last contact in seconds"
    )
    campaign = st.number_input(
        "Contacts This Campaign",
        value=1,
        min_value=1,
        max_value=100,
        step=1,
        help="Number of contacts during this campaign"
    )
    pdays = st.number_input(
        "Days Since Last Contact",
        value=-1,
        min_value=-1,
        max_value=1000,
        step=1,
        help="Days since previous contact (-1 if no previous contact)"
    )
    previous = st.number_input(
        "Previous Contacts",
        value=0,
        min_value=0,
        max_value=100,
        step=1,
        help="Number of previous contacts"
    )
    
    st.markdown("### 💼 Professional & Personal")
    job = st.selectbox("Job Title", JOB_OPTIONS, index=JOB_OPTIONS.index('management'))
    education_str = st.selectbox("Education Level", EDUCATION_STR_OPTIONS, index=EDUCATION_STR_OPTIONS.index('secondary'))
    marital = st.selectbox("Marital Status", MARITAL_OPTIONS, index=MARITAL_OPTIONS.index('married'))
    
    st.markdown("### 💳 Financial Status")
    default = st.selectbox("Credit in Default?", DEFAULT_OPTIONS, index=DEFAULT_OPTIONS.index('no'))
    housing = st.selectbox("Housing Loan?", HOUSING_OPTIONS, index=HOUSING_OPTIONS.index('no'))
    loan = st.selectbox("Personal Loan?", LOAN_OPTIONS, index=LOAN_OPTIONS.index('no'))
    
    st.markdown("### 📅 Campaign Details")
    month_str = st.selectbox("Month of Last Contact", MONTH_STR_OPTIONS, index=MONTH_STR_OPTIONS.index('may'))
    contact = st.selectbox("Contact Type", CONTACT_OPTIONS, index=CONTACT_OPTIONS.index('cellular'))
    poutcome = st.selectbox("Previous Campaign Outcome", POUTCOME_OPTIONS, index=POUTCOME_OPTIONS.index('unknown'))

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Create DataFrame from user inputs
input_data_df = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education_str,
    'default': default,
    'balance': balance,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'day': day,
    'month': month_str,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
}])

# Apply mappings
input_data_df['education'] = input_data_df['education'].map(EDUCATION_MAPPING)
input_data_df['month'] = input_data_df['month'].map(MONTH_MAPPING)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Load model
rf_pipeline, model_error = load_model()
preprocessor = load_preprocessor()

if model_error:
    st.error(f"❌ {model_error}")
    st.stop()

if rf_pipeline is None:
    st.error("❌ Model failed to load. Please ensure 'random_forest_model.joblib' is in the directory.")
    st.stop()

# Create two columns: Prediction & Analytics
col1, col2 = st.columns([1.5, 1])

# ============================================================================
# LEFT COLUMN: PREDICTION RESULT
# ============================================================================
with col1:
    st.markdown("## 🎯 Prediction Result")
    
    # Predict button
    if st.button("🔮 Generate Prediction", use_container_width=True, key="predict_btn"):
        st.session_state.show_prediction = True
    
    if st.session_state.show_prediction:
        try:
            with st.spinner("Analyzing client profile..."):
                prediction = rf_pipeline.predict(input_data_df)
                prediction_proba = rf_pipeline.predict_proba(input_data_df)
            
            prob_no = prediction_proba[0][0]
            prob_yes = prediction_proba[0][1]
            
            # Display result with styling
            if prediction[0] == 1:
                st.markdown("""
                <div class="prediction-success">
                    <h3 style="color: #2ca02c; margin-bottom: 0.5em;">✅ Likely to Subscribe</h3>
                    <p style="font-size: 1.1em; color: #333;">
                        This client has a <strong>high probability</strong> of subscribing to a term deposit.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-danger">
                    <h3 style="color: #d62728; margin-bottom: 0.5em;">❌ Unlikely to Subscribe</h3>
                    <p style="font-size: 1.1em; color: #333;">
                        This client has a <strong>low probability</strong> of subscribing to a term deposit.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence scores
            st.markdown("### 📊 Confidence Scores")
            
            col_yes, col_no = st.columns(2)
            
            with col_yes:
                st.metric(
                    "Probability of 'Yes'",
                    f"{prob_yes*100:.1f}%",
                    delta=f"{prob_yes:.4f}"
                )
                st.progress(prob_yes, text="Yes")
            
            with col_no:
                st.metric(
                    "Probability of 'No'",
                    f"{prob_no*100:.1f}%",
                    delta=f"{prob_no:.4f}"
                )
                st.progress(prob_no, text="No")
            
            # Model confidence
            confidence = max(prob_yes, prob_no)
            if confidence >= 0.75:
                confidence_level = "🟢 Very High"
                confidence_color = "#2ca02c"
            elif confidence >= 0.65:
                confidence_level = "🟡 High"
                confidence_color = "#ff7f0e"
            else:
                confidence_level = "🔴 Moderate"
                confidence_color = "#d62728"
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1em; border-radius: 8px; border-left: 4px solid {confidence_color};">
                <strong>Model Confidence:</strong> {confidence_level} ({confidence*100:.1f}%)
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# ============================================================================
# RIGHT COLUMN: CLIENT PROFILE SUMMARY
# ============================================================================
with col2:
    st.markdown("## 👥 Client Summary")
    
    # Key metrics
    st.metric("Age", f"{age} years")
    st.metric("Account Balance", f"€{balance:,.0f}")
    st.metric("Contact Duration", f"{duration} sec")
    st.metric("Campaign Contacts", campaign)
    
    # Client characteristics
    st.markdown("### 📋 Key Details")
    details_df = pd.DataFrame({
        'Attribute': ['Job', 'Marital', 'Education', 'Contact Type'],
        'Value': [job, marital, education_str, contact]
    })
    st.table(details_df)

# ============================================================================
# EXPANDED FEATURES SECTION
# ============================================================================

st.markdown("---")
st.markdown("## 🔬 Input Features Analysis")

tabs = st.tabs(["📊 Feature Breakdown", "🎛️ Feature Details", "ℹ️ Model Info"])

with tabs[0]:
    # Categorize features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Demographics")
        st.write(f"**Age:** {age}")
        st.write(f"**Job:** {job}")
        st.write(f"**Marital:** {marital}")
        st.write(f"**Education:** {education_str}")
    
    with col2:
        st.markdown("### 💳 Financial")
        st.write(f"**Balance:** €{balance:,}")
        st.write(f"**Default:** {default}")
        st.write(f"**Housing Loan:** {housing}")
        st.write(f"**Personal Loan:** {loan}")
    
    with col3:
        st.markdown("### 📞 Campaign")
        st.write(f"**Month:** {month_str.capitalize()}")
        st.write(f"**Day:** {day}")
        st.write(f"**Duration:** {duration}s")
        st.write(f"**Campaign Calls:** {campaign}")

with tabs[1]:
    # Display the processed input data
    st.markdown("### Raw Input Data")
    st.dataframe(input_data_df, use_container_width=True)
    
    # Display feature statistics
    st.markdown("### Feature Ranges (Based on Dataset)")
    feature_stats = pd.DataFrame({
        'Feature': ['Age', 'Balance', 'Duration', 'Campaign'],
        'Min': [18, -10000, 0, 1],
        'Typical': [45, 1500, 300, 2],
        'Max': [95, 100000, 5000, 100]
    })
    st.table(feature_stats)

with tabs[2]:
    st.markdown("### 🤖 Model Information")
    st.info("""
    **Model Type:** Random Forest Classifier
    
    **Purpose:** Predict whether a bank client will subscribe to a term deposit after a telemarketing campaign.
    
    **Features Used:** 15 input features covering demographics, financial status, and campaign details.
    
    **Training Data:** Portuguese bank marketing dataset (11,162 records)
    
    **Output:** Binary classification (Yes/No) with probability scores
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; padding: 2em 0; font-size: 0.9em;">
    <p>🏦 Bank Telemarketing Success Predictor v1.0</p>
    <p>Powered by Random Forest Machine Learning Model | Real-time Predictions</p>
</div>
""", unsafe_allow_html=True)
