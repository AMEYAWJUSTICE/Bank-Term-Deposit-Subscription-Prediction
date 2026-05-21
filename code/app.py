!pip install streamlit
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
try:
    rf_pipeline = joblib.load('random_forest_model.joblib')
    # The preprocessor is already part of the rf_pipeline, but if you had a separate one,
    # you'd load it here: preprocessor = joblib.load('preprocessor.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'random_forest_model.joblib' is in the current directory.")
    st.stop()

# Define mappings for ordinal features (from previous notebook cells)
education_mapping = {
    'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3
}
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

# Define possible values for categorical features (from df_ct_prep.unique() in notebook)
job_options = ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student']
marital_options = ['married', 'single', 'divorced']
default_options = ['no', 'yes']
housing_options = ['yes', 'no']
loan_options = ['no', 'yes']
contact_options = ['unknown', 'cellular', 'telephone']
poutcome_options = ['unknown', 'failure', 'other', 'success']
month_str_options = list(month_mapping.keys())
education_str_options = list(education_mapping.keys())

st.title("Bank Telemarketing Success Predictor")
st.write("Enter client details to predict if they will subscribe to a term deposit.")

st.sidebar.header("Client Features")

# Create input widgets for each feature
# Numerical Inputs
age = st.sidebar.slider("Age", 18, 95, 30)
balance = st.sidebar.number_input("Balance", value=1500, min_value=-6847, max_value=81204)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
duration = st.sidebar.number_input("Last Contact Duration (seconds)", value=300, min_value=0, max_value=3881)
campaign = st.sidebar.number_input("Number of Contacts this Campaign", value=1, min_value=1, max_value=63)
pdays = st.sidebar.number_input("Days Since Last Contact (pdays, -1 if no previous contact)", value=-1, min_value=-1, max_value=854)
previous = st.sidebar.number_input("Number of Previous Contacts", value=0, min_value=0, max_value=58)

# Ordinal Categorical Inputs (selected as strings, mapped to numeric by the script)
education_str = st.sidebar.selectbox("Education Level", education_str_options, index=education_str_options.index('secondary')) # default to secondary
month_str = st.sidebar.selectbox("Month of Last Contact", month_str_options, index=month_str_options.index('may')) # default to may

# Nominal Categorical Inputs
job = st.sidebar.selectbox("Job Title", job_options, index=job_options.index('management')) # default to management
marital = st.sidebar.selectbox("Marital Status", marital_options, index=marital_options.index('married')) # default to married
default = st.sidebar.selectbox("Has Credit in Default?", default_options, index=default_options.index('no')) # default to no
housing = st.sidebar.selectbox("Has Housing Loan?", housing_options, index=housing_options.index('no')) # default to no
loan = st.sidebar.selectbox("Has Personal Loan?", loan_options, index=loan_options.index('no')) # default to no
contact = st.sidebar.selectbox("Contact Communication Type", contact_options, index=contact_options.index('cellular')) # default to cellular
poutcome = st.sidebar.selectbox("Outcome of Previous Marketing Campaign", poutcome_options, index=poutcome_options.index('unknown')) # default to unknown

# Create a DataFrame from inputs
# Ensure the columns are in the exact order as expected by the preprocessor's fit method (X_ct columns)
input_data_df = pd.DataFrame([{ # Renamed to avoid clash with input_data
    'age': age,
    'job': job,
    'marital': marital,
    'education': education_str, # Will be mapped below
    'default': default,
    'balance': balance,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'day': day,
    'month': month_str, # Will be mapped below
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
}])

# Apply education and month mappings (strings to integers)
input_data_df['education'] = input_data_df['education'].map(education_mapping)
input_data_df['month'] = input_data_df['month'].map(month_mapping)

if st.button("Predict"): # The predict button should be in the main area
    # Make prediction
    prediction = rf_pipeline.predict(input_data_df)
    prediction_proba = rf_pipeline.predict_proba(input_data_df)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"**Yes**, the client is likely to subscribe to a term deposit!")
        st.write(f"Confidence (Probability of 'Yes'): {prediction_proba[0][1]:.2f}")
        st.write(f"Confidence (Probability of 'No'): {prediction_proba[0][0]:.2f}")
    else:
        st.error(f"**No**, the client is not likely to subscribe to a term deposit.")
        st.write(f"Confidence (Probability of 'No'): {prediction_proba[0][0]:.2f}")
        st.write(f"Confidence (Probability of 'Yes'): {prediction_proba[0][1]:.2f}")

    st.markdown("--- Expansion of Predicted Features ---")
    st.write("Note: The model internally transforms these raw inputs into a much larger set of features using one-hot encoding and scaling.")
    
    # This part shows what the preprocessed data looks like (optional but informative)
    try:
        # Access the preprocessor from the pipeline
        preprocessor_from_pipeline = rf_pipeline.named_steps['preprocessor']
        transformed_data = preprocessor_from_pipeline.transform(input_data_df)
        st.subheader("Transformed Input Data (partial view):")
        # Get feature names from OHE
        ohe_feature_names = preprocessor_from_pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features_ct)
        all_feature_names = numerical_features_ct + list(ohe_feature_names)
        transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names) # this is an approximation
        st.dataframe(transformed_df.head())
    except Exception as e:
        st.warning(f"Could not display transformed data: {e}")

    st.markdown("--- Original Input Features Summary ---")
    st.dataframe(input_data_df)

st.markdown("--- Application Information ---")
st.info("To run this Streamlit app: Save the code above as a Python file (e.g., `app.py`) and execute `streamlit run app.py` in your terminal.")