import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

# Load models (cached for performance)
@st.cache_resource
def load_models():
    # Load tree-based models
    with open('ensemble_models.pkl', 'rb') as f:
        ensemble_models = pickle.load(f)

    # Load neural network
    nn_model = load_model('nn_model.h5') 

    # Load preprocessor
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Load best threshold    
    with open('best_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)

    # Create SHAP explainer for LGBM (cached for performance)
    explainer = shap.TreeExplainer(ensemble_models['LGBM Classifier'])
    
    return ensemble_models, nn_model, preprocessor, threshold, explainer

# Title
st.title("Loan Default Prediction System")
st.markdown("Enter applicant information to predict default risk")
st.divider()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Applicant Information")
    
    # Credit Information
    st.subheader("Credit Information")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        fico_range_low = st.number_input(
            "FICO Range Low", 
            min_value=300, 
            max_value=850, 
            value=700,
            help="Assumed minimum FICO score for this applicant."
        )
        dti = st.number_input(
            "DTI (Debt-to-Income)", 
            min_value=0.0, 
            max_value=100.0, 
            value=15.5, 
            step=0.1,
            help="Assumed debt-to-income ratio for the applicant.")
        revol_util = st.number_input(
            "Revolving Utilization (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=30.5, 
            step=0.1,
            help="Assumed proportion of revolving credit utilized.")
        
    with c2:
        fico_range_high = st.number_input(
            "FICO Range High", 
            min_value=300, 
            max_value=850, 
            value=704,
            help="Assumed minimum FICO score for this applicant."
        )
        revol_bal = st.number_input(
            "Revolving Balance ($)", 
            min_value=0, 
            value=5000, 
            step=100,
            help="Assumed outstanding revolving credit balance.")
        annual_inc = st.number_input(
            "Annual Income ($)", 
            min_value=0, 
            value=60000, 
            step=1000,
            help="Total yearly income of the applicant.")
        
    with c3:
        delinq_2yrs = st.number_input(
            "Delinquencies (2 years)", 
            min_value=0, 
            value=0,
            help="Assumed number of delinquencies in the past 2 years")
        pub_rec = st.number_input(
            "Public Records", 
            min_value=0, 
            value=0,
            help="Assumed number of public records in the applicant’s credit history")
        pub_rec_bankruptcies = st.number_input(
            "Bankruptcies", 
            min_value=0, 
            value=0,
            help="Assumed number of bankruptcies in the applicant’s credit history,")
    
    st.divider()
    
    # Loan Details
    st.subheader("Loan Details")
    c4, c5, c6 = st.columns(3)
    
    with c4:
        loan_amnt = st.number_input(
            "Loan Amount ($)", 
            min_value=0, 
            value=10000, 
            step=500,
            help="Assumed loan amount for this applicant")
        term = st.selectbox(
            "Term",
            options=["36 months", "60 months"],
            help="Assumed loan term")
        purpose = st.selectbox(
            "Purpose", 
            options=["debt_consolidation", "credit_card", "home_improvement", 
            "major_purchase", "small_business", "car", "medical", 
            "moving", "vacation", "house", "wedding", "renewable_energy", 
            "educational", "other"],
            help="Assumed loan purpose")
        
    with c5:
        total_acc = st.number_input(
            "Total Accounts", 
            min_value=0, 
            value=10,
            help="Assumed total number of credit accounts for this applicant")
        open_acc = st.number_input(
            "Open Accounts", 
            min_value=0, 
            value=8,
            help="Assumed number of currently open credit accounts")
        inq_last_6mths = st.number_input(
            "Inquiries (6 months)", 
            min_value=0, 
            value=1,
            help="Assumed number of credit inquiries in the past 6 months")
        
    with c6:
        home_ownership = st.selectbox(
            "Home Ownership", 
            options=["RENT", "OWN", "MORTGAGE", "OTHER"],
            help="Assumed home ownership status")
        verification_status = st.selectbox(
            "Verification Status", 
            options=["Verified", "Source Verified", "Not Verified"],
            help="Assumed verification status of applicant information")
        emp_length = st.selectbox(
            "Employment Length", 
            options=["< 1 year", "1 year", "2 years", "3 years", "4 years", 
            "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
            help="Assumed length of employment")
    
    st.divider()
    
    # Additional Information
    st.subheader("Additional Information")
    c7, c8, c9 = st.columns(3)
    
    with c7:
        addr_state = st.selectbox(
            "State", 
            options=[
            "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
            "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
            "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM",
            "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
            "UT", "VA", "VT", "WA", "WI", "WV", "WY"], 
        index=4,
        help="Assumed state of residence")
        
    with c8:
        earliest_cr_line = st.text_input(
            "Earliest Credit Line (YYYY-MM)", 
            value="2010-01",
            help="Assumed start date of the applicant’s earliest credit line")
        
    with c9:
        collections_12_mths_ex_med = st.number_input(
            "Collections (12 months)", 
            min_value=0, 
            value=0,
            help="Assumed number of collections in the last 12 months")

with col2:
    st.header("Prediction")
    
    # Predict button
    if st.button("Predict Default Risk", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'fico_range_low': [fico_range_low],
            'fico_range_high': [fico_range_high],
            'dti': [dti],
            'revol_bal': [revol_bal],
            'revol_util': [revol_util],
            'annual_inc': [annual_inc],
            'delinq_2yrs': [delinq_2yrs],
            'pub_rec': [pub_rec],
            'pub_rec_bankruptcies': [pub_rec_bankruptcies],
            'inq_last_6mths': [inq_last_6mths],
            'loan_amnt': [loan_amnt],
            'term': [term],
            'total_acc': [total_acc],
            'open_acc': [open_acc],
            'earliest_cr_line': [earliest_cr_line],
            'home_ownership': [home_ownership],
            'purpose': [purpose],
            'emp_length': [emp_length],
            'addr_state': [addr_state],
            'verification_status': [verification_status],
            'collections_12_mths_ex_med': [collections_12_mths_ex_med]
        })
        
        try:
            # Load models
            ensemble_models, nn_model, preprocessor, threshold = load_models()

            # Feature Engineering
            input_data = input_data.copy()
            
            # Convert earliest_cr_line to datetime
            input_data['earliest_cr_line'] = pd.to_datetime(
                input_data['earliest_cr_line'])
            
            # Get today's timestamp and calculate credit history in years
            today = pd.Timestamp.today()
            input_data['credit_history_years'] = (
                (today - input_data['earliest_cr_line']).dt.days / 365
            )
            
            # Income to loan ratio
            input_data['income_to_loan_ratio'] = input_data['annual_inc'] / input_data['loan_amnt']
            
            # Fico score
            input_data['fico_score'] = (input_data['fico_range_low'] + input_data['fico_range_high']) / 2
            
            # Account utilization
            input_data['account_utilization'] = input_data['open_acc'] / input_data['total_acc']
            
            # Avg credit per account
            input_data['avg_credit_per_account'] = np.where(
                input_data['open_acc'] > 0,
                input_data['revol_bal'] / input_data['open_acc'],
                0
            )
            
            # Delinquency rate
            input_data['deliquency_rate'] = np.where(
                input_data['total_acc'] > 0,
                input_data['delinq_2yrs'] / input_data['total_acc'],
                0
            )
            
            # Drop irrelevant features after engineering
            input_data = input_data.drop(columns=['earliest_cr_line', 'fico_range_low', 'fico_range_high'])            
            
            # Preprocess
            input_preprocessed = preprocessor.transform(input_data)
            
            # Predict with all 3 models and average
            lgbm_proba = ensemble_models['LGBM Classifier'].predict_proba(input_preprocessed)[0, 1]
            cat_proba = ensemble_models['Cat Boost Classifier'].predict_proba(input_preprocessed)[0, 1]
            nn_proba  = nn_model.predict(input_preprocessed, verbose=0).item()
            
            # Average all predictions
            default_proba = np.mean([lgbm_proba, cat_proba, nn_proba])
            prediction = 1 if default_proba >= threshold else 0

            # Store in session state for SHAP plot
            st.session_state['prediction_made'] = True
            st.session_state['input_preprocessed'] = input_preprocessed
            st.session_state['default_proba'] = default_proba
            st.session_state['prediction'] = prediction
            st.session_state['threshold'] = threshold
            
            # Display results
            st.subheader("Results")
            
            # Probability gauge
            st.metric(
                label="Default Probability", 
                value=f"{default_proba*100:.1f}%",
                delta=f"{(default_proba - threshold)*100:.1f}% from threshold"
            )
            
            # Prediction
            if prediction == 1:
                st.error("High Risk: Likely to Default")
                risk_level = "High Risk"
                risk_color = "red"
            else:
                st.success("Low Risk: Likely to Pay")
                risk_level = "Low Risk"
                risk_color = "green"
            
            # Risk level indicator
            if default_proba > 0.7:
                st.error("Risk Level: CRITICAL")
            elif default_proba > 0.5:
                st.warning("Risk Level: ELEVATED")
            elif default_proba > 0.3:
                st.info("Risk Level: MODERATE")
            else:
                st.success("Risk Level: LOW")
            
            # Show threshold
            st.divider()
            st.caption(f"Optimal Threshold: {threshold:.3f}")
            
            # Progress bar showing probability
            st.progress(default_proba)
            
        except FileNotFoundError:
            st.error("Model files not found! Please ensure the following files are in the same directory:")
            st.code("ensemble_model.pkl\npreprocessor.pkl\nbest_threshold.pkl")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    
    # Information box
    st.divider()
    st.info("""How to use:
            1. Fill in all applicant information
            2. Click 'Predict Default Risk'
            3. Review the prediction results

            Model: Stacking Ensemble
            - XGBoost
            - LightGBM
            - Multilayer Perceptron (MLP)
            """
    )

# SHAP Explanation Section 
if st.session_state.get('prediction_made', False):
    st.divider()
    st.header("🔍 Why Was This Prediction Made?")
    st.markdown("""
    The chart below shows which features had the biggest impact on this prediction:
    - **Red bars** push the prediction towards **default** (increase risk)
    - **Blue bars** push the prediction towards **repayment** (decrease risk)
    - The length of each bar shows how much that feature influenced the decision
    """)

    try:
        # Load explainer
        _, _, _, _, explainer = load_models()

        # Get preprocessed input from session state
        input_preprocessed = st.session_state['input_preprocessed']

        # Compute SHAP values
        with st.spinner("Computing feature importance..."):
            shap_values = explainer(input_preprocessed)

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()

        # Display in streamlit
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Interpretation
        with st.expander("ℹ️ How to interpret this chart"):
            st.markdown("""
            **Understanding the SHAP Waterfall Plot:**

            - **E[f(x)]**: The average prediction across all applicants (baseline)
            - **f(x)**: This applicant's prediction
            - **Features**: Listed from most to least impactful
            - **Values in brackets**: The actual feature value for this applicant
            - **Arrow direction**: 
                - Right (red) = increases default probability
                - Left (blue) = decreases default probability
            
            **Example:** If you see "fico_score = 702" with a blue bar pointing left, 
            this means the applicant's FICO score of 702 reduces their default risk 
            compared to the average applicant.            
            """)
        
    except Exception as e:
        st.error(f"Error generating SHAP plot: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# Footer
st.divider()
st.caption("This prediction is for informational purposes only and should not be the sole basis for lending decisions.")
