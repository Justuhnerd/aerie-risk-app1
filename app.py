 import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io
import requests
import json

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AERIE Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Model Artifacts (with caching)
# -------------------------------
@st.cache_resource
def load_model():
    """Load model, scaler, and feature list (cached for performance)"""
    model = joblib.load('aerie_model.pkl')
    scaler = joblib.load('aerie_scaler.pkl')
    with open('feature_list.pkl', 'rb') as f:
        import pickle
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, feature_list = load_model()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.sidebar.info("Please ensure model files are in the same directory")
    st.stop()

# -------------------------------
# Helper Functions
# -------------------------------
def predict_single(input_dict):
    """Predict for a single incident"""
    df = pd.DataFrame([input_dict])[feature_list]
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0][1]
    pred = model.predict(scaled)[0]
    return pred, proba

def predict_batch(df):
    """Predict for multiple incidents"""
    # Ensure all required features exist
    missing = set(feature_list) - set(df.columns)
    if missing:
        return None, f"Missing columns: {missing}"
    
    df_input = df[feature_list].copy()
    # Fill any missing values with median (simplified)
    df_input = df_input.fillna(df_input.median())
    
    scaled = scaler.transform(df_input)
    probas = model.predict_proba(scaled)[:, 1]
    preds = model.predict(scaled)
    return pd.DataFrame({
        'predicted_major_event': preds,
        'probability': probas
    }), None

def generate_scenarios(prompt, api_token, model="mistralai/Mistral-7B-Instruct-v0.1"):
    """Call Hugging Face Inference API to generate text."""
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🛡️ AERIE")
st.sidebar.markdown("**A**daptive **E**nterprise **R**isk **I**ntelligence **E**ngine")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Choose Mode",
    ["🔍 Single Prediction", "📤 Batch Upload", "🎮 Scenario Simulator", "📊 Model Info", "🤖 AI Scenario Generator"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Feature List")
for i, feat in enumerate(feature_list):
    st.sidebar.text(f"{i+1}. {feat}")

# -------------------------------
# Mode 1: Single Prediction Form
# -------------------------------
if app_mode == "🔍 Single Prediction":
    st.title("🔍 Single Incident Risk Predictor")
    st.markdown("Enter incident details below to predict if it's a **Major Event**.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity = st.slider("Severity (1-5)", 1, 5, 3)
            downtime = st.number_input("Downtime (hours)", 0.0, 100.0, 5.0)
            financial_impact = st.number_input("Financial Impact ($)", 0, 500000, 50000)
            regulatory_flag = st.selectbox("Regulatory Flag", [0, 1])
        
        with col2:
            data_sensitivity = st.slider("Data Sensitivity (0-1)", 0.0, 1.0, 0.5)
            criticality = st.slider("Criticality (1-5)", 1, 5, 3)
            asset_incident_prev_count = st.number_input("Prior Incidents on Asset", 0, 20, 0)
            days_since_audit = st.number_input("Days Since Last Audit", 0, 365, 30)
        
        with col3:
            st.markdown("### Auto-calculated")
            severity_x_data_sensitivity = severity * data_sensitivity
            st.metric("Severity × Sensitivity", f"{severity_x_data_sensitivity:.2f}")
            
            # Preview of input vector
            input_preview = {
                'severity': severity,
                'downtime': downtime,
                'financial_impact': financial_impact,
                'regulatory_flag': regulatory_flag,
                'data_sensitivity': data_sensitivity,
                'criticality': criticality,
                'severity_x_data_sensitivity': severity_x_data_sensitivity,
                'asset_incident_prev_count': asset_incident_prev_count,
                'days_since_audit': days_since_audit
            }
            st.json(input_preview)
        
        submitted = st.form_submit_button("🚀 Predict Risk", use_container_width=True)
        
        if submitted:
            input_dict = {
                'severity': severity,
                'downtime': downtime,
                'financial_impact': financial_impact,
                'regulatory_flag': regulatory_flag,
                'data_sensitivity': data_sensitivity,
                'criticality': criticality,
                'severity_x_data_sensitivity': severity_x_data_sensitivity,
                'asset_incident_prev_count': asset_incident_prev_count,
                'days_since_audit': days_since_audit
            }
            
            pred, proba = predict_single(input_dict)
            
            # Display results
            st.markdown("---")
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                if pred == 1:
                    st.error(f"🚨 **MAJOR EVENT**")
                else:
                    st.success(f"✅ **Minor / Routine**")
            
            with col_res2:
                st.metric("Probability", f"{proba:.1%}")
            
            with col_res3:
                # Risk level indicator
                if proba < 0.3:
                    st.info("🟢 Low Risk")
                elif proba < 0.7:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")
            
            # Feature importance visualization
            st.subheader("📊 What Drove This Prediction?")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({
                    'feature': feature_list,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(feat_imp_df, 
                            x='importance', 
                            y='feature',
                            orientation='h',
                            title="Global Feature Importance",
                            color='importance',
                            color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Mode 2: Batch Upload
# -------------------------------
elif app_mode == "📤 Batch Upload":
    st.title("📤 Batch Risk Scoring")
    st.markdown("Upload a CSV file with multiple incidents to score them all at once.")
    
    # Template download
    template_df = pd.DataFrame(columns=feature_list)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Template CSV",
        data=csv_template,
        file_name="aerie_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data")
        st.dataframe(df.head())
        
        if st.button("🔮 Score All Incidents"):
            with st.spinner("Scoring..."):
                results, error = predict_batch(df)
                
                if error:
                    st.error(error)
                else:
                    # Combine with original data
                    output_df = pd.concat([df, results], axis=1)
                    
                    st.success(f"✅ Scored {len(output_df)} incidents")
                    st.write("### Results Preview")
                    st.dataframe(output_df.head())
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Major Events Predicted", results['predicted_major_event'].sum())
                    col2.metric("Average Probability", f"{results['probability'].mean():.1%}")
                    col3.metric("High Risk (>70%)", (results['probability'] > 0.7).sum())
                    
                    # Download button
                    csv_output = output_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Results",
                        data=csv_output,
                        file_name="aerie_predictions.csv",
                        mime="text/csv"
                    )

# -------------------------------
# Mode 3: Scenario Simulator
# -------------------------------
elif app_mode == "🎮 Scenario Simulator":
    st.title("🎮 What-If Scenario Simulator")
    st.markdown("Adjust the sliders to explore how different factors affect risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity = st.slider("Severity", 1, 5, 3, key="sim_severity")
        downtime = st.slider("Downtime (hours)", 0.0, 100.0, 5.0, key="sim_downtime")
        financial = st.slider("Financial Impact ($)", 0, 500000, 50000, key="sim_financial")
        reg_flag = st.selectbox("Regulatory Flag", [0, 1], key="sim_reg")
    
    with col2:
        data_sens = st.slider("Data Sensitivity", 0.0, 1.0, 0.5, key="sim_data")
        crit = st.slider("Criticality", 1, 5, 3, key="sim_crit")
        prev_count = st.slider("Prior Incidents", 0, 20, 0, key="sim_prev")
        audit_days = st.slider("Days Since Audit", 0, 365, 30, key="sim_audit")
    
    # Auto-calc interaction
    sev_x_data = severity * data_sens
    
    input_dict = {
        'severity': severity,
        'downtime': downtime,
        'financial_impact': financial,
        'regulatory_flag': reg_flag,
        'data_sensitivity': data_sens,
        'criticality': crit,
        'severity_x_data_sensitivity': sev_x_data,
        'asset_incident_prev_count': prev_count,
        'days_since_audit': audit_days
    }
    
    pred, proba = predict_single(input_dict)
    
    # Visualize risk
    st.markdown("---")
    st.subheader("📊 Risk Assessment")
    
    # Gauge chart
    fig = px.bar(x=["Risk Level"], y=[proba], range_y=[0, 1],
                 color=[proba], color_continuous_scale=['green', 'yellow', 'red'])
    fig.update_layout(showlegend=False, title=f"Risk Probability: {proba:.1%}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction result
    if pred == 1:
        st.error("🚨 This scenario would be classified as a **MAJOR EVENT**")
    else:
        st.success("✅ This scenario would be classified as **Minor / Routine**")

# -------------------------------
# Mode 4: Model Information
# -------------------------------
elif app_mode == "📊 Model Info":
    st.title("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Type")
        st.info(f"**Random Forest Classifier**")
        st.write(f"Number of trees: {model.n_estimators}")
        st.write(f"Features used: {len(feature_list)}")
        
    with col2:
        st.subheader("Class Weights")
        st.write(model.class_weight_)
    
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                 title="Global Feature Importance",
                 color='Importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("How to Use This Model")
    st.markdown("""
    - **Single Prediction**: Enter incident details manually
    - **Batch Upload**: Score multiple incidents from CSV
    - **Scenario Simulator**: Explore "what-if" scenarios
    - **All features must be provided** in the exact order shown in the sidebar
    """)

# -------------------------------
# Mode 5: AI Scenario Generator
# -------------------------------
elif app_mode == "🤖 AI Scenario Generator":
    st.title("🤖 Generate Incident Scenarios with AI")
    st.markdown("Use a free Hugging Face model to create plausible incident descriptions. Then you can manually enter them in other modes, or (optionally) parse and score them automatically.")
    
    # Try to get API token from secrets, else ask user
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    except:
        HF_TOKEN = st.text_input("Hugging Face API Token", type="password", help="Get a free token at huggingface.co/settings/tokens")
    
    if not HF_TOKEN:
        st.warning("Please enter your Hugging Face API token to use this feature.")
        st.stop()
    
    # Prompt input
    default_prompt = "Generate 5 realistic IT security incidents with varying severity, downtime, and financial impact for a financial services company. Output as a bullet list."
    user_prompt = st.text_area("Describe the kind of scenarios you want", value=default_prompt, height=150)
    
    if st.button("🚀 Generate Scenarios"):
        with st.spinner("AI is thinking... (may take 10-30 seconds)"):
            try:
                result = generate_scenarios(user_prompt, HF_TOKEN)
                # The API returns a list with generated text, or an error dict
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', str(result))
                elif isinstance(result, dict) and 'error' in result:
                    st.error(f"API error: {result['error']}")
                    st.stop()
                else:
                    generated_text = str(result)
                
                st.subheader("✨ Generated Scenarios")
                st.write(generated_text)
                
                # Optional: add a button to parse into structured format (advanced)
                # You can extend this by asking the AI to output CSV and then parsing.
            except Exception as e:
                st.error(f"Error calling API: {e}")
