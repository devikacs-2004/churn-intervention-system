import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import google.generativeai as genai
import matplotlib.pyplot as plt

#page config
st.set_page_config(
    page_title="Churn Intervention System",
    page_icon="🔴",
    layout="wide"
)

#Title
st.title("🔴 Churn Intervention Intelligence System")
st.markdown("*Upload customer data to identify churn risk and generate personalised interventions*")
st.divider()

#load model artifacts
@st.cache_resource    #What @st.cache_resource does:Loads the model only once when the app starts — not every time a user interacts. Makes the app fast.
def load_artifacts():
    with open("model_artifacts.pkl","rb") as f:
        artifacts=pickle.load(f)
    return artifacts
artifacts=load_artifacts()
model=artifacts["model"]
scaler=artifacts["scaler"]
kmeans=artifacts["kmeans"]
threshold=artifacts["threshold"]
feature_names=artifacts["feature_names"]

#load gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini=genai.GenerativeModel("gemini-2.5-flash")

# Sample data download button
sample_df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv').sample(50, random_state=42)
csv = sample_df.to_csv(index=False)

st.download_button(
    label="Download Sample Data",
    data=csv,
    file_name="sample_customers.csv",
    mime="text/csv"
)
#file upload
st.subheader("📂 Upload Customer Data")
uploaded_file=st.file_uploader("Upload CSV file",type="csv")
if uploaded_file is not None:
    df_input=pd.read_csv(uploaded_file)
    st.success(f" Loaded {len(df_input)} customers")
    st.dataframe(df_input.head())
else:
    st.info("Please upload a CSV file to get started")
    st.stop()


#preprocessing
def preprocess(df):
    #Convert TotalCharges
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    df=df.dropna(subset=["TotalCharges"])
    #engineer Features
    df["spend_per_month"]=df["TotalCharges"]/df["tenure"]
    df["high_value"]=(df["spend_per_month"]>df["spend_per_month"].median()).astype(int)
    df["short_tenure"]=(df["tenure"]<12).astype(int)

    #drop unnecessary columns
    drop_cols=["customerID","Churn","Segment","Segment_Name"]
    existing_drops=[c for c in drop_cols if c in df.columns]
    df_model=df.drop(columns=existing_drops)

    #drop monthlyCharges if exists
    if "MonthlyCharges" in df_model.columns:
        df_model=df_model.drop(columns=["MonthlyCharges"])

    #One Hot Encode
    df_model=pd.get_dummies(df_model).astype(int)

    #Align columns with training data
    df_model=df_model.reindex(columns=feature_names,fill_value=0) #What reindex does: Makes sure the uploaded CSV has exactly the same columns as training data — fills missing columns with 0. Critical for the model to work on new data.

    return df,df_model
df_input, df_model=preprocess(df_input)

#predictions
st.subheader("Churn Risk Analysis")

#churn probabilities
churn_probs=model.predict_proba(df_model)[:,1]

#segments
cluster_features = df_model[['tenure', 'TotalCharges', 'spend_per_month']].copy()
cluster_scaled=scaler.transform(cluster_features)
segments=kmeans.predict(cluster_scaled)
segment_map={0:"Loyal Premiums",1:"New Basics",2:"Loyal Basics"}
segment_names=[segment_map[s] for s in segments]

#risk labels
def risk_labels(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

#result dataframe
results_df=pd.DataFrame({
    "Customer": df_input["customerID"] if "customerID" in df_input.columns else range(len(df_input)),
    "Churn Probability": [f"{p:.1%}" for p in churn_probs],
    "Risk Level": [risk_labels(p) for p in churn_probs],
    "Segment": segment_names
})

st.dataframe(results_df, use_container_width=True)

#Shap + GenAI
st.subheader("Individual Customer Analysis")

#SHAP explainer
explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(df_model)
def get_top_reasons(idx,shap_vals,feature_names,top_n=3):
    shap_series=pd.Series(shap_vals[idx],index=feature_names)
    top_factors=shap_series.nlargest(top_n)
    return top_factors
def generate_message(churn_prob,segment,top_factors,customer_name="Valued Customer"):
    reasons="\n".join([f"- {feat}:impact {val:.3f}"
                       for feat, val in top_factors.items()])
    prompt=f"""You are a customer retention specialist for a telecom company.
    Customer name: {customer_name}
    Customer churn probability: {churn_prob:.1%}
    Customer segment:{segment}
    Top churn reason:
    {reasons}
    write a short personlised retention message (2-3 sentences).
    To be warm, proffessional and specific to their situation."""
    response= gemini.generate_content(prompt)
    return response.text

#show top 5 high risk customers
st.markdown("***Top 5 Highest Risk Customers***")
top5_idx=np.argsort(churn_probs)[-5:][::-1]
for idx in top5_idx:
    prob=churn_probs[idx]
    segment=segment_names[idx]
    top_factors=get_top_reasons(idx,shap_values,feature_names)
    with st.expander(f"Customer {results_df["Customer"].iloc[idx]}-{prob:.1%} churn risk"):
        col1,col2=st.columns(2)

        with col1:
            st.markdown("**Top Churn Reasons (SHAP)**")
            for feat, val in top_factors.items():
                st.markdown(f"-> {feat}:'+ {val:.3f}'")
        with col2:
            st.markdown("**AI Retention Message**")
            if st.button(f"Generate Message", key=f"btn_{idx}"):
                with st.spinner("Generating...."):
                    message=generate_message(prob,segment,top_factors)
                st.write(message)
            else:
                st.info("Click to generate retention message")

#ROI Calculator
st.divider()
st.subheader("Business ROI Calculator")
col1,col2=st.columns(2)

with col1:
    intervention_rate=st.slider(
        "Intervention Success Rate (%)",
        min_value=10,
        max_value=50,
        value=25
    )/100
    top_pct=st.slider(
        "Top % of customers to target",
        min_value=10,
        max_value=30,
        value=20
    )/100
with col2:
    n_intervene=int(len(churn_probs)*top_pct)
    avg_spend=df_input["spend_per_month"].mean()
    recall=0.75
    churners_caught=n_intervene*recall
    customers_saved=churners_caught*intervention_rate
    revenue_saved=customers_saved*avg_spend*12
    st.metric("Customers targeted",n_intervene)
    st.metric("Estimated Churners caught",f"{churners_caught:.0f}")
    st.metric("Customers saved",f"{customers_saved:.0f}")
    st.metric("Annual Revenue Saved",f"${revenue_saved:,.0f}")

