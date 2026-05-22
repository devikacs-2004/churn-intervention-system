# 🔴 Churn Intervention Intelligence System

A production-grade machine learning system that predicts customer churn, 
explains why, quantifies business impact, and generates AI-powered 
personalised retention messages.

**Live App:** https://churn-intervention-system-jbftk3ovuxu7xrttoha6l2.streamlit.app/

---

## 🎯 Business Problem

26.58% of telecom customers churn — nearly 1 in 4. 
Most churn models stop at "this customer will leave." 
This system goes further — it tells you who, why, when, 
and what to say to keep them.

---

## 🏗️ System Architecture — 6 Layers

| Layer | What It Does |
|-------|-------------|
| 1️⃣ Customer Segmentation | K-Means clustering identifies 3 behavioral segments |
| 2️⃣ Churn Prediction | XGBoost classifier per segment — F1: 0.62, ROC-AUC: 0.76 |
| 3️⃣ SHAP Explainability | Global + local explanations for every prediction |
| 4️⃣ Business ROI Calculator | Quantifies revenue saved from interventions |
| 5️⃣ Survival Analysis | Kaplan-Meier + Cox model predicts time-to-churn |
| 6️⃣ GenAI Intervention Engine | Gemini API generates personalised retention messages |

---

## 📊 Key Findings

- Month-to-month customers churn 3x more than yearly contracts
- First 10 months is the highest churn risk window
- New Basics segment: 42.8% churn rate, $99,967 monthly revenue at risk
- Model saves $16,000–$57,000 annually depending on intervention success rate

---

## 🛠️ Tech Stack

- **ML:** Python, XGBoost, Scikit-learn, SHAP, Lifelines
- **Data:** Pandas, NumPy, Matplotlib, Seaborn
- **AI:** Google Gemini API
- **App:** Streamlit
- **Dataset:** IBM Telco Customer Churn (Kaggle)

---

## 🚀 How To Use

1. Open the live app
2. Click **Download Sample Data** to get a test CSV
3. Upload the CSV
4. View churn predictions, risk levels, and segments
5. Expand individual customers to see SHAP explanations
6. Click **Generate Message** for AI retention recommendations
7. Adjust ROI calculator sliders to estimate revenue impact

---

## 📁 Project Structure
churn-intervention-system/
├── app.py                    # Streamlit web application
├── churn_project.ipynb       # Full analysis notebook
├── model_artifacts.pkl       # Saved model and preprocessing objects
├── requirements.txt          # Python dependencies
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset

---

## 👤 About

Built by **Devika C S** — aspiring Data Scientist based in UAE.

[LinkedIn] www.linkedin.com/in/devika-cs-594511286) | [Kaggle]https://www.kaggle.com/devikasubij
