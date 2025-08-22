import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache_resource
def load_model():
    return joblib.load("churn_xgb_model.pkl")

@st.cache_resource
def load_features():
    return joblib.load("features.pkl")  

model = load_model()
feature_names = load_features()

st.title("Telco Churn Prediction Dashboard")


# Preprocessing
def preprocess_input(df):
    drop_cols = ["customer_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)

    df = pd.get_dummies(df, drop_first=True)

    # Align with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df


uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(raw_df.head())

    df = preprocess_input(raw_df)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = raw_df.copy()
    results["Churn_Prediction"] = predictions
    results["Churn_Probability"] = probabilities

    st.write("### Predictions")
    st.dataframe(results.head())

    churn_rate = results["Churn_Prediction"].mean() * 100
    st.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")

    # Charts
    st.subheader("Churn Distribution")
    fig1 = px.histogram(results, x="Churn_Prediction", color="Churn_Prediction",
                        labels={"Churn_Prediction": "Churn (0=No, 1=Yes)"})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Churn Probability Distribution")
    fig2 = px.histogram(results, x="Churn_Probability", nbins=20,
                        title="Distribution of Churn Probabilities")
    st.plotly_chart(fig2, use_container_width=True)

    if "contract" in results.columns:
        st.subheader("Churn by Contract Type")
        fig3 = px.histogram(results, x="contract", color="Churn_Prediction",
                            barmode="group", title="Churn by Contract Type")
        st.plotly_chart(fig3, use_container_width=True)

    # SHAP Global Explainability
    st.subheader("Global Feature Importance (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("Detailed Feature Impact (SHAP Beeswarm)")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, df, show=False)
    st.pyplot(fig2)

    # Customer Drilldown
    st.subheader("Customer Drilldown Analysis")

    # Select customer by index
    customer_index = st.number_input("Select Customer Index", min_value=0, max_value=len(results)-1, step=1)

    customer_row = results.iloc[customer_index]
    st.write("### Customer Details")
    st.write(customer_row)

    # Show churn probability
    churn_prob = customer_row["Churn_Probability"]
    churn_pred = "Yes" if customer_row["Churn_Prediction"] == 1 else "No"
    st.metric("Churn Prediction", churn_pred)
    st.metric("Churn Probability", f"{churn_prob:.2%}")

    # SHAP force plot
    st.write("### SHAP Explanation for Selected Customer")
    shap.initjs()
    customer_shap = shap_values[customer_index]

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(values=customer_shap,
                                         base_values=explainer.expected_value,
                                         data=df.iloc[customer_index],
                                         feature_names=df.columns))
    st.pyplot(fig3)
