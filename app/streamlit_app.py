# app/streamlit_app.py
# Bike Sales AI Pipeline — Hebrew University 2026 Final Project — Streamlit Web App
# Author: Rachel Barazani — AI Developer
# Course: AI Developer Program — Hebrew University 2026

import os
import sys
import pathlib
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from utils.references import CASH_PAYMENT_CITATIONS

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bike Sales AI Pipeline — Hebrew University 2026 Final Project",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HELPERS
# ============================================================

def artifact_exists(path):
    return os.path.exists(path)

def load_clean_data():
    if artifact_exists("artifacts/clean_data.csv"):
        return pd.read_csv("artifacts/clean_data.csv")
    return None

def load_features():
    if artifact_exists("artifacts/features.csv"):
        return pd.read_csv("artifacts/features.csv")
    return None

def load_encoders():
    if artifact_exists("artifacts/models/label_encoders.pkl"):
        return joblib.load("artifacts/models/label_encoders.pkl")
    return None

def load_model(path):
    if artifact_exists(path):
        return joblib.load(path)
    return None

def get_model_features(model):
    """Get the exact feature names the model was trained on."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except Exception:
        return 0

def build_feature_matrix(model, all_features):
    """Build a DataFrame using only the features the model was trained on."""
    model_features = get_model_features(model)
    if model_features:
        return pd.DataFrame(
            [[all_features.get(f, 0) for f in model_features]],
            columns=model_features,
        )
    return pd.DataFrame([all_features])

def read_markdown(path):
    if artifact_exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🚲 Bike Sales AI")
    st.markdown("**Rachel Barazani**  \nAI Developer")
    st.markdown("Hebrew University 2026")
    st.divider()

    st.markdown("### 📁 Pipeline Status")
    artifacts = {
        "clean_data.csv":           "artifacts/clean_data.csv",
        "eda_report.html":          "artifacts/eda_report.html",
        "insights.md":              "artifacts/insights.md",
        "dataset_contract.json":    "artifacts/dataset_contract.json",
        "features.csv":             "artifacts/features.csv",
        "model.pkl":                "artifacts/models/model.pkl",
        "evaluation_report.md":     "artifacts/evaluation_report.md",
        "model_card.md":            "artifacts/model_card.md",
    }
    for name, path in artifacts.items():
        icon = "✅" if artifact_exists(path) else "⏳"
        st.markdown(f"{icon} `{name}`")

    st.divider()
    df = load_clean_data()
    if df is not None:
        st.markdown("### 📊 Dataset")
        st.markdown(f"**Rows:** {len(df):,}")
        st.markdown(f"**Stores:** {df['Store_Location'].nunique()}")
        st.markdown(f"**Bike Models:** {df['Bike_Model'].nunique()}")

    st.divider()
    st.markdown("### ▶️ Run Pipeline")
    st.code("python flow/pipeline.py", language="bash")

# ============================================================
# PAGES
# ============================================================
page = st.selectbox(
    "Navigate",
    [
        "🏠 Business Overview",
        "📦 Prediction 1: Quantity",
        "🚲 Prediction 2: Bike Model",
        "💳 Prediction 3: Cash Payment",
        "📋 Model Documentation"
    ]
)

# ============================================================
# PAGE 1 — BUSINESS OVERVIEW
# ============================================================
if page == "🏠 Business Overview":
    st.title("🚲 Bike Sales AI Pipeline — Hebrew University 2026 Final Project")
    st.markdown("**Author:** Rachel Barazani — AI Developer | **Course:** AI Developer Program — Hebrew University 2026")
    st.divider()

    df = load_clean_data()
    if df is None:
        st.warning("⏳ Pipeline has not been run yet. Run `python flow/pipeline.py` first.")
        st.stop()

    # Metric cards
    df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Store Locations", df["Store_Location"].nunique())
    col3.metric("Bike Models", df["Bike_Model"].nunique())
    col4.metric("Avg Customer Age", f"{df['Customer_Age'].mean():.0f}")

    st.divider()

    # EDA Report
    st.subheader("📊 Exploratory Data Analysis")
    if artifact_exists("artifacts/eda_report.html"):
        with open("artifacts/eda_report.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning("EDA report not found. Run the pipeline first.")

    st.divider()

    # Business Insights
    st.subheader("💡 Business Insights")
    insights = read_markdown("artifacts/insights.md")
    if insights:
        st.markdown(insights)
    else:
        st.warning("Insights not found. Run the pipeline first.")


# ============================================================
# PAGE 2 — PREDICTION 1: QUANTITY
# ============================================================
elif page == "📦 Prediction 1: Quantity":
    st.title("📦 Prediction 1 — Purchase Quantity")
    st.markdown("*How many bikes will this customer buy?*")
    st.divider()

    df = load_features()
    encoders = load_encoders()

    if df is None or encoders is None:
        st.warning("⏳ Run the pipeline first.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Try It Live")
        age = st.slider("Customer Age", 18, 70, 35)
        gender = st.selectbox("Customer Gender", ["Male", "Female", "Unknown"])
        bike_model = st.selectbox("Bike Model", sorted(df["Bike_Model"].unique()))
        price = st.slider("Price ($)", 200, 5000, 1000)
        store = st.selectbox("Store Location", sorted(df["Store_Location"].unique()))
        payment = st.selectbox("Payment Method",
                               sorted(df["Payment_Method"].unique()))
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

        if st.button("🔮 Predict Quantity", type="primary"):
            try:
                price_tier = "Budget" if price < 500 else "Mid" if price <= 2000 else "Premium"
                age_group = (
                    "18-24" if age <= 24 else
                    "25-34" if age <= 34 else
                    "35-44" if age <= 44 else
                    "45-54" if age <= 54 else
                    "55-64" if age <= 64 else "65+"
                )

                all_features = {
                    "Price": price,
                    "Customer_Age": age,
                    "Salesperson_ID": 500,
                    "Month": 6,
                    "Day_of_Week": 1,
                    "Is_Weekend": 0,
                    "Bike_Model_enc": safe_encode(encoders["Bike_Model"], bike_model),
                    "Store_Location_enc": safe_encode(encoders["Store_Location"], store),
                    "Customer_Gender_enc": safe_encode(encoders["Customer_Gender"], gender),
                    "Payment_Method_enc": safe_encode(encoders["Payment_Method"], payment),
                    "Season_enc": safe_encode(encoders["Season"], season),
                    "Age_Group_enc": safe_encode(encoders["Age_Group"], age_group),
                    "Price_Tier_enc": safe_encode(encoders["Price_Tier"], price_tier),
                }

                rf_model = load_model("artifacts/models/quantity_rf.pkl")

                if rf_model:
                    X = build_feature_matrix(rf_model, all_features)
                    pred = rf_model.predict(X)[0]
                    proba = rf_model.predict_proba(X)[0]

                    st.success(f"🛒 Predicted Quantity: **{pred} bike(s)**")

                    st.markdown("**Probability by Quantity:**")
                    classes = rf_model.classes_
                    prob_df = pd.DataFrame({
                        "Quantity": [f"{c} bike(s)" for c in classes],
                        "Probability": [f"{p*100:.1f}%" for p in proba]
                    })
                    st.dataframe(prob_df, hide_index=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.subheader("📊 Model Comparison")

        if artifact_exists("artifacts/models/accuracy_quantity.png"):
            st.image("artifacts/models/accuracy_quantity.png",
                    caption="Accuracy: LR vs RF")
        if artifact_exists("artifacts/models/f1_quantity.png"):
            st.image("artifacts/models/f1_quantity.png",
                    caption="F1 Score: LR vs RF")

        st.subheader("🔍 Feature Importance")
        if artifact_exists("artifacts/models/feature_importance_quantity.png"):
            st.image("artifacts/models/feature_importance_quantity.png")

        st.subheader("📉 Confusion Matrices")
        c1, c2 = st.columns(2)
        with c1:
            if artifact_exists("artifacts/models/cm_lr_quantity.png"):
                st.image("artifacts/models/cm_lr_quantity.png",
                        caption="Logistic Regression")
        with c2:
            if artifact_exists("artifacts/models/cm_rf_quantity.png"):
                st.image("artifacts/models/cm_rf_quantity.png",
                        caption="Random Forest")


# ============================================================
# PAGE 3 — PREDICTION 2: BIKE MODEL
# ============================================================
elif page == "🚲 Prediction 2: Bike Model":
    st.title("🚲 Prediction 2 — Bike Model Recommendation")
    st.markdown("*Which bike should we recommend to this customer?*")
    st.divider()

    df = load_features()
    encoders = load_encoders()

    if df is None or encoders is None:
        st.warning("⏳ Run the pipeline first.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Try It Live")
        age = st.slider("Customer Age", 18, 70, 35, key="age2")
        gender = st.selectbox("Customer Gender",
                              ["Male", "Female", "Unknown"], key="gender2")
        price = st.slider("Price ($)", 200, 5000, 1000, key="price2")
        quantity = st.slider("Quantity", 1, 5, 1, key="qty2")
        store = st.selectbox("Store Location",
                             sorted(df["Store_Location"].unique()), key="store2")
        payment = st.selectbox("Payment Method",
                               sorted(df["Payment_Method"].unique()), key="pay2")
        season = st.selectbox("Season",
                              ["Winter", "Spring", "Summer", "Fall"], key="season2")

        if st.button("🚲 Recommend Bike", type="primary"):
            try:
                price_tier = "Budget" if price < 500 else "Mid" if price <= 2000 else "Premium"
                age_group = (
                    "18-24" if age <= 24 else
                    "25-34" if age <= 34 else
                    "35-44" if age <= 44 else
                    "45-54" if age <= 54 else
                    "55-64" if age <= 64 else "65+"
                )

                all_features = {
                    "Price": price,
                    "Quantity": quantity,
                    "Customer_Age": age,
                    "Salesperson_ID": 500,
                    "Month": 6,
                    "Day_of_Week": 1,
                    "Is_Weekend": 0,
                    "Store_Location_enc": safe_encode(encoders["Store_Location"], store),
                    "Customer_Gender_enc": safe_encode(encoders["Customer_Gender"], gender),
                    "Payment_Method_enc": safe_encode(encoders["Payment_Method"], payment),
                    "Season_enc": safe_encode(encoders["Season"], season),
                    "Age_Group_enc": safe_encode(encoders["Age_Group"], age_group),
                    "Price_Tier_enc": safe_encode(encoders["Price_Tier"], price_tier),
                }

                rf_model = load_model("artifacts/models/bike_model_rf.pkl")

                if rf_model:
                    X = build_feature_matrix(rf_model, all_features)
                    pred_enc = rf_model.predict(X)[0]
                    proba = rf_model.predict_proba(X)[0]

                    # Decode prediction
                    bike_encoder = encoders["Bike_Model"]
                    pred_label = bike_encoder.inverse_transform([pred_enc])[0]
                    classes_decoded = bike_encoder.inverse_transform(
                        rf_model.classes_
                    )

                    st.success(f"🚲 Recommended Bike: **{pred_label}**")

                    st.markdown("**Probability by Bike Model:**")
                    prob_df = pd.DataFrame({
                        "Bike Model": classes_decoded,
                        "Probability": [f"{p*100:.1f}%" for p in proba]
                    }).sort_values("Probability", ascending=False)
                    st.dataframe(prob_df, hide_index=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.subheader("📊 Model Comparison")
        if artifact_exists("artifacts/models/accuracy_bike_model.png"):
            st.image("artifacts/models/accuracy_bike_model.png",
                    caption="Accuracy: LR vs RF")
        if artifact_exists("artifacts/models/f1_bike_model.png"):
            st.image("artifacts/models/f1_bike_model.png",
                    caption="F1 Score: LR vs RF")

        st.subheader("🔍 Feature Importance")
        if artifact_exists("artifacts/models/feature_importance_bike_model.png"):
            st.image("artifacts/models/feature_importance_bike_model.png")

        st.subheader("📉 Confusion Matrices")
        c1, c2 = st.columns(2)
        with c1:
            if artifact_exists("artifacts/models/cm_lr_bike_model.png"):
                st.image("artifacts/models/cm_lr_bike_model.png",
                        caption="Logistic Regression")
        with c2:
            if artifact_exists("artifacts/models/cm_rf_bike_model.png"):
                st.image("artifacts/models/cm_rf_bike_model.png",
                        caption="Random Forest")


# ============================================================
# PAGE 4 — PREDICTION 3: CASH PAYMENT
# ============================================================
elif page == "💳 Prediction 3: Cash Payment":
    st.title("💳 Prediction 3 — Cash Payment Intelligence")
    st.markdown("*Would a universal cash discount attract new customers?*")
    st.divider()

    df = load_features()
    encoders = load_encoders()
    clean_df = load_clean_data()

    if df is None or encoders is None:
        st.warning("⏳ Run the pipeline first.")
        st.stop()

    # Section 1 — Purchase frequency by age group
    st.subheader("📊 Purchase Frequency by Age Group")
    clean_df["Age_Group"] = pd.cut(
        clean_df["Customer_Age"],
        bins=[17, 24, 34, 44, 54, 64, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    )
    age_counts = clean_df["Age_Group"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=age_counts.index.astype(str),
                y=age_counts.values, ax=ax)
    ax.set_title("Transaction Count by Age Group", fontweight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Transactions")
    st.pyplot(fig)
    plt.close()

    # Section 2 — Cash rate by age group
    st.subheader("💵 Cash Payment Rate by Age Group")
    clean_df["Is_Cash"] = (clean_df["Payment_Method"] == "Cash").astype(int)
    cash_rate = clean_df.groupby(
        "Age_Group", observed=True
    )["Is_Cash"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=cash_rate.index.astype(str),
                y=cash_rate.values, ax=ax)
    ax.set_title("Cash Payment Rate by Age Group (%)", fontweight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Cash Rate (%)")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Section 3 — Live prediction
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Cash Payment Probability")
        age = st.slider("Customer Age", 18, 70, 35, key="age3")
        gender = st.selectbox("Customer Gender",
                              ["Male", "Female", "Unknown"], key="gender3")
        bike_model = st.selectbox("Bike Model",
                                  sorted(df["Bike_Model"].unique()), key="bike3")
        price = st.slider("Price ($)", 200, 5000, 1000, key="price3")
        store = st.selectbox("Store Location",
                             sorted(df["Store_Location"].unique()), key="store3")
        season = st.selectbox("Season",
                              ["Winter", "Spring", "Summer", "Fall"],
                              key="season3")

        if st.button("💳 Predict Cash Probability", type="primary"):
            try:
                price_tier = "Budget" if price < 500 else \
                             "Mid" if price <= 2000 else "Premium"
                age_group = (
                    "18-24" if age <= 24 else
                    "25-34" if age <= 34 else
                    "35-44" if age <= 44 else
                    "45-54" if age <= 54 else
                    "55-64" if age <= 64 else "65+"
                )

                all_features = {
                    "Price": price,
                    "Quantity": 1,
                    "Customer_Age": age,
                    "Salesperson_ID": 500,
                    "Month": 6,
                    "Day_of_Week": 1,
                    "Is_Weekend": 0,
                    "Bike_Model_enc": safe_encode(
                        encoders["Bike_Model"], bike_model),
                    "Store_Location_enc": safe_encode(
                        encoders["Store_Location"], store),
                    "Customer_Gender_enc": safe_encode(
                        encoders["Customer_Gender"], gender),
                    "Season_enc": safe_encode(encoders["Season"], season),
                    "Age_Group_enc": safe_encode(
                        encoders["Age_Group"], age_group),
                    "Price_Tier_enc": safe_encode(
                        encoders["Price_Tier"], price_tier),
                }

                rf_model = load_model("artifacts/models/payment_rf.pkl")

                if rf_model:
                    X = build_feature_matrix(rf_model, all_features)
                    proba = rf_model.predict_proba(X)[0]
                    cash_prob = proba[1] * 100

                    if cash_prob >= 50:
                        st.success(
                            f"💵 **{cash_prob:.1f}%** probability of cash payment"
                        )
                    else:
                        st.info(
                            f"💳 **{cash_prob:.1f}%** probability of cash payment"
                        )

                    # Gauge
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.barh(["Cash Probability"],
                            [cash_prob], color="#55A868")
                    ax.barh(["Cash Probability"],
                            [100 - cash_prob], left=[cash_prob],
                            color="#E8E8E8")
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Probability (%)")
                    ax.text(cash_prob / 2, 0,
                            f"{cash_prob:.1f}%",
                            ha="center", va="center",
                            fontweight="bold", color="white")
                    ax.set_title("Cash Payment Probability")
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.subheader("📊 Model Comparison")
        if artifact_exists("artifacts/models/accuracy_payment.png"):
            st.image("artifacts/models/accuracy_payment.png",
                    caption="Accuracy: LR vs RF")
        if artifact_exists("artifacts/models/f1_payment.png"):
            st.image("artifacts/models/f1_payment.png",
                    caption="F1 Score: LR vs RF")

    st.divider()

    # Section 4 — Business Recommendation
    st.subheader("💡 Universal Cash Discount Recommendation")

    total = len(clean_df)
    cash_total = clean_df["Is_Cash"].sum()
    cash_pct = round(cash_total / total * 100, 1)
    low_freq = age_counts.sort_values().head(2).index.tolist()

    st.markdown(f"""
    <div style="background:#f0f7ff; padding:24px; border-radius:12px;
                border-left:4px solid #2E75B6;">

    <h4>📊 Cash Payment Analysis</h4>

    <p><strong>Current cash transaction rate:</strong>
       {cash_pct}% of all sales ({cash_total:,} / {total:,})</p>

    <p><strong>Underrepresented age groups:</strong><br>
    → Ages <strong>{low_freq[0]}</strong>: lowest purchase frequency<br>
    → Ages <strong>{low_freq[1]}</strong>: second lowest frequency</p>

    <h4>💡 Recommendation</h4>
    <p>A <strong>universal cash discount</strong> — available equally to
    <strong>ALL customers</strong> — removes the payment barrier for
    customers who may not have access to credit cards, financing,
    or digital payment methods.</p>

    <p><strong>Expected segments most likely to respond:</strong><br>
    → Ages 18–24: limited credit history<br>
    → Ages 65+: may prefer or rely on cash transactions</p>

    <h4>⚠️ Important</h4>
    <p>This discount applies <strong>universally to all customers</strong>.
    The analysis identifies who benefits most — it does <strong>not</strong>
    determine who qualifies. No customer is excluded or targeted based on
    age, gender, or any personal characteristic.</p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(CASH_PAYMENT_CITATIONS)


# ============================================================
# PAGE 5 — MODEL DOCUMENTATION
# ============================================================
elif page == "📋 Model Documentation":
    st.title("📋 Model Documentation")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Model Card",
        "📊 Evaluation Report",
        "🔒 Dataset Contract",
        "⬇️ Downloads"
    ])

    with tab1:
        content = read_markdown("artifacts/model_card.md")
        if content:
            st.markdown(content)
        else:
            st.warning("Model card not found. Run the pipeline first.")

    with tab2:
        content = read_markdown("artifacts/evaluation_report.md")
        if content:
            st.markdown(content)
        else:
            st.warning("Evaluation report not found. Run the pipeline first.")

    with tab3:
        if artifact_exists("artifacts/dataset_contract.json"):
            with open("artifacts/dataset_contract.json", "r") as f:
                contract = json.load(f)
            st.json(contract)
        else:
            st.warning("Dataset contract not found. Run the pipeline first.")

    with tab4:
        st.subheader("⬇️ Download Artifacts")
        downloads = {
            "clean_data.csv":       "artifacts/clean_data.csv",
            "features.csv":         "artifacts/features.csv",
            "evaluation_report.md": "artifacts/evaluation_report.md",
            "model_card.md":        "artifacts/model_card.md",
            "insights.md":          "artifacts/insights.md",
            "dataset_contract.json":"artifacts/dataset_contract.json",
        }
        for name, path in downloads.items():
            if artifact_exists(path):
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ {name}",
                        data=f,
                        file_name=name,
                        mime="application/octet-stream"
                    )
            else:
                st.markdown(f"⏳ `{name}` — not generated yet")