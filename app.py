"""
Bank Customer Churn Risk Dashboard
-----------------------------------
Built for: Predictive Modeling and Risk Scoring for Bank Customer Churn
A BBA capstone project | European Central Bank case study

Run with:  streamlit run app.py
Requires:  model_artifact.pkl in the same folder (created by train_and_save.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bank Churn Risk Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------
# Load model + data
# ---------------------------------------------------------------
@st.cache_resource
def load_artifact():
    with open("model_artifact.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_artifact()
model = artifact["model"]
feature_columns = artifact["feature_columns"]
X_test = artifact["X_test"]
y_test = artifact["y_test"]
test_probabilities = artifact["test_probabilities"]
raw_sample = artifact["raw_sample"]

FEATURE_LABELS = {
    "CreditScore": "Credit Score",
    "Age": "Age",
    "Tenure": "Tenure (years)",
    "Balance": "Account Balance",
    "NumOfProducts": "Number of Products",
    "HasCrCard": "Has Credit Card",
    "IsActiveMember": "Active Member",
    "EstimatedSalary": "Estimated Salary",
    "BalanceSalaryRatio": "Balance-to-Salary Ratio",
    "ProductDensity": "Product Density",
    "EngagementProductInteraction": "Engagement x Product Score",
    "AgeTenureInteraction": "Age x Tenure Score",
    "ZeroBalanceFlag": "Zero Balance Flag",
    "Geography_Germany": "Geography: Germany",
    "Geography_Spain": "Geography: Spain",
    "Gender_Male": "Gender: Male",
}


def build_feature_row(credit_score, geography, gender, age, tenure, balance,
                       num_products, has_cr_card, is_active, salary):
    """Turn raw, human-friendly inputs into the engineered feature row the model expects."""
    row = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary,
    }
    row["BalanceSalaryRatio"] = balance / (salary + 1)
    row["ProductDensity"] = num_products / (tenure + 1)
    row["EngagementProductInteraction"] = int(is_active) * num_products
    row["AgeTenureInteraction"] = age * tenure
    row["ZeroBalanceFlag"] = 1 if balance == 0 else 0
    row["Geography_Germany"] = 1 if geography == "Germany" else 0
    row["Geography_Spain"] = 1 if geography == "Spain" else 0
    row["Gender_Male"] = 1 if gender == "Male" else 0
    return pd.DataFrame([row])[feature_columns]


def risk_band(prob):
    if prob < 0.30:
        return "Low Risk", "#2ecc71"
    elif prob < 0.60:
        return "Medium Risk", "#f39c12"
    else:
        return "High Risk", "#e74c3c"


# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("🏦 Churn Risk Dashboard")
st.sidebar.caption("Predictive Modeling & Risk Scoring for Bank Customer Churn")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Churn Risk Calculator", "Probability Distribution",
     "Feature Importance", "What-If Simulator"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** Random Forest Classifier  \n"
    "**Test ROC-AUC:** 0.868  \n"
    "**Dataset:** 10,000 European bank customers"
)

# ---------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------
if page == "Overview":
    st.title("Bank Customer Churn — Risk Intelligence Dashboard")
    st.markdown(
        "This dashboard turns a bank's raw customer data into **early-warning churn scores**, "
        "so retention teams can act *before* a customer leaves rather than after."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Analyzed", "10,000")
    c2.metric("Historical Churn Rate", "20.4%")
    c3.metric("Model Accuracy", "82.6%")
    c4.metric("Model ROC-AUC", "0.868")

    st.markdown("### What this dashboard does")
    st.markdown(
        "- **Churn Risk Calculator** — enter one customer's details and get an instant churn probability.\n"
        "- **Probability Distribution** — see how risk is spread across the whole customer base.\n"
        "- **Feature Importance** — understand *which* factors drive churn the most.\n"
        "- **What-If Simulator** — test how a retention action (e.g. re-engaging a customer) would change their risk score."
    )

    st.markdown("### Churn rate by key segments")
    col1, col2 = st.columns(2)
    with col1:
        geo_churn = raw_sample.groupby("Geography")["Exited"].mean().reset_index()
        geo_churn["Exited"] = geo_churn["Exited"] * 100
        fig = px.bar(geo_churn, x="Geography", y="Exited", color="Geography",
                     labels={"Exited": "Churn Rate (%)"}, title="Churn Rate by Country")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        prod_churn = raw_sample.groupby("NumOfProducts")["Exited"].mean().reset_index()
        prod_churn["Exited"] = prod_churn["Exited"] * 100
        fig = px.bar(prod_churn, x="NumOfProducts", y="Exited",
                     labels={"Exited": "Churn Rate (%)", "NumOfProducts": "Number of Products"},
                     title="Churn Rate by Number of Products")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# PAGE: Churn Risk Calculator
# ---------------------------------------------------------------
elif page == "Churn Risk Calculator":
    st.title("🎯 Churn Risk Calculator")
    st.markdown("Enter a customer's profile to calculate their probability of churning.")

    with st.form("calculator_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            age = st.slider("Age", 18, 92, 40)
            tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
            balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
        with col3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
            has_cr_card = st.checkbox("Has Credit Card", value=True)
            is_active = st.checkbox("Active Member", value=True)
            salary = st.number_input("Estimated Salary (€)", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)

        submitted = st.form_submit_button("Calculate Churn Risk", type="primary", use_container_width=True)

    if submitted:
        X_input = build_feature_row(credit_score, geography, gender, age, tenure,
                                     balance, num_products, has_cr_card, is_active, salary)
        prob = model.predict_proba(X_input)[0, 1]
        band, color = risk_band(prob)

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#eafaf1"},
                        {"range": [30, 60], "color": "#fef5e7"},
                        {"range": [60, 100], "color": "#fdedec"},
                    ],
                },
                title={"text": "Churn Probability"},
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"### Risk Level: :{'green' if band=='Low Risk' else 'orange' if band=='Medium Risk' else 'red'}[{band}]")
            if band == "High Risk":
                st.error("This customer shows a high likelihood of churning. Recommend immediate, personalized retention outreach.")
            elif band == "Medium Risk":
                st.warning("This customer shows elevated churn risk. Consider proactive engagement or a loyalty offer.")
            else:
                st.success("This customer shows low churn risk. Standard engagement is likely sufficient.")

            st.markdown("**Key factors for this customer:**")
            notes = []
            if not is_active:
                notes.append("Inactive membership status increases risk substantially.")
            if num_products >= 3:
                notes.append("Holding 3+ products is strongly associated with higher churn.")
            if age >= 50:
                notes.append("Older customers churn more frequently in this dataset.")
            if geography == "Germany":
                notes.append("German customers show higher churn rates than France/Spain.")
            if not notes:
                notes.append("No major red-flag factors detected for this customer.")
            for n in notes:
                st.markdown(f"- {n}")

# ---------------------------------------------------------------
# PAGE: Probability Distribution
# ---------------------------------------------------------------
elif page == "Probability Distribution":
    st.title("📊 Churn Probability Distribution")
    st.markdown("How churn risk is distributed across the bank's customer base (test set of 2,000 customers).")

    dist_df = pd.DataFrame({
        "Probability": test_probabilities,
        "Actual Outcome": np.where(y_test.values == 1, "Churned", "Retained")
    })

    fig = px.histogram(dist_df, x="Probability", color="Actual Outcome", nbins=40,
                        barmode="overlay", opacity=0.7,
                        color_discrete_map={"Churned": "#e74c3c", "Retained": "#2874a6"},
                        title="Distribution of Predicted Churn Probabilities")
    fig.update_layout(xaxis_title="Predicted Churn Probability", yaxis_title="Number of Customers")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Risk segmentation")
    bands = pd.cut(dist_df["Probability"], bins=[0, 0.3, 0.6, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"])
    band_counts = bands.value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Low Risk Customers", int(band_counts["Low Risk"]))
    c2.metric("Medium Risk Customers", int(band_counts["Medium Risk"]))
    c3.metric("High Risk Customers", int(band_counts["High Risk"]))

    fig2 = px.pie(values=band_counts.values, names=band_counts.index,
                  color=band_counts.index,
                  color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"},
                  title="Share of Customers by Risk Band")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------
# PAGE: Feature Importance
# ---------------------------------------------------------------
elif page == "Feature Importance":
    st.title("🔍 What Drives Churn? — Feature Importance")
    st.markdown("Ranking of which customer attributes matter most to the model's predictions.")

    importance = pd.Series(model.feature_importances_, index=feature_columns)
    importance = importance.rename(index=FEATURE_LABELS).sort_values(ascending=True)

    fig = px.bar(importance, orientation="h",
                 labels={"value": "Importance Score", "index": "Feature"},
                 title="Feature Importance (Random Forest)")
    fig.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Business interpretation")
    st.markdown(
        "- **Age** is the single biggest driver — older customers are meaningfully more likely to churn.\n"
        "- **Number of Products** matters a lot, but non-linearly: 2 products is the 'sweet spot'; 3–4 products signals trouble, not loyalty.\n"
        "- **Engagement x Product Score** — an inactive customer with multiple products is a distinctly higher-risk combination than either factor alone.\n"
        "- **Account Balance** and **Geography: Germany** round out the top drivers, pointing to wealth-related flight risk and a country-specific issue worth investigating."
    )

# ---------------------------------------------------------------
# PAGE: What-If Simulator
# ---------------------------------------------------------------
elif page == "What-If Simulator":
    st.title("🧪 What-If Scenario Simulator")
    st.markdown(
        "Start from a customer profile, then adjust engagement or product usage to see "
        "**how much a retention action could reduce their churn risk.**"
    )

    st.markdown("#### Step 1 — Baseline customer profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 600, key="wi_credit")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"], key="wi_geo")
        gender = st.selectbox("Gender", ["Male", "Female"], key="wi_gender")
    with col2:
        age = st.slider("Age", 18, 92, 52, key="wi_age")
        tenure = st.slider("Tenure (years)", 0, 10, 3, key="wi_tenure")
        balance = st.number_input("Account Balance (€)", 0.0, 300000.0, 120000.0, step=1000.0, key="wi_balance")
    with col3:
        salary = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 90000.0, step=1000.0, key="wi_salary")
        has_cr_card = st.checkbox("Has Credit Card", value=True, key="wi_cc")

    st.markdown("#### Step 2 — Baseline engagement (before retention action)")
    col1, col2 = st.columns(2)
    with col1:
        base_products = st.selectbox("Current Number of Products", [1, 2, 3, 4], index=2, key="base_products")
    with col2:
        base_active = st.radio("Current Activity Status", ["Inactive", "Active"], index=0, key="base_active")

    st.markdown("#### Step 3 — Simulated retention action (after intervention)")
    col1, col2 = st.columns(2)
    with col1:
        new_products = st.selectbox("Adjusted Number of Products", [1, 2, 3, 4], index=1, key="new_products")
    with col2:
        new_active = st.radio("Adjusted Activity Status", ["Inactive", "Active"], index=1, key="new_active")

    base_row = build_feature_row(credit_score, geography, gender, age, tenure, balance,
                                  base_products, has_cr_card, base_active == "Active", salary)
    new_row = build_feature_row(credit_score, geography, gender, age, tenure, balance,
                                 new_products, has_cr_card, new_active == "Active", salary)

    base_prob = model.predict_proba(base_row)[0, 1]
    new_prob = model.predict_proba(new_row)[0, 1]
    delta = new_prob - base_prob

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Churn Risk", f"{base_prob*100:.1f}%")
    c2.metric("Simulated Churn Risk", f"{new_prob*100:.1f}%", delta=f"{delta*100:.1f} pp", delta_color="inverse")
    band, color = risk_band(new_prob)
    c3.metric("New Risk Band", band)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Baseline", "After Retention Action"], y=[base_prob*100, new_prob*100],
                          marker_color=["#95a5a6", color], text=[f"{base_prob*100:.1f}%", f"{new_prob*100:.1f}%"],
                          textposition="outside"))
    fig.update_layout(title="Impact of Retention Action on Churn Probability",
                       yaxis_title="Churn Probability (%)", yaxis_range=[0, 100], height=400)
    st.plotly_chart(fig, use_container_width=True)

    if delta < 0:
        st.success(f"This action reduces churn risk by {abs(delta)*100:.1f} percentage points.")
    elif delta > 0:
        st.warning(f"This change actually increases churn risk by {delta*100:.1f} percentage points.")
    else:
        st.info("This change has no meaningful effect on churn risk.")
