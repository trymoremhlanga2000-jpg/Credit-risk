from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import category_encoders as ce

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Trymore | Credit Risk System",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------
st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #dbeafe;
}

/* Main background */
.main {
    background-color: #f8fafc;
}

/* Headings */
h1, h2, h3 {
    color: #1e3a8a;
}

/* Footer branding */
.footer {
    position: fixed;
    bottom: 10px;
    right: 20px;
    font-size: 12px;
    color: gray;
}

/* Card styling */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA & MODEL
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def load_model():
    return joblib.load("credit_model_pipeline.pkl")

df = load_data()
model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("**Developed by Trymore Mhlanga**")

page = st.sidebar.radio(
    "Go to:",
    ["Overview", "Exploratory Analysis", "Credit Risk Prediction"]
)

# ---------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------
if page == "Overview":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.title("💳 Credit Risk Assessment System")
    st.markdown("""
    This system evaluates the **probability of customer default** using historical,
    demographic, and behavioral indicators.

    **Developed by:** *Trymore Mhlanga*
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(df))

    with col2:
        st.metric("Default Rate", f"{df['DefaultStatus'].mean()*100:.1f}%")

    with col3:
        st.metric("Non-Default Rate", f"{(1-df['DefaultStatus'].mean())*100:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------
elif page == "Exploratory Analysis":

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📊 Exploratory Data Analysis")
    st.markdown("**Developed by Trymore Mhlanga**")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            df, x="Age", nbins=30,
            title="Age Distribution",
            color_discrete_sequence=["#2563eb"]
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(
            df, names="DefaultStatus",
            title="Default Distribution",
            hole=0.4,
            color_discrete_sequence=["#16a34a", "#dc2626"]
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.bar(
            df, x="Employment_Status", y="DefaultStatus",
            title="Default by Employment",
            color="Employment_Status"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.bar(
            df, x="Level_of_Education", y="DefaultStatus",
            title="Default by Education Level",
            color="Level_of_Education"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------
elif page == "Credit Risk Prediction":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.title("🔍 Credit Risk Prediction")
    st.markdown("**Developed by Trymore Mhlanga**")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col2:
        employment = st.selectbox("Employment Status", ["Employed", "Unemployed"])
        education = st.selectbox("Level of Education", ["O Level", "A Level", "College"])
        residence = st.selectbox("Residence Area", ["Low Density", "Medium Density", "High Density"])

    with col3:
        dependants = st.slider("Number of Dependants", 0, 10, 0)
        quantity = st.slider("Product Quantity", 1, 20, 1)
        amount = st.number_input("Total Amount", 0.0)

    tenure = st.selectbox("Tenure", ["6 Months", "12 Months"])
    home = st.selectbox("Home Ownership", ["Tenant", "Employer", "Relative/Guardian"])
    other_solar = st.selectbox("Other Solar Product on Installments", ["Yes", "No"])

    input_df = pd.DataFrame([{
        "Age": age,
        "Number_of_Dependants": dependants,
        "Product_Quantity": quantity,
        "Total_Amount": amount,
        "Gender": gender,
        "Marital_Status": marital_status,
        "Employment_Status": employment,
        "Home_Ownership": home,
        "Level_of_Education": education,
        "Residence_Area": residence,
        "Tenure": tenure,
        "Other_Solar_Product_purchase_on_installments": other_solar
    }])

    if st.button("Predict Risk"):
        prob = model.predict_proba(input_df)[0][1]

        if prob < 0.3:
            risk_label = "LOW RISK"
            color = "green"
        elif prob < 0.6:
            risk_label = "MEDIUM RISK"
            color = "orange"
        else:
            risk_label = "HIGH RISK"
            color = "red"

        st.markdown(f"""
        <div class="card">
            <h3 style="color:{color};">Risk Level: {risk_label}</h3>
            <h4>Probability of Default: {prob:.2%}</h4>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob * 100))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown(
    "<div class='footer'>Developed by Trymore Mhlanga</div>",
    unsafe_allow_html=True
)
