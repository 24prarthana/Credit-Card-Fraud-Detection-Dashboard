import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD FILES ----------------

df = pd.read_csv("creditcard.csv")

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ----------------

page = st.sidebar.radio(
    "📂 Navigation",
    [
        "Dashboard",
        "Dataset Analysis",
        "Model Performance",
        "Prediction",
        "About Project"
    ]
)

# ================= DASHBOARD =================

if page == "Dashboard":

    st.title("💳 Credit Card Fraud Detection Dashboard")

    fraud = int(df["Class"].sum())
    genuine = len(df) - fraud
    fraud_rate = (fraud / len(df)) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{len(df):,}"
    )

    c2.metric(
        "Fraud Transactions",
        fraud
    )

    c3.metric(
        "Genuine Transactions",
        genuine
    )

    c4.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("Fraud Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["Class"].value_counts()

    ax.bar(
        ["Genuine", "Fraud"],
        [counts[0], counts[1]],
        color=["green", "red"]
    )

    ax.set_title("Fraud vs Genuine Transactions")
    ax.set_ylabel("Count")

    st.pyplot(fig)

# ================= DATASET ANALYSIS =================

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"Rows: {df.shape[0]}")

    with col2:
        st.info(f"Columns: {df.shape[1]}")

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={
                "index": "Column",
                0: "Missing Values"
            }
        )
    )

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# ================= MODEL PERFORMANCE =================

elif page == "Model Performance":

    st.title("📈 Random Forest Model Performance")

    accuracy = 0.9991533623609096
    precision = 0.7432432432432432
    recall = 0.859375
    f1 = 0.7971014492753623

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    c2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    c3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

    st.divider()

    st.subheader("Performance Metrics Comparison")

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        metrics.keys(),
        metrics.values(),
        color=["blue", "green", "orange", "red"]
    )
    ax.set_ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.3f}",
            ha="center"
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest Performance")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height:.3f}",
            ha="center"
        )

    st.pyplot(fig)

    st.success("""
✅ Accuracy: 99.92%

✅ Precision: 74.32%

✅ Recall: 85.94%

✅ F1 Score: 79.71%
""")

# ================= PREDICTION PAGE =================

elif page == "Prediction":

    st.title("🤖 Random Forest Fraud Prediction")

    st.info(
        "Select any transaction from the dataset and let the model predict whether it is fraudulent or genuine."
    )

    row_number = st.slider(
        "Select Transaction",
        0,
        len(df)-1,
        0
    )

    sample = df.drop("Class", axis=1).iloc[row_number]

    st.dataframe(sample.to_frame())

    if st.button("Predict Fraud"):

        transaction = np.array(sample).reshape(1, -1)

        transaction_scaled = scaler.transform(
            transaction
        )

        prediction = model.predict(
            transaction_scaled
        )[0]

        probability = model.predict_proba(
            transaction_scaled
        )[0][1]

        if prediction == 1:

            st.error(
                f"🚨 FRAUD DETECTED\n\nConfidence: {probability:.2%}"
            )

        else:

            st.success(
                f"✅ GENUINE TRANSACTION\n\nFraud Probability: {probability:.2%}"
            )

            st.balloons()

# ================= ABOUT PAGE =================

else:

    st.title("📖 About Project")

    st.markdown("""
## Credit Card Fraud Detection Using Machine Learning

### Technologies Used
- Python
- Pandas
- Scikit-Learn
- Random Forest
- Streamlit
- Matplotlib

### Machine Learning Workflow
- Data Cleaning
- Duplicate Removal
- Feature Scaling
- Class Imbalance Handling
- Logistic Regression
- Random Forest Classification
- Model Evaluation

### Results
- Accuracy: 99.92%
- Precision: 74.32%
- Recall: 85.94%
- F1 Score: 79.71%

### Goal
Detect fraudulent credit card transactions using machine learning and provide interactive analytics through a Streamlit dashboard.
""")
