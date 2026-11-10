
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Federated Healthcare AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ---------------- THEME ----------------
st.markdown("""
<style>
    .stApp {
        background: #f4f8fc;
        color: #172b4d;
    }
    [data-testid="stSidebar"] {
        background: #102a43;
    }
    [data-testid="stSidebar"] * {
        color: #f0f7ff !important;
    }
    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #087e8b);
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #e0f7fa;
        margin-bottom: 0;
    }
    .metric-note {
        color: #627d98;
        font-size: 13px;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border: 1px solid #d9e5f2;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOADING ----------------
def load_json(filename):
    path = RESULTS / filename
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

flower = load_json("flower_metrics.json")

comparison_path = RESULTS / "model_comparison.csv"
comparison = (
    pd.read_csv(comparison_path)
    if comparison_path.exists()
    else pd.DataFrame()
)

confusion_path = RESULTS / "flower_confusion_matrix.csv"
confusion = (
    pd.read_csv(confusion_path, index_col=0)
    if confusion_path.exists()
    else pd.DataFrame()
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🏥 HealthAI")
    st.caption("Federated Learning Platform")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Model Comparison",
            "Confusion Matrix",
            "Hospital Network",
            "About Project",
        ],
    )
    st.divider()
    st.caption("Academic research prototype")
    st.caption("No real patient data used")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>Federated Healthcare Intelligence</h1>
    <p>
        Privacy-conscious collaborative model training across
        three simulated hospitals.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("Model Performance Overview")

    if flower:
        metric_cols = st.columns(4)
        metric_items = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1_score"),
        ]

        for col, (label, key) in zip(metric_cols, metric_items):
            value = flower.get(key)
            if value is not None:
                col.metric(label, f"{value * 100:.2f}%")
            else:
                col.metric(label, "N/A")

        auc = flower.get("roc_auc")
        samples = flower.get("test_samples", "N/A")

        c1, c2 = st.columns(2)
        c1.metric(
            "ROC-AUC",
            f"{auc * 100:.2f}%" if auc is not None else "N/A",
        )
        c2.metric("Test Samples", samples)
    else:
        st.warning(
            "flower_metrics.json could not be loaded. "
            "Check the results folder."
        )

    st.subheader("Performance Across Models")
    if not comparison.empty:
        metric_names = [
            "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"
        ]
        available = [
            col for col in metric_names if col in comparison.columns
        ]

        chart_data = comparison.melt(
            id_vars="Model",
            value_vars=available,
            var_name="Metric",
            value_name="Score (%)",
        )

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score (%)",
            color="Model",
            barmode="group",
            template="plotly_white",
            color_discrete_sequence=["#94a3b8", "#0891b2", "#0f766e"],
        )
        fig.update_layout(
            height=430,
            legend_title="Training approach",
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("model_comparison.csv is not available.")

# ---------------- MODEL COMPARISON ----------------
elif page == "Model Comparison":
    st.subheader("Compare Training Approaches")

    if not comparison.empty:
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        if "Accuracy" in comparison.columns:
            fig = px.bar(
                comparison,
                x="Model",
                y="Accuracy",
                color="Model",
                text="Accuracy",
                template="plotly_white",
                color_discrete_sequence=["#94a3b8", "#0891b2", "#0f766e"],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model comparison data is missing.")

# ---------------- CONFUSION MATRIX ----------------
elif page == "Confusion Matrix":
    st.subheader("Federated Model — Confusion Matrix")

    if not confusion.empty:
        st.dataframe(confusion, use_container_width=True)

        fig = px.imshow(
            confusion,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Teal",
            labels=dict(x="Predicted label", y="Actual label", color="Count"),
        )
        fig.update_layout(template="plotly_white", height=430)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("flower_confusion_matrix.csv is missing.")

# ---------------- HOSPITAL NETWORK ----------------
elif page == "Hospital Network":
    st.subheader("Simulated Hospital Network")

    hospital_data = pd.DataFrame({
        "Hospital": ["Hospital 1", "Hospital 2", "Hospital 3"],
        "Training Samples": [117, 117, 119],
        "Client Status": ["Participated", "Participated", "Participated"],
    })

    a, b = st.columns(2)
    a.metric("Simulated Hospitals", 3)
    b.metric("Completed Federated Rounds", 5)

    st.dataframe(
        hospital_data,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Each simulated hospital trains locally. The Flower FedAvg server "
        "aggregates model parameters; raw training rows are not intentionally "
        "sent to the server by this prototype."
    )

# ---------------- ABOUT ----------------
elif page == "About Project":
    st.subheader("About This Research Prototype")

    st.write("""
    This project demonstrates federated learning with three simulated
    hospital clients using Flower and Federated Averaging (FedAvg).

    **Dataset limitation:** The experiment uses the scikit-learn diabetes
    disease-progression dataset, with its continuous target converted into
    binary above-median/below-median labels. It is not a real diabetes
    diagnosis dataset and this application is not clinical software.

    Federated learning can reduce the need to centralize raw training data,
    but model updates can still leak information. This prototype does not
    implement or claim differential privacy or secure aggregation.
    """)

    st.write("Training rounds: 5")
    st.write("Test samples: 89")

st.divider()
st.caption(
    "Federated Healthcare AI • Educational prototype • "
    "Not intended for clinical decision-making"
)
