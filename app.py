"""Interactive Streamlit Dashboard for Student Performance ML Pipeline."""

import plotly.express as px
import streamlit as st
from src.pipeline import generate_uci_like_dataset, predict_student, train_models

st.set_page_config(page_title="Student Performance ML | Ikram Elkarkadi", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance ML Predictor")
st.caption("Built by Ikram Elkarkadi · Bachelor in Data Science (SD) · EST Kelaâ des Sraghna")

@st.cache_resource
def load_bundle():
    df = generate_uci_like_dataset()
    return df, train_models(df)

df, bundle = load_bundle()
metrics = bundle["metrics"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("R² Score", f"{metrics['r2']:.3f}")
c2.metric("MAE (pts)", f"{metrics['mae']:.2f}")
c3.metric("Pass Accuracy", f"{metrics['accuracy']*100:.1f}%")
c4.metric("F1 Score", f"{metrics['f1']:.3f}")

st.sidebar.header("Student Input Features")
studytime = st.sidebar.slider("Weekly Study Time (1-4)", 1, 4, 3)
failures = st.sidebar.slider("Past Failures (0-3)", 0, 3, 0)
absences = st.sidebar.slider("Absences (days)", 0, 25, 2)
g1 = st.sidebar.slider("First Period Grade (G1)", 0.0, 20.0, 14.0, 0.5)
g2 = st.sidebar.slider("Second Period Grade (G2)", 0.0, 20.0, 15.0, 0.5)

pred = predict_student(bundle, studytime, failures, absences, g1, g2)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Live Prediction")
    st.metric("Predicted Final Grade (G3)", f"{pred['predicted_g3']} / 20")
    st.progress(min(100, int(pred["pass_probability"])), text=f"Pass Probability: {pred['pass_probability']}%")

with col_right:
    st.subheader("Feature Importance (Random Forest)")
    imp = bundle["feature_importances"]
    fig = px.bar(x=list(imp.values()), y=list(imp.keys()), orientation="h", labels={"x": "Importance", "y": "Feature"})
    st.plotly_chart(fig, use_container_width=True)
