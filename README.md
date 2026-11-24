# 🎓 Student Performance ML Pipeline & Interactive App

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=0B1120)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random_Forest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white&labelColor=0B1120)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=0B1120)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Live_Simulator-Portfolio_Data_Lab-06B6D4?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0B1120)](https://ikram-elkarkadi.github.io/#datalab)

</div>

An end-to-end, reproducible **Machine Learning Pipeline & Interactive Streamlit Application** for predicting student academic performance (`G3` score `0–20` and pass/fail classification), inspired by the UCI Student Performance dataset.

---

## 🏗️ Architecture & Project Structure

```text
student-performance-ml-app/
├── src/
│   └── pipeline.py        # Data generation, StandardScaler + RandomForest pipeline, metrics
├── tests/
│   └── test_pipeline.py   # Pytest unit tests for schema, R², accuracy & prediction bounds
├── app.py                 # Interactive Streamlit dashboard with Plotly feature importance
├── requirements.txt       # Reproducible Python dependencies
└── README.md              # Academic documentation
```

## 🔬 Methodology
1. **Preprocessing**: `StandardScaler` inside a `scikit-learn.pipeline.Pipeline` to prevent data leakage.
2. **Dual Modeling**:
   - **Regression**: `RandomForestRegressor(n_estimators=200, max_depth=8)` predicting final grade `G3`.
   - **Classification**: `RandomForestClassifier(n_estimators=200, max_depth=8)` predicting pass probability (`G3 >= 10`).
3. **Evaluation**: `R²`, `MAE`, `Accuracy`, `F1-Score`, and Feature Importance ranking (`G2`, `G1`, `studytime`, `failures`, `absences`).

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python -m src.pipeline
pytest -v
streamlit run app.py
```

---
> Developed by **Ikram Elkarkadi** · Bachelor in Data Science (SD) @ EST Kelaâ des Sraghna, Université Cadi Ayyad 🇲🇦
