"""
CardioCheck - Streamlit interface (Step 6)
Name: Azwa Noor | Reg No: B24F0659AI069 | Class: BS AI F24 Red
Run:  streamlit run app.py   (after running cardiocheck_model.py once)
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

st.set_page_config(page_title="CardioCheck", page_icon="❤️", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load the trained pipeline, its saved metrics and the test-set probabilities."""
    model = joblib.load("cardiocheck_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta, np.load("test_proba.npy"), np.load("test_y.npy")


model, meta, test_proba, test_y = load_artifacts()

st.title("❤️ CardioCheck: Heart Disease Risk Screening Assistant")
st.warning("This tool is for education only and is not a medical diagnosis.")

# ------------------------------------------------------------ sidebar inputs
st.sidebar.header("Patient details")
age = st.sidebar.slider("Age (years)", 20, 80, 54)
sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female")
cp = st.sidebar.selectbox("Chest pain type (cp)", [1, 2, 3, 4], index=3, format_func={
    1: "1 - Typical angina", 2: "2 - Atypical angina",
    3: "3 - Non-anginal pain", 4: "4 - Asymptomatic"}.get)
trestbps = st.sidebar.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 120, 570, 245)
fbs = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                           format_func=lambda v: "Yes" if v else "No")
restecg = st.sidebar.selectbox("Resting ECG result", [0, 1, 2], format_func={
    0: "0 - Normal", 1: "1 - ST-T abnormality", 2: "2 - LV hypertrophy"}.get)
thalach = st.sidebar.slider("Maximum heart rate achieved", 70, 205, 150)
exang = st.sidebar.selectbox("Chest pain during exercise", [0, 1],
                             format_func=lambda v: "Yes" if v else "No")
oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
slope = st.sidebar.selectbox("Slope of peak exercise ST segment", [1, 2, 3], index=1,
                             format_func={1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}.get)
ca = st.sidebar.selectbox("Major vessels coloured by fluoroscopy (ca)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (thal)", [3, 6, 7], index=0, format_func={
    3: "3 - Normal", 6: "6 - Fixed defect", 7: "7 - Reversible defect"}.get)

threshold = st.sidebar.slider("Decision threshold (bonus)", 0.30, 0.70, 0.50, 0.05,
                              help="Lower threshold catches more sick patients but gives more false alarms.")

row = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                     exang, oldpeak, slope, ca, thal]],
                   columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                            "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

# ------------------------------------------------------------ prediction
st.subheader("Prediction")
prob = float(model.predict_proba(row)[0, 1])
c1, c2 = st.columns([1, 2])
with c1:
    if prob >= threshold:
        st.error(f"### 🔴 High risk")
    else:
        st.success(f"### 🟢 Low risk")
    st.metric("Probability of heart disease", f"{prob*100:.1f}%")
    st.caption(f"Decision threshold: {threshold:.2f}  |  Model: {meta['final_model']}")
with c2:
    st.progress(min(max(prob, 0.0), 1.0))
    st.write("The probability is the model's estimated chance that the patient has heart "
             "disease, based on the 13 values entered in the sidebar.")

# ------------------------------------------------------------ reliability
st.subheader(f"Model reliability (test set, {meta['n_test']} patients)")
pred_t = (test_proba >= threshold).astype(int)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy_score(test_y, pred_t):.3f}")
m2.metric("Precision", f"{precision_score(test_y, pred_t):.3f}")
m3.metric("Recall", f"{recall_score(test_y, pred_t):.3f}")
m4.metric("F1-score", f"{f1_score(test_y, pred_t):.3f}")

left, right = st.columns(2)
with left:
    st.markdown(f"**Confusion matrix (threshold = {threshold:.2f})**")
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ConfusionMatrixDisplay(confusion_matrix(test_y, pred_t),
                           display_labels=["No disease", "Disease"]).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    st.pyplot(fig)
with right:
    st.markdown("**Model comparison (5-fold CV and test set)**")
    st.dataframe(pd.DataFrame(meta["comparison"]).set_index("Model"), width="stretch")
    st.caption("Lowering the threshold raises recall (fewer missed patients) "
               "but adds false alarms.")

st.divider()
st.caption("CardioCheck | Azwa Noor (B24F0659AI069) | BS AI F24 Red | Trained on the UCI Cleveland "
           "dataset (303 patients). This tool is for education only and is not a medical diagnosis.")
