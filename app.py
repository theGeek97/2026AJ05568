# app.py

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from utils import MODEL_FILES, compute_metrics

# Fix NameError: Use __file__ only if it exists in the globals namespace, otherwise use current working directory
CURRENT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
MODEL_DIR = CURRENT_DIR / "model"
st.set_page_config(page_title="Obesity Level Classifier", layout="wide")


# cache_resource: load the models only once, not on every click (makes app faster)
@st.cache_resource
def load_saved_files():
    info = json.loads((MODEL_DIR / "meta.json").read_text())
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models = {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}
    return info, label_encoder, models


info, label_encoder, models = load_saved_files()
target = info["target"]          # name of the answer column
features = info["features"]      # names of the input columns
classes = info["classes"]        # the 7 obesity categories

# ---------------- Page heading ----------------
st.title("Obesity Level Classification")
st.caption("Pick a model, upload test data and see how well it performs. "
           "The models are already trained, nothing is retrained here.")

# ---------------- Sidebar: upload file + choose model ----------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    use_sample = st.checkbox("Use the sample test_data.csv instead")
    model_name = st.selectbox("Select model", list(models.keys()))

# ---------------- Read the CSV ----------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample and (CURRENT_DIR / "test_data.csv").exists():
    df = pd.read_csv(CURRENT_DIR / "test_data.csv")
else:
    # Nothing uploaded yet, so show instructions and stop here
    st.info("Please upload a test CSV from the sidebar. It must contain the input "
            f"columns and the target column `{target}`.")
    with st.expander("Columns expected in the file"):
        st.write(features + [target])
    st.stop()

# ---------------- Basic checks on the uploaded file ----------------
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Your file is missing these columns: {missing}")
    st.stop()

unknown = sorted(set(df[target].astype(str)) - set(classes))
if unknown:
    st.error(f"Unknown values in `{target}`: {unknown}. Allowed values: {classes}")
    st.stop()

st.subheader("Data preview")
st.write(f"{df.shape[0]} rows and {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

# Split the uploaded file into inputs and the true answers
X = df[features]
y_true = label_encoder.transform(df[target].astype(str))
n_classes = len(classes)


def run_model(name):
    """Use a saved model to predict on the uploaded data."""
    pipe = models[name]
    return pipe.predict(X), pipe.predict_proba(X)


# ---------------- Scores of the selected model ----------------
y_pred, y_proba = run_model(model_name)
scores = compute_metrics(y_true, y_pred, y_proba, n_classes)

st.subheader(f"Evaluation metrics: {model_name}")
columns = st.columns(6)
for col, (metric, value) in zip(columns, scores.items()):
    col.metric(metric, "n/a" if value != value else f"{value:.4f}")  # value != value checks for NaN
st.caption("Precision, Recall and F1 are macro averages. AUC is one-vs-rest (macro).")

left, right = st.columns(2)

# ---------------- Confusion matrix ----------------
# Rows = real class, columns = predicted class. Numbers on the diagonal are correct predictions.
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Classification report ----------------
with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                   target_names=classes, output_dict=True,
                                   zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------------- Compare all 5 models together ----------------
st.subheader("Comparison of all models on this data")
all_scores = {}
for name in models:
    pred, proba = run_model(name)
    all_scores[name] = compute_metrics(y_true, pred, proba, n_classes)
comparison = pd.DataFrame(all_scores).T.round(4)
comparison.index.name = "ML Model Name"
st.dataframe(comparison, use_container_width=True)