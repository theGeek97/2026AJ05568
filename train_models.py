# train_models.py
# What it does:
#   1. loads the obesity dataset
#   2. cleans / prepares the data
#   3. splits it into training data and test data
#   4. trains the 5 models and calculates their scores
#   5. saves the trained models and test_data.csv so the Streamlit app can use them

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# The 5 models required in the assignment
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utils import MODEL_FILES, TARGET, compute_metrics

SEED = 42                 # fixed number so results are the same every time we run
MODEL_DIR = Path("model") # folder where trained models will be saved

# Places where the program will look for the dataset file
CSV_CANDIDATES = [Path("data/ObesityDataSet_raw_and_data_sinthetic.csv"),
                  Path("ObesityDataSet_raw_and_data_sinthetic.csv"),
                  Path("data/ObesityDataSet.csv")]


def load_data():
    """Step 1: get the dataset."""
    # First try to find the CSV file on the computer
    for path in CSV_CANDIDATES:
        if path.exists():
            print("Loading", path)
            return pd.read_csv(path)

    # If the file is not there, download it directly from the UCI website
    print("CSV not found, downloading from UCI (needs: pip install ucimlrepo)...")
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=544)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # Save a copy so we don't need to download it again
    Path("data").mkdir(exist_ok=True)
    df.to_csv(CSV_CANDIDATES[0], index=False)
    return df


def make_preprocessor(X):
    """Step 2: prepare the data so the models can understand it.

    Models only understand numbers, but some columns contain words
    (e.g. Gender = Male/Female). So:
      - number columns -> scaled to a similar range (StandardScaler)
      - text columns   -> converted to 0/1 columns (OneHotEncoder)
    """
    text_cols = X.select_dtypes(exclude="number").columns.tolist()
    number_cols = [c for c in X.columns if c not in text_cols]

    return ColumnTransformer([
        ("num", StandardScaler(), number_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), text_cols),
    ])


def main():
    # ---- Load the data and remove repeated rows ----
    df = load_data().drop_duplicates().reset_index(drop=True)
    assert TARGET in df.columns, "Target column not found in the dataset"
    print("Dataset:", df.shape[0], "rows,", df.shape[1] - 1, "features,",
          df[TARGET].nunique(), "classes")

    # ---- Separate inputs (X) and the answer we want to predict (y) ----
    X = df.drop(columns=[TARGET])
    y_text = df[TARGET]

    # Turn the class names (like "Normal_Weight") into numbers 0..6
    label_encoder = LabelEncoder().fit(y_text)
    y = label_encoder.transform(y_text)
    n_classes = len(label_encoder.classes_)

    # ---- Step 3: split into 80% training data and 20% test data ----
    # stratify=y keeps the same class proportions in both parts
    X_train, X_test, y_train, y_test, _, y_test_text = train_test_split(
        X, y, y_text, test_size=0.2, stratify=y, random_state=SEED)

    # Save the test part as a CSV. This is the file we upload in the Streamlit app.
    test_csv = X_test.copy()
    test_csv[TARGET] = y_test_text.values
    test_csv.to_csv("test_data.csv", index=False)

    # ---- Step 4: the 5 models ----
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=SEED),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}

    for name, model in models.items():
        # A pipeline = data preparation + model packed together.
        # This way the app can accept the raw CSV and prepare it automatically.
        pipe = Pipeline([("prep", make_preprocessor(X)), ("clf", model)])

        pipe.fit(X_train, y_train)               # train on the 80% training data

        predictions = pipe.predict(X_test)       # predict on the unseen 20%
        probabilities = pipe.predict_proba(X_test)

        # Calculate all 6 scores for this model
        results[name] = compute_metrics(y_test, predictions, probabilities, n_classes)

        # Save the trained model to disk
        joblib.dump(pipe, MODEL_DIR / MODEL_FILES[name])
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    # ---- Step 5: save extra info the app needs ----
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    info = {"target": TARGET,
            "features": X.columns.tolist(),
            "classes": label_encoder.classes_.tolist()}
    (MODEL_DIR / "meta.json").write_text(json.dumps(info, indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # Make the comparison table (copy this into the README)
    table = pd.DataFrame(results).T.round(4)
    lines = ["| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
             "|---|---|---|---|---|---|---|"]
    for name, row in table.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    Path("results_table.md").write_text("\n".join(lines))

    print("\n" + "\n".join(lines))
    print("\nBest model by MCC:", table["MCC"].idxmax())


if __name__ == "__main__":
    main()
