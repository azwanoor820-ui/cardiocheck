"""
CardioCheck: Heart Disease Risk Screening Assistant
Machine Learning - Assignment 3 / CCP Project
Name: Azwa Noor        Reg No: B24F0659AI069        Class: BS AI F24 Red

This script performs Steps 1-5 of the assignment:
  1. Data preprocessing (load, impute, scale, stratified split, EDA plots)
  2. Algorithm selection (LR, DT, RF + KNN, SVM extras) with 5-fold CV
  3. Feature importance (Random Forest)
  4. Confusion matrices and metrics
  5. Final model choice, saved for the Streamlit app
Run:  python cardiocheck_model.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_FILE = "processed.cleveland.data"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "target"]
FEATURES = COLUMNS[:-1]


# ---------------------------------------------------------------- Step 1
def load_data(path=DATA_FILE):
    """Load the Cleveland file, name the columns, replace '?' with NaN."""
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)   # 0 = no disease, 1 = disease
    return df


def eda_plots(df):
    """Class-balance bar chart and correlation heatmap."""
    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values,
                  color=["#4C9F70", "#D1495B"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                int(b.get_height()), ha="center", fontweight="bold")
    ax.set_ylabel("Number of patients")
    ax.set_title("Class balance of the target")
    plt.tight_layout(); plt.savefig("fig_class_balance.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Correlation heatmap")
    plt.tight_layout(); plt.savefig("fig_correlation.png", dpi=200); plt.close()


def make_pipeline(model):
    """Imputer + scaler + model in one Pipeline so test data never leaks."""
    pre = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())]), FEATURES)])
    return Pipeline([("prep", pre), ("model", model)])


def main():
    df = load_data()
    print("Shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nClass counts:\n", df["target"].value_counts())
    print("\nSex counts (1=male):\n", df["sex"].value_counts())
    eda_plots(df)

    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print("\nTrain:", X_train.shape, " Test:", X_test.shape)

    # ------------------------------------------------------------ Step 2
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "KNN (extra)": KNeighborsClassifier(n_neighbors=7),
        "SVM (extra)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, cms, fitted = [], {}, {}

    for name, model in models.items():
        pipe = make_pipeline(model)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        # ---------------------------------------------------------- Step 4/5
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cms[name] = confusion_matrix(y_test, pred)
        fitted[name] = pipe
        results.append({
            "Model": name,
            "CV Mean": round(cv_scores.mean(), 4),
            "CV Std": round(cv_scores.std(), 4),
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred), 4),
            "Recall": round(recall_score(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4),
        })
    res = pd.DataFrame(results)
    print("\n", res.to_string(index=False))
    res.to_csv("model_comparison.csv", index=False)

    # Confusion matrices for the three required models
    core = ["Logistic Regression", "Decision Tree", "Random Forest"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, name in zip(axes, core):
        ConfusionMatrixDisplay(cms[name], display_labels=["No disease", "Disease"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    plt.tight_layout(); plt.savefig("fig_confusion_matrices.png", dpi=200); plt.close()

    # ------------------------------------------------------------ Step 3
    rf = fitted["Random Forest"].named_steps["model"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    print("\nRF importances:\n", imp.sort_values(ascending=False).round(3))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot.barh(ax=ax, color="#2E6F95")
    ax.set_xlabel("Importance"); ax.set_title("Random Forest feature importance")
    plt.tight_layout(); plt.savefig("fig_feature_importance.png", dpi=200); plt.close()

    # Optional: tune the Random Forest with GridSearchCV
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
                        {"model__n_estimators": [100, 300],
                         "model__max_depth": [3, 5, None],
                         "model__min_samples_leaf": [1, 3]},
                        cv=cv, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("\nGridSearch best params:", grid.best_params_, " CV recall:", round(grid.best_score_, 4))

    # ------------------------------------------------------------ Step 5
    # Final model chosen after comparison (see report): Logistic Regression.
    final_name = "Logistic Regression"
    final = fitted[final_name]
    joblib.dump(final, "cardiocheck_model.joblib")
    proba = final.predict_proba(X_test)[:, 1]
    np.save("test_proba.npy", proba); np.save("test_y.npy", y_test.values)
    meta = {
        "final_model": final_name,
        "metrics": res.set_index("Model").loc[final_name].to_dict(),
        "comparison": res.to_dict(orient="records"),
        "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "feature_importance": imp.sort_values(ascending=False).to_dict(),
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)

    # Bonus: threshold analysis
    print("\nThreshold analysis (final model):")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        p = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
        print(f"t={t}: TN={tn} FP={fp} FN={fn} TP={tp} recall={tp/(tp+fn):.3f}")


if __name__ == "__main__":
    main()
