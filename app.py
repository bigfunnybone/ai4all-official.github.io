import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

# -----------------
# CONFIG
# -----------------
DEFAULT_FILENAME = "Credit Card Defaulter Prediction.csv"
TARGET_COL = "default "  # note the space at the end!


def load_data(uploaded_file=None):
    """Load data from uploaded file or default CSV on disk."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded file loaded successfully.")
    else:
        if not os.path.exists(DEFAULT_FILENAME):
            st.error(
                f"File '{DEFAULT_FILENAME}' not found. "
                "Upload a CSV or place the file in this folder."
            )
            st.stop()
        df = pd.read_csv(DEFAULT_FILENAME)
        st.info(f"Loaded default file: {DEFAULT_FILENAME}")
    return df


def prepare_data(df: pd.DataFrame):
    """Clean and prepare data: drop ID, encode target, dummies for X."""
    df = df.copy()

    # Drop ID so model doesn't memorize IDs
    df.drop(["ID"], axis=1, inplace=True, errors="ignore")

    # Map Y/N to 1/0 in target column
    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' not found in data.")
        st.stop()

    df[TARGET_COL] = df[TARGET_COL].map({"N": 0, "Y": 1})
    if df[TARGET_COL].isna().any():
        st.warning(
            "Some values in the target column could not be mapped from 'N'/'Y' to 0/1."
        )

    st.write("Target value counts:")
    st.write(df[TARGET_COL].value_counts())

    # Define X and y
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """Train RandomForest and compute metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # Predictions & probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    clf_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation on full data
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    return {
        "model": model,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": clf_report,
        "confusion_matrix": cm,
        "cv_scores": cv_scores,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def main():
    st.title("Credit Card Defaulter Prediction (Random Forest)")
    st.write(
        "This Streamlit app trains a Random Forest model on the credit card default dataset "
        "and shows performance metrics."
    )

    # -------------
    # DATA INPUT
    # -------------
    st.sidebar.header("Data Options")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (optional)", type=["csv"]
    )

    df = load_data(uploaded_file)

    with st.expander("Raw Data Preview"):
        st.write(df.head())
        st.write("Summary statistics:")
        st.write(df.describe(include="all"))
        st.write("Info:")
        buffer = []
        df.info(buf=buffer)
        info_str = "\n".join(buffer)
        st.text(info_str)

    # -------------
    # TRAINING
    # -------------
    if st.button("Train Model"):
        with st.spinner("Preparing data and training model..."):
            X, y = prepare_data(df)
            results = train_and_evaluate(X, y)

        st.success("Model training complete! ✅")

        # Metrics
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
        with col2:
            st.metric("ROC-AUC", f"{results['roc_auc']:.3f}")

        st.subheader("Classification Report")
        st.text(results["classification_report"])

        st.subheader("Confusion Matrix [ [TN FP] [FN TP] ]")
        st.write(results["confusion_matrix"])

        st.subheader("Cross-Validation Accuracy Scores (5-fold)")
        st.write(results["cv_scores"])
        st.write(f"True Average Accuracy: {results['cv_scores'].mean():.3f}")

        # -------------
        # SAVE MODEL
        # -------------
        joblib.dump(results["model"], "rf_model.sav")
        st.success("Model saved to `rf_model.sav` in this folder.")


if __name__ == "__main__":
    main()
