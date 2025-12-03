import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


DATA_FILENAME = "Credit Card Defaulter Prediction.csv"
TARGET_COL = "default "  # note the space in your original file


@st.cache_resource
def load_models_and_columns():
    """Load your trained models and rebuild the feature columns from the original data."""
    # Load models saved by model.py
    rf_model = joblib.load("rf_model.sav")
    log_reg_model = joblib.load("log_reg_model.sav")
    smote_rf_model = joblib.load("smote_rf_model.sav")

    # Rebuild the dummy columns from the original dataset
    df = pd.read_csv(DATA_FILENAME)
    df = df.drop(columns=["ID"], errors="ignore")

    # Map target to 0/1 if present
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"N": 0, "Y": 1})
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df

    X = pd.get_dummies(X, drop_first=True)
    feature_columns = X.columns.tolist()

    return rf_model, log_reg_model, smote_rf_model, feature_columns


def preprocess_for_model(df: pd.DataFrame, feature_columns):
    """Apply same preprocessing as training: drop ID, map target (if present), one-hot encode & align columns."""
    df = df.copy()

    # Drop ID if present
    df = df.drop(columns=["ID"], errors="ignore")

    y = None
    if TARGET_COL in df.columns:
        # Map N/Y to 0/1 and separate out target
        df[TARGET_COL] = df[TARGET_COL].map({"N": 0, "Y": 1})
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df

    # One-hot encode and align columns with training columns
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    return X, y


def main():
    st.title("Credit Card Defaulter Prediction")
    st.write("Use your trained models to predict probability of default for new customers.")

    # Check for data & model files
    missing = []
    for f in [DATA_FILENAME, "rf_model.sav", "log_reg_model.sav", "smote_rf_model.sav"]:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        st.error(
            "The following required files are missing in this folder:\n"
            + "\n".join(f"- {f}" for f in missing)
        )
        st.stop()

    # Load models & columns
    with st.spinner("Loading models and feature metadata..."):
        rf_model, log_reg_model, smote_rf_model, feature_columns = load_models_and_columns()

    model_choice = st.selectbox(
        "Choose model",
        ("Random Forest", "Logistic Regression", "SMOTE + Random Forest"),
    )

    if model_choice == "Random Forest":
        model = rf_model
    elif model_choice == "Logistic Regression":
        model = log_reg_model
    else:
        model = smote_rf_model

    st.subheader("Upload Data")

    uploaded_file = st.file_uploader(
        "Upload a CSV with the same columns as the original dataset",
        type=["csv"],
    )

    use_example = st.checkbox("Or use example data (first 100 rows of training file)")

    if not uploaded_file and not use_example:
        st.info("Upload a file or check 'use example data' to continue.")
        st.stop()

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(raw_df.head())
    else:
        raw_df = pd.read_csv(DATA_FILENAME).head(100)
        st.write("Using first 100 rows from the original data as example:")
        st.dataframe(raw_df.head())

    # Preprocess
    X, y_true = preprocess_for_model(raw_df, feature_columns)

    # Predict
    with st.spinner("Running predictions..."):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

    result_df = raw_df.copy()
    result_df["predicted_default"] = y_pred
    result_df["default_probability"] = y_proba

    st.subheader("Prediction Results")
    st.write("1 = predicted default, 0 = predicted no default")
    st.dataframe(result_df.head(50))

    # If labels exist in uploaded data, show quick metrics
    if y_true is not None and y_true.notna().all():
        try:
            acc = accuracy_score(y_true, y_pred)
            st.metric("Accuracy on provided data", f"{acc:.3f}")
        except Exception:
            pass

    # Option to download results
    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_out,
        file_name="predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
