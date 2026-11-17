
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from pathlib import Path


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

DATA_PATH = (
    BASE_DIR / "data" / "fraud_intelligence"
    / "test_data.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "explainability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# SETTINGS
# =====================================================

SAMPLE_SIZE = 300
RANDOM_STATE = 42

COLORS = {
    "background": "#0B1120",
    "card": "#151F32",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "green": "#22C55E",
    "red": "#EF4444",
    "cyan": "#06B6D4",
    "purple": "#A78BFA",
    "orange": "#F59E0B"
}

plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["muted"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "font.size": 11,
    "savefig.facecolor": COLORS["background"]
})


# =====================================================
# LOAD SAVED MODEL
# =====================================================

def load_model():

    print("\nLoading FraudShield model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    pipeline = package["model"]
    features = package["features"]
    threshold = package["threshold"]

    print("Model:", package["model_name"])
    print("Decision threshold:", round(threshold, 4))
    print("Number of original features:", len(features))

    return pipeline, features, threshold


# =====================================================
# LOAD TEST DATA
# =====================================================

def load_data(features):

    print("\nLoading test transactions...")

    df = pd.read_csv(DATA_PATH)

    missing = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    if "Class" not in df.columns:
        raise ValueError("Class column is missing")

    X = df[features].copy()

    for column in features:

        if X[column].dtype == "object":
            X[column] = X[column].replace({
                "True": 1,
                "False": 0,
                "true": 1,
                "false": 0
            })

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("Total test transactions:", len(df))

    return df, X


# =====================================================
# PREPARE DATA FOR SHAP
# =====================================================

def prepare_shap_data(pipeline, X):

    print("\nPreparing features for SHAP...")

    imputer = pipeline.named_steps["imputer"]

    classifier = pipeline.named_steps["classifier"]

    # Apply the SAME imputer fitted during training.
    transformed = imputer.transform(X)

    try:
        feature_names = (
            imputer.get_feature_names_out(
                X.columns.to_numpy()
            ).tolist()
        )

    except AttributeError:
        feature_names = list(X.columns)

        indicator = imputer.indicator_

        if indicator is not None:
            for index in indicator.features_:
                feature_names.append(
                    f"missingindicator_{X.columns[index]}"
                )

    if transformed.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature names do not match "
            "transformed feature dimensions"
        )

    X_transformed = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X.index
    )

    print(
        "Transformed feature count:",
        X_transformed.shape[1]
    )

    return classifier, X_transformed


# =====================================================
# CREATE SHAP EXPLAINER
# =====================================================

def create_explainer(classifier, X_transformed):

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(classifier)

    explanation = explainer(X_transformed)

    # Binary classifiers normally return one value
    # per feature. Some SHAP versions return an extra
    # output dimension for the two classes.
    if explanation.values.ndim == 3:

        explanation = explanation[:, :, 1]

    print("SHAP calculations completed!")

    return explanation


# =====================================================
# SAVE CHART
# =====================================================

def save_chart(filename):

    plt.gcf().set_facecolor(
        COLORS["background"]
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        bbox_inches="tight",
        facecolor=COLORS["background"]
    )

    plt.close("all")


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def feature_importance(explanation):

    print("\nGenerating feature importance chart...")

    importance = np.abs(
        explanation.values
    ).mean(axis=0)

    feature_names = list(
        explanation.feature_names
    )

    report = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": importance
    })

    report = report.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    report.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    print("\nTOP 10 IMPORTANT FEATURES")

    print(
        report.head(10).to_string(index=False)
    )

    top = report.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))

    ax.barh(
        top["feature"],
        top["mean_absolute_shap"],
        color=COLORS["cyan"]
    )

    ax.set_title(
        "FraudShield - SHAP Feature Importance",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Mean Absolute SHAP Value"
    )

    ax.grid(
        axis="x",
        alpha=0.15,
        linestyle="--"
    )

    ax.set_axisbelow(True)

    save_chart("feature_importance.png")

    return report


# =====================================================
# SHAP SUMMARY PLOT
# =====================================================

def shap_summary(explanation):

    print("\nGenerating SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        explanation.values,
        features=explanation.data,
        feature_names=explanation.feature_names,
        max_display=15,
        show=False
    )

    plt.title(
        "FraudShield - SHAP Summary",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    save_chart("shap_summary.png")


# =====================================================
# INDIVIDUAL WATERFALL EXPLANATION
# =====================================================

def explain_transaction(
    explanation,
    position,
    filename,
    title
):

    print(f"\nGenerating {title}...")

    plt.figure(figsize=(11, 8))

    shap.plots.waterfall(
        explanation[position],
        max_display=15,
        show=False
    )

    plt.title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=18
    )

    save_chart(filename)


# =====================================================
# SAVE INDIVIDUAL SHAP VALUES
# =====================================================

def save_shap_report(
    explanation,
    selected_df,
    probabilities,
    threshold
):

    print("\nSaving transaction explanations...")

    report = pd.DataFrame(
        explanation.values,
        columns=[
            f"shap_{name}"
            for name in explanation.feature_names
        ]
    )

    report.insert(
        0,
        "transaction_id",
        selected_df["transaction_id"].to_numpy()
    )

    report.insert(
        1,
        "actual_class",
        selected_df["Class"].to_numpy()
    )

    report.insert(
        2,
        "fraud_probability",
        probabilities
    )

    report.insert(
        3,
        "predicted_class",
        (probabilities >= threshold).astype(int)
    )

    report.to_csv(
        OUTPUT_DIR / "transaction_explanations.csv",
        index=False
    )


# =====================================================
# MAIN EXPLAINABILITY PIPELINE
# =====================================================

def run_explainability():

    print("\n" + "=" * 60)
    print("FRAUDSHIELD - SHAP EXPLAINABILITY")
    print("=" * 60)

    pipeline, features, threshold = load_model()

    df, X = load_data(features)

    # Select a representative random sample for
    # global explanations. Include both classes.
    fraud_indices = df.index[
        df["Class"] == 1
    ].to_numpy()

    normal_indices = df.index[
        df["Class"] == 0
    ].to_numpy()

    if len(fraud_indices) == 0 or len(normal_indices) == 0:
        raise ValueError(
            "Both fraud and legitimate records are required"
        )

    rng = np.random.default_rng(RANDOM_STATE)

    # Random sample from the test distribution.
    sampled_indices = rng.choice(
        df.index.to_numpy(),
        size=min(SAMPLE_SIZE, len(df)),
        replace=False
    )

    # Add one example of each actual class if needed.
    selected_indices = list(sampled_indices)

    selected_classes = set(
        df.loc[selected_indices, "Class"]
    )

    if 1 not in selected_classes:
        selected_indices.append(
            int(rng.choice(fraud_indices))
        )

    if 0 not in selected_classes:
        selected_indices.append(
            int(rng.choice(normal_indices))
        )

    selected_indices = list(
        dict.fromkeys(selected_indices)
    )

    selected_df = df.loc[
        selected_indices
    ].reset_index(drop=True)

    selected_X = X.loc[
        selected_indices
    ].reset_index(drop=True)

    probabilities = pipeline.predict_proba(
        selected_X
    )[:, 1]

    classifier, X_transformed = prepare_shap_data(
        pipeline,
        selected_X
    )

    explanation = create_explainer(
        classifier,
        X_transformed
    )

    feature_importance(explanation)

    shap_summary(explanation)

    # Explain actual fraud and legitimate examples.
    fraud_position = int(
        np.flatnonzero(
            selected_df["Class"].to_numpy() == 1
        )[0]
    )

    normal_position = int(
        np.flatnonzero(
            selected_df["Class"].to_numpy() == 0
        )[0]
    )

    print("\nFRAUD EXAMPLE")
    print(
        "Transaction:",
        selected_df.loc[
            fraud_position,
            "transaction_id"
        ]
    )
    print(
        "Predicted fraud probability:",
        f"{probabilities[fraud_position] * 100:.2f}%"
    )

    explain_transaction(
        explanation,
        fraud_position,
        "fraud_waterfall.png",
        "Fraud Transaction - SHAP Explanation"
    )

    print("\nLEGITIMATE EXAMPLE")
    print(
        "Transaction:",
        selected_df.loc[
            normal_position,
            "transaction_id"
        ]
    )
    print(
        "Predicted fraud probability:",
        f"{probabilities[normal_position] * 100:.2f}%"
    )

    explain_transaction(
        explanation,
        normal_position,
        "legitimate_waterfall.png",
        "Legitimate Transaction - SHAP Explanation"
    )

    save_shap_report(
        explanation,
        selected_df,
        probabilities,
        threshold
    )

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETED!")
    print("=" * 60)

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run_explainability()
