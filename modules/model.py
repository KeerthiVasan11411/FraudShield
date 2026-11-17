
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "fraud_intelligence"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = DATA_DIR / "model_reports"

TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# FEATURES
# =====================================================

BASIC_FEATURES = [
    "amount",
    "card_present",
    "is_international",
    "transaction_hour",
    "transaction_day",
    "is_weekend",
    "account_age_days"
]

PERSONALIZED_FEATURES = [
    "distance_from_home_km",
    "device_change",
    "transactions_last_1h",
    "transactions_last_24h",
    "failed_attempts_24h",
    "customer_prior_transaction_count",
    "customer_avg_amount_prior",
    "customer_std_amount_prior",
    "amount_deviation_ratio",
    "unusual_transaction_time",
    "new_merchant_for_customer",
    "new_payee",
    "has_time_history",
    "has_merchant_history",
    "has_payee_history"
]

ALL_FEATURES = BASIC_FEATURES + PERSONALIZED_FEATURES


# =====================================================
# CHART THEME
# =====================================================

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
# LOAD AND SPLIT DATA
# =====================================================

def load_data():

    print("\nLoading FraudShield datasets...")

    earlier_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    required = ALL_FEATURES + ["Class", "timestamp"]

    for name, df in [
        ("Earlier data", earlier_data),
        ("Testing data", test_data)
    ]:
        missing = [
            column for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

        if not df["Class"].isin([0, 1]).all():
            raise ValueError(
                f"{name} contains invalid Class labels"
            )

    earlier_data["timestamp"] = pd.to_datetime(
        earlier_data["timestamp"]
    )

    test_data["timestamp"] = pd.to_datetime(
        test_data["timestamp"]
    )

    earlier_data = earlier_data.sort_values(
        ["timestamp", "transaction_id"]
    ).reset_index(drop=True)

    test_data = test_data.sort_values(
        ["timestamp", "transaction_id"]
    ).reset_index(drop=True)

    if (
        earlier_data["timestamp"].max()
        >= test_data["timestamp"].min()
    ):
        raise ValueError(
            "Training and testing periods overlap"
        )

    # Split the earlier 40,000 transactions into
    # approximately 30,000 training and 10,000 validation.
    split_index = int(len(earlier_data) * 0.75)

    cutoff_time = earlier_data.loc[
        split_index, "timestamp"
    ]

    train_data = earlier_data[
        earlier_data["timestamp"] < cutoff_time
    ].copy()

    validation_data = earlier_data[
        earlier_data["timestamp"] >= cutoff_time
    ].copy()

    if train_data.empty or validation_data.empty:
        raise ValueError(
            "Chronological validation split is empty"
        )

    for name, df in [
        ("Training", train_data),
        ("Validation", validation_data),
        ("Testing", test_data)
    ]:
        if df["Class"].nunique() != 2:
            raise ValueError(
                f"{name} must contain both fraud classes"
            )

        print(
            f"{name}: {len(df):,} records | "
            f"Fraud rate: {df['Class'].mean() * 100:.2f}%"
        )

    print("\nTraining ends:", train_data["timestamp"].max())
    print(
        "Validation begins:",
        validation_data["timestamp"].min()
    )
    print(
        "Validation ends:",
        validation_data["timestamp"].max()
    )
    print("Testing begins:", test_data["timestamp"].min())

    return train_data, validation_data, test_data


# =====================================================
# PREPARE FEATURES
# =====================================================

def prepare_features(df, features):

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

    y = df["Class"].astype(int)

    return X, y


# =====================================================
# CREATE MODELS
# =====================================================

def create_models(y_train):

    fraud_count = int((y_train == 1).sum())
    normal_count = int((y_train == 0).sum())

    imbalance_ratio = normal_count / fraud_count

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=imbalance_ratio,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }


# =====================================================
# FIND BEST VALIDATION THRESHOLD
# =====================================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    # The last precision/recall point has no threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / np.maximum(
            precision + recall,
            1e-12
        )
    )

    best_index = int(np.argmax(f1_scores))

    best_threshold = float(
        thresholds[best_index]
    )

    return best_threshold, float(
        f1_scores[best_index]
    )


# =====================================================
# EVALUATION
# =====================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold,
    model_name
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    ap = average_precision_score(
        y_true,
        probabilities
    )

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": ap,
        "predictions": predictions,
        "probabilities": probabilities
    }


# =====================================================
# CONFUSION MATRIX
# =====================================================

def plot_confusion_matrix(y_true, predictions):

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.imshow(matrix, cmap="Blues")

    labels = [
        ["TN", "FP"],
        ["FN", "TP"]
    ]

    for i in range(2):
        for j in range(2):

            value = matrix[i, j]

            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{value:,}",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=(
                    "white"
                    if value > matrix.max() / 2
                    else COLORS["muted"]
                )
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticklabels(["Legitimate", "Fraud"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(
        "Final FraudShield Confusion Matrix",
        pad=18
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "final_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# PRECISION-RECALL CURVE
# =====================================================

def plot_precision_recall(y_true, probabilities):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    ap = average_precision_score(
        y_true,
        probabilities
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        recall,
        precision,
        color=COLORS["cyan"],
        linewidth=2.5,
        label=f"FraudShield (AP={ap:.4f})"
    )

    ax.axhline(
        y_true.mean(),
        color=COLORS["orange"],
        linestyle="--",
        label="Fraud prevalence"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_title(
        "Final FraudShield Precision-Recall Curve",
        pad=18
    )

    ax.legend()
    ax.grid(alpha=0.15)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "final_precision_recall.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def train_models():

    print("\n" + "=" * 60)
    print("FRAUDSHIELD - VALIDATED MACHINE LEARNING")
    print("=" * 60)

    train, validation, test = load_data()

    experiments = {
        "Basic": BASIC_FEATURES,
        "Personalized": ALL_FEATURES
    }

    validation_results = []

    best_model = None
    best_name = None
    best_features = None
    best_threshold = None

    best_ap = -1
    best_f1 = -1

    # -------------------------------------------------
    # TRAINING AND VALIDATION
    # -------------------------------------------------

    for experiment_name, features in experiments.items():

        print("\n" + "=" * 60)
        print("EXPERIMENT:", experiment_name)
        print("=" * 60)

        X_train, y_train = prepare_features(
            train, features
        )

        X_val, y_val = prepare_features(
            validation, features
        )

        models = create_models(y_train)

        for model_name, estimator in models.items():

            full_name = (
                f"{model_name} - {experiment_name}"
            )

            print("\nTraining:", full_name)

            pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True
                    )
                ),
                ("classifier", estimator)
            ])

            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(
                X_val
            )[:, 1]

            threshold, validation_f1 = (
                find_best_threshold(
                    y_val,
                    probabilities
                )
            )

            result = evaluate_model(
                y_val,
                probabilities,
                threshold,
                full_name
            )

            print(
                f"Best threshold: {threshold:.4f}"
            )
            print(
                f"Validation Precision: "
                f"{result['precision']:.4f}"
            )
            print(
                f"Validation Recall: "
                f"{result['recall']:.4f}"
            )
            print(
                f"Validation F1: "
                f"{result['f1_score']:.4f}"
            )
            print(
                f"Validation AP: "
                f"{result['pr_auc']:.4f}"
            )

            validation_results.append({
                "model": full_name,
                "threshold": threshold,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "pr_auc": result["pr_auc"]
            })

            # Select by validation AP.
            # Use validation F1 as a tie-breaker.
            if (
                result["pr_auc"] > best_ap
                or (
                    np.isclose(
                        result["pr_auc"],
                        best_ap,
                        rtol=0,
                        atol=1e-12
                    )
                    and validation_f1 > best_f1
                )
            ):

                best_ap = result["pr_auc"]
                best_f1 = validation_f1

                best_model = pipeline
                best_name = full_name
                best_features = features.copy()
                best_threshold = threshold

    # -------------------------------------------------
    # SAVE VALIDATION RESULTS
    # -------------------------------------------------

    comparison = pd.DataFrame(
        validation_results
    ).sort_values(
        ["pr_auc", "f1_score"],
        ascending=False
    )

    print("\n" + "=" * 60)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False
    )

    print("\nSelected model:", best_name)
    print(
        f"Selected threshold: {best_threshold:.4f}"
    )

    # -------------------------------------------------
    # FINAL UNTOUCHED TEST EVALUATION
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    X_test, y_test = prepare_features(
        test,
        best_features
    )

    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    final_result = evaluate_model(
        y_test,
        test_probabilities,
        best_threshold,
        best_name
    )

    print("\nModel:", best_name)
    print(
        f"Threshold: {best_threshold:.4f}"
    )
    print(
        f"Precision: {final_result['precision']:.4f}"
    )
    print(
        f"Recall: {final_result['recall']:.4f}"
    )
    print(
        f"F1 Score: {final_result['f1_score']:.4f}"
    )
    print(
        f"PR-AUC: {final_result['pr_auc']:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            final_result["predictions"],
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        final_result["predictions"],
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    print("\nCONFUSION MATRIX RESULTS")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    # -------------------------------------------------
    # SAVE FINAL REPORTS
    # -------------------------------------------------

    final_metrics = pd.DataFrame([{
        "model": best_name,
        "threshold": best_threshold,
        "precision": final_result["precision"],
        "recall": final_result["recall"],
        "f1_score": final_result["f1_score"],
        "pr_auc": final_result["pr_auc"],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }])

    final_metrics.to_csv(
        REPORT_DIR / "final_test_metrics.csv",
        index=False
    )

    plot_confusion_matrix(
        y_test,
        final_result["predictions"]
    )

    plot_precision_recall(
        y_test,
        test_probabilities
    )

    # Save predictions for future error analysis.
    prediction_report = pd.DataFrame({
        "transaction_id": test["transaction_id"].values,
        "actual_class": y_test.values,
        "fraud_probability": test_probabilities,
        "predicted_class": final_result["predictions"]
    })

    prediction_report.to_csv(
        REPORT_DIR / "final_test_predictions.csv",
        index=False
    )

    # -------------------------------------------------
    # SAVE SELECTED MODEL
    # -------------------------------------------------

    model_package = {
        "model": best_model,
        "features": best_features,
        "model_name": best_name,
        "threshold": best_threshold,
        "validation_average_precision": best_ap
    }

    joblib.dump(
        model_package,
        MODEL_DIR / "fraud_model.pkl"
    )

    print("\n" + "=" * 60)
    print("FRAUDSHIELD MODEL TRAINING COMPLETED!")
    print("=" * 60)

    print("\nSaved model:")
    print(MODEL_DIR / "fraud_model.pkl")

    print("\nSaved reports:")
    print(REPORT_DIR)


if __name__ == "__main__":
    train_models()
