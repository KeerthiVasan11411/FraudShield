
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "feature_engineered_data.csv"

OUTPUT_DIR = BASE_DIR / "data" / "fraud_intelligence"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DARK THEME
# ============================================================

COLORS = {
    "background": "#0B1120",
    "card": "#151F32",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "green": "#22C55E",
    "red": "#EF4444",
    "cyan": "#06B6D4",
    "purple": "#A78BFA",
    "orange": "#F59E0B",
    "blue": "#3B82F6"
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
    "figure.dpi": 120,
    "savefig.facecolor": COLORS["background"]
})


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading FraudShield intelligence dataset...")

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "Class",
        "fraud_type",
        "transaction_type",
        "merchant_id",
        "payee_id",
        "device_change",
        "unusual_transaction_time",
        "new_merchant_for_customer",
        "new_payee",
        "amount_deviation_ratio",
        "distance_from_home_km"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )

    if df[required_columns].isna().any().any():
        # Merchant/payee IDs and spending ratios may be
        # legitimately missing. Check essential fields only.
        essential = [
            "transaction_id",
            "customer_id",
            "timestamp",
            "Class",
            "fraud_type",
            "transaction_type",
            "device_change",
            "unusual_transaction_time",
            "new_merchant_for_customer",
            "new_payee",
            "distance_from_home_km"
        ]

        if df[essential].isna().any().any():
            raise ValueError(
                "Essential intelligence fields contain missing values"
            )

    if not df["Class"].isin([0, 1]).all():
        raise ValueError(
            "Class must contain only 0 and 1"
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected"
        )

    df = df.sort_values(
        ["timestamp", "transaction_id"],
        kind="stable"
    ).reset_index(drop=True)

    print("Dataset loaded successfully!")
    print("Total transactions:", len(df))

    return df


# ============================================================
# CHART HELPERS
# ============================================================

def format_chart(ax, title, xlabel, ylabel):

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)

    ax.grid(
        axis="y",
        alpha=0.15,
        linestyle="--"
    )

    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_chart(filename):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


def plot_fraud_rates(summary, category, title, filename):

    if summary.empty:
        print("No applicable records for:", title)
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = summary[category].astype(str).tolist()

    rates = summary["fraud_rate"].fillna(0).to_numpy()

    palette = [
        COLORS["cyan"],
        COLORS["purple"],
        COLORS["orange"],
        COLORS["red"],
        COLORS["green"],
        COLORS["blue"]
    ]

    colors = [
        palette[i % len(palette)]
        for i in range(len(labels))
    ]

    bars = ax.bar(
        labels,
        rates,
        color=colors,
        width=0.65
    )

    for bar, rate, count in zip(
        bars,
        rates,
        summary["transactions"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{rate:.2f}%\n(n={count:,})",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["text"]
        )

    ax.set_ylim(
        0,
        max(float(np.max(rates)) * 1.25, 2)
    )

    format_chart(
        ax,
        title,
        category.replace("_", " ").title(),
        "Fraud Rate (%)"
    )

    ax.tick_params(axis="x", rotation=15)

    save_chart(filename)


# ============================================================
# SUMMARY CALCULATION
# ============================================================

def calculate_fraud_summary(df, column):

    summary = (
        df.groupby(column, observed=True, dropna=False)["Class"]
        .agg(
            transactions="count",
            fraud_cases="sum",
            fraud_rate="mean"
        )
        .reset_index()
    )

    summary["fraud_rate"] *= 100

    return summary


# ============================================================
# OVERALL DATASET OVERVIEW
# ============================================================

def dataset_overview(df):

    print("\n" + "=" * 55)
    print("1. DATASET OVERVIEW")
    print("=" * 55)

    total = len(df)
    fraud = int(df["Class"].sum())
    legitimate = total - fraud

    print("Total transactions:", total)
    print("Legitimate transactions:", legitimate)
    print("Fraudulent transactions:", fraud)

    print(
        f"Overall fraud rate: {df['Class'].mean() * 100:.2f}%"
    )

    print(
        "Date range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max()
    )


# ============================================================
# BINARY BEHAVIORAL FEATURE ANALYSIS
# ============================================================

def analyze_binary_feature(
    df,
    feature,
    title,
    filename,
    applicable_types=None
):

    print(f"\nAnalyzing {feature}...")

    data = df.copy()

    if applicable_types is not None:
        data = data[
            data["transaction_type"].isin(applicable_types)
        ].copy()

    summary = calculate_fraud_summary(data, feature)

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / f"{feature}_report.csv",
        index=False
    )

    plot_fraud_rates(
        summary,
        feature,
        title,
        filename
    )

    return summary


# ============================================================
# PERSONALIZED SPENDING ANALYSIS
# ============================================================

def analyze_amount_deviation(df):

    print("\n" + "=" * 55)
    print("2. PERSONALIZED AMOUNT DEVIATION ANALYSIS")
    print("=" * 55)

    data = df.copy()

    bins = [-np.inf, 1, 2, 5, 10, np.inf]

    labels = [
        "Below 1x",
        "1x - 2x",
        "2x - 5x",
        "5x - 10x",
        "Above 10x"
    ]

    data["amount_risk_group"] = pd.cut(
        data["amount_deviation_ratio"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Missing history is a separate group.
    data["amount_risk_group"] = (
        data["amount_risk_group"]
        .cat.add_categories(["No History"])
        .fillna("No History")
    )

    summary = calculate_fraud_summary(
        data,
        "amount_risk_group"
    )

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "amount_deviation_report.csv",
        index=False
    )

    plot_fraud_rates(
        summary,
        "amount_risk_group",
        "Fraud Rate by Personalized Spending Deviation",
        "amount_deviation_risk.png"
    )

    return summary


# ============================================================
# LOCATION-BASED FRAUD ANALYSIS
# ============================================================

def analyze_location_risk(df):

    print("\n" + "=" * 55)
    print("3. LOCATION-BASED FRAUD ANALYSIS")
    print("=" * 55)

    data = df.copy()

    bins = [-np.inf, 10, 50, 200, 1000, np.inf]

    labels = [
        "0-10 km",
        "10-50 km",
        "50-200 km",
        "200-1000 km",
        "1000+ km"
    ]

    data["distance_group"] = pd.cut(
        data["distance_from_home_km"],
        bins=bins,
        labels=labels
    )

    summary = calculate_fraud_summary(
        data,
        "distance_group"
    )

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "location_risk_report.csv",
        index=False
    )

    plot_fraud_rates(
        summary,
        "distance_group",
        "Fraud Rate by Distance from Registered Location",
        "location_fraud_risk.png"
    )

    return summary


# ============================================================
# FRAUD TYPE ANALYSIS
# ============================================================

def analyze_fraud_types(df):

    print("\n" + "=" * 55)
    print("4. FRAUD TYPE ANALYSIS")
    print("=" * 55)

    fraud_df = df[df["Class"] == 1].copy()

    summary = (
        fraud_df["fraud_type"]
        .value_counts()
        .rename_axis("fraud_type")
        .reset_index(name="fraud_cases")
    )

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "fraud_type_report.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        summary["fraud_type"],
        summary["fraud_cases"],
        color=[
            COLORS["red"],
            COLORS["purple"],
            COLORS["orange"]
        ][:len(summary)]
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(int(bar.get_height())),
            ha="center",
            fontweight="bold"
        )

    ax.set_ylim(
        0,
        max(summary["fraud_cases"].max() * 1.2, 1)
    )

    format_chart(
        ax,
        "Fraud Cases by Fraud Type",
        "Fraud Type",
        "Number of Fraud Cases"
    )

    save_chart("fraud_types_intelligence.png")

    # Compare behavior within each fraud category.
    indicators = [
        "device_change",
        "unusual_transaction_time",
        "new_merchant_for_customer",
        "new_payee"
    ]

    behavioral_summary = (
        fraud_df.groupby("fraud_type")[indicators]
        .mean()
        .mul(100)
        .round(2)
    )

    print("\nPercentage of fraud cases with each indicator:")
    print(behavioral_summary.to_string())

    behavioral_summary.to_csv(
        OUTPUT_DIR / "fraud_type_behavior_report.csv"
    )

    return summary


# ============================================================
# COMBINED BEHAVIORAL FLAGS
# ============================================================

def analyze_combined_risk(df):

    print("\n" + "=" * 55)
    print("5. COMBINED BEHAVIORAL ANALYSIS")
    print("=" * 55)

    data = df.copy()

    # Exploratory count, NOT a validated risk score.
    # Missing amount history does not count as a flag.
    data["behavioral_flags"] = (
        data["device_change"]
        + data["unusual_transaction_time"]
        + data["new_merchant_for_customer"]
        + data["new_payee"]
        + (
            data["distance_from_home_km"] > 200
        ).astype(int)
        + (
            data["amount_deviation_ratio"] > 5
        ).fillna(False).astype(int)
    )

    summary = calculate_fraud_summary(
        data,
        "behavioral_flags"
    )

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "combined_behavioral_report.csv",
        index=False
    )

    plot_fraud_rates(
        summary,
        "behavioral_flags",
        "Fraud Rate by Number of Behavioral Flags",
        "combined_behavioral_risk.png"
    )

    return summary


# ============================================================
# SUSPICIOUS RELATIONSHIP / SYNTHETIC DATA AUDIT
# ============================================================

def detect_suspicious_patterns(df):

    print("\n" + "=" * 55)
    print("6. SYNTHETIC DATA QUALITY AUDIT")
    print("=" * 55)

    indicators = [
        "device_change",
        "unusual_transaction_time",
        "new_merchant_for_customer",
        "new_payee"
    ]

    warnings = []

    for feature in indicators:

        summary = calculate_fraud_summary(
            df, feature
        )

        for _, row in summary.iterrows():

            count = int(row["transactions"])
            rate = float(row["fraud_rate"])

            if count >= 30 and (
                rate >= 99 or rate <= 0.01
            ):
                warnings.append({
                    "feature": feature,
                    "group": str(row[feature]),
                    "transactions": count,
                    "fraud_rate": rate,
                    "warning": "Near-perfect class separation"
                })

    # Audit distance groups as well.
    distance_data = df.copy()

    distance_data["distance_group"] = pd.cut(
        distance_data["distance_from_home_km"],
        bins=[-np.inf, 10, 50, 200, 1000, np.inf],
        labels=[
            "0-10 km",
            "10-50 km",
            "50-200 km",
            "200-1000 km",
            "1000+ km"
        ]
    )

    distance_summary = calculate_fraud_summary(
        distance_data,
        "distance_group"
    )

    for _, row in distance_summary.iterrows():

        count = int(row["transactions"])
        rate = float(row["fraud_rate"])

        if count >= 30 and (
            rate >= 99 or rate <= 0.01
        ):
            warnings.append({
                "feature": "distance_group",
                "group": str(row["distance_group"]),
                "transactions": count,
                "fraud_rate": rate,
                "warning": "Near-perfect class separation"
            })

    warning_df = pd.DataFrame(
        warnings,
        columns=[
            "feature",
            "group",
            "transactions",
            "fraud_rate",
            "warning"
        ]
    )

    if warning_df.empty:
        print(
            "No near-perfect relationships detected "
            "by the current audit rules."
        )
    else:
        print("\nPotential synthetic-data shortcuts:")
        print(warning_df.to_string(index=False))

    warning_df.to_csv(
        OUTPUT_DIR / "synthetic_data_warnings.csv",
        index=False
    )

    print(
        "\nNote: These warnings are indicators, "
        "not proof of data leakage."
    )

    return warning_df


# ============================================================
# CHRONOLOGICAL TRAIN / TEST PREPARATION
# ============================================================

def prepare_time_based_split(df):

    print("\n" + "=" * 55)
    print("7. TIME-BASED DATASET SPLIT")
    print("=" * 55)

    # 80% earliest records for training,
    # 20% latest records for testing.
    # Do not split identical timestamps across sets.
    data = df.sort_values(
        ["timestamp", "transaction_id"],
        kind="stable"
    ).reset_index(drop=True)

    target_index = int(len(data) * 0.8)

    if target_index <= 0 or target_index >= len(data):
        raise ValueError(
            "Not enough records for chronological splitting"
        )

    cutoff_time = data.loc[
        target_index, "timestamp"
    ]

    train_df = data[
        data["timestamp"] < cutoff_time
    ].copy()

    test_df = data[
        data["timestamp"] >= cutoff_time
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Unable to create nonempty time-based splits"
        )

    train_path = OUTPUT_DIR / "train_data.csv"
    test_path = OUTPUT_DIR / "test_data.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))

    print(
        "Training fraud rate:",
        f"{train_df['Class'].mean() * 100:.2f}%"
    )

    print(
        "Testing fraud rate:",
        f"{test_df['Class'].mean() * 100:.2f}%"
    )

    print("Training ends:", train_df["timestamp"].max())
    print("Testing begins:", test_df["timestamp"].min())

    print("\nFiles saved:")
    print(train_path)
    print(test_path)

    return train_df, test_df


# ============================================================
# MAIN INTELLIGENCE PIPELINE
# ============================================================

def run_intelligence():

    print("\n" + "=" * 55)
    print("FRAUDSHIELD - BEHAVIORAL FRAUD INTELLIGENCE")
    print("=" * 55)

    df = load_data()

    dataset_overview(df)

    print("\n" + "=" * 55)
    print("BEHAVIORAL INDICATOR ANALYSIS")
    print("=" * 55)

    analyze_binary_feature(
        df,
        "device_change",
        "Fraud Rate by Device Change",
        "device_change_risk.png"
    )

    analyze_binary_feature(
        df,
        "unusual_transaction_time",
        "Fraud Rate by Unusual Transaction Time",
        "unusual_time_risk.png"
    )

    # Only transactions with merchant IDs are applicable.
    merchant_df = df[df["merchant_id"].notna()].copy()

    analyze_binary_feature(
        merchant_df,
        "new_merchant_for_customer",
        "Fraud Rate by New Merchant",
        "new_merchant_risk.png"
    )

    # Only transactions with payee IDs are applicable.
    payee_df = df[df["payee_id"].notna()].copy()

    analyze_binary_feature(
        payee_df,
        "new_payee",
        "Fraud Rate by New Payee",
        "new_payee_risk.png"
    )

    analyze_amount_deviation(df)

    analyze_location_risk(df)

    analyze_fraud_types(df)

    analyze_combined_risk(df)

    detect_suspicious_patterns(df)

    prepare_time_based_split(df)

    print("\n" + "=" * 55)
    print("FRAUD INTELLIGENCE ANALYSIS COMPLETED!")
    print("=" * 55)

    print("\nReports and charts saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run_intelligence()
