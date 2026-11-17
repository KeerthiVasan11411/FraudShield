
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "processed_data.csv"

ANALYSIS_DIR = BASE_DIR / "data" / "analysis"


# ==========================================
# FRAUDSHIELD COLOR THEME
# ==========================================

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

    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlecolor": COLORS["text"],

    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.titlepad": 18,

    "axes.labelsize": 11,

    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],

    "axes.grid": True,
    "axes.axisbelow": True,

    "grid.color": "#334155",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#334155",

    "savefig.facecolor": COLORS["background"],
    "savefig.dpi": 200,

    "font.family": "DejaVu Sans"
})


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

def load_processed_data():

    print("\nLoading processed FraudShield dataset...")

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "processed_data.csv not found. "
            "Run data_processing.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise"
    )

    df["Class"] = pd.to_numeric(
        df["Class"],
        errors="raise"
    ).astype(int)

    print("Dataset loaded:", df.shape)

    return df


# ==========================================
# COMMON CHART FORMATTING
# ==========================================

def format_chart(ax, title, xlabel, ylabel):

    ax.set_title(title, pad=18)

    ax.set_xlabel(xlabel, labelpad=10)

    ax.set_ylabel(ylabel, labelpad=10)

    ax.grid(axis="y", alpha=0.3)

    ax.grid(axis="x", visible=False)

    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")


def save_chart(filename):

    plt.tight_layout()

    plt.savefig(
        ANALYSIS_DIR / filename,
        dpi=200,
        bbox_inches="tight",
        facecolor=COLORS["background"]
    )

    plt.close()


# ==========================================
# DATASET OVERVIEW
# ==========================================

def dataset_overview(df):

    print("\n========== DATASET OVERVIEW ==========")

    print("\nTotal Transactions:", len(df))

    print("\nDataset Information:")

    df.info()

    print("\nStatistical Summary:")

    print(df.describe())

    print("\nTransaction Types:")

    print(df["transaction_type"].value_counts())

    print("\nCurrencies:")

    print(df["currency"].value_counts())


# ==========================================
# GENUINE VS FRAUDULENT TRANSACTIONS
# ==========================================

def analyze_fraud_distribution(df):

    print("\n========== FRAUD DISTRIBUTION ==========")

    counts = (
        df["Class"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    percentages = counts / len(df) * 100

    print("\nTransaction Counts:")

    print(
        counts.rename({
            0: "Genuine",
            1: "Fraudulent"
        })
    )

    print("\nTransaction Percentages:")

    print(
        percentages.rename({
            0: "Genuine",
            1: "Fraudulent"
        }).round(2)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        ["Genuine", "Fraudulent"],
        counts.values,
        color=[
            COLORS["green"],
            COLORS["red"]
        ],
        width=0.55
    )

    for bar, count, percentage in zip(
        bars, counts.values, percentages.values
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontsize=10,
            fontweight="bold"
        )

    format_chart(
        ax,
        "Genuine vs Fraudulent Transactions",
        "Transaction Category",
        "Number of Transactions"
    )

    save_chart("fraud_distribution.png")


# ==========================================
# FRAUD CATEGORY ANALYSIS
# ==========================================

def analyze_fraud_types(df):

    print("\n========== FRAUD TYPE ANALYSIS ==========")

    fraud_data = df[df["Class"] == 1]

    fraud_counts = fraud_data["fraud_type"].value_counts()

    print("\nFraud Categories:")

    print(fraud_counts)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        fraud_counts.index,
        fraud_counts.values,
        color=[
            COLORS["red"],
            COLORS["purple"],
            COLORS["orange"]
        ][:len(fraud_counts)],
        width=0.6
    )

    for bar in bars:

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontweight="bold"
        )

    format_chart(
        ax,
        "Fraud Distribution by Category",
        "Fraud Type",
        "Number of Fraudulent Transactions"
    )

    save_chart("fraud_types.png")


# ==========================================
# FRAUD BY TRANSACTION TYPE
# ==========================================

def analyze_transaction_types(df):

    print("\n========== FRAUD BY TRANSACTION TYPE ==========")

    result = df.groupby("transaction_type").agg(
        total_transactions=("Class", "count"),
        fraudulent_transactions=("Class", "sum")
    )

    result["fraud_rate"] = (
        result["fraudulent_transactions"]
        / result["total_transactions"]
    ) * 100

    print(result.round(2))

    result = result.sort_values(
        "fraud_rate",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        result.index,
        result["fraud_rate"],
        color=COLORS["cyan"],
        width=0.6
    )

    for bar in bars:

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontsize=9
        )

    format_chart(
        ax,
        "Fraud Rate by Transaction Type",
        "Transaction Type",
        "Fraud Rate (%)"
    )

    plt.xticks(rotation=25, ha="right")

    save_chart("fraud_by_transaction_type.png")


# ==========================================
# FRAUD BY CURRENCY
# ==========================================

def analyze_currency(df):

    print("\n========== FRAUD BY CURRENCY ==========")

    result = df.groupby("currency").agg(
        total_transactions=("Class", "count"),
        fraudulent_transactions=("Class", "sum")
    )

    result["fraud_rate"] = (
        result["fraudulent_transactions"]
        / result["total_transactions"]
    ) * 100

    print(result.round(2))

    result = result.sort_values(
        "fraud_rate",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        result.index,
        result["fraud_rate"],
        color=COLORS["purple"],
        width=0.6
    )

    for bar in bars:

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            color=COLORS["text"]
        )

    format_chart(
        ax,
        "Fraud Rate by Currency",
        "Currency",
        "Fraud Rate (%)"
    )

    save_chart("fraud_by_currency.png")


# ==========================================
# FRAUD TRENDS OVER TIME
# ==========================================

def analyze_fraud_trends(df):

    print("\n========== FRAUD TRENDS OVER TIME ==========")

    df = df.copy()

    df["week"] = (
        df["timestamp"]
        .dt.to_period("W")
        .dt.start_time
    )

    weekly_data = df.groupby("week").agg(
        total_transactions=("Class", "count"),
        fraudulent_transactions=("Class", "sum")
    )

    weekly_data["fraud_rate"] = (
        weekly_data["fraudulent_transactions"]
        / weekly_data["total_transactions"]
    ) * 100

    print(weekly_data.round(2))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        weekly_data.index,
        weekly_data["fraud_rate"],
        color=COLORS["cyan"],
        linewidth=2.5,
        marker="o",
        markersize=6
    )

    ax.fill_between(
        weekly_data.index,
        weekly_data["fraud_rate"],
        color=COLORS["cyan"],
        alpha=0.12
    )

    format_chart(
        ax,
        "Weekly Fraud Rate Trend",
        "Week",
        "Fraud Rate (%)"
    )

    plt.xticks(rotation=30)

    save_chart("weekly_fraud_trend.png")


# ==========================================
# DOMESTIC VS INTERNATIONAL FRAUD
# ==========================================

def analyze_international_transactions(df):

    print("\n========== INTERNATIONAL TRANSACTION ANALYSIS ==========")

    result = df.groupby("is_international").agg(
        total_transactions=("Class", "count"),
        fraudulent_transactions=("Class", "sum")
    )

    result["fraud_rate"] = (
        result["fraudulent_transactions"]
        / result["total_transactions"]
    ) * 100

    result.index = result.index.map({
        0: "Domestic",
        1: "International"
    })

    print(result.round(2))

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        result.index,
        result["fraud_rate"],
        color=[
            COLORS["green"],
            COLORS["orange"]
        ][:len(result)],
        width=0.55
    )

    for bar in bars:

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontweight="bold"
        )

    format_chart(
        ax,
        "Domestic vs International Fraud Rate",
        "Transaction Category",
        "Fraud Rate (%)"
    )

    save_chart("international_fraud.png")


# ==========================================
# TRANSACTION AMOUNT ANALYSIS
# ==========================================

def analyze_transaction_amounts(df):

    print("\n========== TRANSACTION AMOUNT ANALYSIS ==========")

    # Compare INR transactions only.
    inr_data = df[df["currency"] == "INR"].copy()

    if inr_data.empty:

        print("No INR transactions available.")

        return

    print("\nINR Transaction Statistics:")

    print(
        inr_data.groupby("Class")["amount"].describe()
    )

    genuine_amounts = inr_data.loc[
        inr_data["Class"] == 0,
        "amount"
    ]

    fraud_amounts = inr_data.loc[
        inr_data["Class"] == 1,
        "amount"
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    box = ax.boxplot(
        [genuine_amounts, fraud_amounts],
        tick_labels=["Genuine", "Fraudulent"],
        patch_artist=True,
        widths=0.5
    )

    box["boxes"][0].set_facecolor(COLORS["green"])
    box["boxes"][1].set_facecolor(COLORS["red"])

    for median in box["medians"]:

        median.set_color(COLORS["text"])
        median.set_linewidth(2)

    format_chart(
        ax,
        "INR Transaction Amount Distribution",
        "Transaction Category",
        "Transaction Amount (INR)"
    )

    save_chart("transaction_amounts_inr.png")


# ==========================================
# MAIN ANALYSIS FUNCTION
# ==========================================

def run_analysis():

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_processed_data()

    dataset_overview(df)

    analyze_fraud_distribution(df)

    analyze_fraud_types(df)

    analyze_transaction_types(df)

    analyze_currency(df)

    analyze_fraud_trends(df)

    analyze_international_transactions(df)

    analyze_transaction_amounts(df)

    print("\n========== FRAUDSHIELD ANALYSIS COMPLETED ==========")

    print("\nAnalysis charts saved in:")

    print(ANALYSIS_DIR)


# ==========================================
# PROGRAM EXECUTION
# ==========================================

if __name__ == "__main__":

    run_analysis()
