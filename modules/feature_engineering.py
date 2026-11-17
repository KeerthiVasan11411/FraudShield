
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "feature_engineered_data.csv"


def load_data():
    print("\nLoading processed dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["account_creation_date"] = pd.to_datetime(
        df["account_creation_date"]
    )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Stable chronological order
    df = df.sort_values(
        ["timestamp", "transaction_id"],
        kind="stable"
    ).reset_index(drop=True)

    print("Dataset loaded successfully!")
    print("Total records:", len(df))

    return df


def create_time_features(df):
    print("\nCreating transaction time features...")

    df["transaction_hour"] = df["timestamp"].dt.hour
    df["transaction_day"] = df["timestamp"].dt.dayofweek

    df["is_weekend"] = (
        df["transaction_day"].isin([5, 6]).astype(int)
    )

    print("Time features created successfully!")
    return df


def create_location_features(df):
    print("\nCreating location-based features...")

    columns = [
        "customer_registered_latitude",
        "customer_registered_longitude",
        "transaction_latitude",
        "transaction_longitude"
    ]

    for column in columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    lat1 = np.radians(df["customer_registered_latitude"])
    lon1 = np.radians(df["customer_registered_longitude"])
    lat2 = np.radians(df["transaction_latitude"])
    lon2 = np.radians(df["transaction_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    df["distance_from_home_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )

    print("Location feature created successfully!")
    return df


def create_customer_features(df):
    print("\nCreating customer-based features...")

    df["account_age_days"] = (
        df["timestamp"] - df["account_creation_date"]
    ).dt.days

    # Previous device only; simultaneous events are not
    # treated as known history.
    previous_device = df.groupby("customer_id")["device_id"].shift(1)
    previous_time = df.groupby("customer_id")["timestamp"].shift(1)

    valid_previous = previous_time < df["timestamp"]

    df["device_change"] = (
        valid_previous
        & previous_device.notna()
        & df["device_id"].notna()
        & (df["device_id"] != previous_device)
    ).astype(int)

    print("Customer-based features created successfully!")
    return df


def create_transaction_frequency_features(df):
    print("\nCreating transaction frequency features...")

    df["transactions_last_1h"] = 0
    df["transactions_last_24h"] = 0
    df["failed_attempts_24h"] = 0

    one_hour = pd.Timedelta(hours=1).value
    twenty_four_hours = pd.Timedelta(hours=24).value

    for customer_id, group in df.groupby("customer_id", sort=False):
        indices = group.index.to_numpy()

        times = (
            group["timestamp"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )

        # Exclude the current timestamp and all simultaneous
        # transactions from historical calculations.
        previous_end = np.searchsorted(times, times, side="left")

        start_1h = np.searchsorted(
            times, times - one_hour, side="left"
        )

        start_24h = np.searchsorted(
            times, times - twenty_four_hours, side="left"
        )

        df.loc[indices, "transactions_last_1h"] = (
            previous_end - start_1h
        )

        df.loc[indices, "transactions_last_24h"] = (
            previous_end - start_24h
        )

        failed = (
            group["transaction_status"]
            .eq("FAILED")
            .to_numpy(dtype=int)
        )

        cumulative_failed = np.concatenate(
            ([0], np.cumsum(failed))
        )

        df.loc[indices, "failed_attempts_24h"] = (
            cumulative_failed[previous_end]
            - cumulative_failed[start_24h]
        )

    print("Transaction frequency features created successfully!")
    return df


def create_customer_spending_features(df):
    print("\nCreating personalized spending features...")

    df["customer_prior_transaction_count"] = 0
    df["customer_avg_amount_prior"] = np.nan
    df["customer_std_amount_prior"] = np.nan
    df["amount_deviation_ratio"] = np.nan

    # Historical spending is calculated separately for each
    # customer and currency.
    for (customer_id, currency), group in df.groupby(
        ["customer_id", "currency"], sort=False
    ):
        indices = group.index.to_numpy()
        times = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        amounts = group["amount"].to_numpy(dtype=float)

        # Only strictly earlier timestamps are eligible.
        end = np.searchsorted(times, times, side="left")

        # Successful, positive, finite historical amounts
        # establish the customer's spending baseline.
        eligible = (
            group["transaction_status"].eq("SUCCESS").to_numpy()
            & np.isfinite(amounts)
            & (amounts > 0)
        )

        historical_amounts = np.where(eligible, amounts, 0.0)

        counts = np.concatenate(([0], np.cumsum(eligible.astype(int))))
        sums = np.concatenate(([0.0], np.cumsum(historical_amounts)))
        squares = np.concatenate((
            [0.0], np.cumsum(historical_amounts ** 2)
        ))

        n = counts[end]
        total = sums[end]
        total_squares = squares[end]

        mean = np.full(len(group), np.nan)
        std = np.full(len(group), np.nan)

        np.divide(total, n, out=mean, where=n > 0)

        has_two = n >= 2

        variance = np.zeros(len(group))

        variance[has_two] = (
            total_squares[has_two]
            - total[has_two] ** 2 / n[has_two]
        ) / (n[has_two] - 1)

        std[has_two] = np.sqrt(
            np.maximum(variance[has_two], 0)
        )

        ratio = np.full(len(group), np.nan)

        np.divide(
            amounts,
            mean,
            out=ratio,
            where=np.isfinite(mean) & (mean > 0)
        )

        df.loc[indices, "customer_prior_transaction_count"] = n
        df.loc[indices, "customer_avg_amount_prior"] = mean
        df.loc[indices, "customer_std_amount_prior"] = std
        df.loc[indices, "amount_deviation_ratio"] = ratio

    print("Personalized spending features created successfully!")
    return df


def create_behavioral_features(df):
    print("\nCreating additional behavioral features...")

    df["unusual_transaction_time"] = 0
    df["new_merchant_for_customer"] = 0
    df["new_payee"] = 0

    df["has_time_history"] = 0
    df["has_merchant_history"] = 0
    df["has_payee_history"] = 0

    for customer_id, group in df.groupby("customer_id", sort=False):
        # Process one timestamp at a time so simultaneous
        # transactions cannot see one another.
        seen_hours = []
        seen_merchants = set()
        seen_payees = set()

        for timestamp, same_time in group.groupby(
            "timestamp", sort=True
        ):
            indices = same_time.index

            if seen_hours:
                # Circular hour distance handles midnight.
                hours = same_time["transaction_hour"].to_numpy()
                previous_hours = np.array(seen_hours)

                differences = np.abs(
                    hours[:, None] - previous_hours[None, :]
                )

                circular_distances = np.minimum(
                    differences, 24 - differences
                )

                # Unusual when the hour is more than 4 hours
                # away from every previous transaction hour.
                unusual = (
                    circular_distances.min(axis=1) > 4
                ).astype(int)

                df.loc[indices, "unusual_transaction_time"] = unusual
                df.loc[indices, "has_time_history"] = 1

            for index in indices:
                merchant = df.at[index, "merchant_id"]
                payee = df.at[index, "payee_id"]

                if pd.notna(merchant):
                    df.at[index, "has_merchant_history"] = int(
                        len(seen_merchants) > 0
                    )
                    df.at[index, "new_merchant_for_customer"] = int(
                        merchant not in seen_merchants
                    )

                if pd.notna(payee):
                    df.at[index, "has_payee_history"] = int(
                        len(seen_payees) > 0
                    )
                    df.at[index, "new_payee"] = int(
                        payee not in seen_payees
                    )

            # Update history only after processing this timestamp.
            seen_hours.extend(
                same_time["transaction_hour"].tolist()
            )

            seen_merchants.update(
                same_time["merchant_id"].dropna().tolist()
            )

            seen_payees.update(
                same_time["payee_id"].dropna().tolist()
            )

    print("Additional behavioral features created successfully!")
    return df


def validate_engineered_features(df):
    print("\n" + "=" * 55)
    print("FRAUDSHIELD FEATURE QUALITY REPORT")
    print("=" * 55)

    feature_columns = [
        "transaction_hour",
        "transaction_day",
        "is_weekend",
        "distance_from_home_km",
        "account_age_days",
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

    print("\n1. MISSING VALUE REPORT")
    print(df[feature_columns].isnull().sum())

    print("\n2. NEGATIVE VALUE CHECK")

    non_negative_columns = [
        "distance_from_home_km",
        "account_age_days",
        "transactions_last_1h",
        "transactions_last_24h",
        "failed_attempts_24h",
        "customer_prior_transaction_count"
    ]

    for column in non_negative_columns:
        print(column, ":", (df[column] < 0).sum())

    print("\n3. VALID RANGE CHECK")
    print(
        "Invalid hours:",
        (~df["transaction_hour"].between(0, 23)).sum()
    )
    print(
        "Invalid days:",
        (~df["transaction_day"].between(0, 6)).sum()
    )

    binary_columns = [
        "is_weekend",
        "device_change",
        "unusual_transaction_time",
        "new_merchant_for_customer",
        "new_payee",
        "has_time_history",
        "has_merchant_history",
        "has_payee_history"
    ]

    for column in binary_columns:
        invalid = (~df[column].isin([0, 1])).sum()
        print(f"Invalid {column}: {invalid}")

    print("\n4. FREQUENCY CONSISTENCY CHECK")

    print(
        "1h count greater than 24h count:",
        (
            df["transactions_last_1h"]
            > df["transactions_last_24h"]
        ).sum()
    )

    print(
        "Failed count greater than 24h count:",
        (
            df["failed_attempts_24h"]
            > df["transactions_last_24h"]
        ).sum()
    )

    print("\n5. DEVICE CHANGE DISTRIBUTION")
    print(df["device_change"].value_counts())

    print("\n6. CUSTOMER SPENDING SUMMARY")
    print(
        df[
            [
                "customer_prior_transaction_count",
                "customer_avg_amount_prior",
                "customer_std_amount_prior",
                "amount_deviation_ratio"
            ]
        ].describe()
    )

    print("\n7. BEHAVIORAL FEATURE DISTRIBUTION")

    for column in [
        "unusual_transaction_time",
        "new_merchant_for_customer",
        "new_payee"
    ]:
        print(f"\n{column}")
        print(df[column].value_counts())

    print("\n8. TARGET LABEL INFORMATION")
    print("Class column present:", "Class" in df.columns)
    print("Fraud type column present:", "fraud_type" in df.columns)

    print("\nFeature quality report completed!")


def main():
    df = load_data()

    df = create_time_features(df)
    df = create_location_features(df)
    df = create_customer_features(df)
    df = create_transaction_frequency_features(df)

    df = create_customer_spending_features(df)
    df = create_behavioral_features(df)

    validate_engineered_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature-engineered dataset saved!")
    print("Output:", OUTPUT_PATH)

    print("\nCUSTOMER SPENDING EXAMPLE")

    example_customer = df["customer_id"].iloc[0]

    print(
        df[df["customer_id"] == example_customer][
            [
                "timestamp",
                "currency",
                "amount",
                "customer_avg_amount_prior",
                "customer_std_amount_prior",
                "amount_deviation_ratio"
            ]
        ].to_string(index=False)
    )

    print("\nTotal records:", len(df))
    print("Total columns:", len(df.columns))


if __name__ == "__main__":
    main()
