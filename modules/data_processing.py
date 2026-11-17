
import pandas as pd
from pathlib import Path


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "FraudShield_Dataset.xlsx"

OUTPUT_PATH = BASE_DIR / "data" / "processed_data.csv"

REJECTED_PATH = BASE_DIR / "data" / "rejected_records.csv"

ERROR_REPORT_PATH = BASE_DIR / "data" / "error_report.csv"


# ==========================================
# REQUIRED COLUMNS
# ==========================================

TRANSACTION_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "currency",
    "merchant_id",
    "merchant_category",
    "transaction_type",
    "transaction_subtype",
    "payee_id",
    "device_id",
    "transaction_status",
    "card_present",
    "is_international",
    "transaction_latitude",
    "transaction_longitude",
    "Class",
    "fraud_type"
]

CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_registered_latitude",
    "customer_registered_longitude",
    "account_creation_date"
]


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("\nLoading FraudShield dataset...")

    transactions = pd.read_excel(
        DATASET_PATH,
        sheet_name="Transactions",
        engine="openpyxl"
    )

    customers = pd.read_excel(
        DATASET_PATH,
        sheet_name="Customer Profiles",
        engine="openpyxl"
    )

    print("Transactions loaded:", transactions.shape)
    print("Customer profiles loaded:", customers.shape)

    return transactions, customers


# ==========================================
# VALIDATE COLUMNS
# ==========================================

def validate_columns(transactions, customers):

    print("\nValidating columns...")

    missing_transaction_columns = [
        col for col in TRANSACTION_COLUMNS
        if col not in transactions.columns
    ]

    missing_customer_columns = [
        col for col in CUSTOMER_COLUMNS
        if col not in customers.columns
    ]

    if missing_transaction_columns:
        raise ValueError(
            f"Missing transaction columns: {missing_transaction_columns}"
        )

    if missing_customer_columns:
        raise ValueError(
            f"Missing customer columns: {missing_customer_columns}"
        )

    print("All required columns are present.")


# ==========================================
# DATA TYPE CONVERSION
# ==========================================

def convert_data_types(transactions, customers):

    print("\nConverting data types...")

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="coerce"
    )

    customers["account_creation_date"] = pd.to_datetime(
        customers["account_creation_date"],
        errors="coerce"
    )

    numeric_columns = [
        "amount",
        "card_present",
        "is_international",
        "transaction_latitude",
        "transaction_longitude",
        "Class"
    ]

    for col in numeric_columns:
        transactions[col] = pd.to_numeric(
            transactions[col],
            errors="coerce"
        )

    customer_numeric_columns = [
        "customer_registered_latitude",
        "customer_registered_longitude"
    ]

    for col in customer_numeric_columns:
        customers[col] = pd.to_numeric(
            customers[col],
            errors="coerce"
        )

    print("Data type conversion completed.")

    return transactions, customers


# ==========================================
# VALIDATE TRANSACTIONS
# ==========================================

def validate_transactions(transactions):

    print("\nValidating transaction records...")

    errors = []

    def add_error(mask, column, reason):

        for index in transactions.index[mask]:

            errors.append({
                "record_type": "transaction",
                "record_id": transactions.at[index, "transaction_id"],
                "column": column,
                "reason": reason
            })

    # Missing critical values

    critical_columns = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "currency",
        "transaction_type",
        "transaction_subtype"
    ]

    for col in critical_columns:

        add_error(
            transactions[col].isnull(),
            col,
            "Missing critical value"
        )

    # Invalid amount

    add_error(
        transactions["amount"].notnull()
        & (transactions["amount"] <= 0),
        "amount",
        "Amount must be greater than zero"
    )

    # Invalid class label

    add_error(
        transactions["Class"].notnull()
        & ~transactions["Class"].isin([0, 1]),
        "Class",
        "Class must be 0 or 1"
    )

    # Fraud type and class consistency

    add_error(
        (transactions["Class"] == 0)
        & (transactions["fraud_type"] != "NONE"),
        "fraud_type",
        "Genuine transaction must have fraud_type NONE"
    )

    add_error(
        (transactions["Class"] == 1)
        & (
            ~transactions["fraud_type"].isin(
                ["CNP", "ATO", "MONEY_MULE"]
            )
        ),
        "fraud_type",
        "Invalid fraud type for fraudulent transaction"
    )

    # Binary field validation

    for col in ["card_present", "is_international"]:

        add_error(
            transactions[col].notnull()
            & ~transactions[col].isin([0, 1]),
            col,
            "Value must be 0 or 1"
        )

    # Currency validation

    valid_currencies = [
        "INR", "USD", "EUR", "GBP", "AED", "SGD"
    ]

    add_error(
        transactions["currency"].notnull()
        & ~transactions["currency"].isin(valid_currencies),
        "currency",
        "Unsupported currency"
    )

    # Geographic coordinate validation

    add_error(
        transactions["transaction_latitude"].notnull()
        & ~transactions["transaction_latitude"].between(-90, 90),
        "transaction_latitude",
        "Latitude must be between -90 and 90"
    )

    add_error(
        transactions["transaction_longitude"].notnull()
        & ~transactions["transaction_longitude"].between(-180, 180),
        "transaction_longitude",
        "Longitude must be between -180 and 180"
    )

    # Duplicate transaction IDs

    add_error(
        transactions["transaction_id"].notnull()
        & transactions["transaction_id"].duplicated(keep=False),
        "transaction_id",
        "Duplicate transaction ID"
    )

    # Transaction-specific merchant/payee validation

    merchant_types = [
        "MERCHANT_PAYMENT",
        "ONLINE_PURCHASE"
    ]

    payee_types = [
        "P2P_TRANSFER",
        "WALLET_TRANSFER"
    ]

    add_error(
        transactions["transaction_subtype"].isin(merchant_types)
        & transactions["merchant_id"].isnull(),
        "merchant_id",
        "Merchant ID required for this transaction subtype"
    )

    add_error(
        transactions["transaction_subtype"].isin(payee_types)
        & transactions["payee_id"].isnull(),
        "payee_id",
        "Payee ID required for this transaction subtype"
    )

    # Build rejected transaction list

    rejected_indices = {
        error_index
        for error_index in transactions.index
        if any(
            error["record_id"] == transactions.at[error_index, "transaction_id"]
            and error["record_type"] == "transaction"
            for error in errors
        )
    }

    # Use error rows to identify rejected records
    error_ids = {
        error["record_id"]
        for error in errors
        if error["record_type"] == "transaction"
    }

    rejected_mask = transactions["transaction_id"].isin(error_ids)

    valid_data = transactions[~rejected_mask].copy()
    rejected_data = transactions[rejected_mask].copy()

    print("Valid transactions:", len(valid_data))
    print("Rejected transactions:", len(rejected_data))

    return valid_data, rejected_data, errors


# ==========================================
# VALIDATE CUSTOMER PROFILES
# ==========================================

def validate_customers(customers):

    print("\nValidating customer profiles...")

    errors = []

    # Missing customer ID

    missing_id = customers["customer_id"].isnull()

    for index in customers.index[missing_id]:

        errors.append({
            "record_type": "customer_profile",
            "record_id": None,
            "column": "customer_id",
            "reason": "Missing customer ID"
        })

    # Missing or invalid account creation date

    invalid_date = customers["account_creation_date"].isnull()

    for index in customers.index[invalid_date]:

        errors.append({
            "record_type": "customer_profile",
            "record_id": customers.at[index, "customer_id"],
            "column": "account_creation_date",
            "reason": "Missing or invalid account creation date"
        })

    # Duplicate customer IDs

    duplicate_ids = (
        customers["customer_id"].notnull()
        & customers["customer_id"].duplicated(keep=False)
    )

    for index in customers.index[duplicate_ids]:

        errors.append({
            "record_type": "customer_profile",
            "record_id": customers.at[index, "customer_id"],
            "column": "customer_id",
            "reason": "Duplicate customer ID"
        })

    # Validate customer coordinates

    for col, minimum, maximum in [
        ("customer_registered_latitude", -90, 90),
        ("customer_registered_longitude", -180, 180)
    ]:

        invalid = (
            customers[col].notnull()
            & ~customers[col].between(minimum, maximum)
        )

        for index in customers.index[invalid]:

            errors.append({
                "record_type": "customer_profile",
                "record_id": customers.at[index, "customer_id"],
                "column": col,
                "reason": "Invalid geographic coordinate"
            })

    # Both coordinates must be present or both missing

    one_coordinate_missing = (
        customers["customer_registered_latitude"].isnull()
        != customers["customer_registered_longitude"].isnull()
    )

    for index in customers.index[one_coordinate_missing]:

        errors.append({
            "record_type": "customer_profile",
            "record_id": customers.at[index, "customer_id"],
            "column": "customer_coordinates",
            "reason": "Only one customer coordinate is missing"
        })

    error_ids = {
        error["record_id"]
        for error in errors
        if error["record_id"] is not None
    }

    rejected_mask = (
        customers["customer_id"].isin(error_ids)
        | customers["customer_id"].isnull()
    )

    valid_customers = customers[~rejected_mask].copy()

    print("Valid customer profiles:", len(valid_customers))
    print("Rejected customer profiles:", rejected_mask.sum())

    return valid_customers, errors


# ==========================================
# VALIDATE CUSTOMER REFERENCES
# ==========================================

def validate_customer_references(transactions, customers):

    print("\nValidating customer references...")

    unmatched = ~transactions["customer_id"].isin(
        customers["customer_id"]
    )

    rejected = transactions[unmatched].copy()
    valid = transactions[~unmatched].copy()

    errors = []

    for index in rejected.index:

        errors.append({
            "record_type": "transaction",
            "record_id": rejected.at[index, "transaction_id"],
            "column": "customer_id",
            "reason": "Customer profile not found"
        })

    print("Transactions with unknown customers:", len(rejected))

    return valid, rejected, errors


# ==========================================
# MERGE CUSTOMER PROFILES
# ==========================================

def merge_customer_profiles(transactions, customers):

    print("\nMerging customer profiles...")

    processed_data = transactions.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    print("Merged dataset shape:", processed_data.shape)

    return processed_data


# ==========================================
# SAVE OUTPUT FILES
# ==========================================

def save_outputs(valid_data, rejected_data, errors):

    print("\nSaving output files...")

    valid_data = valid_data.sort_values("timestamp")

    valid_data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    rejected_data.to_csv(
        REJECTED_PATH,
        index=False
    )

    error_report = pd.DataFrame(errors)

    error_report.to_csv(
        ERROR_REPORT_PATH,
        index=False
    )

    print("\nAll output files saved successfully!")

    print("Processed data:", OUTPUT_PATH)
    print("Rejected records:", REJECTED_PATH)
    print("Error report:", ERROR_REPORT_PATH)


# ==========================================
# MAIN DATA PROCESSING FUNCTION
# ==========================================

def process_data():

    transactions, customers = load_data()

    validate_columns(transactions, customers)

    transactions, customers = convert_data_types(
        transactions,
        customers
    )

    valid_transactions, rejected_transactions, transaction_errors = (
        validate_transactions(transactions)
    )

    valid_customers, customer_errors = validate_customers(customers)

    valid_transactions, unmatched_transactions, reference_errors = (
        validate_customer_references(
            valid_transactions,
            valid_customers
        )
    )

    rejected_transactions = pd.concat(
        [rejected_transactions, unmatched_transactions],
        ignore_index=True
    )

    all_errors = (
        transaction_errors
        + customer_errors
        + reference_errors
    )

    processed_data = merge_customer_profiles(
        valid_transactions,
        valid_customers
    )

    save_outputs(
        processed_data,
        rejected_transactions,
        all_errors
    )

    print("\n========== DATA PROCESSING COMPLETED ==========")

    return processed_data


# ==========================================
# PROGRAM EXECUTION
# ==========================================

if __name__ == "__main__":

    process_data()
