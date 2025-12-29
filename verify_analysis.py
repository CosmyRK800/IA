import pandas as pd
import time
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- Configuration ---
DATA_PATH = "/home/ubuntu/data_fao/consumption_user.csv"
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5

def load_and_preprocess(file_path):
    """
    Loads the data and performs the preprocessing steps described in the document:
    1. Defines a transaction as a unique combination of SUBJECT, ROUND, SURVEY_DAY, and CONSUMPTION_TIME_HOUR.
    2. Uses FOODEX2_INGR_DESCR as the item (ingredient).
    3. Filters out transactions with only one item.
    """
    print("Loading data...")
    # Use a subset of columns to save memory and time
    cols_to_use = [
        "SUBJECT", "ROUND", "SURVEY_DAY", "CONSUMPTION_TIME_HOUR",
        "FOODEX2_INGR_DESCR"
    ]
    df = pd.read_csv(file_path, usecols=cols_to_use)

    # Clean up the ingredient description column
    df['FOODEX2_INGR_DESCR'] = df['FOODEX2_INGR_DESCR'].astype(str).str.strip()

    # Define the transaction ID
    df['TRANSACTION_ID'] = df['SUBJECT'].astype(str) + '_' + \
                           df['ROUND'].astype(str) + '_' + \
                           df['SURVEY_DAY'].astype(str) + '_' + \
                           df['CONSUMPTION_TIME_HOUR'].astype(str)

    print("Grouping items into transactions...")
    # Group by transaction ID and aggregate ingredients into a list
    transactions = df.groupby('TRANSACTION_ID')['FOODEX2_INGR_DESCR'].apply(list).reset_index(name='Items')

    # Filter out transactions with only one item (as per the document)
    initial_count = len(transactions)
    transactions['Item_Count'] = transactions['Items'].apply(len)
    transactions_filtered = transactions[transactions['Item_Count'] >= 2]
    final_count = len(transactions_filtered)

    print(f"Initial number of transactions: {initial_count}")
    print(f"Final number of transactions (>= 2 items): {final_count}")

    # Convert the list of lists into a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions_filtered['Items']).transform(transactions_filtered['Items'])
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    return df_encoded, final_count

def run_fpgrowth(df_encoded):
    """Runs the FP-Growth algorithm and extracts association rules."""
    print("\nRunning FP-Growth...")
    start_time = time.time()
    frequent_itemsets_fp = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    time_fp = time.time() - start_time
    print(f"FP-Growth execution time: {time_fp:.4f} seconds")

    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules_fp = rules_fp.sort_values(by='lift', ascending=False)

    return time_fp, frequent_itemsets_fp, rules_fp

def run_apriori(df_encoded):
    """Runs the Apriori algorithm and extracts association rules."""
    print("\nRunning Apriori...")
    start_time = time.time()
    frequent_itemsets_ap = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    time_ap = time.time() - start_time
    print(f"Apriori execution time: {time_ap:.4f} seconds")

    rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules_ap = rules_ap.sort_values(by='lift', ascending=False)

    return time_ap, frequent_itemsets_ap, rules_ap

if __name__ == "__main__":
    try:
        df_encoded, final_transaction_count = load_and_preprocess(DATA_PATH)

        # 1. Verify transaction count
        print(f"\n--- Verification of Preprocessing ---")
        print(f"Document's transaction count: 207,849")
        print(f"Calculated transaction count: {final_transaction_count}")

        # 2. Run FP-Growth and Apriori for verification
        time_fp, frequent_itemsets_fp, rules_fp = run_fpgrowth(df_encoded)
        time_ap, frequent_itemsets_ap, rules_ap = run_apriori(df_encoded)

        # 3. Verify results
        print(f"\n--- Verification of Results (MinSup={MIN_SUPPORT}, MinConf={MIN_CONFIDENCE}) ---")
        print(f"FP-Growth Time: {time_fp:.4f}s | Document Time: 1.5103s")
        print(f"Apriori Time: {time_ap:.4f}s | Document Time: 9.3182s")
        print(f"Number of Frequent Itemsets (FP): {len(frequent_itemsets_fp)}")
        print(f"Number of Rules (FP): {len(rules_fp)}")

        print("\nTop 5 Rules (FP-Growth, sorted by Lift):")
        # Format the output to match the document's table structure
        rules_output = rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5)
        print(rules_output.to_markdown(index=False, floatfmt=".4f"))

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        # If mlxtend is not installed, this will catch the error
        # and I will attempt to install it again.
        if "No module named 'mlxtend'" in str(e):
            print("\nmlxtend is not installed. Attempting installation...")
            # The previous installation failed, I will try again with the recommended flag
            import subprocess
            subprocess.run(["pip3", "install", "mlxtend", "--break-system-packages"])
            print("Please re-run the script after successful installation.")
