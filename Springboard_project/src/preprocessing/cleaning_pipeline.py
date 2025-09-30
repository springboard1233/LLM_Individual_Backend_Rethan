
import argparse, os, re, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

def parse_datetime(series):
    s = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
    if s.isna().mean() > 0.7:
        try: s = pd.to_datetime(series.astype(float), unit='s', errors='coerce')
        except: pass
    return s

def coerce_amount(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)
    x = re.sub(r"[₹$,]", "", str(x)).replace("INR", "").strip()
    try: return float(x)
    except: return np.nan

def preprocess(input_path, output_dir):
    print(f"📥 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print("Initial shape:", df.shape)

    rename_map = {"TransactionID":"transaction_id","TransactionDate":"timestamp",
                  "TransactionAmount":"transaction_amount","Channel":"channel"}
    df.rename(columns={c:rename_map[c] for c in df.columns if c in rename_map}, inplace=True)

    if "kyc_verified" not in df.columns:
        df["kyc_verified"] = "No"
    else:
        df["kyc_verified"].fillna("No", inplace=True)

    if "transaction_amount" in df.columns:
        df["transaction_amount"] = df["transaction_amount"].apply(coerce_amount)
        df.dropna(subset=["transaction_amount"], inplace=True)

    if "transaction_id" in df.columns:
        df.drop_duplicates(subset="transaction_id", inplace=True)
    else:
        df.drop_duplicates(inplace=True)

    if "timestamp" in df.columns:
        df["timestamp"] = parse_datetime(df["timestamp"])

    if "channel" in df.columns:
        df["channel"] = df["channel"].astype(str).str.title().replace({"Nan":np.nan})

    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.weekday
    if "transaction_amount" in df.columns:
        df["is_high_value"] = (df["transaction_amount"] > 50000).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir,"transactions_processed.csv"), index=False)
    print("✅ Saved transactions_processed.csv")

    if "is_fraud" in df.columns and df["is_fraud"].nunique()>1:
        train, test = train_test_split(df, test_size=0.2, stratify=df["is_fraud"], random_state=42)
    else:
        train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv(os.path.join(output_dir,"train.csv"), index=False)
    test.to_csv(os.path.join(output_dir,"test.csv"), index=False)
    print("✅ Saved train.csv and test.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--outdir", required=True, help="Folder to save processed data")
    args = parser.parse_args()
    preprocess(args.input, args.outdir)
