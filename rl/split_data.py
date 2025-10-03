import pandas as pd
import os

data_file = "data/BTCUSD_5m_2y.parquet"
df = pd.read_parquet(data_file)

# 確保時間序列排序
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df = df.sort_values("ts").drop_duplicates(subset=["ts"])

n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.9)

os.makedirs("data/split", exist_ok=True)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

train_df.to_parquet("data/split/train.parquet")
val_df.to_parquet("data/split/val.parquet")
test_df.to_parquet("data/split/test.parquet")

print("Train / Val / Test saved:")
print(f"train: {len(train_df)} rows")
print(f"val: {len(val_df)} rows")
print(f"test: {len(test_df)} rows")
