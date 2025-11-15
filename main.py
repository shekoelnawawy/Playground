import joblib
import math
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

percent_drift = 30
indexer = (100-percent_drift)/100

base_dir = Path("/home/mnawawy/Downloads/MIMIC/original/data/csv")
out_dir = Path("/home/mnawawy/Downloads/MIMIC/processed_data/drift/patient")
os.makedirs(out_dir, exist_ok=True)

for item_path in base_dir.rglob("*"):
    if item_path.is_dir():  # Check if the item is a directory
        directory_name = item_path.name
        os.makedirs(os.path.join(out_dir, directory_name), exist_ok=True)

        for file_path in base_dir.rglob("*.csv"):
            if file_path.name.endswith("dynamic.csv"):
                try:
                    df_columns = pd.read_csv(file_path, header=None, nrows=1)
                    print(df_columns)
                    df = pd.read_csv(file_path, header=1)
                    print(df)
                    df.loc[math.floor(len(df) * indexer):, "224751"] += np.random.randint(low=20, high=30, size=len(df["224751"][math.floor(len(df) * indexer):]))
                    dst_path = Path(os.path.join(out_dir, directory_name, file_path.name))

                    old_header = df.columns.tolist()  # save header as a list
                    print(pd.DataFrame([old_header]))

                    df.columns = range(df.shape[1])
                    print(df)
                    df = pd.concat([df_columns, pd.DataFrame([old_header]), df], ignore_index=True)
                    df.columns = range(df.shape[1])
                    print(df)
                    exit(1)
                except Exception as e:
                    print(f"Failed to load: {e}")
            else:
                dst_path = Path(os.path.join(out_dir, directory_name, file_path.name))
                if file_path.resolve() != dst_path.resolve():
                    shutil.copyfile(file_path, dst_path)


# Suppose df is your existing dataframe
old_header = df.columns.tolist()       # save header as a list

# Insert header list as the first row


# Remove the header by renaming columns to generic names
df.columns = range(df.shape[1])