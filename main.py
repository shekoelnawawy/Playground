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
                    print(file_path)
                    df = pd.read_csv(file_path, header=1)
                    print(df["224751"])
                    exit(1)
                    # df.loc[math.floor(len(df) * indexer):, "glucose"] += np.random.randint(low=20, high=30, size=len(df['glucose'][math.floor(len(df) * indexer):]))
                    # if "2018" in str(file_path):
                    #     dst_path = Path(os.path.join(base_dir, "drift", "patient", "2018data", file_path.name))
                    #     joblib.dump(df, dst_path)
                    # elif "2020" in str(file_path):
                    #     dst_path = Path(os.path.join(base_dir, "drift", "patient", "2020data", file_path.name))
                    #     joblib.dump(df, dst_path)
                except Exception as e:
                    print(f"Failed to load: {e}")
            # else:
            #     dst_path = Path(os.path.join(out_dir, directory_name, file_path.name))
            #     if file_path.resolve() != dst_path.resolve():
            #         shutil.copyfile(file_path, dst_path)
