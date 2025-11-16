import math
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

percent_drift = 30
indexer = (100-percent_drift)/100

base_dir = Path("~/Downloads/Sepsis/files/challenge-2019/1.0.0/training")
# out_dir = Path("/home/mnawawy/Downloads/Sepsis/processed_data/drift/patient")
out_dir = Path("/home/mnawawy/SepsisDrift")

os.makedirs(out_dir, exist_ok=True)

shutil.copyfile(Path(base_dir / 'index.html'), Path(out_dir) / 'index.html')

for item_path in base_dir.rglob("*"):
    if item_path.is_dir():  # Check if the item is a directory
        directory_name = item_path.name
        os.makedirs(os.path.join(out_dir, directory_name), exist_ok=True)

        for file_path in item_path.rglob("*"):
            dst_path = Path(os.path.join(out_dir, directory_name, file_path.name))
            if dst_path.name.endswith("*.psv"):
                try:
                    df = pd.read_csv(file_path, sep='|')
                    df.loc[math.floor(len(df) * indexer):, "HR"] += np.random.randint(low=20, high=30, size=len(df["HR"][math.floor(len(df) * indexer):]))
                    df.to_csv(dst_path, index=False, sep='|')

                except Exception as e:
                    print(f"Failed to load: {e}")
            else:
                if file_path.resolve() != dst_path.resolve():
                    shutil.copyfile(file_path, dst_path)