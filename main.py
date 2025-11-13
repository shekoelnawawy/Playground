import joblib
import math
import os
import shutil
import numpy as np
from pathlib import Path

percent_drift = 30
indexer = (100-percent_drift)/100

base_dir = Path("/home/mnawawy/Downloads/OhioT1DM/processed_data/")
os.makedirs(os.path.join(base_dir, "drift", "patient", "2018data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "drift", "patient", "2020data"), exist_ok=True)

for file_path in base_dir.rglob("*data/*.pkl"):
    if file_path.name.endswith(".test.pkl"):
        try:
            obj = joblib.load(file_path)
            obj.loc[math.floor(len(obj) * indexer):, "glucose"] += np.random.randint(low=20, high=30, size=len(obj['glucose'][math.floor(len(obj) * indexer):]))
            if "2018" in str(file_path):
                dst_path = Path(os.path.join(base_dir, "drift", "patient", "2018data", file_path.name))
                joblib.dump(obj, dst_path)
            elif "2020" in str(file_path):
                dst_path = Path(os.path.join(base_dir, "drift", "patient", "2020data", file_path.name))
                joblib.dump(obj, dst_path)
        except Exception as e:
            print(f"Failed to load: {e}")
    else:
        if "2018" in str(file_path):
            dst_path = Path(os.path.join(base_dir, "drift", "patient", "2018data", file_path.name))
            if file_path.resolve() != dst_path.resolve():
                shutil.copyfile(file_path, dst_path)
        elif "2020" in str(file_path):
            dst_path = Path(os.path.join(base_dir, "drift", "patient", "2020data", file_path.name))
            if file_path.resolve() != dst_path.resolve():
                shutil.copyfile(file_path, dst_path)