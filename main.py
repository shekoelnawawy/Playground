import joblib
import math
import os
import shutil
import numpy as np
from pathlib import Path

base_dir = Path("/home/mnawawy/Downloads/OhioT1DM/processed_data/")
os.makedirs(os.path.join(base_dir, "drift", "patient", "2018data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "drift", "patient", "2020data"), exist_ok=True)

for file_path in base_dir.rglob("*data/*.pkl"):
    if file_path.name.endswith(".test.pkl"):
        print(f"\n🔍 Checking: {file_path}")
        try:
            obj = joblib.load(file_path)
            obj.loc[math.floor(len(obj) * 0.7):, "glucose"] += np.random.randint(low=20, high=30, size=len(obj['glucose'][math.floor(len(obj) * 0.7):]))
            if "2018" in str(file_path):
                dst_path = Path(os.path.join(base_dir, "drift", "patient", "2018data", file_path.name))
                print(obj)
                exit(1)
                joblib.dump(obj)
            elif "2020" in str(file_path):
                dst_path = Path(os.path.join(base_dir, "drift", "patient", "2020data", file_path.name))

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