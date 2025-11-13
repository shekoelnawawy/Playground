import joblib
import math
import os
import shutil
import numpy as np
from pathlib import Path

base_dir = Path("/home/mnawawy/Downloads/OhioT1DM/processed_data/")
os.makedirs(os.path.join(base_dir, "drift", "patient", "2018data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "drift", "patient", "2020data"), exist_ok=True)

for file_path in base_dir.rglob("/*data/*.pkl"):
    if file_path.name.endswith(".test.pkl"):
        print(f"\n🔍 Checking: {file_path}")
        # try:
        #     obj = joblib.load(file_path)
        #     try:
        #         length = len(obj)
        #         print(f"✅ Length: {length}")
        #     except TypeError:
        #         print("⚠️ Object has no measurable length (no __len__).")
        #
        #     obj['glucose'][math.floor(len(obj) * 0.7):] += np.random.randint(low=20, high=30, size=len(obj['glucose'][math.floor(len(obj) * 0.7):]))
        #
        # except Exception as e:
        #     print(f"❌ Failed to load: {e}")
    else:
        if "2018" in str(file_path):
            shutil.copyfile(file_path, os.path.join(base_dir, "drift", "patient", "2018data", file_path.name))
        elif "2020" in str(file_path):
            shutil.copyfile(file_path, os.path.join(base_dir, "drift", "patient", "2020data", file_path.name))
