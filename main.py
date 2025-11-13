import joblib
import math
import os
import numpy as np
from pathlib import Path

base_dir = Path("/home/mnawawy/Downloads/OhioT1DM/processed_data/")

for file_path in base_dir.rglob("*data/*.pkl"):
    # for file_path in src_dir.rglob("*.pkl"):
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
        # else:
