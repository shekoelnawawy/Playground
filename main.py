import joblib
import math
import os
import numpy as np
from pathlib import Path

base_dir = Path("/home/mnawawy/Downloads/OhioT1DM/processed_data/")
years = ["2020data", "2018data"]

for src_dir in base_dir.rglob("*data"):
    print(src_dir)
exit(1)
# src_dir = Path(os.path.join(base_dir,))
# dst_dir =

for file_path in src_dir.rglob("*.test.pkl"):
    print(f"\n🔍 Checking: {file_path}")
    try:
        obj = joblib.load(file_path)
        try:
            length = len(obj)
            print(f"✅ Length: {length}")
        except TypeError:
            print("⚠️ Object has no measurable length (no __len__).")

        obj['glucose'][math.floor(len(obj) * 0.7):] += np.random.randint(low=20, high=30, size=len(obj['glucose'][math.floor(len(obj) * 0.7):]))

    except Exception as e:
        print(f"❌ Failed to load: {e}")