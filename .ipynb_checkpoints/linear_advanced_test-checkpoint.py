from typing import Any


from zipfile import ZIP_BZIP2
import pandas as pd
import numpy as np

# split
from sklearn.model_selection import train_test_split


# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# model
from sklearn.linear_model import LinearRegression

# evaluation
from sklearn.metrics import mean_squared_error, r2_score


# 1️⃣ load dataset
df = pd.read_csv("energy.csv")

# 2️⃣ features & label
X = df[
    ["area", "num_people", "avg_temp", "insulation_score", "building_age"]
]
y = df["monthly_energy_consumption"]

# 3️⃣ train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4️⃣ pipeline (very important)
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# 5️⃣ train
pipeline.fit(X_train, y_train)

# 6️⃣ predict
y_pred = pipeline.predict(X_test)

# 7️⃣ evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2:", r2)


wights = pipeline.named_steps["model"].coef_
fetures = X.columns


for f, w in zip(fetures, wights):
    print(f"{f}: {w}")



