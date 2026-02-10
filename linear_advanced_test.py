from typing import Any


from zipfile import ZIP_BZIP2
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("energy.csv")

X = df[
    ["area", "num_people", "avg_temp", "insulation_score", "building_age"]
]
y = df["monthly_energy_consumption"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2:", r2)


wights = pipeline.named_steps["model"].coef_
fetures = X.columns


for f, w in zip(fetures, wights):
    print(f"{f}: {w}")




