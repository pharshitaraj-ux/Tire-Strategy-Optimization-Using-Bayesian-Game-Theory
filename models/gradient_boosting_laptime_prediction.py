import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = os.path.join(
    "outputs",
    "feature_engineering",
    "f1_2025_feature_engineered.csv"
)

OUTPUT_DIR = os.path.join(
    "outputs",
    "advanced_analytics",
    "gradient_boosting_laptime_prediction"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "LapTime"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in dataset."
    )


# ============================================================
# 4. SELECT FEATURES
# ============================================================
#
# LapTime is intentionally NOT included as an input.
# This prevents target leakage.
#

candidate_features = [
    "Compound",
    "TyreLife",
    "StintLength",
    "WeatherState",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Rainfall",
    "WindSpeed",
    "CompoundCode",
    "PitStop",
    "Team",
    "Circuit"
]

features = [
    col for col in candidate_features
    if col in df.columns
]

print("\nFeatures used:")
for feature in features:
    print(f" - {feature}")


# ============================================================
# 5. PREPARE MODEL DATA
# ============================================================

model_df = df[features + [TARGET]].copy()

# Convert target to numeric.
model_df[TARGET] = pd.to_numeric(
    model_df[TARGET],
    errors="coerce"
)

# Remove invalid target values.
model_df = model_df.dropna(subset=[TARGET])

# Remove non-positive lap times.
model_df = model_df[model_df[TARGET] > 0]

print(f"\nRows after LapTime cleaning: {len(model_df)}")


# ============================================================
# 6. DEFINE FEATURE TYPES
# ============================================================

numeric_features = [
    col for col in features
    if pd.api.types.is_numeric_dtype(model_df[col])
]

categorical_features = [
    col for col in features
    if col not in numeric_features
]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    verbose_feature_names_out=False
)


# ============================================================
# 8. GRADIENT BOOSTING MODEL
# ============================================================

gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    loss="squared_error"
)


# ============================================================
# 9. PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("gradient_boosting", gb_model)
    ]
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X = model_df[features]
y = model_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Gradient Boosting model...")

model.fit(X_train, y_train)

print("Gradient Boosting training completed.")


# ============================================================
# 12. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 13. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("GRADIENT BOOSTING MODEL RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f} seconds")
print(f"RMSE : {rmse:.4f} seconds")
print(f"R²   : {r2:.4f}")


# ============================================================
# 14. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_metrics.csv"
    ),
    index=False
)


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "ActualLapTime": y_test.values,
    "PredictedLapTime": y_pred
})

prediction_df["AbsoluteError"] = (
    prediction_df["ActualLapTime"]
    - prediction_df["PredictedLapTime"]
).abs()

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_predictions.csv"
    ),
    index=False
)


# ============================================================
# 16. PREDICTED VS ACTUAL GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.35
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Lap Time (seconds)")
plt.ylabel("Predicted Lap Time (seconds)")
plt.title("Gradient Boosting: Actual vs Predicted Lap Time")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted_laptime.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

# Get transformed feature names.
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    model.named_steps["gradient_boosting"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 18. TOP 15 FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(9, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Gradient Boosting Features")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance_top15.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 19. SAVE SUMMARY
# ============================================================

summary = f"""
Gradient Boosting LapTime Prediction Summary
=============================================

Dataset:
f1_2025_feature_engineered.csv

Target:
LapTime

Number of input features:
{len(features)}

Training samples:
{len(X_train)}

Testing samples:
{len(X_test)}

Model:
GradientBoostingRegressor

Parameters:
n_estimators = 150
learning_rate = 0.05
max_depth = 3

Evaluation:
MAE  = {mae:.4f} seconds
RMSE = {rmse:.4f} seconds
R2   = {r2:.4f}

LapTime was intentionally excluded from the
input features to avoid target leakage.
"""

with open(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("GRADIENT BOOSTING IMPLEMENTATION COMPLETE")
print("=" * 60)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print(" - gradient_boosting_metrics.csv")
print(" - gradient_boosting_predictions.csv")
print(" - actual_vs_predicted_laptime.png")
print(" - feature_importance.csv")
print(" - feature_importance_top15.png")
print(" - gradient_boosting_summary.txt")