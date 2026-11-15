import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)


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
    "svm_pace_prediction"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. CHECK TARGET
# ============================================================

TARGET = "PaceCategory"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# 4. SELECT FEATURES
# ============================================================
#
# We intentionally do NOT use LapTime because PaceCategory
# is a performance-related target. Using LapTime would cause
# target leakage.
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

# Keep only features that actually exist in the dataset.
features = [col for col in candidate_features if col in df.columns]

missing_features = [
    col for col in candidate_features if col not in df.columns
]

print("\nFeatures used:")
for feature in features:
    print(f" - {feature}")

if missing_features:
    print("\nFeatures not found and therefore skipped:")
    for feature in missing_features:
        print(f" - {feature}")


# ============================================================
# 5. CREATE MODEL DATASET
# ============================================================

model_df = df[features + [TARGET]].copy()

# Remove rows where the target is missing.
model_df = model_df.dropna(subset=[TARGET])

print(f"\nRows after target cleaning: {len(model_df)}")


# ============================================================
# 6. HANDLE MISSING FEATURE VALUES
# ============================================================

numeric_features = model_df[features].select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = [
    col for col in features
    if col not in numeric_features
]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# Fill numeric missing values with median.
for col in numeric_features:
    model_df[col] = model_df[col].fillna(
        model_df[col].median()
    )

# Fill categorical missing values with "Unknown".
for col in categorical_features:
    model_df[col] = model_df[col].fillna("Unknown")


# ============================================================
# 7. SPLIT INPUT AND TARGET
# ============================================================

X = model_df[features]
y = model_df[TARGET]

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 9. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 10. SVM MODEL
# ============================================================

svm_model = LinearSVC(
    random_state=42,
    class_weight="balanced",
    max_iter=10000
)


# ============================================================
# 11. PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("svm", svm_model)
    ]
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nTraining SVM model...")

model.fit(X_train, y_train)

print("SVM training completed.")


# ============================================================
# 13. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 14. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 60)
print("SVM MODEL RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Weighted F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.csv"
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=sorted(y.unique())
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=sorted(y.unique())
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("SVM Pace Category - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "svm_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

predictions = X_test.copy()

predictions["ActualPaceCategory"] = y_test.values
predictions["PredictedPaceCategory"] = y_pred

predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "svm_predictions.csv"
    ),
    index=False
)


# ============================================================
# 18. SAVE SUMMARY
# ============================================================

summary = f"""
SVM Pace Prediction Summary
===========================

Dataset:
f1_2025_feature_engineered.csv

Target:
PaceCategory

Number of input features:
{len(features)}

Training samples:
{len(X_train)}

Testing samples:
{len(X_test)}

Accuracy:
{accuracy:.4f}

Weighted F1 Score:
{f1:.4f}

Model:
Linear Support Vector Machine (LinearSVC)

Purpose:
Predict Formula 1 lap pace category using tyre,
weather, stint, team and circuit information.

LapTime was intentionally excluded from the input
features to avoid target leakage.
"""

with open(
    os.path.join(
        OUTPUT_DIR,
        "svm_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SVM IMPLEMENTATION COMPLETE")
print("=" * 60)

print(f"\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print(" - classification_report.csv")
print(" - svm_confusion_matrix.png")
print(" - svm_predictions.csv")
print(" - svm_summary.txt")