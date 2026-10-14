"""
Student Performance Analytics and AI-Based Prediction
Author: Shahrukh Khan

Educational demonstration using a synthetic dataset.
Run: python student_performance_ai_project.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_performance_sample_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "assignments_completed_out_of_10",
    "quiz_average_out_of_10",
    "sleep_hours_per_day",
]
TARGET = "final_score"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Keep the CSV in the same folder as this script."
        )

    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print("\\nSTUDENT PERFORMANCE ANALYTICS AND AI-BASED PREDICTION")
    print("=" * 55)
    print(f"Dataset records: {len(data)}")
    print("\\nDescriptive statistics:")
    print(data[required].describe().round(2).to_string())

    # Visualisation 1: Study hours vs final score
    plt.figure(figsize=(8, 5))
    plt.scatter(data["study_hours_per_day"], data[TARGET], alpha=0.8)
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study hours per day")
    plt.ylabel("Final score")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_hours_vs_score.png", dpi=160)
    plt.close()

    # Visualisation 2: Correlation heatmap (Matplotlib only)
    corr = data[FEATURES + [TARGET]].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, label="Correlation")
    labels = [label.replace("_", "\\n") for label in corr.columns]
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title("Correlation Between Performance Indicators")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=42, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Save observed and predicted scores for the held-out test set
    prediction_table = X_test.copy()
    prediction_table.insert(0, "student_id", data.loc[X_test.index, "student_id"].values)
    prediction_table["actual_final_score"] = y_test.values
    prediction_table["predicted_final_score"] = predictions.round(1)
    prediction_table["absolute_error"] = (
        prediction_table["actual_final_score"] - prediction_table["predicted_final_score"]
    ).abs().round(1)
    prediction_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Visualisation 3: Actual vs predicted scores
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.85)
    lower = min(float(y_test.min()), float(predictions.min())) - 3
    upper = max(float(y_test.max()), float(predictions.max())) + 3
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.title("Actual vs Predicted Final Scores")
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    metrics = (
        "Student Performance Prediction Model\\n"
        f"Records: {len(data)}\\n"
        f"Training records: {len(X_train)}\\n"
        f"Test records: {len(X_test)}\\n"
        f"Mean Absolute Error (MAE): {mae:.2f}\\n"
        f"R-squared (R2): {r2:.3f}\\n"
    )
    (OUTPUT_DIR / "model_metrics.txt").write_text(metrics, encoding="utf-8")

    print("\\nModel evaluation (held-out test set):")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.3f}")
    print(f"\\nFiles saved in: {OUTPUT_DIR}")
    print("- study_hours_vs_score.png")
    print("- correlation_heatmap.png")
    print("- actual_vs_predicted.png")
    print("- test_predictions.csv")
    print("- model_metrics.txt")
    print("\\nNote: Results are educational only; the dataset is synthetic and very small.")

if __name__ == "__main__":
    main()
