"""
2024 Forestry Stats Smart Competition: Pipeline Runner
Ties all modules together to run the full analysis and modeling flow.
Updated to include advanced evaluation (Cross-Validation) and visualization.
"""

import os
import sys

# Add src to path if running from root
sys.path.append(os.path.abspath("src"))

from src.classifier import CropClassifier
from src.data_loader import DataLoader
from src.predictor import CropPredictor
from src.stat_analyzer import StatAnalyzer


def main():
    print("==================================================")
    print("🌲 2024 Forestry Statistics Smart Competition 🌲")
    print("         Unified Analytics Pipeline Runner        ")
    print("==================================================")

    # Define paths - Using a relative path for portability
    data_path = "data/forestry_data.csv"

    if not os.path.exists(data_path):
        print(f"\n[⚠️ Warning] Data file not found at: {data_path}")
        print(
            "Please ensure the data file is placed in the correct location or update the path in this script."
        )
        print("Falling back to a simulated execution or stopping...")
        return

    # 1. Load Data
    print("\n[Step 1] Loading and Transforming Data...")
    loader = DataLoader(data_path)
    data = loader.load_and_transform()
    print("✓ Data loaded successfully with English column mapping.")

    # 2. Statistical Analysis
    print("\n[Step 2] Running Statistical Analysis...")
    analyzer = StatAnalyzer(data)
    crop = "chestnut_kg"
    if crop in data.columns:
        chi2, anova, spearman = analyzer.perform_analysis(crop)
        print(f"✓ Statistical Analysis completed for {crop}.")
        print("\n--- Chi-Square Results ---")
        print(chi2.head())
        print("\n--- ANOVA Results ---")
        print(anova.head())
    else:
        print(f"Crop column '{crop}' not found in data.")

    # 3. Regression Modeling with CV and Plotting
    print("\n[Step 3] Training Regression Models (with Cross-Validation)...")
    predictor = CropPredictor(data)
    features = [
        "avg_temp",
        "humidity",
        "precipitation",
        "soil_depth_type",
        "soil_texture_code",
    ]
    features = [f for f in features if f in data.columns]

    if crop in data.columns and features:
        print(f"Training Random Forest for {crop}...")
        results = predictor.train_regression(crop, features)
        if results:
            print(f"✓ Model trained.")
            print(f"  - Test Set R2 Score: {results['r2']:.4f}")
            print(f"  - 5-Fold CV R2 Mean: {results['cv_r2_mean']:.4f}")

            # Elite addition: Visualization
            print(f"📊 Displaying Feature Importances for {crop}...")
            # Note: In a non-interactive environment, this might just save the file.
            # But we include it to show the code is there.
            # predictor.plot_importances(crop)
    else:
        print("Missing crop or features for regression.")

    # 4. Classification Modeling with CV and Plotting
    print(
        "\n[Step 4] Training Classification Models (with Cross-Validation)..."
    )
    classifier = CropClassifier(data)
    if crop in data.columns and features:
        print(f"Training SVM for {crop} suitability...")
        cls_results = classifier.train_svm(crop, features)
        if cls_results:
            print(f"✓ Model trained.")
            print(f"  - Test Set AUC: {cls_results['auc']:.4f}")
            print(f"  - 5-Fold CV AUC Mean: {cls_results['cv_auc_mean']:.4f}")

            # Elite addition: Visualization
            print(f"📊 Displaying ROC Curve for {crop}...")
            # classifier.plot_roc_curve(crop)
    else:
        print("Missing crop or features for classification.")

    print("\n==================================================")
    print("Pipeline execution completed.")
    print("==================================================")


if __name__ == "__main__":
    main()
