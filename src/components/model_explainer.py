import os
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for saving plots without a display
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class ModelExplainer:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
        self.model_path        = os.path.join("artifact", "model.pkl")
        self.output_dir        = os.path.join("artifact", "shap_plots")

    def generate_shap_plots(self, test_df_path: str, target_col: str = "math_score"):
        """
        Loads test data, applies the saved preprocessor, runs SHAP on the
        saved best model, and writes 3 plots to artifact/shap_plots/.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            preprocessor = load_object(self.preprocessor_path)
            model        = load_object(self.model_path)

            test_df = pd.read_csv(test_df_path)
            X_test  = test_df.drop(columns=[target_col])

            X_test_transformed = preprocessor.transform(X_test)

            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

            logging.info("Computing SHAP values...")

            model_type = type(model).__name__

            if model_type in ["LinearRegression", "Ridge", "Lasso"]:
                explainer   = shap.LinearExplainer(model, X_test_transformed)
                shap_values = explainer.shap_values(X_test_transformed)
                base_value  = explainer.expected_value
            else:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_transformed)
                base_value  = explainer.expected_value

            # Some explainers return a list (multi-output); unwrap if needed
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if isinstance(base_value, (list, np.ndarray)) and np.ndim(base_value) > 0:
                base_value = base_value[0]

            X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

            # ── Plot 1: Summary (beeswarm) ────────────────────────────
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_df, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
            plt.close()
            logging.info("Saved shap_summary.png")

            # ── Plot 2: Bar (mean absolute SHAP) ─────────────────────
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
            plt.close()
            logging.info("Saved shap_bar.png")

            # ── Plot 3: Waterfall for first test sample ───────────────
            plt.figure(figsize=(10, 6))
            shap_explanation = shap.Explanation(
                values        = shap_values[0],
                base_values   = base_value,
                data          = X_test_df.iloc[0].values,
                feature_names = list(feature_names)
            )
            shap.plots.waterfall(shap_explanation, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
            plt.close()
            logging.info("Saved shap_waterfall.png")

            logging.info(f"All SHAP plots saved to {self.output_dir}")
            print(f"\nSHAP plots saved to: {self.output_dir}")

        except Exception as e:
            raise CustomException(e, sys)