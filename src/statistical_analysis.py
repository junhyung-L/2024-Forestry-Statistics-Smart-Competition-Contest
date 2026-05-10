"""
2024 Forestry Stats Smart Competition: Statistical Analysis Module
Refactored from notebooks/data_preprocessing.ipynb
Fully translated to English.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, spearmanr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ForestryStatAnalyzer:
    """Performs statistical analysis (Chi-Square, ANOVA, Spearman) on forestry data."""

    def __init__(self, data_path: str, encoding: str = "cp949"):
        self.data_path = data_path
        self.encoding = encoding
        self.data = None

        # Column mapping for translation
        self.column_mapping = {
            "토양깊이유형": "soil_depth_type",
            "토성코드": "soil_texture_code",
            "토양형코드": "soil_type_code",
            "토양유효수분량": "soil_effective_moisture",
            "밤 (kg)": "chestnut_kg",
            "복분자딸기 (kg)": "blackberry_kg",
            "오갈피 (kg)": "ogapi_kg",
            "마 (kg)": "yam_kg",
            "도라지 (kg)": "doraji_kg",
            "더덕 (kg)": "deodeok_kg",
            "생표고 (kg)": "shiitake_kg",
        }

        self.categorical_columns = [
            "soil_depth_type",
            "soil_texture_code",
            "soil_type_code",
            "soil_effective_moisture",
        ]
        self.production_columns = [
            "chestnut_kg",
            "blackberry_kg",
            "ogapi_kg",
            "yam_kg",
            "doraji_kg",
            "deodeok_kg",
            "shiitake_kg",
        ]

    def load_data(self) -> pd.DataFrame:
        """Loads data, renames columns to English, and ensures correct types."""
        logging.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path, encoding=self.encoding)

        # Rename columns to English
        self.data.rename(columns=self.column_mapping, inplace=True)

        # Ensure categorical columns are treated as objects
        for col in self.categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype("object")

        # Ensure production columns are numeric
        for col in self.production_columns:
            if col in self.data.columns:
                self.data[col] = (
                    pd.to_numeric(self.data[col], errors="coerce").fillna(0)
                )

        return self.data

    def perform_analysis_for_production(
        self, production_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform Chi-Square and ANOVA analysis for a given production column."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Filter the data to exclude rows where production is 0
        filtered_data = self.data[self.data[production_col] != 0].copy()

        # Chi-Square test
        chi2_results = {}
        for cat_col in self.categorical_columns:
            if cat_col in filtered_data.columns:
                crosstab = pd.crosstab(
                    filtered_data[cat_col], filtered_data[production_col]
                )
                _, p, _, _ = chi2_contingency(crosstab)
                chi2_results[cat_col] = {"p-value": p}

        # ANOVA test
        anova_results = {}
        for cat_col in filtered_data.columns:
            if cat_col in filtered_data.columns:
                groups = [
                    group[production_col].values
                    for name, group in filtered_data.groupby(cat_col)
                ]
                if (
                    len(groups) > 1
                ):  # ANOVA requires at least 2 groups
                    _, p = f_oneway(*groups)
                    anova_results[cat_col] = {"p-value": p}

        chi2_df = pd.DataFrame.from_dict(
            chi2_results, orient="index", columns=["p-value"]
        )
        anova_df = pd.DataFrame.from_dict(
            anova_results, orient="index", columns=["p-value"]
        )

        return chi2_df, anova_df

    def calculate_spearman_correlation(
        self, production_col: str
    ) -> pd.DataFrame:
        """Calculate Spearman correlation coefficients."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        filtered_data = self.data[self.data[production_col] != 0].copy()

        # Ensure categorical columns are numeric for correlation calculation
        temp_data = filtered_data.copy()
        for col in self.categorical_columns:
            if col in temp_data.columns:
                temp_data[col] = pd.factorize(temp_data[col])[0]

        spearman_results = {}
        for col in self.categorical_columns:
            if col in temp_data.columns:
                coef, p = spearmanr(temp_data[col], temp_data[production_col])
                spearman_results[col] = {
                    "Spearman Correlation": coef,
                    "p-value": p,
                }

        return pd.DataFrame.from_dict(spearman_results, orient="index")

    def run_full_analysis(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Runs full analysis for all production columns."""
        if self.data is None:
            self.load_data()

        results = {}
        for prod_col in self.production_columns:
            if prod_col in self.data.columns:
                logging.info(f"Analyzing {prod_col}...")
                chi2, anova = self.perform_analysis_for_production(prod_col)
                spearman = self.calculate_spearman_correlation(prod_col)
                results[prod_col] = {
                    "Chi-Square": chi2,
                    "ANOVA": anova,
                    "Spearman": spearman,
                }
        return results


if __name__ == "__main__":
    print("Forestry Statistical Analysis Module loaded.")
