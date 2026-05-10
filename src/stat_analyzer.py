"""
2024 Forestry Stats Smart Competition: Statistical Analyzer Module
Performs Chi-Square, ANOVA, and Spearman correlation.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, spearmanr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class StatAnalyzer:
    """Performs statistical tests on forestry data."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.categorical_columns = [
            "soil_depth_type",
            "soil_texture_code",
            "soil_type_code",
            "soil_effective_moisture",
        ]

    def perform_analysis(
        self, production_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform Chi-Square, ANOVA, and Spearman analysis for a given production column."""
        # Filter the data to exclude rows where production is 0
        filtered_data = self.data[self.data[production_col] > 0].copy()

        if len(filtered_data) < 10:
            logging.warning(f"Not enough data for {production_col}. Skipping.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
        for cat_col in self.categorical_columns:
            if cat_col in filtered_data.columns:
                groups = [
                    group[production_col].values
                    for name, group in filtered_data.groupby(cat_col)
                ]
                if len(groups) > 1:
                    _, p = f_oneway(*groups)
                    anova_results[cat_col] = {"p-value": p}

        # Spearman correlation
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

        chi2_df = pd.DataFrame.from_dict(
            chi2_results, orient="index", columns=["p-value"]
        )
        anova_df = pd.DataFrame.from_dict(
            anova_results, orient="index", columns=["p-value"]
        )
        spearman_df = pd.DataFrame.from_dict(spearman_results, orient="index")

        return chi2_df, anova_df, spearman_df


if __name__ == "__main__":
    print("Statistical Analyzer Module loaded.")
