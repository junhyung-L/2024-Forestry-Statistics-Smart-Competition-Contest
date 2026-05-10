"""
2024 Forestry Stats Smart Competition: Data Loader Module
Handles loading, cleaning, and English translation of forestry data.
"""

import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    """Handles loading, cleaning, and English translation of forestry data."""

    def __init__(self, data_path: str, encoding: str = "cp949"):
        self.data_path = data_path
        self.encoding = encoding
        self.column_mapping = {
            "토성코드": "soil_texture_code",
            "토양깊이유형": "soil_depth_type",
            "토양형코드": "soil_type_code",
            "토양유효수분량": "soil_effective_moisture",
            "연평균기온": "avg_temp",
            "최고기온": "max_temp",
            "최저기온": "min_temp",
            "상대습도(%)": "humidity",
            "강수량(mm)": "precipitation",
            "온도차이": "temp_diff",
            "밤 (kg)": "chestnut_kg",
            "복분자딸기 (kg)": "blackberry_kg",
            "오갈피 (kg)": "ogapi_kg",
            "더덕 (kg)": "deodeok_kg",
            "생표고 (kg)": "shiitake_kg",
            "마 (kg)": "yam_kg",
            "도라지 (kg)": "doraji_kg",
        }

    def load_and_transform(self) -> pd.DataFrame:
        """Loads data, renames columns, and fills missing values."""
        logging.info(f"Loading data from {self.data_path}")
        data = pd.read_csv(self.data_path, encoding=self.encoding)

        # Rename columns to English
        data.rename(columns=self.column_mapping, inplace=True)

        # Feature engineering
        if (
            "temp_diff" not in data.columns
            and "max_temp" in data.columns
            and "min_temp" in data.columns
        ):
            data["temp_diff"] = data["max_temp"] - data["min_temp"]

        # Fill missing values for climate variables
        climate_vars = [
            "avg_temp",
            "max_temp",
            "min_temp",
            "humidity",
            "precipitation",
            "temp_diff",
        ]
        for col in climate_vars:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mean())

        # Ensure categorical columns are objects
        cat_vars = [
            "soil_texture_code",
            "soil_depth_type",
            "soil_type_code",
            "soil_effective_moisture",
        ]
        for col in cat_vars:
            if col in data.columns:
                data[col] = data[col].astype("object")

        return data


if __name__ == "__main__":
    print("Data Loader Module loaded.")
