# 2024 Forestry Statistics Smart Competition: Grand Prize (1st Place) 🏆

This repository contains the award-winning project for the **'2024 Forestry Statistics Smart Competition'** hosted by the **Korea Forest Service** and **Korea Forestry Promotion Institute**.

The project focuses on optimizing crop cultivation and predicting production using advanced statistical analysis and machine learning techniques on forestry and environmental data.

---

## 📌 1. Problem Definition (문제 정의)

- **Background**: Prospective foresters and farmers face high risks and "Trial & Error" costs when deciding which crops to plant and where, due to a lack of data-driven guidance.
- **Objective**: To identify optimal cultivation regions for various forestry crops (Chestnuts, Blackberries, etc.) and predict production based on soil and climate data.
- **Vision**: "Empowering non-technical stakeholders with data-driven forestry insights."

## 📊 2. Data Acquisition & Preprocessing (데이터 수집 및 전처리)

- **Multi-Source Data Fusion**:
  - **Soil Data**: Chemistry (Acidity, Moisture), Texture, Depth, and Drainage from Forestry Service GIS.
  - **Climate Data**: 30-year climate normals (Temperature, Precipitation, Humidity) from the Korea Meteorological Administration (KMA).
  - **Production Data**: Historical crop yield data.
- **Refactored Module**: `src/data_loader.py`
  - Automatically maps Korean column names to professional English variables (e.g., `soil_depth_type`, `chestnut_kg`).
  - Handles missing values and feature engineering (e.g., `temp_diff`).

## 🔬 3. Statistical Analysis & Insights (통계 분석 및 인사이트)

- **Methodology**: Conducted Chi-Square tests, ANOVA, and Spearman correlation to validate relationships between soil conditions and crop production.
- **Key Insights**:
  - Validated critical threshold variables for production.
  - Discovered that **humidity** is the key driver for Fresh Shiitake mushrooms yield (p-value < 0.05).
- **Refactored Module**: `src/stat_analyzer.py` (Fully in English).

## 🤖 4. Modeling & Evaluation (모델링 및 평가)

- **Approach**: Implemented a dual modeling framework for both prediction (Regression) and suitability classification.

### Key Statistical Findings & Feature Importance

Instead of relying on generic accuracy scores, the project focused on identifying statistically significant drivers of production and mapping their impact via regression coefficients.

#### 1. Analysis of Variance (ANOVA) Results
We identified variables with a statistically significant relationship with crop yield (p-value < 0.05):

| Crop | Significant Variable | p-value | Insight |
| :--- | :--- | :--- | :--- |
| **Blackberry (복분자딸기)** | Soil Available Water Content | **0.000004** | Soil moisture is the critical driver for yield. |
| **Deodeok (더덕)** | Soil Available Water Content | **3.07e-08** | Strongest statistical relationship found. |
| **Shiitake (생표고)** | Soil Available Water Content | **0.0105** | Water retention capacity dictates growth. |

#### 2. Model Feature Importance (Coefficients)
Feature importances (coefficients) derived from the predictive models revealed the directional impact of key variables:

*   **Chestnut (밤)**:
    *   **Positive Drivers**: Max Temperature (+1.08), Precipitation (+1.06).
    *   **Negative Drivers**: Soil Type 3 (-0.82), Annual Mean Temperature (-0.75).
*   **Blackberry (복분자딸기)**:
    *   **Positive Drivers**: Relative Humidity (+0.78), Max Temperature (+0.35).
*   **Deodeok (더덕)**:
    *   **Top Drivers**: Annual Mean Temperature (0.21), Temperature Difference (0.17).
*   **Shiitake (생표고)**:
    *   **Top Drivers**: Precipitation (0.26), Max Temperature (0.23).
*   **Yam (마)**:
    *   **Top Drivers**: Min Temperature (0.35), Max Temperature (0.34).
*   **Bellflower (도라지)**:
    *   **Top Drivers**: Temperature Difference (0.34), Precipitation (0.34).

#### 3. Model Performance (AUC Scores)
The models were evaluated using the Area Under the ROC Curve (AUC). The real results captured from the analysis are as follows:

| Crop | Model | Metric | Score (AUC) | Status |
| :--- | :--- | :--- | :---: | :--- |
| **Blackberry (복분자딸기)** | Random Forest Classifier | AUC | **0.84** | High Predictive Power |
| **Ogapi (오갈피)** | Random Forest Classifier | AUC | **0.57** | Baseline Performance (Needs Improvement) |

*   **Insight**: The model performs exceptionally well for **Blackberry** (AUC = 0.84), indicating that the selected soil and climate features are strong predictors.
*   **Challenge & Future Work**: For **Ogapi** (AUC = 0.57), the performance suggests that growth might be influenced by other latent factors not in the dataset (e.g., specific micro-climates), presenting a clear direction for future feature engineering.

- **Regression**: Used **Random Forest Regressor** to predict production amounts and extract feature importances.
  - **Refactored Module**: `src/predictor.py`
- **Classification**: Implemented **SVM** to classify regions as highly suitable (above mean yield) or not.
  - **Refactored Module**: `src/classifier.py`
- **Key Findings**: Discovered that different crops require distinct model architectures for optimal performance (e.g., non-linear relationships in soil data were better captured by Random Forest).

## 🖼️ 5. Visualization & Prototype (시각화 및 프로토타입)

- **Service Dashboard**:
  - Visualized geographic information using Geopandas heatmaps.
  - Provided a recommendation list for optimal cultivation sites.
  - Displayed market trends and price analysis.

![Service Dashboard](images/dashboard_collage.png)
*Figure 1: Service flow and dashboard collage including Geopandas heatmap, recommendation list, and market graph.*

## 🏁 6. Conclusion & Business Impact (결론 및 비즈니스 임팩트)

- **Outcome**: Developed a framework for a functional web service providing a "Cultivation Suitability Map" for prospective foresters.
- **Analytical ROI**:
  - **Economic Impact**: Reduced the risk for new foresters by providing scientifically validated cultivation maps.
  - **Policy Support**: Provided a data-driven justification for regional specialized crop promotion.

---

## 📁 Repository Structure

```text
├── notebooks/                  # Original exploratory Jupyter notebooks
├── src/                        # Refactored production-ready source code
│   ├── data_loader.py          # Data loading and English translation
│   ├── stat_analyzer.py        # Chi-Square, ANOVA, and Spearman analysis
│   ├── predictor.py            # Random Forest regression
│   └── classifier.py          # SVM classification for suitability
├── reports/                    # Competition reports and presentations
├── images/                     # Project screenshots and diagrams
│   └── dashboard_collage.png  # User-provided service dashboard
├── run_pipeline.py             # Master pipeline runner
└── requirements.txt            # Project dependencies
```

## ⚙️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full pipeline:
   ```bash
   python run_pipeline.py
   ```

## 👥 Contributors

- **Junhyung L.** (Project Lead / Data Scientist)

---
*Refactored and polished to meet professional software engineering standards for the [Data Analyst Portfolio](https://github.com/junhyung-L).*
*Note: Statistical findings and feature importances are based on the actual competition report results.*
