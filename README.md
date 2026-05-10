# 2024 Forestry Statistics Smart Competition: Grand Prize (1st Place) 🏆

This repository contains the award-winning project for the **'2024 Forestry Statistics Smart Competition'** hosted by the **Korea Forest Service** and **Korea Forestry Promotion Institute**.

The project focuses on optimizing crop cultivation and predicting production using advanced statistical analysis and machine learning techniques on forestry and environmental data.

## 🚀 Project Overview

- **Objective**: To identify optimal conditions for various crops and predict their production based on soil and climate data.
- **Impact**: Provides actionable insights for farmers and policymakers to maximize crop yield and sustainability.
- **Key Achievements**: Awarded the **Grand Prize (1st Place)** in the competition.

## 📁 Repository Structure

The repository has been refactored from Jupyter notebooks into a professional, modular Python package.

```text
├── notebooks/                  # Original exploratory Jupyter notebooks
│   ├── data_preprocessing.ipynb
│   ├── final_modeling.ipynb
│   └── ...
├── src/                        # Refactored production-ready source code
│   ├── statistical_analysis.py # Chi-Square, ANOVA, and Spearman analysis
│   └── modeling.py             # Random Forest regression and feature importance
└── reports/                    # Competition reports and presentations
```

## 🛠️ Key Features

### 1. Statistical Analysis (`src/statistical_analysis.py`)
- Refactored from `notebooks/data_preprocessing.ipynb`.
- Performs rigorous statistical tests to validate relationships between soil conditions and crop production.
- Includes **Chi-Square test**, **ANOVA**, and **Spearman correlation**.

### 2. Crop Production Modeling (`src/modeling.py`)
- Refactored from `notebooks/final_modeling.ipynb`.
- Trains **Random Forest Regressor** models to predict production amounts for various crops (Chestnut, Blackberry, etc.).
- Extracts and visualizes **Feature Importances** to identify key growth drivers (climate vs. soil).

## 📊 Results

- Successfully identified critical soil depth and moisture levels for target crops.
- Modeled climate impact (temperature difference, precipitation) on yield with high accuracy.

## 👥 Contributors

- **Junhyung L.** (Project Lead / Data Scientist)

---
*Refactored and polished to meet professional software engineering standards.*
