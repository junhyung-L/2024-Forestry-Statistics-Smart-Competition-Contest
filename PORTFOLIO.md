# Forestry Crop Suitability Analysis: Matching Models to Crops and Conditions

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Overview

This project combines soil, climate, and production data to help prospective
foresters reason about crop suitability. It received the 2024 Grand Prize
(Korea Forest Service Commissioner Award, first place).

## Approach

Levene, ANOVA, and Chi-square tests narrow environmental candidates before
modelling. Classification paths compare SVM, random forest, and logistic
regression against mean-binarised production; production regression remains a
separate task. The final conditions are mapped back to candidate regions.

## Result

The final report records SVM as the strongest approach for yam, random forest
for blackberry, shiitake, deodeok, and acanthopanax, and logistic regression
for chestnut and balloon flower. Retained ROC figures show AUC 0.94 for
chestnut, 0.91 for yam, and 0.84 for blackberry.

## Limitations

Raw data and historical split provenance are incomplete. The retained figures
are project artifacts, not current reproducible benchmarks; spatial and time
validation is the next step.

## Evidence

- Final report in `reports/`
- ROC figures in `images/`
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)

## From analysis to a cultivation decision

The report does more than label a place as suitable. Soil, climate, and yield sources are first aligned by their spatial and analytical criteria; crop-specific relationships are then screened before modelling. Levene’s test checks variance assumptions, while ANOVA and chi-square analyses narrow the environmental signals that move with yield or suitability. This creates an interpretable candidate set instead of feeding every available column into one model.

Suitability and yield are treated as different questions. The classification path binarises whether yield is at least the crop average, while the regression path estimates yield itself. The documented choices reflect that distinction: SVM for yam; Random Forest for blackberry, fresh shiitake, deodeok, and acanthopanax; and Logistic Regression for chestnut and balloon flower. No single algorithm is presented as correct for every crop.

The intended interface is a GeoPandas heatmap with suitable, moderate, and unsuitable labels. It narrows locations for field review rather than replacing a grower’s decision. The retained AUC values are historical artefacts; incomplete split and spatial-validation provenance means they do not establish cultivation success rates.
