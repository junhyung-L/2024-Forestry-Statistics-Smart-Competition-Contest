# Project Review

## Evidence basis

This review uses the maintained source files, checked-in notebooks and images,
and the data-source note. The raw analysis CSV and a rerunnable historical
experiment are not present.

## Assessment

| Area | Assessment | Evidence |
|---|---|---|
| Problem framing | Clear exploratory crop-production and suitability use case | `README.md`, `run_pipeline.py` |
| Data complexity | Multi-source soil, climate, and production context is documented, but the actual join keys and grain are unknown | `data/임업통계 데이터 셋 출처(이준형).txt`; raw data absent |
| EDA and statistics | Multiple association methods and retained figures provide useful exploration | `src/stat_analyzer.py`, `images/correlation_heatmap.png` |
| Modeling | Separate regression and classification paths make the decision framing explicit | `src/predictor.py`, `src/classifier.py` |
| Experimental rigor | Improved: seeded splits and fold-local preprocessing are now explicit; external, temporal, and spatial validation remain absent | `src/config.py`, model modules |
| Evaluation | Correct metrics are available for each maintained task (R²/MSE and AUC), but no current result artifact can be generated without data | `run_pipeline.py`; data absent |
| Interpretability | Correlations and random-forest impurity importances are available, but neither provides causal attribution | `images/correlation_heatmap.png`, `src/predictor.py` |
| Reproducibility | Improved CLI and saved tables/summary; blocked by missing raw data and data dictionary | `run_pipeline.py`, `data/` |

## Strengths

- The code makes the two problem formulations visible instead of conflating
  production prediction with suitability classification.
- The checked-in ROC and heatmap artifacts give the repository concrete visual
  evidence, with their historical provenance clearly bounded.
- The maintained classifier now prevents a prior leakage risk by fitting the
  imputer and scaler inside the split/CV pipeline.
- A CLI now records its statistic tables and metrics rather than only printing
  them to the console.

## Limitations

- The repository has no raw data, data dictionary, model artifact, or stored
  current run, so all historical AUC labels are non-reproducible.
- Suitability is defined by an above-mean label; that operational rule needs
  agronomic and stakeholder validation.
- Associations, factorized-category Spearman values, and repeated p-values
  should not be presented as causal effects without a pre-specified analysis
  plan and multiple-testing control.
- A random split may overstate deployment performance if observations are
  spatially or temporally correlated.

## Priority improvements

1. Version the processed input schema plus a small de-identified fixture and
   document the geographic/time unit.
2. Evaluate spatial and temporal holdouts, then report baselines and
   uncertainty intervals next to AUC/R².
3. Pre-register or control the statistical-testing family, and define a
   domain-valid suitability threshold.
4. Save trained models, feature schema, and configuration with each run.
