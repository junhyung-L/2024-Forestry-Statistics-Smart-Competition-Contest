# Architecture

The reproducible path starts at `run_pipeline.py`, not the historical
notebooks. Supply the CSV path, target crop column, and feature columns; the
CLI validates what is available after the stored header mapping.

```mermaid
flowchart LR
    A[Input CSV] --> B[Data loading and cleaning]
    B --> C[Exploratory statistics]
    B --> D[Positive-yield random forest]
    B --> E[Above-mean linear SVM]
    C --> F[CSV tables]
    D --> G[JSON summary]
    E --> G
```

`CropPredictor` and `CropClassifier` keep imputation—and classification
scaling—inside sklearn pipelines. This prevents those transforms from being
fit on holdout rows or cross-validation validation folds.

The maintained CLI and `src/` modules are the reference for component details
and historical-code boundaries.
