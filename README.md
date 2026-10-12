# F1 Race Winner Prediction using XGBoost

## Overview
This project aims to predict the winner of a future Formula 1 Grand Prix using historical race data collected through the [FastF1](https://docs.fastf1.dev/) Python library. The goal is to develop a machine learning model using XGBoost.

## Current Progress
- Set up the Python environment and imported the required libraries.
- Enabled FastF1's local caching to reduce repeated data downloads.
- Created a data collection function for historical qualifying and race results.
- Started collecting data for the 2022–2025 F1 seasons.
- Prepared code to combine qualifying positions, grid positions, finishing positions, driver names, and team information.
- Added a binary target variable, `Winner`, where `1` represents a race winner and `0` represents other drivers.
- Configured local storage for generated datasets and cache files.

## Tech Stack
- Python
- FastF1
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Planned Workflow
1. Collect and validate historical F1 data.
2. Handle missing values and clean the dataset.
3. Engineer features using historical driver and team performance.
4. Train an XGBoost classifier.
5. Evaluate performance using chronological train/test splits.
6. Predict and rank the probable winners of upcoming Grands Prix.

## Project Structure
```text
F1-Race-Winner-Prediction/
├── data/
│   └── fastf1_cache/
├── notebooks/
├── .gitignore
└── README.md
```

*Note: The project structure may evolve as development progresses.*

## Status
**In progress** — Data collection and initial setup are underway. Model training, evaluation, and predictions are planned next.

## Disclaimer
Predictions are probabilistic estimates based on historical data and selected features. Actual race outcomes may differ due to unpredictable events such as weather, incidents, and strategy changes.
