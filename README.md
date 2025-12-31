Flood Susceptibility Modeling with Explainable AI (SHAP)

This project implements a machine-learning–based flood susceptibility analysis using raster-derived environmental variables and explainable AI techniques.

🔍 What the code does

Loads raster-based predictor variables exported to CSV

Trains multiple regression models to predict flood occurrence:

XGBoost

Random Forest

LightGBM

Decision Tree

Applies SHAP (SHapley Additive exPlanations) to interpret model predictions

Quantifies feature importance and sensitivity

Compares feature influence across models

Generates and exports publication-quality SHAP plots

🤖 Machine Learning Models

All models are trained on the same dataset to ensure fair comparison:

Tree-based ensemble methods capture nonlinear relationships

A Decision Tree is used as a transparent baseline

🧠 Model Explainability (SHAP)

SHAP values explain how each environmental variable contributes to flood prediction

Summary (beeswarm) plots show feature impact distribution

Bar plots show global feature importance

Mean absolute SHAP values are used to compare feature sensitivity across models

📊 Outputs

SHAP summary plots (PNG/JPG, 300 DPI)

SHAP feature importance bar plots

Cross-model feature sensitivity comparison table

Combined feature importance visualization

🎯 Purpose

The workflow enables transparent, interpretable flood modeling, helping identify the most influential environmental drivers of flooding while ensuring model reliability and scientific reproducibility.

🛠 Tools & Libraries

Python, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, SHAP, Matplotlib, Rasterio
