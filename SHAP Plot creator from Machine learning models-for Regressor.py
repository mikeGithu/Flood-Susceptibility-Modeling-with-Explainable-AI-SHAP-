#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgbm
import shap
import matplotlib.pyplot as plt
import graphviz
from sklearn.inspection import permutation_importance


# In[2]:


os.chdir(r"D:\PAU document\Thesis\Analysis\Raster Datas\Input_Machine")


# In[3]:


df = pd.read_csv("Input_raster_variables_observed.csv")
print(df.head())
print(df.shape)


# In[6]:


y = df["Flood"]
X = df.drop(columns=["row", "col", "Flood"])


# ## XGBoots

# In[7]:


models = {}

# XGBoost
models["XGBoost"] = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# In[8]:


# Random Forest
models["RandomForest"] = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


# In[9]:


# LightGBM
models["LightGBM"] = lgbm.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)


# In[10]:


# Decision Tree
models["DecisionTree"] = DecisionTreeRegressor(
    max_depth=None,
    random_state=42
)


# In[11]:


# Train all models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X, y)


# In[12]:


shap_values_models = {}

# Sample for SHAP
X_sample = shap.sample(X, 500, random_state=42)

for name, model in models.items():
    print(f"Computing FAST SHAP for {name}...")

    if name in ["XGBoost", "RandomForest", "LightGBM", "DecisionTree"]:
        
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent"
        )

        shap_values = explainer.shap_values(
            X_sample,
            check_additivity=False
        )

        # LightGBM sometimes returns list — take only the first
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values_models[name] = shap_values


# ## Produce SHAP Summary Plots

# In[13]:


# Set global font
plt.rcParams["font.family"] = "Times New Roman"


# In[14]:


for name in ["XGBoost", "RandomForest", "LightGBM", "DecisionTree"]:
    print(f"\n### SHAP summary plot: {name} ###")

    shap_values = shap_values_models[name]

    # Downsample X to speed up SHAP plotting
    X_plot = shap.sample(X, 500, random_state=42)

    # Downsample SHAP values to the SAME number of rows
    shap_plot = shap_values[: X_plot.shape[0], :]

    # Create figure with title before plotting
    plt.figure()
    plt.title(f"{name}", fontfamily="Times New Roman", fontsize=16)

    shap.summary_plot(
        shap_plot,
        X_plot,
        max_display=20,
        show=True
    )


# ## Produce SHAP Bar Plots

# In[15]:


for name in models.keys():
    print(f"\n### SHAP BAR plot: {name} ###")

    shap_values = shap_values_models[name]

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False,
        title=f"SHAP Feature Importance — {name}"
    )

    plt.title(f"Feature Importance — {name}", fontsize=16, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.show()


# ## Create a Comparison Table of Feature Sensitivity

# In[39]:


# Make sure feature names exist 
feature_names = X.columns.tolist()


# In[40]:


comparison = {}

for name in models.keys():
    shap_vals = shap_values_models[name]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    comparison[name] = mean_abs

df_compare = pd.DataFrame(comparison, index=feature_names)
df_compare["Average"] = df_compare.mean(axis=1)
df_compare = df_compare.sort_values("Average", ascending=False)

df_compare


# ## Plot a Combined Sensitivity Comparison

# In[41]:


df_compare.drop(columns=["Average"]).plot(kind="bar", figsize=(12, 6))
plt.title("Feature Sensitivity Across Models (Mean Absolute SHAP)")
plt.ylabel("Mean |SHAP value|")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ## Export SHAP Summary Plots (PNG/JPG, 300 dpi)

# In[42]:


for name in ["XGBoost", "RandomForest", "LightGBM", "DecisionTree"]:
    print(f"Saving SHAP summary plot for {name}...")

    shap_values = shap_values_models[name]

    # Downsample together → must match shapes
    X_plot = shap.sample(X, 500, random_state=42)
    shap_plot = shap_values[:len(X_plot)]

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_plot,
        X_plot,
        max_display=20,
        show=False
    )

    plt.title(f"SHAP Summary Plot — {name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"SHAP_summary_{name}.jpg", dpi=300)
    plt.close()


# ## Export SHAP Bar Plots (PNG/JPG, 300 dpi)

# In[43]:


for name in ["XGBoost", "RandomForest", "LightGBM", "DecisionTree"]:
    print(f"Saving SHAP BAR plot for {name}...")

    shap_values = shap_values_models[name]

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

    plt.title(f"SHAP Bar Plot — {name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"SHAP_bar_{name}.jpg", dpi=300)
    plt.close()


# In[ ]:




