#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shap
import joblib
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# 0. OUTPUT FOLDERS
os.chdir(r"D:\PAU document\Thesis\Analysis\Raster Datas\Input_Machine")
output_dir = os.path.join(os.getcwd(), "Machine_Output")
shap_dir = os.path.join(output_dir, "shap")
os.makedirs(shap_dir, exist_ok=True)


# In[3]:


# 1. LOAD CSV
df = pd.read_csv("Input_raster_variables_observed.csv")
print(df.head())
print(df.shape)


# ## SHAP Summary

# In[16]:


# Collect models into a dictionary
models = {
    "RandomForest": rf_model,
    "XGBoost": xgb_model,
    "DecisionTree": dt_model,
    "LightGBM": lgbm_model
}


# In[23]:


# Sample for SHAP (speed up plotting)
X_sample = shap.sample(X_train,1500, random_state=42)

shap_values_models = {}

for name, model in models.items():
    print(f"Computing SHAP values for {name}...")

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )

    shap_values = explainer.shap_values(
        X_sample,
        check_additivity=False
    )

    # Binary classification → select Flood class (1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values_models[name] = shap_values


# In[18]:


# 4. GLOBAL FONT

plt.rcParams["font.family"] = "Times New Roman"


# ## Shap Waterfall

# In[28]:


# ---------------------------------------------
# SELECT ONE FLOOD SAMPLE
# ---------------------------------------------

sample_idx = np.where(y_train == 1)[0][0]

X_instance = X_train[sample_idx]
X_instance_df = pd.DataFrame([X_instance], columns=feature_names)

# ---------------------------------------------
# WATERFALL SHAP (ROBUST VERSION)
# ---------------------------------------------

for name, model in models.items():
    print(f"\n### SHAP Waterfall Plot: {name} ###")

    explainer = shap.TreeExplainer(model)

    shap_exp = explainer(X_instance_df)

    # 🔑 HANDLE BOTH SHAP OUTPUT TYPES
    if len(shap_exp.shape) == 3:
        # XGBoost / LightGBM → select Flood class (1)
        shap_single = shap_exp[0, :, 1]
    else:
        # RandomForest / DecisionTree → already Flood class
        shap_single = shap_exp[0]

    plt.figure(figsize=(9, 6))

    shap.plots.waterfall(
        shap_single,
        max_display=15,
        show=False
    )

    plt.title(f"{name} — SHAP Waterfall (Flood Class)", fontsize=16)

    output_path = os.path.join(output_dir, f"{name}_SHAP_waterfall.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    print(f"Saved to: {output_path}")


# In[ ]:




