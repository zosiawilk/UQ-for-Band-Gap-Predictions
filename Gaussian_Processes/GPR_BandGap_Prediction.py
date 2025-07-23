#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# ### Read and preprocess the data

# In[2]:


df_mp = pd.read_csv('/home/april-ai/Desktop/UQ/MP_Data_Corrected/mp_data_100k_cleaned2.csv')


# In[3]:


df_mp = df_mp[(df_mp['band_gap'] > 0.2)]
y = df_mp['band_gap']


# In[4]:


df_mp.head()


# ### Train-test split

# In[5]:


# Step 1: Define and clean inputs
y = df_mp['band_gap'].values
excluded = ["material_id", "composition", "formula_pretty", "symmetry", "structure", "sites", 'HOMO_character', 'HOMO_element', 'LUMO_character', 'LUMO_element']
X = df_mp.drop(columns=excluded + ["band_gap"], errors='ignore')
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

#
original_indices = df_mp.index.to_numpy()
formulas = df_mp.loc[original_indices, "formula_pretty"].values

# Step 4: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Feature selection AFTER scaling
#X_selected = SelectKBest(score_func=f_regression, k=30).fit_transform(X_scaled, y)

# Step 6: Train-test split on selected features and aligned y
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, original_indices, test_size=0.2, random_state=42
)


# In[6]:


# Normalise training data
#scaler_x = StandardScaler()
#scaler_y = StandardScaler()


#X_train = scaler_x.fit_transform(X_train_raw) #computes mean and std from training data and scales it
#X_test = scaler_x.transform(X_test_raw) #uses the same mean and std to scale test data


#y_train = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
#y_test = scaler_y.transform(y_test_raw.values.reshape(-1, 1)).ravel() #ravel() flattens the output back to 1D, which is typically needed for PyTorch models expecting targets as flat vectors.

#scaler_x = StandardScaler()
#X_train = scaler_x.fit_transform(X_train)
#X_test = scaler_x.transform(X_test)


# In[ ]:





# ## Run Gaussian Process Regression 

# In[ ]:


kernel = (
    ConstantKernel(1.0, (1e-2, 1e2)) *
    (
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
        Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)) +
        RationalQuadratic(length_scale=1.0, alpha=1.0,
                          length_scale_bounds=(1e-2, 1e2),
                          alpha_bounds=(1e-2, 1e3))
    ) +
    WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1))
)
#kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
gpr.fit(X_train, y_train)

# Predict with uncertainty
y_pred, y_std = gpr.predict(X_test, return_std=True)

# Evaluate
print("R² score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("Optimized kernel:", gpr.kernel_)


# In[ ]:


# Just the GPR model, no feature selection
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)

# Grid of GPR hyperparameters
param_grid = {
    'alpha': [1e-6, 1e-5]  # Adjust for stability
}

# Grid search (no pipeline needed)
search = GridSearchCV(gpr, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
search.fit(X_scaled, y)

# Output results
print("Best R² score:", search.best_score_)
print("Best params:", search.best_params_)


# In[ ]:


scores = cross_val_score(gpr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("CV RMSE:", np.sqrt(-scores).mean())


# In[ ]:


plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', label='Ideal')
plt.xlabel('True Band Gap (eV)')
plt.ylabel('Predicted Band Gap (eV)')
plt.title('GPR Band Gap Prediction')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Get cross-validated predictions ( 5-fold)
y_pred_cv = cross_val_predict(gpr, X_train, y_train, cv=5)

# y_pred_cv = cross_val_predict(gpr, X_train, y_train, cv=5)
r2 = r2_score(y_train, y_pred_cv)
print("Cross-validated R² score:", r2)


# In[ ]:


# Plot true vs predicted (cross-validated)
plt.figure(figsize=(6,6))
plt.scatter(y_train, y_pred_cv, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Ideal')
plt.xlabel('True Band Gap (eV)')
plt.ylabel('Predicted Band Gap (eV)')
plt.title('Cross-Validated GPR Predictions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


y_true = np.array(y_test)
y_predicted = np.array(y_pred)
y_std_dev = np.array(y_std)

# Sort by true band gap values
sorted_idx = np.argsort(y_true)
y_true_sorted = y_true[sorted_idx]
y_pred_sorted = y_predicted[sorted_idx]
y_std_sorted = y_std_dev[sorted_idx]

# 95% confidence interval
lower = y_pred_sorted -  y_std_sorted
upper = y_pred_sorted + y_std_sorted

# Optional: get formulas for annotation
formulas = df_mp.loc[idx_test, "formula_pretty"].values
formulas_sorted = formulas[sorted_idx]

# Calculate prediction error
errors = np.abs(y_true_sorted - y_pred_sorted)
outlier_mask = errors > 1
outlier_indices = np.where(outlier_mask)[0]

# Plot
plt.figure(figsize=(14, 6))

# Confidence interval band
plt.fill_between(range(len(y_true_sorted)), lower, upper, alpha=0.3, label='Prediction ±σ', color='blue')

# Predicted mean
plt.plot(y_pred_sorted, 'o', markersize=3, label='Predicted Band Gaps', color='blue')

# True values
plt.plot(y_true_sorted, 'k.', markersize=3, label='True Band Gaps')

# Outliers
plt.plot(outlier_indices, y_pred_sorted[outlier_mask], 'ro', markerfacecolor='none', markersize=10, label='Outliers')

# Annotate outliers
for i in outlier_indices:
    plt.text(i, y_pred_sorted[i] + 0.4, f"{formulas_sorted[i]}\n{y_true_sorted[i]:.2f} eV", 
             fontsize=8, color='brown', ha='center')

# Final formatting
plt.xlabel("Sorted Test Sample Index")
plt.ylabel("Band Gap (eV)")
plt.title("GPR Band Gap Predictions ±σ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Convert y_test_raw and y_pred to numpy arrays for indexing
y_true = np.array(y_test)
y_predicted = np.array(y_pred)
y_std_dev = np.array(y_std)

# Sort by true band gap values
sorted_idx = np.argsort(y_true)
y_true_sorted = y_true[sorted_idx]
y_pred_sorted = y_predicted[sorted_idx]
y_std_sorted = y_std_dev[sorted_idx]

# 95% confidence interval
lower = y_pred_sorted - 1.96 * y_std_sorted
upper = y_pred_sorted + 1.96 * y_std_sorted

# Optional: get formulas for annotation
formulas = df_mp.loc[idx_test, "formula_pretty"].values
formulas_sorted = formulas[sorted_idx]

# Calculate prediction error
errors = np.abs(y_true_sorted - y_pred_sorted)
outlier_mask = errors > 1.5
outlier_indices = np.where(outlier_mask)[0]

# Plot
plt.figure(figsize=(14, 6))

# Confidence interval band
plt.fill_between(range(len(y_true_sorted)), lower, upper, alpha=0.3, label='95% Confidence Interval', color='blue')

# Predicted mean
plt.plot(y_pred_sorted, 'o', markersize=3, label='Predicted Band Gaps', color='blue')

# True values
plt.plot(y_true_sorted, 'k.', markersize=3, label='True Band Gaps')

# Outliers
plt.plot(outlier_indices, y_pred_sorted[outlier_mask], 'ro', markerfacecolor='none', markersize=10, label='Outliers')

# Annotate outliers
for i in outlier_indices:
    plt.text(i, y_pred_sorted[i] + 0.4, f"{formulas_sorted[i]}\n{y_true_sorted[i]:.2f} eV", 
             fontsize=8, color='brown', ha='center')

# Final formatting
plt.xlabel("Sorted Test Sample Index")
plt.ylabel("Band Gap (eV)")
plt.title("GPR Band Gap Predictions with 95% Confidence Intervals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Log-transformed data training

# In[ ]:


# Before fitting
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
gpr.fit(X_train, y_train_log)
y_pred_log, y_std_log = gpr.predict(X_test, return_std=True)
y_pred = np.expm1(y_pred_log)
y_std = np.expm1(y_pred_log + y_std_log) - y_pred  # Approximate std in original space

# Step 9: Evaluate
print("R² score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("Optimized kernel:", gpr.kernel_)


# In[ ]:


# Visualize log-transformed GPR predictions with 95% confidence interval

# Convert to numpy arrays
y_true_log = np.array(y_test_log)
y_pred_log = np.array(y_pred_log)
y_std_log = np.array(y_std_log)

# Sort by true log band gap values
sorted_idx = np.argsort(y_true_log)
y_true_log_sorted = y_true_log[sorted_idx]
y_pred_log_sorted = y_pred_log[sorted_idx]
y_std_log_sorted = y_std_log[sorted_idx]

# 95% confidence interval in log space
lower_log = y_pred_log_sorted - 1.96 * y_std_log_sorted
upper_log = y_pred_log_sorted + 1.96 * y_std_log_sorted

# Optional: get formulas for annotation
formulas = df_mp.loc[idx_test, "formula_pretty"].values
formulas_sorted = formulas[sorted_idx]

# Calculate prediction error (in log space)
errors_log = np.abs(y_true_log_sorted - y_pred_log_sorted)
outlier_mask = errors_log > 1.5  # adjust threshold as needed for log space
outlier_indices = np.where(outlier_mask)[0]

# Plot
plt.figure(figsize=(14, 6))

# Confidence interval band
plt.fill_between(range(len(y_true_log_sorted)), lower_log, upper_log, alpha=0.3, label='95% Confidence Interval', color='cornflowerblue')

# Predicted mean
plt.plot(y_pred_log_sorted, 'o', markersize=3, label='Predictive Mean', color='blue')

# True values
plt.plot(y_true_log_sorted, 'k.', markersize=4, label='True log(Band Gaps)')

# Outliers
plt.plot(outlier_indices, y_true_log_sorted[outlier_mask], 'ro', markerfacecolor='none', markersize=10, label='Outliers')

# Annotate outliers
for i in outlier_indices:
    plt.text(i, y_true_log_sorted[i] + 0.2, f"{formulas_sorted[i]}\n{y_true_log_sorted[i]:.2f}", 
             fontsize=8, color='brown', ha='center')

# Final formatting
plt.xlabel("Sorted Test Sample Index")
plt.ylabel("log(Band Gap + 1) (eV)")
plt.title("GPR Band Gap Predictions with 95% Confidence Intervals (log-transformed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# After predicting in log space, inverse-transform to original band gap scale and plot
# Inverse-transform log-predicted and true values
y_true = np.expm1(y_test_log)
y_pred = np.expm1(y_pred_log)
y_std = np.expm1(y_pred_log + y_std_log) - y_pred  # Approximate std in original space

# Sort by true band gap values for a clearer plot
sorted_idx = np.argsort(y_true)
y_true_sorted = y_true[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]
y_std_sorted = y_std[sorted_idx]

# 95% confidence interval
lower = y_pred_sorted - 1.96 * y_std_sorted
upper = y_pred_sorted + 1.96 * y_std_sorted

# Optional: get formulas for annotation
formulas = df_mp.loc[idx_test, "formula_pretty"].values
formulas_sorted = formulas[sorted_idx]

# Calculate prediction error
errors = np.abs(y_true_sorted - y_pred_sorted)
outlier_mask = errors > 1
outlier_indices = np.where(outlier_mask)[0]

# Plot
plt.figure(figsize=(14, 6))

# Confidence interval band
plt.fill_between(range(len(y_true_sorted)), lower, upper, alpha=0.3, label='95% Confidence Interval', color='blue')

# Predicted mean
plt.plot(y_pred_sorted, 'o', markersize=3, label='Predicted Band Gaps', color='blue')

# True values
plt.plot(y_true_sorted, 'k.', markersize=4, label='True Band Gaps')

# Outliers
plt.plot(outlier_indices, y_pred_sorted[outlier_mask], 'ro', markerfacecolor='none', markersize=10, label='Outliers')

# Annotate outliers
for i in outlier_indices:
    plt.text(i, y_pred_sorted[i] + 0.4, f"{formulas_sorted[i]}\n{y_true_sorted[i]:.2f} eV", 
             fontsize=8, color='brown', ha='center')

# Final formatting
plt.xlabel("Sorted Test Sample Index")
plt.ylabel("Band Gap (eV)")
plt.title("GPR Band Gap Predictions (Inverse-Transformed from Log) with 95% Confidence Intervals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.pipeline           import Pipeline
from sklearn.compose            import TransformedTargetRegressor
from sklearn.preprocessing      import StandardScaler, FunctionTransformer
from sklearn.feature_selection  import SelectKBest, f_regression
from sklearn.gaussian_process   import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, RBF, Matern,
                                              RationalQuadratic, WhiteKernel)
import numpy as np


# In[ ]:


log1p = FunctionTransformer(np.log1p, inverse_func=np.expm1)


# In[ ]:


n_feat = X_train.shape[1]               # 137 in your data

kernel = (ConstantKernel(1.0, (0.1, 10.0)) *
          (RBF(length_scale=1.0,      length_scale_bounds=(1e-2, 1e3)) +
           Matern(length_scale=1.0,    length_scale_bounds=(1e-2, 1e3), nu=2.5) +
           RationalQuadratic(length_scale=1.0, alpha=1.0,
                             length_scale_bounds=(1e-2, 1e3),
                             alpha_bounds=(1e-4, 1e4)))
          + WhiteKernel(noise_level=1e-4,
                        noise_level_bounds=(1e-9, 1e1)))



# In[ ]:


gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-5,                 # fixed jitter added to the diagonal
        optimizer="fmin_l_bfgs_b",  # default
        n_restarts_optimizer=8,
        normalize_y=False,
        random_state=42)

pipe = Pipeline([
    ("scale",  StandardScaler()),                 # MUST come first
    ("select", SelectKBest(f_regression, k=60)),  # k is tunable
    ("gpr",    gpr)
])

model = TransformedTargetRegressor(regressor=pipe,
                                   transformer=log1p)  # log ↔ expm1 on y


# In[ ]:


from sklearn.model_selection import GridSearchCV, KFold

param_grid = {
    "regressor__select__k":   [40, 60, 90],   # try a few
    "regressor__gpr__alpha":  [1e-6, 1e-5]    # bigger α = safer numerically
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(model,
                      param_grid   = param_grid,
                      scoring      = "r2",
                      cv           = cv,
                      n_jobs       = -1,
                      verbose      = 1)
search.fit(X_train, y_train)

print("CV-best R² :", search.best_score_)
print("Params     :", search.best_params_)
print("Kernel     :", search.best_estimator_.regressor_["gpr"].kernel_)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# ── 1 ▸ grab the best fitted model ────────────────────────────────────────────
best_model   = search.best_estimator_          # TransformedTargetRegressor
reg_pipeline = best_model.regressor_           # the X-pipeline + DynamicGPR

# ── 2 ▸ quick helper to invert log targets ────────────────────────────────────
def inv(x): return np.expm1(x)

# ── 3 ▸ scatter: predicted vs true ────────────────────────────────────────────
y_pred_lin = inv(reg_pipeline.predict(X_test))        # μ in eV
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, y_pred_lin, s=25, alpha=0.7)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], linestyle="--")
ax.set_xlabel("True band gap (eV)")
ax.set_ylabel("Predicted band gap (eV)")
ax.set_title("GPR — predicted vs. true")
ax.grid(True)
plt.tight_layout()
plt.show()

# ── 4 ▸ sorted curve with 95 % CI ─────────────────────────────────────────────
y_pred_log, y_std_log = reg_pipeline.predict(X_test, return_std=True)
y_true_log            = log1p.transform(y_test.reshape(-1, 1)).ravel()

y_true = inv(y_true_log)
y_pred = inv(y_pred_log)
y_std  = inv(y_pred_log + y_std_log) - y_pred    # ≈ σ in eV

idx          = np.argsort(y_true)
y_true_s     = y_true[idx]
y_pred_s     = y_pred[idx]
y_std_s      = y_std[idx]

lower = y_pred_s - 1.96 * y_std_s
upper = y_pred_s + 1.96 * y_std_s

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(np.arange(len(y_true_s)), lower, upper, alpha=0.30,
                label="95 % CI")
ax.plot(y_pred_s, "o", ms=3, label="Predicted")
ax.plot(y_true_s, ".", ms=4, label="True")
ax.set_xlabel("Sorted test-sample index")
ax.set_ylabel("Band gap (eV)")
ax.set_title("Predictions with 95 % confidence band")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

