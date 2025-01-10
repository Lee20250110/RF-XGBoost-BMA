import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt



def compute_posterior_weights(y_test, y_pred_rf, y_pred_xgb):

    epsilon = 1e-10
    residuals_rf = (y_test - y_pred_rf) / np.std(y_test)
    residuals_xgb = (y_test - y_pred_xgb) / np.std(y_test)
    sigma_rf = np.var(residuals_rf) + epsilon
    sigma_xgb = np.var(residuals_xgb) + epsilon


    log_likelihood_rf = -0.5 * np.sum(residuals_rf**2) / sigma_rf
    log_likelihood_xgb = -0.5 * np.sum(residuals_xgb**2) / sigma_xgb


    max_log_likelihood = max(log_likelihood_rf, log_likelihood_xgb)
    likelihood_rf = np.exp(log_likelihood_rf - max_log_likelihood)
    likelihood_xgb = np.exp(log_likelihood_xgb - max_log_likelihood)


    prior_rf = 1
    prior_xgb = 1


    likelihood_sum = likelihood_rf * prior_rf + likelihood_xgb * prior_xgb
    posterior_rf = (likelihood_rf * prior_rf) / likelihood_sum
    posterior_xgb = (likelihood_xgb * prior_xgb) / likelihood_sum

    return posterior_rf, posterior_xgb


def evaluate_regress(y_pre, y_true):
    MAE = np.mean(np.abs(y_pre - y_true))
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)  # 避免目标值为零导致的除以零
    MAPE = np.mean(np.abs((y_pre - y_true) / y_true_safe))
    MSE = np.mean((y_pre - y_true) ** 2)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_true, y_pre)
    return {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE, 'RMSE': RMSE, 'R2': R2}

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'R2': r2, 'RMSE': rmse}


file_path = r"sample.xlsx"
df = pd.read_excel(file_path)
print(df.head())


scaler = StandardScaler()
columns_to_scale = df.columns[1:]
df_scaled = df.copy()
df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
#
print("标准化后的数据：")
print(df_scaled.head())




X = df.drop(columns=['y'])
y = df['y']


# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

########## 随机森林模型 ##########
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
# 评估随机森林性能
rf_metrics = evaluate_model(y_test, y_pred_rf)
print("随机森林模型评估：", rf_metrics)
print(evaluate_regress(y_pred_rf,y_test))
print('#################################################')

########## XGBoost 模型 ##########
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
# 评估 XGBoost 性能
xgb_metrics = evaluate_model(y_test, y_pred_xgb)
print("XGBoost 模型评估：", xgb_metrics)
print(evaluate_regress(y_pred_xgb,y_test))
print('#################################################')

########## 贝叶斯模型平均 ##########
# 计算后验权重
posterior_rf, posterior_xgb = compute_posterior_weights(y_test, y_pred_rf, y_pred_xgb)
print(f"后验权重: 随机森林 = {posterior_rf:.3f}, XGBoost = {posterior_xgb:.3f}")
# 融合预测
y_pred_bma = posterior_rf * y_pred_rf + posterior_xgb * y_pred_xgb

# 评估融合模型
bma_metrics = evaluate_model(y_test, y_pred_bma)
print("贝叶斯模型平均评估：", bma_metrics)
print(evaluate_regress(y_pred_bma,y_test))
print('#################################################')


explainer_rf = shap.Explainer(rf_model)

shap_values_rf = explainer_rf(X_test)
mean_shap_values_rf = np.abs(shap_values_rf.values).mean(axis=0)
importance_df_rf = pd.DataFrame({
    'Feature': X.columns,
    'Mean Absolute SHAP Value': mean_shap_values_rf
}).sort_values(by='Mean Absolute SHAP Value', ascending=False)
print("随机森林特征重要性 (平均绝对 SHAP 值)：")
print(importance_df_rf)


explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_test)
mean_shap_values_xgb = np.abs(shap_values_xgb.values).mean(axis=0)
importance_df_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Mean Absolute SHAP Value': mean_shap_values_xgb
}).sort_values(by='Mean Absolute SHAP Value', ascending=False)
print("XGBoost特征重要性 (平均绝对 SHAP 值)：")
print(importance_df_xgb)

shap_values_bma = posterior_rf * shap_values_rf.values + posterior_xgb * shap_values_xgb.values
mean_shap_values_bma = np.abs(shap_values_bma).mean(axis=0)
importance_df_bma = pd.DataFrame({
    'Feature': X.columns,
    'Mean Absolute SHAP Value': mean_shap_values_bma
}).sort_values(by='Mean Absolute SHAP Value', ascending=False)
print("bma结合后特征重要性 (平均绝对 SHAP 值)：")
print(importance_df_bma)

######### 三模型绘图 ##########
def summary_plot(shap_values,name_path,color):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    # 配色   viridis  Spectral   coolwarm  RdYlGn  RdYlBu  RdBu  RdGy  PuOr  BrBG PRGn  PiYG
    shap.summary_plot(shap_values, X_test, feature_names=X.columns,cmap=color, show=False)
    plt.savefig(name_path, dpi=300, bbox_inches='tight')
    plt.close()

# summary_plot(shap_values_rf.values,'shap_summary_plot_rf.png','RdYlBu_r')
# summary_plot(shap_values_xgb.values,'shap_summary_plot_xgb.png','coolwarm')
summary_plot(shap_values_bma,'shap_summary_plot_bma_1111.png','Spectral_r')

########## 贝叶斯模型结合 Heatmap ##########
shap_values_bma_expl = shap.Explanation(
    values=shap_values_bma,
    base_values=posterior_rf * explainer_rf.expected_value + posterior_xgb * explainer_xgb.expected_value,
    data=X_test.values,
    feature_names=X.columns
)


plt.figure()
shap.plots.heatmap(shap_values_bma_expl, cmap='Spectral_r', show=False)
plt.savefig(r'shap_heatmap_bma_11111.png', dpi=300, bbox_inches='tight')
plt.close()

mean_shap_rf = pd.DataFrame({
    "Feature": X.columns,
    "Mean SHAP": shap_values_rf.values.mean(axis=0),
    "Positive Impact": (shap_values_rf.values > 0).mean(axis=0),
    "Negative Impact": (shap_values_rf.values < 0).mean(axis=0)
}).sort_values(by="Mean SHAP", ascending=False)

print("随机森林特征的全局贡献：")
print(mean_shap_rf)

mean_shap_xgb = pd.DataFrame({
    "Feature": X.columns,
    "Mean SHAP": shap_values_xgb.values.mean(axis=0),
    "Positive Impact": (shap_values_xgb.values > 0).mean(axis=0),
    "Negative Impact": (shap_values_xgb.values < 0).mean(axis=0)
}).sort_values(by="Mean SHAP", ascending=False)

print("XGBoost 特征的全局贡献：")
print(mean_shap_xgb)