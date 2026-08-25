import numpy as np 
import os

def calculate_metrics(y_true, y_pred):
    """计算 R²、皮尔逊相关系数、MSE、MAE 和 RMSE"""
    # 计算 R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 计算皮尔逊相关系数
    correlation_coefficient = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    # 计算 MSE、MAE 和 RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    return r_squared, correlation_coefficient, mse, mae, rmse

# 存储每个文件的指标
r_squared_list = []
pearson_list = []
mse_list = []
mae_list = []
rmse_list = []

# 存储所有数据（用于整体计算）
all_y_true = []
all_y_pred = []

# 遍历所有子文件夹
root = '/ayt/anyitong/TIEPN/experiment/ayt_test_result/3200_test/'

for folder in os.listdir(root):
    label_path = os.path.join(root, folder, 'label_cn2.npy')
    pred_path = os.path.join(root, folder, 'cn2_out.npy')
    
    if os.path.exists(label_path) and os.path.exists(pred_path):
        y_true = np.load(label_path)*1e-9
        y_pred = np.load(pred_path)*1e-9

        # 计算单个文件的指标
        r2, pearson, mse, mae, rmse = calculate_metrics(y_true, y_pred)
        
        # 存入列表
        r_squared_list.append(r2)
        pearson_list.append(pearson)
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)

        # 存入整体数据
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

# 计算所有文件合并后的整体指标
if all_y_true and all_y_pred:
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    overall_r2, overall_pearson, overall_mse, overall_mae, overall_rmse = calculate_metrics(all_y_true, all_y_pred)

    print("整体 R²:", overall_r2)
    print("整体皮尔逊相关系数:", overall_pearson)
    print("整体 MSE:", overall_mse)
    print("整体 MAE:", overall_mae)
    print("整体 RMSE:", overall_rmse)
else:
    print("未找到有效的数据文件。")

# # 计算单个文件的指标均值
# if r_squared_list and pearson_list and mse_list and mae_list and rmse_list:
#     mean_r2 = np.mean(r_squared_list)
#     mean_pearson = np.mean(pearson_list)
#     mean_mse = np.mean(mse_list)
#     mean_mae = np.mean(mae_list)
#     mean_rmse = np.mean(rmse_list)

#     print("单个文件的平均 R²:", mean_r2)
#     print("单个文件的平均皮尔逊相关系数:", mean_pearson)
#     print("单个文件的平均 MSE:", mean_mse)
#     print("单个文件的平均 MAE:", mean_mae)
#     print("单个文件的平均 RMSE:", mean_rmse)
# else:
#     print("未计算出单个文件的指标。")
