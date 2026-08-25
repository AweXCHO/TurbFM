import os
import numpy as np

# ===========================================
# 路径配置
# ===========================================
root_dir = "./experiment/test_result"      # 包含 0,1,2,... 文件夹
index_file = "/ayt1/anyitong/TurbMAE/3D/selected_indices_mean_order.txt"

# 读取索引
with open(index_file, "r") as f:
    valid_indices = set(int(line.strip()) for line in f if line.strip().isdigit())

print("有效样本数:", len(valid_indices))

# -------------------------------------------
#   第一遍：计算 MAE、RMSE、MSLE 以及 GT 全局均值
# -------------------------------------------
MAE = RMSE = MSLE = 0.0
Cn2_mean = 0.0
count = 0

for idx in valid_indices:
    folder = os.path.join(root_dir, str(idx))
    pred = np.load(os.path.join(folder, "cn2_out.npy"))
    label = np.load(os.path.join(folder, "label_cn2.npy"))

    count += 1

    # 指标（逐样本）
    mae = np.mean(np.abs(label - pred))
    rmse = np.sqrt(np.mean((label - pred) ** 2))
    msle = np.mean((np.log(1 + label) - np.log(1 + pred)) ** 2)

    MAE += mae
    RMSE += rmse
    MSLE += msle

    # GT 均值（用于 R²）
    Cn2_mean += np.mean(label)

Cn2_mean /= count

# -------------------------------------------
#   第二遍：计算 R² & Pearson (全局)
# -------------------------------------------
Cn2_molecule = 0.0
Cn2_denominator = 0.0
gt_all = []
pre_all = []

for idx in valid_indices:
    folder = os.path.join(root_dir, str(idx))
    pred = np.load(os.path.join(folder, "cn2_out.npy"))
    label = np.load(os.path.join(folder, "label_cn2.npy"))

    Cn2_molecule += np.sum((label - pred) ** 2)
    Cn2_denominator += np.sum((label - Cn2_mean) ** 2)

    gt_all.append(label.flatten())
    pre_all.append(pred.flatten())

gt_array = np.concatenate(gt_all)
pre_array = np.concatenate(pre_all)

pearson_corr = np.corrcoef(gt_array, pre_array)[0, 1]

# -------------------------------------------
#   输出
# -------------------------------------------
print(
    "Cn2 -- MAE:{:.4f}, RMSE:{:.4f}, MSLE:{:.4f}, R2:{:.4f}, Pearson:{:.4f}".format(
        MAE / count,
        RMSE / count,
        MSLE / count,
        1 - Cn2_molecule / Cn2_denominator,
        pearson_corr,
    )
)
