# def filter_lines(file1_path, file2_path, file3_path, threshold, output_path):
#     # 读取所有文件的每一行，转为 float 数组
#     with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(file3_path, 'r') as f3:
#         values1 = [float(line.strip()) for line in f1]
#         values2 = [float(line.strip()) for line in f2]
#         values3 = [float(line.strip()) for line in f3]

#     # 检查行数一致性
#     if not (len(values1) == len(values2) == len(values3)):
#         raise ValueError("三个文件的行数不一致！")

#     # 查找满足条件的行索引
#     matched_indices = []
#     for i, (v1, v2, v3) in enumerate(zip(values1, values2, values3)):
#         if v1 > threshold and v1 < v2 and v1 < v3:
#             matched_indices.append(i)

#     # 将结果写入输出文件
#     with open(output_path, 'w') as f_out:
#         for idx in matched_indices:
#             f_out.write(f"{idx}\n")

#     print(f"找到 {len(matched_indices)} 个满足条件的行，结果已保存到 {output_path}")

# # 用法示例（你可以根据实际路径和阈值修改）
# if __name__ == "__main__":
#     filter_lines(
#         file1_path='/ayt/anyitong/PBCL_ori/code/Cn2_R2.txt',
#         file2_path='/ayt/anyitong/PBCL_TMM+MAE/code/Cn2_R2_strategy1.txt',
#         file3_path='/ayt/anyitong/PBCL_TMM+MAE/code/Cn2_R2_strategy1.txt',
#         threshold=0.7,
#         output_path='index_strategy1.txt'
#     )






def filter_lines(file1_path, file2_path, file3_path, file4_path, file5_path, file6_path, threshold, output_path):
    # 读取所有文件的每一行，转为 float 数组
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, \
         open(file3_path, 'r') as f3, open(file4_path, 'r') as f4, \
         open(file5_path, 'r') as f5, open(file6_path, 'r') as f6:
        values1 = [float(line.strip()) for line in f1]
        values2 = [float(line.strip()) for line in f2]
        values3 = [float(line.strip()) for line in f3]
        values4 = [float(line.strip()) for line in f4]
        values5 = [float(line.strip()) for line in f5]
        values6 = [float(line.strip()) for line in f6]

    # 检查行数一致性
    length = len(values1)
    if not all(len(lst) == length for lst in [values2, values3, values4, values5, values6]):
        raise ValueError("所有文件的行数必须一致！")

    # 查找满足所有条件的行索引
    matched_indices = []
    for i, (v1, v2, v3, v4, v5, v6) in enumerate(zip(values1, values2, values3, values4, values5, values6)):
        if v1 > threshold and v1-0.008 < v2 and v3 < v4 and v3>31:
            matched_indices.append(i)

    # 将结果写入输出文件
    with open(output_path, 'w') as f_out:
        for idx in matched_indices:
            f_out.write(f"{idx}\n")

    print(f"找到 {len(matched_indices)} 个满足条件的行，结果已保存到 {output_path}")


# 用法示例
if __name__ == "__main__":
    filter_lines(
        file1_path='/ayt/anyitong/PBCL_ori/code/Cn2_R2.txt',
        file2_path='/ayt/anyitong/PBCL_TMM+MAE/code/Cn2_R2_strategy1.txt',
        file3_path='/ayt/anyitong/PBCL_ori/code/psnr.txt',
        file4_path='/ayt/anyitong/PBCL_TMM+MAE/code/psnr_strategy1.txt',
        file5_path='/ayt/anyitong/PBCL_ori/code/ssim.txt',
        file6_path='/ayt/anyitong/PBCL_TMM+MAE/code/ssim_strategy1.txt',
        threshold=-1,
        output_path='index_strategy1.txt'
    )
