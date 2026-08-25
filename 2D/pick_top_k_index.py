def top_n_common_indices(file1, file2, top_n, output_file):
    # 读取 file1 和 file2 的数据
    with open(file1, 'r') as f1:
        data1 = [(i, float(line.strip())) for i, line in enumerate(f1)]

    with open(file2, 'r') as f2:
        data2 = [(i, float(line.strip())) for i, line in enumerate(f2)]

    # 检查两个文件行数是否一致（可选）
    if len(data1) != len(data2):
        raise ValueError("两个文件行数不一致")

    # 分别取 top_n 个最大值对应的索引
    top_indices_1 = set(i for i, _ in sorted(data1, key=lambda x: x[1], reverse=True)[:top_n])
    top_indices_2 = set(i for i, _ in sorted(data2, key=lambda x: x[1], reverse=True)[:top_n])

    # 求交集
    common_indices = top_indices_1 & top_indices_2

    # 按照 file1 中的值从大到小排序这些共同索引
    common_sorted = sorted(
        [(i, data1[i][1]) for i in common_indices],
        key=lambda x: x[1],
        reverse=True
    )

    # 写入输出文件
    with open(output_file, 'w') as f_out:
        for idx, _ in common_sorted:
            f_out.write(f"{idx}\n")

    print(f"共同存在于两个文件前 {top_n} 中的行号共 {len(common_sorted)} 个，已写入 {output_file}")

# 用法示例
if __name__ == "__main__":
    top_n_common_indices(
        file1='/ayt/anyitong/PBCL_TMM+MAE/code/Cn2_R2_strategy2.txt',
        file2='/ayt/anyitong/PBCL_TMM+MAE/code/psnr_strategy2.txt',
        top_n=3000,  # 根据需要调整
        output_file='index_strategy2.txt'
    )
