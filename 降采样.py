# 降采样
import pandas as pd
from tqdm import tqdm
import os

def downsample_csv_by_label(input_file, output_file, n, label_column='label'):
    """
    input_file: 原始大文件路径
    output_file: 降采样后的保存路径
    n: 每个 label 保留的最大行数
    label_column: label 所在的列名
    """
    
    # 1. 初始化计数器和进度条
    print(f"正在分析文件: {input_file}...")
    with open(input_file, 'rb') as f:
        total_lines = sum(1 for _ in f) - 1
    
    # 记录每个 label 已经采集了多少条
    label_counts_dict = {}
    chunk_size = 1000000 
    is_first_write = True

    # 2. 分块读取
    reader = pd.read_csv(input_file, chunksize=chunk_size)

    with tqdm(total=total_lines, desc="降采样进度", unit="行") as pbar:
        for chunk in reader:
            # 定义一个函数，用于处理当前块中每一行是否保留
            def should_keep(label):
                if pd.isna(label): return False # 跳过空标签
                
                current_count = label_counts_dict.get(label, 0)
                if current_count < n:
                    label_counts_dict[label] = current_count + 1
                    return True
                return False

            # 应用筛选逻辑
            # 注意：此处使用 apply 会逐行判断，确保精确控制数量
            mask = chunk[label_column].apply(should_keep)
            filtered_chunk = chunk[mask]

            # 3. 如果块中有符合条件的数据，追加写入
            if not filtered_chunk.empty:
                filtered_chunk.to_csv(
                    output_file, 
                    mode='a', 
                    index=False, 
                    header=is_first_write, 
                    encoding='utf-8-sig'
                )
                is_first_write = False
            
            pbar.update(len(chunk))

    print(f"\n处理完成！每类最多保留 {n} 条数据。")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    config = {
        "input_file": "data.csv",
        "output_file": "data_desample10000.csv",
        "n": 10000,               # 每个 label 最多保留 500 条
        "label_column": "scene_label"
    }
    
    if os.path.exists(config["input_file"]):
        downsample_csv_by_label(**config)
