# 筛选指定label
import pandas as pd
from tqdm import tqdm
import os

def filter_csv_by_labels(input_file, output_file, target_labels, label_column='scene_label'):
    """
    input_file: 原始大文件路径
    output_file: 筛选后的保存路径
    target_labels: 包含目标标签的列表, 例如 ['apple', 'cherry']
    label_column: label 所在的列名
    """
    
    # 1. 快速获取总行数用于进度条显示
    print(f"正在分析文件: {input_file}...")
    with open(input_file, 'rb') as f:
        total_lines = sum(1 for _ in f) - 1
    
    chunk_size = 1000000  # 每次处理 100 万行
    
    # 2. 分块读取并过滤
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    
    # 初始化：第一次写入需要表头，之后追加不需要
    is_first_chunk = True
    
    with tqdm(total=total_lines, desc="筛选进度", unit="行") as pbar:
        for chunk in reader:
            # 筛选出 label 列在目标列表中的行
            # isin 会自动处理并忽略 NaN 值
            filtered_chunk = chunk[chunk[label_column].isin(target_labels)]
            
            # 如果这一块中有符合条件的数据，则执行写入
            if not filtered_chunk.empty:
                # mode='a' 表示追加模式
                # header=is_first_chunk 确保只有文件开头有表头
                filtered_chunk.to_csv(
                    output_file, 
                    mode='a', 
                    index=False, 
                    header=is_first_chunk, 
                    encoding='utf-8'
                )
                is_first_chunk = False # 第一次写入后，关闭表头开关
            
            pbar.update(len(chunk))

    print(f"\n处理完成！筛选后的数据已保存至: {output_file}")


if __name__ == "__main__":
    # 配置参数
    config = {
        "input_file": "../data/单框架语义化整合0322_muban_merged.csv",       # 原始文件名
        "output_file": "../data/单框架语义化整合0322_muban_merged_filtered.csv",    # 保存的文件名
        "target_labels": [
            "行程规划",
            
            "抵达始发高铁站",
            "在高铁站候车",
            "高铁行程途中",
            "抵达终点高铁站",
            "离开终点高铁站",
            
            "抵达始发地铁站",
            "乘坐地铁中",
            "抵达终点地铁站",
            
            "抵达始发机场",
            "机场内活动",
            "飞机行程途中",
            "抵达终点机场",
            "离开终点机场",
            
            
            "旅游参观",
            "旅游中途休息",
            "旅游住宿休息",
            "酒店办理入住",
            "旅游中逛街",
            "旅游中用餐",
            
            "等待网约车",
            "乘坐网约车行程中",
            "网约车到达终点",
            
            "户外运动",
            "文化场馆参观",
            "亲子游玩",
            
            "上班通勤",
            "下班通勤",
            
            "自驾途中",
            "驾车抵达终点",
            "驾驶途中加油、充电",
            "服务区休息",
            "去停车场停车/取车",
            ],  # 目标 label 列表
        "label_column": "scene_label"                # CSV 中 label 的列名
    }
    
    # 检查文件是否存在
    if os.path.exists(config["input_file"]):
        filter_csv_by_labels(**config)
    else:
        print("错误：未找到输入文件，请确认路径是否正确。")
