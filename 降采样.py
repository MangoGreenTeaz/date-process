# 降采样
import os

import polars as pl


def downsample_csv_by_label(input_file, output_file, n, label_column='label'):
    """
    input_file: 原始大文件路径
    output_file: 降采样后的保存路径
    n: 每个 label 保留的最大行数
    label_column: label 所在的列名
    """
    batch_size = 200_000
    label_counts = {}
    processed_rows = 0
    is_first_write = True

    print(f"正在分析文件: {input_file}...")

    if os.path.exists(output_file):
        os.remove(output_file)

    batches = pl.scan_csv(input_file).collect_batches(
        chunk_size=batch_size,
        maintain_order=True,
        engine="streaming",
    )

    for batch in batches:
        if label_column not in batch.columns:
            raise ValueError(f"CSV 中未找到标签列: {label_column}")

        labels = batch.get_column(label_column).to_list()
        keep_mask = []

        for label in labels:
            if label is None:
                keep_mask.append(False)
                continue

            current_count = label_counts.get(label, 0)
            if current_count < n:
                label_counts[label] = current_count + 1
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        filtered_batch = batch.filter(pl.Series("_keep", keep_mask))

        if filtered_batch.height > 0:
            if is_first_write:
                filtered_batch.write_csv(output_file, include_bom=True)
                is_first_write = False
            else:
                with open(output_file, "a", encoding="utf-8", newline="") as f:
                    f.write(filtered_batch.write_csv(file=None, include_header=False))

        processed_rows += batch.height
        print(f"已处理行数: {processed_rows}", end="\r", flush=True)

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
