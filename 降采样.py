# 降采样
import os
import random

import polars as pl


def downsample_csv_by_label(input_file, output_file, n, label_column='label'):
    """
    input_file: 原始大文件路径
    output_file: 降采样后的保存路径
    n: 每个 label 保留的最大行数
    label_column: label 所在的列名
    """
    batch_size = 200_000
    label_seen_counts = {}
    reservoirs = {}
    processed_rows = 0
    output_columns = None

    print(f"正在分析文件: {input_file}...")

    if os.path.exists(output_file):
        os.remove(output_file)

    batches = pl.scan_csv(input_file).collect_batches(
        chunk_size=batch_size,
        maintain_order=True,
        engine="streaming",
    )

    for batch in batches:
        if output_columns is None:
            output_columns = batch.columns

        if label_column not in batch.columns:
            raise ValueError(f"CSV 中未找到标签列: {label_column}")

        for row in batch.iter_rows(named=True):
            label = row.get(label_column)
            if label is None:
                continue

            seen_count = label_seen_counts.get(label, 0)
            bucket = reservoirs.setdefault(label, [])

            if len(bucket) < n:
                bucket.append(row)
            else:
                replace_idx = random.randint(0, seen_count)
                if replace_idx < n:
                    bucket[replace_idx] = row

            label_seen_counts[label] = seen_count + 1

        processed_rows += batch.height
        print(f"已处理行数: {processed_rows}", end="\r", flush=True)

    sampled_rows = [row for bucket in reservoirs.values() for row in bucket]

    if output_columns is None:
        output_columns = []

    if sampled_rows:
        pl.DataFrame(sampled_rows, schema=output_columns).write_csv(output_file, include_bom=True)
    else:
        pl.DataFrame({col: [] for col in output_columns}).write_csv(output_file, include_bom=True)

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
