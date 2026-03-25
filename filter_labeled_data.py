"""筛选标签不为空的数据。"""

import os

import polars as pl
from tqdm import tqdm


def filter_csv_by_non_empty_label(input_file, output_file, label_column="scene_label"):
    """
    筛选 label 不为空的数据并保存（分块处理，适合大文件）。

    input_file: 原始大文件路径
    output_file: 筛选后的保存路径
    label_column: label 所在的列名
    """

    print(f"正在分析文件: {input_file}...")

    if os.path.exists(output_file):
        os.remove(output_file)

    # 1. 先快速获取总行数（用于进度条）
    with open(input_file, "rb") as f:
        total_lines = max(sum(1 for _ in f) - 1, 0)  # 减去表头，避免负数

    print(f"文件总行数: {total_lines:,}")

    # 2. 分块处理：使用 Polars 的 batched reader，避免重复从文件头扫描
    batch_size = 1_000_000  # 每批 100 万行
    reader = pl.read_csv_batched(
        input_file,
        has_header=True,
        batch_size=batch_size,
    )

    output_handle = None

    try:
        with tqdm(total=total_lines, desc="筛选进度", unit="行") as pbar:
            while True:
                batches = reader.next_batches(1)
                if not batches:
                    break

                for batch in batches:
                    # label 不为空：非 null 且去空白后不为空串
                    filtered_batch = batch.filter(
                        pl.col(label_column)
                        .cast(pl.Utf8, strict=False)
                        .str.strip_chars()
                        .fill_null("")
                        != ""
                    )

                    if not filtered_batch.is_empty():
                        if output_handle is None:
                            output_handle = open(output_file, "w", encoding="utf-8", newline="")
                            filtered_batch.write_csv(output_handle)
                        else:
                            filtered_batch.write_csv(output_handle, include_header=False)

                    pbar.update(len(batch))
    finally:
        if output_handle is not None:
            output_handle.close()

    print(f"\n处理完成！筛选后的数据已保存至: {output_file}")

    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            output_lines = max(sum(1 for _ in f) - 1, 0)
        print(f"输出文件行数: {output_lines:,}")


if __name__ == "__main__":
    config = {
        "input_file": "../data/单框架语义化整合0322_muban_merged.csv",
        "output_file": "../data/单框架语义化整合0322_muban_merged_filtered.csv",
        "label_column": "scene_label",
    }

    if os.path.exists(config["input_file"]):
        filter_csv_by_non_empty_label(**config)
    else:
        print("错误：未找到输入文件，请确认路径是否正确。")
