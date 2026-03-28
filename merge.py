import os
import csv
import polars as pl
from collections import deque
from tqdm import tqdm


def process_and_merge_final_streaming_polars(
    input_file: str,
    output_file: str,
    n: int = 10,
    batch_size: int = 200_000,
):
    """
    使用 Polars 分块处理并流式写出，将 label 列统一为 scene_label。

    数据假设：
    1. 已按 udid, time 排好序
    2. 同一个 udid 的记录连续出现

    MERGED_TEXT 规则：
    [current]当前text[/current]
    + 依次追加 [previous-1]...[/previous-1] 到 [previous-n]...[/previous-n]
    （仅限同 udid 的历史记录）

    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        n: 历史窗口大小
        batch_size: 每批读取行数
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    output_cols = [
        "time",
        "udid",
        "scene_label",
        "MERGED_TEXT",
        "context",
        "history_usage",
        "service_click",
    ]

    print("开始使用 Polars 分块读取并拼接（不排序，流式写出）...")

    # 跨 chunk 保留状态
    last_udid = None
    prev_texts = deque(maxlen=n)

    # 先删掉旧文件，避免追加到历史结果后面
    if os.path.exists(output_file):
        os.remove(output_file)

    # 用 csv writer 做真正的流式写出
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(output_cols)

        # batched=True + batch_size 可以分批取
        reader = pl.read_csv_batched(
            input_file,
            batch_size=batch_size,
            has_header=True,
            columns=[
                "time",
                "udid",
                "scene_label",
                "text",
                "context",
                "history_usage",
                "service_click",
            ],
            schema_overrides={
                "time": pl.Utf8,
                "udid": pl.Utf8,
                "scene_label": pl.Utf8,
                "text": pl.Utf8,
                "context": pl.Utf8,
                "history_usage": pl.Utf8,
                "service_click": pl.Utf8,
            },
            ignore_errors=True,
            encoding="utf8",
        )

        pbar = tqdm(desc="处理batch", unit="batch")

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break

            batch = batches[0]

            # 空值统一处理
            batch = batch.with_columns([
                pl.col("time").fill_null(""),
                pl.col("udid").fill_null(""),
                pl.col("scene_label").fill_null(""),
                pl.col("text").fill_null(""),
                pl.col("context").fill_null(""),
                pl.col("history_usage").fill_null(""),
                pl.col("service_click").fill_null(""),
            ])

            # 转成行迭代需要的形式
            rows = batch.iter_rows(named=True)

            out_rows = []

            for row in rows:
                time_val = row["time"]
                udid = row["udid"]
                scene_label = row["scene_label"]
                text = row["text"]
                context = row["context"]
                history_usage = row["history_usage"]
                service_click = row["service_click"]

                # udid 切换时清空历史窗口
                if udid != last_udid:
                    prev_texts.clear()
                    last_udid = udid

                cur_text = text or ""
                merged = f"[current]{cur_text}[/current]"

                if prev_texts:
                    for i, ptxt in enumerate(reversed(prev_texts), start=1):
                        merged += f" [previous-{i}]{ptxt}[/previous-{i}]"

                out_rows.append([
                    time_val,
                    udid,
                    scene_label,
                    merged,
                    context,
                    history_usage,
                    service_click,
                ])

                prev_texts.append(cur_text)

            writer.writerows(out_rows)
            pbar.update(1)

        pbar.close()

    print(f"✅ 处理成功！结果已保存至: {output_file}")


# 运行配置
config = {
    "input_file": "../data/单框架戏剧替换_test1_feature_label.csv",
    "output_file": "../data/单框架戏剧替换_test1_muban_merged.csv",
    "n": 10,
    "batch_size": 200000,
}

if __name__ == "__main__":
    if os.path.exists(config["input_file"]):
        process_and_merge_final_streaming_polars(**config)
    else:
        print(f"错误：未找到输入文件 {config['input_file']}")
