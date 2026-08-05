# 数据处理

> [English](README.md)

本包将异构 Agent 轨迹转换为 detector 消费的唯一统一 schema。

## 统一 schema

每个 `data/unified/<dataset>/<task_id>.json` 包含：

```json
{
  "messages": [
    {"step": 0, "role": "user", "name": "human", "content": "..."}
  ],
  "metadata": {
    "dataset": "example",
    "task_id": "task-1",
    "task_description": "...",
    "reward": 0,
    "annotation": {
      "critical_error_step": 7,
      "critical_error_type": "act.WrongTool"
    },
    "extra": {}
  }
}
```

必要约束：

- `messages` 非空，且 `messages[i].step == i`。
- `role` 为 `user`、`assistant`、`tool`、`system` 之一。
- `content` 为字符串。
- `reward` 为 `0` 或 `1`。
- 非空 `critical_error_step` 必须是有效的统一消息下标，且不能指向 `user` 消息。
- 成功轨迹的关键错误标注必须为空。

`schema.py` 定义并校验此契约。

## 构建内置数据集

从仓库根目录执行：

```bash
python -m data_processing.build_unified_dataset --all
python -m data_processing.build_unified_dataset --dataset tau2bench
python -m data_processing.build_unified_dataset --dataset swebenchpro
```

已注册键为 `alfworld`、`gaia`、`webshop`、`whoandwhen`、`whoandwhen_algorithm`、`tau2bench`、`swebenchpro`。默认输出到 `data/unified/<dataset>`。

## 接入新数据集

1. 新增 `data_processing/<name>.py`，提供 `convert_directory(src, out[, labels])`。
2. 归一化角色，并分配连续的零索引 `step`。
3. 构造必要 metadata；有标签时映射为 `<module>.<subtype>`。
4. 对每个输出调用 `validate_unified`。
5. 在 `build_unified_dataset.py` 的 `DATASETS` 中注册转换器。
6. 必要时在 `dataset_config.json` 中添加 judging/compression 设置。

与 ALFWorld、GAIA、WebShop 相似的消息格式可复用 `_standard_messages.convert_standard_record`。其他结构可直接构造 schema；参考 `whoandwhen.py`、`tau2bench.py`、`swebenchpro.py`。

构建并运行：

```bash
python -m data_processing.build_unified_dataset \
  --dataset my_dataset \
  --src data/MyDataset \
  --out data/unified/my_dataset

DATASETS="my_dataset" DETECTOR_MODEL=your-model bash run_pipeline.sh
```

detector 默认输出到 `outputs`；数据转换过程不会写 detector 输出。
