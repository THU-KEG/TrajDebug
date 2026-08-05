# Detector

> [English](README.md)

detector 通过 OpenAI-compatible LLM 端点，在失败轨迹中定位关键错误。

## 流水线

```text
data/unified/<dataset>/*.json
  → Stage A：多粒度压缩
  → Stage B：基于证据的错误 trigger
  → Stage C1：按对象聚合错误实例
  → Stage C2：判断修复状态与终态影响
  → Stage C3：候选引导的因果归因
  → outputs/<dataset>_final
```

入口脚本：

- `stage_a_diagnosis.py`：生成 `*_stage_a.json`。
- `stage_b_per_step.py`：生成带逐字冲突证据的 `*_stage_b.json`。
- `stage_c_phase1_cluster.py`：将重复 trigger 聚合成错误实例。
- `stage_c_phase2_state.py`：判断修复状态、实例状态、终态连接和因果链成员关系。
- `stage_c_phase3_assemble.py`：选择关键步并写入 `*_final.json`。
- `score_steps.py`：对照 `metadata.annotation` 评测。

## 核心约定

所有步号都是统一 `messages` 数组中的零索引位置。每个 trigger 必须同时引用错误承诺与被违反参照。实例按具体被违反对象聚合，而不是只按 taxonomy 标签聚合。Stage C 保留与终态相关的实例，并在候选中归因最终失败。

五类 Agent 错误模块为 `plan`、`reason`、`act`、`obs`、`verify`，定义见 `utils/error_definitions.py`。

## 运行

从仓库根目录执行：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export DETECTOR_MODEL="your-model-name"

DATASETS="alfworld" bash run_pipeline.sh
```

runner 支持断点续跑，默认写入 `outputs`：

```text
outputs/<dataset>_stage_a/
outputs/<dataset>_stage_b/
outputs/<dataset>_phase1/
outputs/<dataset>_phase2/
outputs/<dataset>_final/
outputs/score_<dataset>.json
```

用 `UNIFIED_ROOT` 和 `OUTPUT_ROOT` 覆盖路径；用 `FILE_CONCURRENCY`、`LLM_CONCURRENCY`、`MAX_TOKENS`、`TEMPERATURE` 调整请求。

## 评测

```bash
python detector/score_steps.py \
  --unified-dir data/unified/alfworld \
  --pred-dir outputs/alfworld_final \
  --out outputs/score_alfworld.json

python detector/score_steps_breakdown.py \
  --unified-dir data/unified/alfworld \
  --pred-dir outputs/alfworld_final \
  --stage-b-dir outputs/alfworld_stage_b \
  --phase1-dir outputs/alfworld_phase1 \
  --phase2-dir outputs/alfworld_phase2 \
  --out outputs/breakdown_alfworld.json
```

评分器报告 exact、loose-1、loose-2 步准确率，并在标签可用时报告 taxonomy 指标。成功轨迹与 system failure 不参与评分。

## 可选的论文 §6 反馈应用

`applications/generate_feedback.py` 基于已完成的 Stage C 输出实现论文 §6 反馈应用。它不是核心 detector stage，`run_pipeline.sh` 不会调用它。

```bash
python applications/generate_feedback.py \
  --final_dir outputs/alfworld_final \
  --trajectory_dir data/unified/alfworld \
  --stage_a_dir outputs/alfworld_stage_a \
  --output_dir outputs/alfworld_report \
  --base_url "$OPENAI_BASE_URL" \
  --model "$DETECTOR_MODEL" \
  --api_key "$OPENAI_API_KEY" \
  --resume
```

报告路径为 `outputs/<dataset>_report/<task_id>_report.json`。最终 report 的 `fix_suggestion.hint_sentence` 字段包含可操作建议，路径与 viewer 约定一致。

## 开发参考

- 从仓库根目录调用 CLI，确保 detector 工具模块导入一致。
- 保持 `messages[i].step == i`，关键错误步不能指向 `user` 消息。
- Stage B 证据必须逐字保留；taxonomy 应保持稳定，否则已有标注会失效。
- 修改并发代码时，压缩状态必须保持为逐轨迹局部状态。
- 新数据集只需生成统一 schema 文件，见 [`../data_processing/README_zh.md`](../data_processing/README_zh.md)。
