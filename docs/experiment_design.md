# CG-CWRA 实验体系设计

## 1. 研究目标与核心假设

本文方法的目标是在高分辨率文档多模态理解场景下，显式恢复窗口切分后丢失的长程结构关系。实验体系围绕以下三条核心假设展开：

1. `H1`：在需要跨窗口聚合信息的样本上，CG-CWRA 相比无交互或仅局部交互基线具有更高的准确率。
2. `H2`：坐标引导的全连接窗口关系建模，相比局部位移交互或多尺度上下文，更能提升跨窗口实体关联能力。
3. `H3`：在冻结视觉编码器和语言模型的条件下，仅训练轻量关系模块与连接器，能够以较低参数增量获得稳定收益。

## 2. 评价指标

- `DocVQA`: ANLS
- `WikiTableQuestions`: Exact Match Accuracy
- `ChartQA`: Exact Match Accuracy
- `CWEPA`: 跨窗口实体配对准确率，仅在需要跨窗口实体对齐的子集上统计
- `Trainable Params`: 新增可训练参数量
- `Latency / Throughput`: 训练与评测吞吐，用于分析额外计算开销

## 3. 对照组设置

- `Monkey`: 无显式窗口交互
- `TextMonkey`: 局部窗口交互
- `Mini-Monkey`: 多尺度上下文增强
- `CG-CWRA`: 本文提出的坐标引导全连接窗口关系模块

在统一实验协议下，四类模型共享如下条件：

- 相同的数据划分
- 相同的窗口裁剪策略
- 相同的训练轮数、优化器、batch size、评估间隔
- 相同的冻结策略与日志记录格式

## 4. 消融实验规划

- 去除跨窗口关系模块
- 仅保留相邻窗口交互
- 去除相对坐标偏置
- 关系层数 `L=1/2`
- 去除结构对齐损失
- 替换全连接图为稀疏局部图

## 5. 多维验证场景

- `任务维度`: 文档问答、表格问答、图表问答
- `难度维度`: 单窗口样本、相邻窗口样本、远距离跨窗口样本
- `资源维度`: CPU smoke test、单卡训练、多卡并行搜索
- `泛化维度`: 不同窗口网格划分、不同基座模型接入

## 6. 数据与预处理规范

仓库统一采用 JSONL 标准样本格式，每条记录包含：

- `sample_id`, `dataset_name`, `split`
- `question`, `answer`, `label`
- `window_features`, `coordinates`
- `window_mask`, `patch_mask`
- `metadata.cross_window_required`

当前仓库提供两层实现：

1. `synthetic_cross_window`：用于 Docker / CI / 代码正确性冒烟验证
2. `normalize_annotation_dataset(...)`：用于将 DocVQA、WTQ、ChartQA 的任务特定导出结果转为统一格式

## 7. 基线复现与统一运行

统一 runner 位于 `src/cgcwra/experiments/runner.py`，负责：

- 构建 train/val/test dataloader
- 训练所有启用的 baseline
- 输出统一 metrics
- 产出 JSON 报告与曲线图

外部大模型代码接入通过 `scripts/bootstrap_baselines.sh` 完成。当前默认接入 `Yuliang-Liu/Monkey` 仓库，它包含 Monkey、TextMonkey、Mini-Monkey 的上游实现。仓库内的轻量 baseline 则用于统一协议和可复现实验骨架验证。

## 8. 资源与并行策略

- 默认 CPU smoke test 使用 `configs/train/research_smoke.yaml`
- 实际训练时建议：
  - 单卡：固定 seed 复现实验
  - 多卡：数据并行或不同 seed / 超参并行
  - 超参搜索：优先搜索学习率、权重衰减、关系层数

## 9. 产出物

- `outputs/experiments/<name>/research_report.json`
- `outputs/experiments/<name>/plots/*.png`
- `reports/environment_report.json`
- `reports/docker_validation_report.md`
