# CG-CWRA

Coordinate-Guided Cross-Window Relation Adapter for high-resolution document MLLMs.

这个仓库用于承载你的课程论文《坐标引导的跨窗口关系适配器（CG-CWRA）》的代码、配置、训练脚本和实验记录。仓库设计目标是：

- 本地只负责整理和推送代码到 GitHub
- 服务器 `git clone` 后直接补数据、模型和环境
- 数据集、checkpoints、缓存、日志、推理结果默认不入库

## 当前论文文件

论文主文件当前在仓库根目录：

- `bare_jrnl.tex`

## 建议工作流

1. 本地维护代码、配置、README、Docker 和脚本
2. 推送到 GitHub
3. 服务器克隆仓库后执行 `scripts/bootstrap_server.sh`
4. 将数据集和预训练权重放入已忽略目录
5. 使用 Docker 或宿主机 Python 环境运行训练/评测

## 目录结构

```text
CG-CWRA/
├─ bare_jrnl.tex
├─ README.md
├─ .gitignore
├─ .gitattributes
├─ .dockerignore
├─ .env.example
├─ pyproject.toml
├─ requirements/
│  ├─ base.txt
│  └─ dev.txt
├─ configs/
│  └─ train/
│     └─ example.yaml
├─ docker/
│  ├─ Dockerfile
│  └─ docker-compose.yml
├─ docs/
│  └─ server_setup.md
├─ scripts/
│  ├─ bootstrap_server.sh
│  ├─ train.sh
│  └─ eval.sh
├─ src/
│  └─ cgcwra/
│     ├─ __init__.py
│     ├─ data/
│     ├─ models/
│     ├─ train/
│     └─ utils/
├─ tests/
├─ data/                 # 数据集，不上传内容
├─ artifacts/            # checkpoints / pretrained / HF cache，不上传内容
└─ outputs/              # 日志、预测结果、tensorboard，不上传内容
```

## 哪些内容会上传到 GitHub

会上传：

- 源代码
- 训练与评测配置
- 轻量脚本
- 文档与论文源码
- 空目录占位文件

不会上传：

- 原始数据集
- 预处理数据
- 模型 checkpoints
- 预训练权重
- Hugging Face 缓存
- 日志、可视化、预测结果
- LaTeX 编译中间文件

## 服务器初始化

```bash
git clone <your-repo-url>
cd CG-CWRA
bash scripts/bootstrap_server.sh
```

然后把内容按下面位置放好：

- 数据集：`data/raw/`、`data/external/`
- 处理后数据：`data/processed/`
- 预训练模型：`artifacts/pretrained/`
- 训练输出 checkpoint：`artifacts/checkpoints/`
- Hugging Face 缓存：`artifacts/hf_cache/`

## Docker 用法

构建镜像：

```bash
docker compose -f docker/docker-compose.yml build
```

你可以通过复制 `docker/.env.example` 为 `docker/.env` 并按需修改其中的路径变量（如 `HOST_DATA_DIR`），来将服务器上的数据和模型缓存目录挂载进容器。

进入容器：

```bash
docker compose -f docker/docker-compose.yml run --rm cgcwra
```

或者启动后台容器再通过 exec 进入：

```bash
docker compose -f docker/docker-compose.yml up -d
docker exec -it cgcwra-container /bin/bash
```

容器内默认工作目录为 `/workspace/cgcwra`，并挂载当前仓库。缓存和输出都通过卷挂载写回宿主机对应的 `artifacts/`、`outputs/` 和 `data/` 目录。

## 训练入口

干跑检查配置：

```bash
bash scripts/train.sh configs/train/example.yaml --dry-run
```

评测占位入口：

```bash
bash scripts/eval.sh configs/train/example.yaml --dry-run
```

目前已实现论文中的核心 CG-CWRA 模块：对 ViT 输出的
`[batch, windows, patches, hidden]` 特征进行窗口级池化、坐标编码、带相对位置偏置的全连接图注意力，并把关系增量广播回每个 patch。它不改变 patch 数量或顺序，可直接接在既有连接器之前。

```python
from cgcwra.models import CGCWRAAdapter, CGCWRAConfig

adapter = CGCWRAAdapter(CGCWRAConfig(hidden_size=1024, num_heads=8))
enhanced_features = adapter(vit_patch_features, window_grid_coordinates)
```

`scripts/train.sh ... --dry-run` 会校验 YAML 并报告新增参数量。真正的端到端训练需按选定的 TextMonkey 或 Mini-Monkey 基座补充数据与连接器适配，冻结策略由 `configs/train/example.yaml` 明确记录。

## 后续建议

- 先把你计划用的基座模型和数据集清单补到 `docs/server_setup.md`
- 再把真实训练参数填入 `configs/train/example.yaml`
- 最后开始往 `src/cgcwra/` 里逐步实现数据管线、模型适配器和训练循环
