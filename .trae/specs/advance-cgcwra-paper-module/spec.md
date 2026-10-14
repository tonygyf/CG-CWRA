# 论文模块推进 Spec

## Why
当前仓库已经具备 CG-CWRA 的初始代码框架、研究冒烟配置和本地预训练模型资产，但尚未形成一套覆盖模型资产校验、环境冒烟、上游代码同步、模块正式落地与验证报告输出的统一开发闭环。为了支撑论文实现与实验复现，需要把这些阶段性工作收敛为明确、可执行、可验收的规格。

## What Changes
- 新增预训练模型基础校验能力，覆盖 Qwen-VL 基座模型与 TextMonkey 开源模型的权重完整性、版本一致性、架构信息、参数规模和依赖环境盘点
- 新增冒烟测试准备流程，覆盖运行环境初始化、硬件资源探测、最小推理用例与基础可用性判定
- 新增本地 TextMonkey 代码与 GitHub 官方最新主干的同步修复要求，覆盖依赖版本、核心逻辑、配置文件与必要脚本
- 新增 CG-CWRA 模块正式落地要求，覆盖需求澄清、技术方案调研、迭代任务拆解、编码规范、风格统一与类型安全检查
- 新增模块设计验证要求，覆盖单元测试、集成测试、兼容性验证、性能达标验证与验证报告输出

## Impact
- Affected specs: 模型资产管理、环境与冒烟测试、上游基线同步、CG-CWRA 模块实现、测试与验证报告
- Affected code: [bare_jrnl.tex](file:///home/gyf/CG-CWRA/bare_jrnl.tex)、[artifacts/pretrained](file:///home/gyf/CG-CWRA/artifacts/pretrained)、[src/cgcwra](file:///home/gyf/CG-CWRA/src/cgcwra)、[tests](file:///home/gyf/CG-CWRA/tests)、[configs](file:///home/gyf/CG-CWRA/configs)、[scripts](file:///home/gyf/CG-CWRA/scripts)、[pyproject.toml](file:///home/gyf/CG-CWRA/pyproject.toml)

## ADDED Requirements
### Requirement: 预训练模型基础校验
系统 SHALL 对已下载的 Qwen-VL 基座模型和 TextMonkey 开源模型执行本地基础校验，并输出可追溯的核心信息报告。

#### Scenario: 权重与版本校验成功
- **WHEN** 用户在本地已有 `artifacts/pretrained/Qwen-VL` 与 `artifacts/pretrained/TextMonkey` 目录的前提下发起模型校验
- **THEN** 系统应检查权重分片与索引文件是否完整、配置文件是否齐全、模型版本与依赖约束是否可识别
- **THEN** 系统应输出模型架构、关键配置、参数规模估算、依赖环境与潜在异常项

#### Scenario: 发现模型资产缺失或不一致
- **WHEN** 权重文件、索引文件、配置文件或版本声明存在缺失、冲突或无法解析
- **THEN** 系统应明确报告缺失项、冲突项和对应影响范围
- **THEN** 系统应给出后续修复所需的最小行动项

### Requirement: 冒烟测试准备与最小推理验证
系统 SHALL 提供可复用的冒烟测试准备流程，用于验证模型是否可以被本地环境正常加载并完成最小推理。

#### Scenario: 环境准备完成并执行最小推理
- **WHEN** 用户执行冒烟测试流程
- **THEN** 系统应探测 CPU、GPU、显存、系统内存和关键推理依赖的可用性
- **THEN** 系统应能够完成最小化测试用例，验证模型加载、输入构造、前向或生成调用至少有一种成功路径
- **THEN** 系统应输出成功或失败结论以及关键运行证据

#### Scenario: 资源或依赖不足
- **WHEN** 环境中缺少关键依赖、GPU 不可用、显存不足或模型初始化失败
- **THEN** 系统应中止后续正式实验流程
- **THEN** 系统应输出资源约束、失败阶段和最小修复建议

### Requirement: TextMonkey 官方主干同步修复
系统 SHALL 以 GitHub 官方 TextMonkey 最新主干为基准，排查并修复本地代码与上游之间的差异，确保本地副本与上游实现保持一致。

#### Scenario: 本地代码与官方主干比对
- **WHEN** 用户发起同步修复任务
- **THEN** 系统应获取官方仓库最新主干信息并逐项比对本地依赖、核心逻辑、配置文件和关键脚本
- **THEN** 系统应形成差异清单，并按可验证方式修复本地偏差
- **THEN** 修复完成后，本地代码应能够说明与官方主干一致的依据

### Requirement: CG-CWRA 模块正式落地
系统 SHALL 将论文中的 CG-CWRA 方案落地为分阶段可演进的模块实现，并保持统一的代码规范与类型安全约束。

#### Scenario: 模块需求拆解与迭代实现
- **WHEN** 用户进入 CG-CWRA 模块开发阶段
- **THEN** 系统应先梳理核心功能需求、技术指标、插入位置、输入输出契约与兼容边界
- **THEN** 系统应完成技术方案调研并拆解为可迭代实施的子任务
- **THEN** 每一轮实现都应满足既有代码风格、语法规范和类型检查要求

#### Scenario: 模块与现有模型体系集成
- **WHEN** 模块接入现有 TextMonkey 或相关训练/推理流程
- **THEN** 系统应保持对既有配置、连接器流程和冻结策略的兼容
- **THEN** 系统应避免破坏现有 smoke 配置与实验入口

### Requirement: 模块验证与报告输出
系统 SHALL 为 CG-CWRA 模块提供完整的验证闭环，包括测试用例、兼容性验证、性能验证和结果报告。

#### Scenario: 验证闭环完成
- **WHEN** CG-CWRA 模块整体架构与实现完成
- **THEN** 系统应具备单元测试与集成测试覆盖关键功能路径
- **THEN** 系统应验证功能正确性、性能指标达标情况以及与现有模型体系的兼容性
- **THEN** 系统应输出结构化验证报告，说明通过项、失败项、指标结果与遗留风险

## MODIFIED Requirements
### Requirement: 现有研究冒烟流程
现有研究冒烟流程 SHALL 从仅面向合成数据与内部原型验证，扩展为同时支持真实预训练模型资产校验、环境资源探测和最小推理验证的前置流程。

## REMOVED Requirements
### Requirement: 无
**Reason**: 本次变更以补充和收敛流程为主，不删除既有能力。
**Migration**: 无需迁移。
