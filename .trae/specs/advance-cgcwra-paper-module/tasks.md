# Tasks

- [ ] Task 1: 盘点并校验本地模型资产
  - [ ] SubTask 1.1: 核对 `artifacts/pretrained/Qwen-VL` 与 `artifacts/pretrained/TextMonkey` 的权重文件、索引文件、配置文件和 tokenizer 文件完整性
  - [ ] SubTask 1.2: 提取模型架构、关键配置、参数规模估算、依赖版本约束和异常项
  - [ ] SubTask 1.3: 生成可复核的模型基础校验报告，明确通过项、缺失项和风险项

- [ ] Task 2: 建立冒烟测试准备流程
  - [ ] SubTask 2.1: 梳理并固定最小运行环境，补齐推理所需依赖与环境检测入口
  - [ ] SubTask 2.2: 实现硬件资源探测，包括 CPU、GPU、显存、系统内存和关键运行限制
  - [ ] SubTask 2.3: 编写最小化测试用例，验证模型可加载并完成基础推理或前向调用
  - [ ] SubTask 2.4: 输出冒烟测试结果与失败排查信息

- [ ] Task 3: 同步修复本地 TextMonkey 代码
  - [ ] SubTask 3.1: 拉取或比对 GitHub 官方 TextMonkey 最新主干信息
  - [ ] SubTask 3.2: 逐项排查本地依赖版本、核心逻辑、配置文件和关键脚本差异
  - [ ] SubTask 3.3: 修复本地差异并补充必要验证，确保与官方主干一致
  - [ ] SubTask 3.4: 记录同步修复清单及其验证依据

- [ ] Task 4: 细化 CG-CWRA 模块设计并完成迭代实现
  - [ ] SubTask 4.1: 梳理模块功能需求、技术指标、输入输出契约、插入位置和兼容边界
  - [ ] SubTask 4.2: 调研适配 TextMonkey 体系的技术方案与实现路径
  - [ ] SubTask 4.3: 将模块拆解为可迭代子任务并按阶段完成实现
  - [ ] SubTask 4.4: 在实现过程中统一代码风格、语法规范与类型安全校验

- [ ] Task 5: 建立 CG-CWRA 模块验证闭环
  - [ ] SubTask 5.1: 编写单元测试覆盖坐标编码、跨窗口关系建模、特征注入等核心逻辑
  - [ ] SubTask 5.2: 编写集成测试覆盖与现有训练/推理入口的接入路径
  - [ ] SubTask 5.3: 验证功能正确性、兼容性与关键性能指标达标情况
  - [ ] SubTask 5.4: 输出完整模块验证报告，汇总结论、证据与遗留风险

- [ ] Task 6: 回填论文与实验文档中的实现依据
  - [ ] SubTask 6.1: 根据最终实现与验证结果回填论文主文件中的实现、实验设置和验证描述
  - [ ] SubTask 6.2: 保证论文表述与代码现状、模型资产状态和测试结论一致

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 can run in parallel with Task 2 after Task 1 starts collecting模型现状
- Task 4 depends on Task 3
- Task 5 depends on Task 4
- Task 6 depends on Task 5
