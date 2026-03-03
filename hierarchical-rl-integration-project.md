# 项目方案：Hierarchical RL for Symbolic Integration via RUBI Rule Selection

## 一、项目概述

**题目（暂定）**: Hierarchical Reinforcement Learning for Symbolic Integration with Rule-Based Verification

**核心思想**: 用分层强化学习（Hierarchical RL）训练一个小型神经网络，在 RUBI 的 6,700+ 条积分规则上学习最优的规则选择策略，以 Wolfram Engine 作为规则执行引擎，以求导验证作为 reward signal。

**目标**: 在 RUBI 的 72,944 题标准测试集上，用学习到的策略接近或匹配 RUBI hand-crafted decision tree 的 99.8% 成功率。

---

## 二、研究动机与创新点

### 2.1 已有工作

| 工作 | 方法 | 动作空间 | 局限 |
|------|------|---------|------|
| RUBI (Albert Rich) | Hand-crafted decision tree | 6,700 条规则 | 人工设计，不可泛化 |
| AlphaIntegrator (ETH, 2024.10) | 监督学习 + beam search，10M GPT | 47 个 SymPy 动作 | 动作空间太小，覆盖率有限 |
| SIRD (NeurIPS 2023 Workshop) | 监督学习，Transformer 预测规则 | 24 条 SymPy 规则 | 规模更小 |
| England et al. (2024-2025) | TreeLSTM / Tree Transformer | Maple 的 12 个子算法 | 算法选择而非规则选择 |

### 2.2 本项目的创新点

1. **第一个用 RL 训练的符号积分系统**（已有工作全部是监督学习）
2. **第一个在 RUBI 全部 6,700 条规则上做神经网络选择的系统**（已有工作最多 47 个动作）
3. **分层动作空间设计**：利用 RUBI 的 9 大类层次结构，高层选类别，低层选规则
4. **完美 verifier**：求导验证提供免费、精确的 reward signal
5. **不移植规则**：用 Wolfram Engine 做执行引擎，完全绕开 SymPy 移植 RUBI 失败的工程深坑

---

## 三、技术方案

### 3.1 系统架构

```
输入：被积函数 f(x)（表达式树）
  │
  ├── 特征提取器（Tree-RNN / GNN / small Transformer）
  │     └── 将表达式树编码为向量 h ∈ R^d
  │
  ├── 高层 Policy（MLP）
  │     └── 输出 9 大类的概率分布 → 选择类别 C_k
  │
  ├── 低层 Policy（MLP，每个类别一个或共享+条件）
  │     └── 在类别 C_k 内输出规则概率 → 选择具体规则 R_j
  │
  ├── Wolfram Engine 执行
  │     └── 应用规则 R_j → 得到新表达式 f'(x)
  │
  └── 递归：对 f'(x) 重复上述过程，直到积分完成或超时
```

### 3.2 RUBI 的层次结构（动作空间）

```
Level 0: 9 大类
  1. Algebraic functions        (~2000 条规则)
  2. Exponentials               (~300 条)
  3. Logarithms                 (~300 条)
  4. Trig functions             (~2000 条)
  5. Inverse trig functions     (~600 条)
  6. Hyperbolic functions       (~400 条)
  7. Inverse hyperbolic         (~700 条)
  8. Special functions          (~300 条)
  9. Miscellaneous              (若干)

Level 1: 子类别（每大类内部还有细分）
  例：1. Algebraic → 1.1 Binomial products → 1.1.1 Linear → 具体规则

Level 2: 具体规则编号
```

### 3.3 MDP 形式化

- **State**: 当前待积分的数学表达式（表达式树）
- **Action**: 二层选择 — (类别, 规则编号)
- **Transition**: Wolfram Engine 执行所选规则，返回变换后的表达式
- **Reward**:
  - 终端奖励：求导验证通过 → +1
  - 步骤惩罚：每步 -0.01（鼓励短路径）
  - 中间奖励（可选）：表达式复杂度减少 → potential-based shaping
- **Episode 终止**: 积分成功 / 超过最大步数 / 无规则匹配

### 3.4 训练策略

**阶段 1：数据收集（模仿学习 warm start）**

```python
# 用 Wolfram Engine 跑 RUBI 的 ShowSteps 模式
from wolframclient.evaluation import WolframLanguageSession
session = WolframLanguageSession()
session.evaluate('<<Rubi`')

# 对 72,944 题中的每一题，记录 RUBI decision tree 的选择轨迹
# 得到：(表达式, 选中的规则编号, 变换后的表达式) 三元组序列
```

预期可生成 ~200K-500K 条 (state, action) 训练样本。

**阶段 2：模仿学习（Behavior Cloning）**

- 用阶段 1 的轨迹数据做监督学习
- 训练高层 policy（分类到 9 大类）和低层 policy（类内规则选择）
- 目标：复现 RUBI decision tree 的选择行为
- 预期准确率：80-90%（因为部分表达式特征可能不够区分）

**阶段 3：RL 强化（PPO / REINFORCE）**

- 在模仿学习基础上，用 RL 继续训练
- Reward = 最终积分成功（求导验证）
- 目标：修复模仿学习阶段的错误，发现新策略
- 关键：可能学到 decision tree 没发现的更短路径或替代规则序列

### 3.5 模型规模与计算资源

| 组件 | 规模 |
|------|------|
| 表达式编码器 | ~1-5M 参数（Tree-RNN 或 small Transformer） |
| 高层 policy | ~50K 参数（MLP: d → 256 → 9） |
| 低层 policy | ~500K-2M 参数（MLP: d → 512 → max_rules_per_category） |
| **总计** | **~2-8M 参数** |
| 训练硬件 | 单卡 RTX 3090/4090 或 A100（甚至 CPU 可行） |
| Wolfram Engine | 免费开发者许可，通过 wolframclient 调用 |

---

## 四、评测方案

### 4.1 测试集

RUBI 官方测试集：72,944 道积分题
- 下载地址：https://rulebasedintegration.org/testProblems.html
- 已有 Mathematica 格式，可直接用

### 4.2 评测指标

采用 RUBI 官方的 ABCF 评分体系：
- **Grade A**: 结果不超过最优解 2 倍大小，使用同级函数
- **Grade B**: 结果超过最优解 2 倍
- **Grade C**: 不必要地引入高级函数或复数
- **Grade F**: 未解出 / 超时 / 崩溃

### 4.3 对比 Baseline

| 系统 | 已知成绩 |
|------|---------|
| RUBI 4.16.1 (decision tree) | 99.8% Grade A |
| Mathematica 11.3 (内置) | 72.9% Grade A |
| Maple 2018 | 54.1% Grade A |
| 本项目 - 模仿学习 only | 目标 85-95% |
| 本项目 - 模仿学习 + RL | 目标 90-99% |

### 4.4 消融实验

- Flat policy (6700 维 softmax) vs Hierarchical policy
- 模仿学习 only vs 模仿学习 + RL
- 不同表达式编码器：Tree-RNN vs GNN vs Transformer
- 不同 reward shaping 策略
- 不同规则子集（按类别 ablation）

---

## 五、执行计划

### 人员分工

- **导师（理论物理背景）**: 物理直觉、理论框架设计、论文写作
- **学生（自动化大二）**: 工程实现、数据收集、训练实验

### 时间线（~12 周）

```
第 1-2 周：环境搭建
  - 安装 Wolfram Engine + wolframclient
  - 安装 RUBI 包，验证 ShowSteps 功能
  - 下载 RUBI 测试集，跑通单题端到端流程
  - 学生阅读 AlphaIntegrator 论文 + RL 基础（PPO）

第 3-4 周：数据收集
  - 用 RUBI ShowSteps 跑全部 72,944 题
  - 解析输出，提取 (表达式, 规则编号, 变换结果) 轨迹
  - 数据清洗、统计分析（规则使用频率分布、平均步数等）
  - 设计表达式特征提取方案

第 5-6 周：模仿学习
  - 实现表达式编码器（先用简单的 tree feature → MLP）
  - 实现分层 policy network
  - 训练 Behavior Cloning baseline
  - 在测试集上评测模仿学习的准确率

第 7-9 周：RL 训练
  - 搭建 RL 环境（Gym 接口封装 Wolfram Engine 调用）
  - 实现 PPO 训练循环（用 Stable-Baselines3 或 CleanRL）
  - 实现 reward function（求导验证）
  - 训练 + 调参 + 迭代

第 9-10 周：评测与分析
  - 在 RUBI 完整测试集上跑 benchmark
  - 对比所有 baseline
  - 消融实验
  - 分析 RL 发现的新策略（与 RUBI decision tree 不同的规则选择）
  - 分析失败案例

第 11-12 周：写论文
  - 目标投稿：NeurIPS ML4PS Workshop / ICML AI4Science Workshop / ICLR 2027
```

---

## 六、风险与应对

| 风险 | 概率 | 应对策略 |
|------|------|---------|
| Wolfram Engine 调用太慢（每题数秒） | 中 | 并行调用多个 session；数据收集阶段一次跑完缓存结果 |
| RUBI ShowSteps 输出格式难解析 | 中 | 先用 10 题手工验证格式，写 robust parser |
| 模仿学习准确率不够高 | 低 | 加强特征工程；尝试更强的编码器 |
| RL 训练不稳定 | 中 | 先保证模仿学习 baseline 足够好；RL 只做 fine-tuning |
| 学生 RL 经验不足 | 中 | 前两周集中学习 PPO；用成熟框架（SB3） |
| 72,944 题跑完需要较长时间 | 低 | RUBI 单题在 Mathematica 里 <1s，批量跑一晚上 |

---

## 七、关键参考文献

1. **RUBI**: Rich, A.D. and Jeffrey, D.J. (2018). "Rule-based integration: An extensive system of symbolic integration rules." JOSS.
   - 官网: https://rulebasedintegration.org/
   - GitHub: https://github.com/RuleBasedIntegration/Rubi

2. **AlphaIntegrator**: Unsal, Gehr & Vechev (2024). "AlphaIntegrator: Transformer Action Search for Symbolic Integration Proofs." arXiv:2410.02666.

3. **SIRD**: Sharma, Nagpal & Balin (2023). "Symbolic Integration Rules Dataset." NeurIPS MATH-AI Workshop.

4. **England et al.** (2024). "Symbolic Integration Algorithm Selection with ML: LSTMs vs Tree LSTMs." arXiv:2404.14973.

5. **England et al.** (2025). "Tree-Based Deep Learning for Ranking Symbolic Integration Algorithms." arXiv:2508.06383.

6. **Lample & Charton** (2019). "Deep Learning for Symbolic Mathematics." ICLR 2020. arXiv:1912.01412.

7. **SymPy RUBI 移植失败**: SymPy PR #24315 (2022), 87,806 行代码删除。教训：不要在 Python 里重新实现 RUBI 的模式匹配和执行引擎。

8. **PPO**: Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

9. **Hierarchical RL**: Option Framework — Sutton, Precup & Singh (1999). "Between MDPs and Semi-MDPs."

10. **AlphaProof**: DeepMind (2025). Nature. RL + formal verification for theorem proving.

---

## 八、预期成果

1. **一篇论文**: 投 NeurIPS/ICML Workshop 或 ICLR 主会
2. **一个开源系统**: 分层 RL 积分策略 + Wolfram Engine 接口
3. **实证发现**: RL 能否发现 RUBI decision tree 未覆盖的新规则组合
4. **方法论贡献**: 符号计算系统中用 RL 替代 hand-crafted heuristic 的范式

---

## 九、SymPy 移植 RUBI 失败的教训（附录）

SymPy 团队花了 8 年（2014-2022）、3 个 GSoC 学生，试图将 RUBI 移植到 Python，最终失败并删除了全部 87,806 行代码。核心原因：

1. **模式匹配语义不可移植**: Mathematica 的模式匹配器原生支持交换律/结合律，Python 无等价物
2. **性能灾难**: 加载 6700 条规则需 10 分钟到 1 小时以上
3. **工具函数语义差异**: `GtQ` 不确定时 Mathematica 返回 False，SymPy 返回 None → 规则选择错误 → 无限递归
4. **预编译决策树超出 Python 编译器限制**: 生成的代码太大
5. **依赖 Mathematica 流水线**: 更新规则需要商业 Mathematica 许可
6. **无人持续维护**: 代码从未集成到主 integrate()，测试排除在 CI 外，5 年间悄悄腐烂

**本项目的核心设计原则就是从这个失败中汲取的**：不移植规则，用 Wolfram Engine 做 oracle，只训练选择策略。
