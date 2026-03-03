# IntegrateZero: AlphaZero for Symbolic Integration

## 一、核心思想

利用微分-积分的正反向不对称性（微分是 trivial 的，积分是困难的），训练一个小型 Transformer 模型，以 AlphaZero 的方式学习符号积分：

- **模型自由生成表达式变换**，不依赖预定义的规则库
- **求导做裁判**（完美 verifier）
- **MCTS 做搜索**，探索多条求解路径
- **规则隐式存在于模型权重中**，无需显式枚举

```
State:   表达式 A（可能含有 ∫ 符号）
Action:  生成下一个表达式 B，使得 d/dx(B) = A
终止:    B 中不再有 ∫ 符号
Reward:  终止时给 +1
验证:    每步可通过求导验证合法性
```

---

## 二、研究动机

### 2.1 核心观察：正向简单，反向困难

对于数学表达式，很多操作是正向 trivial、反向极其复杂：

| 正向（简单） | 反向（困难） |
|---|---|
| 微分 | 积分 |
| 乘法展开 | 因式分解 |
| 矩阵乘法 | 矩阵分解 |
| 代入求值 | 方程求解 |

**关键洞察**：正向过程可以无限量、零成本地产生 (输入, 输出) 对，为学习反向过程提供完美的训练信号。

### 2.2 已有工作及局限

| 工作 | 方法 | 核心局限 |
|------|------|---------|
| RUBI (Albert Rich) | 6,700 条手工规则 + decision tree | 人工设计，不可泛化，移植困难（SymPy 8 年移植失败） |
| Lample & Charton (ICLR 2020) | Seq2seq Transformer，一步到位 f→F | 黑盒，无中间步骤，跨分布泛化差 |
| AlphaIntegrator (ETH, 2024) | 10M GPT + 47 个 SymPy 动作，监督学习 | 动作空间受限（47 个），非 RL |
| SIRD (NeurIPS 2023 WS) | Transformer 预测 24 条 SymPy 规则 | 规模小，纯监督 |
| England et al. (2024-25) | TreeLSTM 选择 Maple 子算法 | 仅算法选择，非端到端 |

### 2.3 本项目的创新点

1. **第一个 AlphaZero 式的符号积分系统**：无预定义规则，模型自由生成变换
2. **多步推理 + 每步可验证**：不是 seq2seq 一步到位，而是逐步变换，每步求导验证
3. **MCTS 搜索**：Policy network 提议变换，Value network 评估状态，MCTS 探索最优路径
4. **利用正反向不对称性**：训练数据通过正向微分无限生成
5. **不依赖任何外部规则库或 CAS 引擎**：纯 SymPy + PyTorch

---

## 三、方法

### 3.1 与 AlphaZero 的对应关系

| AlphaZero (围棋) | IntegrateZero (积分) |
|---|---|
| 棋盘状态 | 当前数学表达式（可能含 ∫） |
| 落子（有限动作空间 ~361） | 生成下一个表达式（自回归序列生成） |
| 游戏规则判断合法性 | 求导验证：d/dx(B) == A |
| 获胜 | 表达式无 ∫ 符号 且 验证通过 |
| Policy network | 给定 A，生成候选 B 的概率分布 |
| Value network | 估计 v(A) = "从 A 出发能解出来的概率" |
| MCTS | 在每步探索多个候选变换，选最优路径 |

### 3.2 求解过程示例

```
问题: ∫ x·cos(x) dx

Step 0: A₀ = ∫ x·cos(x) dx

Step 1: 模型生成 A₁ = x·sin(x) - ∫ sin(x) dx
        验证: d/dx[x·sin(x)] - sin(x) = x·cos(x) = 原始被积函数 ✓
        A₁ 仍含 ∫ → 继续

Step 2: 模型生成 A₂ = x·sin(x) + cos(x)
        验证: d/dx[x·sin(x) + cos(x)] = sin(x) + x·cos(x) - sin(x) = x·cos(x) ✓
        A₂ 无 ∫ → 成功！Reward = +1
```

### 3.3 模型架构

```
          当前表达式 A (前缀表达式 token 序列)
               │
          Transformer Encoder
               │
    ┌──────────┴──────────┐
    │                     │
Policy Head           Value Head
(Transformer          (MLP,
 Decoder,              输出标量)
 自回归生成 B)
    │                     │
    ▼                     ▼
候选表达式             v(A) ∈ [0,1]
B₁, B₂, ...Bk        "这个表达式
(采样 k 个)           多大概率能解出来"
    │
    ▼
验证: d/dx(Bᵢ) == A ?
    │
    ▼
合法的 Bᵢ 进入 MCTS
```

**模型规模**:
- Encoder: 4-6 层 Transformer, d=256-384
- Decoder: 4-6 层 Transformer, d=256-384
- Value head: 2 层 MLP
- **总参数量: ~10-30M**
- 单卡 GPU 即可训练

### 3.4 表达式表示

使用前缀表达式 (prefix notation / Polish notation)：

```
sin(x² + 1)  →  [sin, +, ^, x, 2, 1]
x·sin(x) + cos(x)  →  [+, *, x, sin, x, cos, x]
∫ sin(x) dx  →  [INT, sin, x]     ← 用特殊 token 表示未解积分
```

**词表**: ~100-200 tokens
- 运算符: +, -, *, /, ^
- 函数: sin, cos, tan, exp, log, sqrt, arcsin, arctan, sinh, cosh, ...
- 变量/常数: x, a, b, c, 0, 1, 2, 3, ..., pi, e
- 特殊: INT (积分符号), EOS

### 3.5 验证器

```python
import sympy

def verify_step(A_integrand, B_expr, x=sympy.Symbol('x')):
    """验证 d/dx(B) == A"""
    dB = sympy.diff(B_expr, x)

    # 方法 1: 符号化简
    if sympy.simplify(dB - A_integrand) == 0:
        return True

    # 方法 2: 数值抽样（符号化简失败时）
    import random
    for _ in range(20):
        x_val = random.uniform(-5, 5)
        try:
            diff = abs(float(dB.subs(x, x_val) - A_integrand.subs(x, x_val)))
            if diff > 1e-8:
                return False
        except:
            continue
    return True

def is_terminal(expr):
    """检查表达式是否还含有 ∫ 符号"""
    return not expr.has(sympy.Integral)
```

### 3.6 训练流程

#### Phase 1: 数据生成 + 监督预训练

```python
def generate_training_data(num_samples, max_depth):
    """利用正向微分的简单性，无限生成训练数据"""
    data = []
    for _ in range(num_samples):
        # 1. 随机生成表达式 F(x)（这是"答案"）
        F = random_expression(max_depth=max_depth)
        # 2. 求导得到 f(x)（trivial）
        f = sympy.diff(F, x)
        # 3. 训练对: 输入 f，目标输出 F
        data.append((f, F))
        # 4. 也可以记录微分的中间步骤，反转作为多步训练数据
    return data
```

**课程学习**: 从 max_depth=2 开始，逐步增加到 5, 7, 10...

用生成的 (f, F) 对训练 policy（生成 F）和 value（预测是否能解出来）。

#### Phase 2: Self-Play RL (AlphaZero 式)

```
repeat:
    1. 从题库中采样一道积分题 f(x)
    2. 用 MCTS + 当前 policy/value 搜索解法:
       - 在每个 state，policy 采样 k 个候选 B
       - 验证合法性（求导）
       - value 评估每个合法 B
       - MCTS 选择最优路径
    3. 如果解出来 → 记录整条轨迹 (state, action, reward)
       如果超时  → 记录为失败
    4. 用轨迹更新 policy 和 value:
       - policy: 向 MCTS 选择的动作学习
       - value: 向实际结果（成功=1/失败=0）学习
    5. 定期增加题目难度（课程学习）
```

#### Phase 3: 自课程 + 难度自动调节

```
训练循环:
  生成一批 F(x)（复杂度 d）→ 求导得 f(x) → 让模型积分
  → 成功率 > 80%: 增加 d
  → 成功率 < 30%: 降低 d
  → 保持在模型能力边缘训练（最大学习效率）
```

---

## 四、评测方案

### 4.1 测试集

| 测试集 | 来源 | 题目数 |
|--------|------|-------|
| RUBI 官方测试集 | rulebasedintegration.org | 72,944 |
| Lample & Charton FWD 测试集 | arXiv:1912.01412 | ~5,000 |
| Lample & Charton BWD 测试集 | arXiv:1912.01412 | ~5,000 |
| 自生成跨分布测试集 | 不同于训练分布的表达式 | ~10,000 |
| 本科微积分教科书习题 | 手工收集 | ~500 |

### 4.2 对比 Baseline

| 系统 | 类型 |
|------|------|
| RUBI 4.16.1 | 规则系统 (6,700 规则) |
| Mathematica integrate() | 商业 CAS |
| SymPy integrate() | 开源 CAS |
| Lample & Charton (2019) | Seq2seq Transformer |
| AlphaIntegrator (2024) | 监督学习 + 47 动作 |
| IntegrateZero (本项目) | AlphaZero 式 RL |

### 4.3 评测指标

- **求解率**: 在规定步数/时间内成功积分的比例
- **平均步数**: 求解所用的变换步数
- **跨分布泛化**: 在训练分布之外的测试集上的表现
- **模型大小 vs 性能**: 不同参数量的 scaling curve
- **搜索效率**: MCTS 节点数 vs 成功率

### 4.4 消融实验

- MCTS vs 纯 greedy (无搜索)
- 监督预训练 vs 纯 RL (从零开始)
- 课程学习 vs 固定难度
- 前缀表达式 vs 树编码 vs LaTeX
- Value network 的作用（有 vs 无）

---

## 五、执行计划

### 人员分工

- **导师（理论物理）**: 理论框架、物理直觉、论文写作
- **学生（自动化大二）**: 工程实现（PyTorch + SymPy）、训练实验

### 时间线（~12 周）

```
第 1-2 周: 基础设施
  ├── 实现表达式随机生成器（SymPy，可控复杂度）
  ├── 实现前缀表达式 tokenizer（表达式 ↔ token 序列）
  ├── 实现验证器（SymPy 求导 + 数值检查）
  ├── 生成初始数据集（100K-1M 对）
  └── 学生阅读: AlphaZero 论文 + AlphaIntegrator 论文

第 3-4 周: 监督学习 Baseline
  ├── 实现 Encoder-Decoder Transformer（~10M 参数）
  ├── 实现 Value Head
  ├── 训练 seq2seq baseline（一步到位，f → F）
  ├── 训练多步 baseline（逐步变换）
  └── 在简单积分上验证模型能学会基本积分

第 5-7 周: MCTS + Self-Play RL
  ├── 实现 MCTS（参考 AlphaZero 实现）
  ├── 实现 self-play 训练循环
  ├── 实现课程学习（自动难度调节）
  ├── 训练 + 调参
  └── 监控: 求解率、步数、搜索效率

第 8-9 周: 评测
  ├── 在 RUBI 测试集子集上评测
  ├── 对比 SymPy integrate()
  ├── 对比 Lample & Charton baseline
  ├── 消融实验
  └── 分析: 模型学到了什么"规则"？（注意力可视化）

第 10-12 周: 论文
  ├── 写论文
  ├── 目标: NeurIPS/ICML Workshop 或 ICLR 2027 主会
  └── 开源代码
```

### 硬件需求

| 组件 | 需求 |
|------|------|
| 训练 | 单卡 RTX 3090/4090 或 A100 |
| 数据生成 | CPU 即可（SymPy 符号计算） |
| MCTS 搜索 | CPU + GPU 混合 |
| 验证 | CPU（SymPy 求导） |

---

## 六、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 模型生成大量非法表达式 | 高 | 中 | 树结构解码约束合法性；非法表达式给负 reward，模型快速学会避免 |
| 搜索空间太大，MCTS 效率低 | 中 | 高 | 课程学习从简单开始；增加 value network 精度来剪枝 |
| 复杂积分步数太多，奖励稀疏 | 中 | 高 | 中间 reward（表达式变简单）；每步验证提供密集信号 |
| 监督预训练的数据分布偏差 | 中 | 中 | 多种生成策略混合；RL 阶段修正偏差 |
| SymPy 验证器性能瓶颈 | 低 | 中 | 优先数值验证（快），符号验证作为后备 |
| 学生 RL 经验不足 | 中 | 中 | 用成熟的 AlphaZero 开源实现作为起点 |

---

## 七、预期成果

1. **IntegrateZero 系统**: 第一个 AlphaZero 式的符号积分系统
2. **论文**: 投 NeurIPS/ICML Workshop 或 ICLR 2027
3. **开源代码**: 完整的训练 + 推理代码
4. **实证发现**:
   - 模型是否能隐式学会标准积分技巧（u-substitution, 分部积分等）？
   - RL + MCTS 能否发现人类/RUBI 没有的求解路径？
   - 模型在不同复杂度上的 scaling behavior 是什么样的？

---

## 八、关键参考文献

1. **AlphaZero**: Silver, D. et al. (2018). "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science.
2. **AlphaIntegrator**: Unsal, Gehr & Vechev (2024). "AlphaIntegrator: Transformer Action Search for Symbolic Integration Proofs." arXiv:2410.02666.
3. **Lample & Charton**: (2019). "Deep Learning for Symbolic Mathematics." ICLR 2020. arXiv:1912.01412.
4. **RUBI**: Rich, A.D. and Jeffrey, D.J. (2018). "Rule-based integration: An extensive system of symbolic integration rules."
5. **SIRD**: Sharma, Nagpal & Balin (2023). NeurIPS MATH-AI Workshop.
6. **England et al.** (2024). "Symbolic Integration Algorithm Selection with ML." arXiv:2404.14973.
7. **England et al.** (2025). "Tree-Based Deep Learning for Ranking Symbolic Integration Algorithms." arXiv:2508.06383.
8. **AlphaProof**: DeepMind (2025). Nature. RL for theorem proving.
9. **DreamCoder**: Ellis et al. (2020). "DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning."
10. **PPO**: Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

---

## 九、与前期方案（RUBI RL）的关系

本方案是在调研 RUBI 系统和 SymPy 移植失败经验后的进化版本。

| 对比维度 | 前期方案 (RUBI RL) | 本方案 (IntegrateZero) |
|---|---|---|
| 动作空间 | RUBI 6,700 条规则（显式） | 自由生成表达式（隐式） |
| 外部依赖 | Wolfram Engine | 仅 SymPy |
| 规则来源 | 人工总结 (RUBI) | 模型自己学会 |
| 核心算法 | 分层 PPO | AlphaZero (MCTS + Policy + Value) |
| 工程复杂度 | 高（需要 Wolfram Engine 接口） | 低（纯 PyTorch + SymPy） |
| 学术叙事 | "RL 替代 decision tree" | "AlphaZero for Mathematics" |

前期方案文档保留在 `hierarchical-rl-integration-project.md` 供参考。

---

## 十、项目名称

**IntegrateZero** — 致敬 AlphaZero，Zero 代表"从零开始学习积分，不依赖人工规则"。
