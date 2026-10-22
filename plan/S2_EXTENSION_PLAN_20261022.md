# S2 扩展计划（2026-10-22）：从"哪个部件带来精度"到"什么时候需要状态"

> **用途**：在 S1（本仓库现有的论文与 361 个运行）的基础上加入 4 组新实验，合成一篇论文。
> **这是计划，不是冻结协议。** 第 3 节的新假设要在开跑前另存为 `protocol/PROTOCOL_S2_FROZEN_<日期>.md`，记录 SHA-256 后再跑，写法与 S1 协议相同（偏离逐条追加）。
> **附录 A 的试点数字来自另一套实现**（1 层 Mamba4Rec、Mamba4Rec 原始初始化、单个种子、部分数据集不同），只用来决定"做什么"，**不进论文**。论文里的所有新数字都必须用本仓库的外壳（`src/s1_lib.py`）、7 种混合器、调参网格和种子重跑。两套实现的结论在 Amazon 上甚至方向不同（试点：SASRec 比 Mamba4Rec 高 12%；S1：Mamba 比 SASRec 高 5%，同一份 Baby 数据），不能混用。

---

## 1. 合并后的论文结构

| RQ | 问题 | 证据 | 状态 |
|---|---|---|---|
| RQ1 | 选择性带来多少增益？ | H1、H5 | S1 已完成 |
| RQ2 | 近因、最后一个物品能解释多少？ | H2、H3、两种安慰剂、打乱训练 | S1 已完成 |
| RQ3 | 什么样的数据需要状态？ | Amazon（短）vs ML-1M + **ML-20M**（长） | **E2 新增** |
| RQ4 | 同一数据集里，哪些用户需要状态？ | 历史长度三分位（S1 已有）+ **证据距离切片** | **E1 新增** |
| RQ5 | 每种混合器能学会哪种序列结构？ | **可控合成能力任务** | **E3 新增** |
| RQ6 | 后果：隐式注意力解释在哪里可信？ | **解释可信度** | **E4 新增** |
| 稳健性 | d=128、Electronics、效率；**GRU 输出层归一化**；（可选）Δ 初始化、更长耐心 | E5–E7 | 部分新增 |

**主线**：(1) 短历史上，近因和最后一个物品解释了大部分精度（S1）；(2) 状态只在"证据离当前位置很远"时才有用——在数据集层面（ML-1M、ML-20M）和样本层面（证据距离切片，控制历史长度后仍成立）都如此。ML-1M 上的探索性结果显示，连选择性的作用也集中在远程证据用户上（见 E1）；(3) 合成任务说明为什么：各混合器能学会的结构不同；(4) 后果：基于状态的解释只在需要状态的地方忠实。

**标题候选**：
- *When Does the State Matter? Components, Data Regimes, and Capabilities of Mamba-Based Sequential Recommenders*
- 或保留原题，加副标题 *…, and When the State Matters*

---

## 2. 新增实验

### E1 证据距离切片（不需要训练，最先做）

**目的**：S1 的历史长度三分位把"历史长"和"证据远"混在一起。这里直接找出"与目标最相关的历史物品离当前位置很远"的用户，检验状态的优势是否集中在这些用户上，并在每个长度三分位内部重复，排除长度的影响。

**定义**（与试点相同）：
- 关联度：物品–物品 PMI，用训练部分（`data["seqs"]`，每个用户最后 L 个物品去重）的用户级共现计算。
- 每个测试用户的历史 = `seqs[u] + [vmap[u]]` 的最后 L 个（与 `_ranks` 完全一致），目标 = 测试物品。
- `far` = PMI 最大的历史物品离当前位置的步数（0 = 最近一个）；`near` = 最近 5 个物品里的最大 PMI。
- **远程切片**：`far ≥ 20` 且 `near` < 全体 `near` 的中位数（近处没有强证据）。
- **近程切片**：`far < 5`（最强证据就在最近 5 步内）。
- 每个切片要求 ≥ 200 个用户；不够就只报告人数，不做检验。

**做法**：直接读 `results/runs/final_<ds>_<model>_*_s{0..4}.json` 里的 `test.ranks`。排名顺序就是 `load_ds(ds)["test"][0]` 的用户顺序，排名从 0 开始、并列按悲观方式计（`NDCG = 1/log2(r+2)`，r < 10）。代码见附录 B。

**数据集**：ML-1M（主）、ML-20M（E2 之后）。Amazon 中位长度只有 6，远程切片几乎为空，只报告人数。

**输出**：每个混合器相对 Mamba 的 NDCG@10 在两个切片上的差（5 个种子配对）；在 3 个长度三分位内部分别重复。

**已有的探索性结果（2026-10-22，用本仓库已发布的 ML-1M `final` 运行，附录 B 代码原样运行）**：

ML-1M 切分用 `prep_ml1m.py` 重建，3 个文件的哈希与 `SHA256SUMS.txt` 一致。远程切片 1685 人，近程切片 1721 人。Mamba 自身 NDCG@10：远程 0.0937，近程 0.2910，全体 0.1875。下表是相对 Mamba 的 NDCG@10 差，5 个种子平均；"差值"= 远程 − 近程，单位百分点：

| 混合器 | 远程证据用户 | 近程证据用户 | 差值（5 个种子） | 3 个长度三分位内（远程 / 近程） |
|---|---|---|---|---|
| `last` | −32.6% | −22.1% | −10.5（−15.5 ~ −8.4） | — |
| `mean` | −25.8% | −16.7% | −9.1（−15.3 ~ −5.5） | −23.4/−13.4，−19.2/−21.7，−35.3/−16.6 |
| `ema` | −20.1% | −1.2% | **−18.9（−22.0 ~ −15.8）** | −18.1/−3.1，−21.1/−1.8，−19.8/+3.8 |
| `lti` | −6.6% | +0.9% | **−7.5（−13.4 ~ −3.0）** | −6.4/+2.6，−8.0/−0.9，−4.8/+0.1 |
| `sasrec` | −18.2% | −8.5% | −9.8（−13.7 ~ −4.3） | −19.2/−6.3，−16.5/−9.4，−18.8/−11.5 |
| `gru` | −26.8% | −9.7% | −17.1（−21.0 ~ −14.9） | — |

（三分位切点 56 / 155；远程 / 近程人数：短 299/788，中 662/552，长 724/381。）

**读法**：
- 所有混合器在 5 个种子里都是在"远程证据"用户上输给 Mamba 更多。
- 近因滤波（EMA）在近程用户上与 Mamba 持平（−1.2%），在远程用户上落后 20%，3 个长度三分位里都一样，所以不是历史长度造成的。
- **新发现**：S1 中选择性整体只有 +0.9%（H5 不成立），但它对远程证据用户约有 +6.6%（`lti` −6.6%），对近程用户约 −0.9%，平均后就看不出来了。这可以改写 S1 的结论：选择性的作用集中在"证据在远处"的用户上。

**注意**：这些数字是在 S2 协议冻结**之前**看到的，ML-1M 上的 E1 只能算探索性结果。事先写好的 H6a（`mean`，≥ 10 个百分点）在 ML-1M 上没有达到（9.1）。验证性检验放到还没看过的 ML-20M 上（第 3 节 H6），并在论文中披露 H6b–H6d 是看过 ML-1M 后才提出的（与 S1 的 H2 来自试点同样处理）。

### E2 ML-20M：第二个长历史数据集

**目的**：S1 的"长历史"结论只靠 ML-1M 一个数据集，这是最容易被质疑的地方。

**数据准备**（新脚本 `data/prep_ml20m.py`，仿照 `prep_ml1m.py`）：
- 来源：GroupLens 官方 `ml-20m.zip`（官网 MD5 `cd245b17a1ae2cc31bb14903e1204af3`）。
- 全部评分视为隐式交互；每个用户按时间戳排序，同一时间戳保持文件顺序。
- 随机抽 25,000 个用户：`np.random.RandomState(0).choice(sorted_user_ids, 25000, replace=False)`，保留他们的全部交互。全量 138,493 个用户在笔记本上太重。
- 留一法：最后一个 → `test.csv`，倒数第二个 → `val.csv`，其余 → `train.csv`；列与 Amazon 相同（`user_id,item_id,rating,timestamp`）。
- 把哈希加入 `data/SHA256SUMS.txt`，`verify_splits.py` 加入 ml20m。

**代码改动**（只加数据集名，不改计算）：`s1_lib.py` 的 `PATHS` 加 `"ml20m"`，`MAXLEN` 加 `"ml20m": 200`。批大小 128（同 ML-1M）。物品数约 2.7 万，现有的分块交叉熵和自动评测批大小已经能处理。

**调参与种子**：
- 首选：ml20m 上完整网格（lr {1e-3, 3e-3} × dropout {0.2, 0.5}，种子 0），再跑种子 0–4。
- 时间不够时：沿用 ML-1M 选出的配置（类似 Electronics 沿用 Sports），**必须在开跑前写进 S2 协议**，种子至少 0–2。

**时间**：先计时一个运行再排队。试点（另一套代码、batch 32）每个模型约 25–30 分钟；估计全部 10–25 GPU 小时。S1 记录过这台机器在 ML-1M 上蓝屏（0x113），沿用 S1 的措施：批大小 128、`S1_MEMFRAC=0.7`、单进程（`s1_runall.py`）。

**假设**：H7、H8（第 3 节）。所有 S1 的分层、安慰剂会自动生成；ML-20M 跑完后再做一次 E1。

### E3 可控合成能力任务

**目的**：解释 E1/E2 的"为什么"——每种混合器能学会哪种序列结构。思路借鉴语言模型的 Zoology / MQAR（S1 已引用 [49]），但任务是推荐式的。

**任务**（数据种子固定为 0；500 个物品、6000 个用户、每人 50 个物品的训练流，验证目标 = 第 51 个，测试目标 = 第 52 个）：
- `markov1`（对照）：下一个物品 ~ 当前物品的 4 个固定后继（Dirichlet(1) 权重），概率 0.9；否则均匀随机。
- `markov2`：下一个物品 ~ **前两个物品组合**的 4 个后继（3000 个哈希表），概率 0.9；否则均匀随机。
- `period`：每个用户有 3 个"常购品"，周期 5 / 7 / 9；到期就出现，否则是随机填充；验证、测试位置构造成正好有常购品到期。
- 不用试点的 `recall`：试点中没有任何模型学会它，而且对比探针"获胜"是因为全局计数表包含了用户自己的配对，属于设计漏洞。`recall2`（回忆事件概率 0.3）可以放附录。

**与 S1 代码的对接**：训练流 = `seqs[u]`，验证目标 = `vmap[u]`，测试目标 = test；测试历史 = 训练流 + 验证目标，正好就是 `_ranks` 的约定。**不屏蔽历史物品**（重复出现正是任务的一部分）：`seen_val`、`seen_test` 设为空矩阵。生成器与数据字典见附录 C。

**`period` 的指标必须用 HR@1，并加"用户频次探针"**：只要知道用户的 3 个常购品是哪几个、按出现次数排，NDCG@10 就能到约 0.63（目标常购品通常排第 2），根本不需要懂"哪个到期"。试点数据证实了这一点：

| 试点 period 结果 | NDCG@10 | HR@1 |
|---|---|---|
| GRU | 0.608 | 0.001（只记住了常购品是哪几个，完全不懂到期） |
| 1 层 Mamba（加长训练） | 0.697 | 0.278 |
| 2 层 Mamba（加长训练） | 0.778 | 0.425 |

因此 `period` 以 HR@1 为主指标，从已保存的 `test.ranks` 离线计算 `(ranks < 1).mean()`。基线加"用户频次探针"：按用户自己历史中的出现次数给物品打分。`markov*` 任务仍以 NDCG@10 为主指标。合成数据里没有冷门物品，所以 `by_pop` 分层为空，会出现 "Mean of empty slice" 警告，可以忽略。

**模型与预算**：
- 7 种混合器 + 探针（近因加权一阶转移计数，不学习）+ 用户频次探针（`period`）+ 理论上限（马尔可夫任务可算）。
- 每个（任务，混合器）在种子 0 上跑完整网格，再跑种子 0–2。
- **加长训练**：最多 300 轮、耐心 15 次验证（= 30 轮）。试点里慢学习很常见：1 层 Mamba 在 period 上默认预算停在 0.144，加长后到 0.697；2 层 Mamba 在 markov2 上第 298 轮才最好。新写 `src/s2_synth.py`，直接调用 `S.train(..., patience=15)` 并设 `S1_MAXEP=300`。

**时间**：3 个任务 × 7 个混合器 × 约 6 个运行，每个 2–5 分钟，合计约 4–8 GPU 小时。

**假设**：H9、H10。**H10 决定"选择性"那条论点**：试点中关掉选择性的版本在 period 上只有 0.017，但那是默认预算；完整 Mamba 加长后从 0.144 升到 0.697，而关掉选择性的版本从来没有用加长预算跑过，**所以"选择性有能力、但真实基准不需要它"目前只是假设**。

### E4 解释可信度：隐式注意力（HA）在哪里忠实

**目的**：检验基于状态的解释（Mamba 的隐式注意力，Ali et al. [51]）是否只在需要状态的数据上忠实。

**模型**：**1 个块**的 `mamba`（即 Mamba4Rec 的默认层数；S1 用 2 个块是为了和 SASRec 深度一致）。这样 HA 可以精确分解。2 个块的版本用逐块"展开"（rollout）作为次要分析，先写进协议。

**需要保存检查点**：现有 `s1_run.py` 不存权重。加一个阶段 `explain`（`blocks=1`），设 `S1_SAVE=1` 时 `torch.save(model.state_dict())`；数据集 baby、sports、clothing、ml1m、ml20m，种子 0–2，配置沿用该数据集 `mamba` 的调参结果（写进协议）。

**评测**：
- 每个数据集随机 1000 个、历史 ≥ 3 的测试用户。被解释量 `f = <h_T, e_{i*}>`，`i*` = 屏蔽历史后的 top-1 推荐。
- 真值：逐个删掉历史物品后 `f` 的变化（留一删除）。
- 方法：HA（精确的冻结系数分解，附录 D）、积分梯度（20 步，零基线，作用在 LayerNorm 后的嵌入上）、梯度×输入、近因规则。
- 指标：每个用户与真值的 Spearman 相关取平均；top-1 一致率；耗时。

**假设**：H11。

### E5 GRU 参照的输出层归一化（稳健性）

S1 的 GRU4Rec 没有块结构，GRU 输出直接打分。试点里 GRU 输出后加一层 LayerNorm（嵌入 dropout → GRU → LayerNorm），在同一份 Baby 数据上 NDCG@10 = 0.0278（种子 0）。这接近 S1 最好的 EMA（0.0286），明显高于 S1 的 GRU4Rec（0.0233）。在试点的另外 3 个 Amazon 数据集上它也排第一。

加一个变体 `gru_ln`，在 3 个 Amazon 数据集上跑完整网格 + 5 个种子（约 1–2 GPU 小时）。如果 `gru_ln ≥ ema`，S1 里"近因滤波是最准的混合器"这句话要改写。也可以把它写成一个发现：简单的循环结构同样够用。

### E6（可选）Δ 初始化范围

S1 的 Δ 按对数均匀分布初始化在 [1e-3, 0.1]。试点中 Δ 改 200 倍，5 个数据集的 NDCG@10 只变 1.4–4.6%；ML-20M 上换成长记忆初始化只差 +0.8%。可在附录加：baby、ml1m 上把 Δ 范围乘 {0.1, 10}，种子 0–2。

### E7（可选）ML-1M 更长的耐心

试点里把耐心放到 30 轮后，ML-1M 上三个模型都还有提高（SASRec +3.7%、Mamba +2.4%、GRU +2.5%），Mamba 对 SASRec 的领先从 11.9% 变为 10.5%。S1 在 ML-1M 上多数混合器选中了更大的学习率（3e-3），平均最佳轮次 37–64（表 9），离 200 轮上限很远，问题较小。可选：ml1m 上 `mamba`、`sasrec`、`mean` 用耐心 15 次验证、种子 0–2。

---

## 3. 新假设（开跑前冻结；统计方法同 S1）

统计：按（数据集，种子）配对；配对相对差的 95% 百分位 bootstrap 区间（20,000 次）+ 精确 Wilcoxon；同一族做 Holm 校正。种子数 < 5 时只按区间和方向判定，并在文中说明。

| 编号 | 实验 | 假设 | 判据 |
|---|---|---|---|
| H6a | E1 | 状态的优势集中在远程证据用户（看 ML-1M 之前写的原判据，不改） | `mamba` 相对 `mean` 的 NDCG@10 优势，远程切片比近程切片大 ≥ 10 个百分点，差值的 95% 区间下界 > 0；两个切片各 ≥ 200 人。ML-1M 探索性结果为 9.1，未达到；在 ML-20M 上正式检验 |
| H6b | E1（ML-20M） | 近因滤波只在远程证据用户上落后（看过 ML-1M 后提出，需披露） | `ema` 相对 `mamba` 的劣势，远程比近程大 ≥ 10 个百分点，差值区间下界 > 0（ML-1M 探索性结果：18.9） |
| H6c | E1（ML-20M） | 选择性的作用集中在远程证据用户（同上，需披露） | `lti` 相对 `mamba` 的劣势，远程比近程大，差值区间下界 > 0（ML-1M 探索性结果：7.5，5 个种子都为正） |
| H6d | E1（ML-20M） | H6b、H6c 不是历史长度造成的 | 在 3 个长度三分位中至少 2 个内部方向相同（切片 < 100 人的三分位不计）（ML-1M：3/3） |
| H7 | E2 | ML-20M 上序列建模有用（H4 的重复） | `mamba` 与 `sasrec` 都比 `mean` 高 ≥ 5%，区间下界 > 0 |
| H8 | E2 | ML-20M 上选择性有用（H5 的重复） | `mamba` 对 `lti` 的配对增益区间下界 > 0 |
| H9 | E3 | 能力因结构而异 | `markov2` 上 {`mamba`, `lti`} 中较好者的 NDCG@10 比 `sasrec` 高 ≥ 20%；`period` 上 {`mamba`, `lti`} 中较好者的 HR@1 比 `sasrec` 和用户频次探针都高 ≥ 0.10（绝对值）；均为种子 0–2 平均，且 3 个种子方向一致。对照：`markov1` 上所有学习型混合器的 NDCG@10 相差 < 5% |
| H10 | E3 | 选择性在周期结构上有能力 | `period` 上 `mamba` 的 HR@1 比 `lti` 高 ≥ 0.10（绝对值），3 个种子方向一致 |
| H11 | E4 | 解释可信度取决于数据 | 在 ml1m、ml20m 上 HA 的 Spearman ≥ 积分梯度，且比近因规则高 ≥ 0.1；在每个 Amazon 数据集上近因规则 > HA |

H6–H11 中任何一条不成立都照实写进论文，不因结果调整网格、种子或判据。

---

## 4. 执行顺序与预算（一次只跑一个 GPU 任务）

| 步骤 | 内容 | GPU | 估计 |
|---|---|---|---|
| 0 | 写 S2 协议，冻结并记录哈希，提交 | 否 | 半天 |
| 1 | E1：ML-1M 已完成（探索性，见 E1）；Amazon 只报人数 | 否 | 1 小时 |
| 2 | E5：`gru_ln`，3 个 Amazon 数据集，网格 + 5 个种子 | 是 | 1–2 小时 |
| 3 | E3：合成任务，3 个任务 × 7 个混合器 | 是 | 4–8 小时 |
| 4 | E2：ML-20M 准备、调参、最终运行 | 是 | 10–25 小时，分几晚跑 |
| 5 | E1 在 ML-20M 排名上重做 | 否 | 1 小时 |
| 6 | E4：1 块 Mamba 存检查点 + HA / 积分梯度 / 留一删除 | 是 | 编码 1–2 天 + 3–5 GPU 小时 |
| 7 | `analysis/s2_analyze.py` + 图表 | 否 | 1 天 |
| 8 | 改写论文 | — | 3–5 天 |

---

## 5. 论文改动清单

- **摘要 / 引言**：贡献点从"部件归因"扩展为"部件 → 数据 → 样本 → 能力 → 解释"。
- **相关工作**：加入 Petrov et al.（RecSys 2026，arXiv 2608.19833，"Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?"）。这是数据集层面最接近的工作，必须引用并说明区别：它只到数据集层面，没有拆部件、没有切样本。再加合成能力基准（Zoology [49] 已引用；其他候选引用前核对）。**所有新引用投稿前逐条核对。**
- **方法**：新增 3 个小节（证据距离切片、合成任务、HA 分解）。
- **结果**：新增 RQ3–RQ6 小节；ML-20M 并入表 2 / 图 2。
- **讨论**：选择性的说法取决于 H10。成立就写成"选择性有能力、但现有基准不需要"；不成立就保持"没有可靠增益"。
- **附录 A**：S2 协议与偏离；README 加新脚本。
- **篇幅**：合成任务和解释的细节放附录，正文各保留一张图或一张表。

---

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| 2 块外壳下合成任务分不开混合器 | 照实报告，H9 不成立；同时报告 1 块结果作为对照（事先写明） |
| H10 不成立 | 讨论保持 S1 的说法，不写"选择性有能力" |
| ML-20M 太慢或蓝屏 | 先计时一个运行；沿用 S1 的批大小与显存上限；必要时种子减到 0–2（事先写明） |
| 2 块 HA 展开不忠实 | 主分析用 1 块 Mamba（Mamba4Rec 默认），2 块只作次要分析 |
| `gru_ln` 超过 EMA | 改写"近因滤波最准"的表述，作为新发现报告 |

---

## 附录 A　试点数字（另一套实现，仅供参考，不进论文）

试点代码：1 层 Mamba4Rec（d=64、d_state=32、Mamba4Rec 原始初始化）、SASRec、GRU（嵌入 dropout → GRU → LayerNorm），全量交叉熵，屏蔽历史全排序，**种子 0**。数据集：Amazon Scientific、Pantry、Instruments、Baby，ML-1M，ML-20M（25,000 个用户子集）。

| 实验 | 结果 |
|---|---|
| 证据距离切片（ML-1M） | 远程切片 1685 人、近程切片 1721 人。去掉 SSM（只留门控卷积）：远程 −13.2%，近程 −1.4%；关掉选择性：远程 −8.6%，近程 −0.4%；SASRec 相对 Mamba4Rec：远程 −20.7%，近程 −7.2% |
| 去掉 SSM（数据集层面） | 4 个 Amazon 数据集 −1.5% ~ +0.7%；ML-1M −7.9%；ML-20M −11.1%（0.1194 vs 0.1343） |
| Δ 时间尺度 | Δ 改 200 倍，5 个数据集 NDCG@10 只变 1.4–4.6%；ML-20M 上长记忆初始化 +0.8% |
| 解释可信度（与留一删除的 Spearman，1000 个用户） | ML-20M：HA 0.573、积分梯度 0.422、梯度×输入 0.268、近因 0.166；ML-1M：0.600 / 0.429 / 0.295 / 0.187；Amazon 4 个数据集：HA 0.25–0.34、积分梯度 0.32–0.42、近因 0.50–0.62。长序列上 HA 耗时约为积分梯度的 1/8。HA 精确分解相对误差 2.6e-7 – 5e-7 |
| 合成任务 markov1 | 所有模型 ≈ 0.71（理论上限 0.756，探针 0.643） |
| 合成任务 markov2 | SASRec 0.014（加长训练 0.022），1 层 Mamba 0.246，关掉选择性 0.228，GRU 0.047，2 层 Mamba（加长）0.361，探针 0.022，上限 0.755 |
| 合成任务 period | NDCG@10：SASRec 0.036（加长 0.033），1 层 Mamba 0.144（加长 0.697），2 层 Mamba（加长）0.778，关掉选择性 0.017（**仅默认预算**），GRU 0.608，探针 0.035。**HR@1**：GRU 0.001、1 层 Mamba（加长）0.278、2 层 Mamba（加长）0.425、SASRec ≤ 0.009 |
| 合成任务 recall / recall2 | recall：所有模型 ≤ 0.027，低于探针 0.050（任务有设计漏洞）；recall2：GRU 0.194、SASRec 0.175、2 层 Mamba 0.134、探针 0.088、1 层 Mamba 0.023、关掉选择性 0.026 |
| ML-1M 更长耐心（30 轮、最多 300 轮） | SASRec 0.1681 → 0.1744（第 299 轮仍在涨）；Mamba4Rec 0.1882 → 0.1928；GRU 0.1626 → 0.1667。Amazon 上最佳轮次 10–35，加长训练结果不变 |
| GRU（带输出 LayerNorm） | Baby 0.0278、Scientific 0.0743、Pantry 0.0237、Instruments 0.0862；在 4 个 Amazon 数据集上都排第一（SASRec 0.0268 / 0.0723 / 0.0234 / 0.0849） |

---

## 附录 B　E1 切片代码（基于 `s1_lib`，在仓库根目录运行）

```python
import glob, json, sys
import numpy as np, scipy.sparse as sp
sys.path.insert(0, "src"); import s1_lib as S

ds = "ml1m"
data = S.load_ds(ds); L, nI = data["L"], data["nI"]

# item-item co-occurrence over each user's training part (last L items, de-duplicated)
rows, cols = [], []
for s in data["seqs"]:
    tr = np.unique(np.asarray(s[-L:]))
    rows.append(np.repeat(tr, len(tr))); cols.append(np.tile(tr, len(tr)))
C = sp.csr_matrix((np.ones(sum(map(len, rows)), np.float32),
                   (np.concatenate(rows), np.concatenate(cols))), shape=(nI + 1, nI + 1))
cnt = np.asarray(C.diagonal()).ravel(); N = len(data["seqs"])

us, ts = data["test"]                                   # same order as test["ranks"]
far, near, hlen = [], [], []
for u, t in zip(us, ts):
    h = np.asarray(data["seqs"][u] + [data["vmap"][u]])[-L:]   # the test history used by _ranks
    co = np.asarray(C[h, t].todense()).ravel()
    pmi = np.log((co * N + 1e-9) / (cnt[h] * max(cnt[t], 1) + 1e-9)); pmi[co == 0] = -np.inf
    back = len(h) - 1 - np.arange(len(h))                       # 0 = most recent item
    far.append(back[int(np.argmax(pmi))] if np.isfinite(pmi).any() else -1)
    near.append(pmi[back < 5].max()); hlen.append(len(data["seqs"][u]) + 1)
far, near, hlen = map(np.array, (far, near, hlen))
med = np.median(near[np.isfinite(near)])
long_m = (far >= 20) & (near < med)       # strongest evidence >= 20 steps back, no strong evidence nearby
short_m = (far >= 0) & (far < 5)          # strongest evidence among the last 5 items
print("users: long", long_m.sum(), "short", short_m.sum())

nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2), 0.0)
R = {}
for f in glob.glob(f"results/runs/final_{ds}_*_s*.json"):
    r = json.load(open(f)); R[(r["model"], r["seed"])] = nd(np.asarray(r["test"]["ranks"]))
for m in ["last", "mean", "ema", "lti", "sasrec", "gru"]:
    d_long = [R[(m, s)][long_m].mean() / R[("mamba", s)][long_m].mean() - 1 for s in range(5)]
    d_short = [R[(m, s)][short_m].mean() / R[("mamba", s)][short_m].mean() - 1 for s in range(5)]
    print(f"{m:6s} vs mamba: long {np.mean(d_long):+.1%}  short {np.mean(d_short):+.1%}")
# H6b: repeat inside each history-length tertile (cuts = test["len_cuts"] of any run; S1 uses len(hist) incl. val item)
```

## 附录 C　E3 合成数据（与试点生成器相同的随机数顺序）

```python
import numpy as np, scipy.sparse as sp

def make_synth(task, N=500, U=6000, T=50, data_seed=0):
    rng = np.random.RandomState(data_seed)
    streams, vt, tt = [], [], []
    if task in ("markov1", "markov2"):
        H = N + 1 if task == "markov1" else 3000
        succ = rng.randint(1, N + 1, (H, 4)); w = rng.dirichlet(np.ones(4), H)
        idx = (lambda a, b: b) if task == "markov1" else (lambda a, b: (a * 1009 + b * 31) % H)
        def step(a, b):
            k = idx(a, b)
            return succ[k, rng.choice(4, p=w[k])] if rng.rand() < 0.9 else rng.randint(1, N + 1)
        for u in range(U):
            s = list(rng.randint(1, N + 1, 2))
            while len(s) < T + 2:
                s.append(step(s[-2], s[-1]))
            streams.append(s[:T]); vt.append(s[T]); tt.append(s[T + 1])
    elif task == "period":
        for u in range(U):
            st = rng.choice(np.arange(1, N + 1), 3, replace=False); per = [5, 7, 9]
            ph = [T % 5, (T + 1) % 7, rng.randint(9)]          # staple 0 due at T, staple 1 due at T+1
            s = []
            for t in range(T + 2):
                due = [k for k in range(3) if t % per[k] == ph[k]]
                if t == T: due = [0]
                if t == T + 1: due = [1]
                s.append(st[due[0]] if due else rng.randint(1, N + 1))
            streams.append(s[:T]); vt.append(s[T]); tt.append(s[T + 1])
    return streams, np.array(vt), np.array(tt)

def synth_data(task, N=500):
    """data dict in the s1_lib format; NO history masking (repeats are part of the tasks)"""
    streams, vt, tt = make_synth(task, N=N)
    seqs = [[int(x) for x in s] for s in streams]; nU = len(seqs); us = np.arange(nU)
    empty = sp.csr_matrix((nU, N), dtype=np.float32)
    pop = np.bincount(np.concatenate(seqs), minlength=N + 1)[1:]
    return dict(name=f"syn_{task}", nU=nU, nI=N, L=50, seqs=seqs, val=(us, vt), test=(us, tt),
                vmap=dict(zip(us.tolist(), vt.tolist())), seen_val=empty, seen_test=empty, pop=pop)
# runner: set S.L = 50; model, info = S.train(data, kind, seed, lr=lr, drop=drop, patience=15) with S1_MAXEP=300;
# then S.evaluate(model, data, "test", full=True).  Probe: recency-weighted first-order counts,
# score = sum_{k=1..10} 0.5^(k-1) * M[h[-k]], M = transition counts over all training streams.
# Oracle (markov tasks): rank of the target under the true transition distribution.
# User-frequency probe (period): score = count of each item in the user's own test history.
# Primary metric for period = HR@1 = (ranks < 1).mean() from test["ranks"]; markov tasks use NDCG@10.
# Checked 2026-10-22 on CPU: synth_data + S.train(max_epochs=2) + S.evaluate(full=True) run for all three tasks
# (by_pop strata are empty for synthetic data -> harmless "Mean of empty slice" warning).
```

## 附录 D　1 个 Mamba 块的 HA 精确分解（对应 `s1_lib.SSM`）

记一个块在位置 T 的混合器输出为 `out_T`。前向（`SSM.forward`）：
- `xb = in_proj(x)[:di] * mask`，`z = in_proj(x)[di:]`
- 卷积前激活 `p_t = b + Σ_{j=0..3} w_j · xb_{t-3+j}`，`u_t = SiLU(p_t) = σ(p_t)·p_t`（再乘 mask）
- 选择性参数 `dt_t = softplus(dt_proj(·))`、`B_t`、`C_t` 由 `u_t` 计算；`A = -exp(A_log)`
- `y_T = Σ_{t≤T} K_{T,t} ⊙ u_t + D ⊙ u_T`，其中 `K_{T,t,c} = dt_{t,c} · Σ_n C_{T,n} B_{t,n} exp(A_{c,n} Σ_{r=t+1..T} dt_{r,c})`
- `out_T = W_out [ SiLU(z_T) ⊙ y_T ]`

**冻结**所有依赖输入的系数（`dt`、`B`、`C`、`σ(p_t)`、`SiLU(z_T)`）后，`out_T` 对 `xb` 是线性的。物品 s 通过卷积进入 `u_t`（t = s … s+3，权重 `w_{3-(t-s)}`）：

`M_{T,s} = W_out [ SiLU(z_T) ⊙ Σ_{t=s..min(s+3,T)} (K'_{T,t} ⊙ σ(p_t) ⊙ w_{3-(t-s)}) ⊙ xb_s ]`，其中 `K' = K + D·1[t=T]`；
卷积偏置单独成一项：`M_bias = W_out [ SiLU(z_T) ⊙ Σ_t K'_{T,t} ⊙ σ(p_t) ⊙ b ]`。

**精确性检查**：`Σ_s M_{T,s} + M_bias` 必须等于真实的 `out_T`（试点的相对误差 < 1e-6）。
**归因分数**：`score_s = <M_{T,s}, ∂f/∂out_T>`，梯度经过该块的残差 + LayerNorm + FFN 线性化。
**2 个块**：每块按上式得到位置间的贡献矩阵（按行归一化），`Ã = 0.5 I + 0.5 W`，展开 `R = Ã^(2) Ã^(1)`，`score_s = R_{T,s}`（次要分析，事先写入协议）。
