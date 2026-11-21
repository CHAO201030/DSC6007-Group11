# 表 T1 协议

| 项 | 取值 |
|---|---|
| layout | cramped_room |
| horizon | 400 |
| 起点 | standard |
| 动作 | 采样（stochastic） |
| 训练步数 | SP / FCP-scratch 8M；Arm P / Arm S 4M |
| 训练 seed | 1, 2, 3, 42, 526007 |
| 评估 seed | 0, 1, 2（每格 5 局） |
| held-out 强 | ppo_vec_env seed1–5 @8M |
| held-out 弱 | StayBot, RandomBot |

# 表 T2 主结果（每 seed：自对弈 → held-out）

| 臂 | 自对弈（5 seed） | 存活 | held-out |
|---|---|---|---|
| SP | 11, 11, 10, 11, 11 | 5/5 | 0, 0, 0, 0, 0 |
| Arm P | 9, 9, 10, 0, 0 | 3/5 | 0, 0, 0, 0, 0 |
| Arm S | 0, 0, 9, 11, 0 | 2/5 | 0, 0, 0, 0, 0 |
| FCP-scratch | 0, 0, 0, 8, 0 | 1/5 | 0, 0, 0, 7, 0 |

注：
- "存活" = 该 seed 自对弈 ≥ 9 汤；
- T2 的 held-out 每格 = 5 个伙伴（StayBot、RandomBot、vec seed1–3）× 15 局，SP / Arm P / Arm S 的最大值为 1 汤；
- 迁移矩阵（F2）另有 5 ego × 8 个异源伙伴 = 40 格，其中仅 1 格达到 3 汤。
- FCP-scratch seed42：对 StayBot 9、RandomBot 7、异源 checkpoint 8–9（vec seed1 例外，1 汤）。
