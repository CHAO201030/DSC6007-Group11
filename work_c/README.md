# work_c/ — 智能体与评估层（工作 C）

回答两个问题：**给定一个 checkpoint，它和不同伙伴配合得怎么样；伙伴多样性训练能否改善这一点。**

---

## 目录

```
work_c/
├── eval/            环境包装、单局执行、分组扫描、汇总
├── scripts/         入口：评估、批量、交叉矩阵、出图、复现校验
│   └── legacy/      已归档的旧版绘图与流水线脚本（不要使用）
├── results/         CSV / JSON / 图
├── ckpt_final/      四条臂终点 checkpoint 的评估副本（脚本生成，权重本身不入库）
└── README.md
```

## results/ 里有什么

| 文件 | 内容 |
|---|---|
| `decay_s{1,2,3,42,526007}.csv` | 兼容率衰减：同一训练轨迹内，ego 对距离 0–7.5M 步的伙伴 |
| `transfer_stoch.csv` | 15 ego × 10 partner 的迁移矩阵（采样动作） |
| `final_table.csv` | 四臂终点表（自对弈 / held-out / gap） |
| `f42_cross.csv` | FCP-scratch seed42 对异源 checkpoint 的验证 |
| `figures/F1–F6.png` | 报告配图（`plot_all.py` 生成） |
| `raw_final/*.json` | 终点表 20 个 ego 的原始评估结果 |
| `diag/*.json` | 过程中用于定位的诊断（冒烟、塌缩案例等） |
| `TABLES.md` | 报告用的 T1/T2 表 |
| `REPRO_CHECK.md` | 复现校验记录 |

---

## 四条臂

| 目录 | 报告里的名字 | 初始化 | 伙伴 | 步数 |
|---|---|---|---|---|
| `checkpoints/ppo_single_env` | **SP** | 随机 | 自己是同一份权重（联合训练） | 8M |
| `checkpoints/fcp` | **FCP-scratch** | 随机 | 100% 池（5 seed × 16 checkpoint） | 8M |
| `checkpoints/fcp2` | **Arm P** | SP@8M | 50% 自身 8M + 50% 池 | 4M |
| `checkpoints/sp_ref` | **Arm S** | SP@8M | 100% 自身 8M（对照） | 4M |
| `checkpoints/ppo_vec_env` | **held-out** | — | 仅评估使用，从未进入任何池 | — |

---

## 训练入口（train_fcp.py）

位于**仓库根目录**（不在 `work_c/` 下），产出 Arm P / Arm S / FCP-scratch 三条臂。它与 B 的 `train.py`（联合自对弈）的唯一区别是「伙伴从哪来」。

| 参数 | 作用 |
|---|---|
| `--pool-dir` | 伙伴池目录（可给多个），读取其中的 `*.pth` |
| `--init-ckpt` | 热启动：只加载 actor / critic 权重，不续步数 |
| `--self-ckpt` | 固定的自对弈伙伴路径 |
| `--self-prob` | 每局改用 `--self-ckpt` 的概率（0 = 纯池） |
| `--ckpt-interval` | 落盘间隔（默认 500k） |

三条臂的启动差异：

| 臂 | 关键参数 |
|---|---|
| FCP-scratch（8M） | 无 `--init-ckpt`、无 `--self-*`；`--pool-dir` 给 5 个 seed 目录 |
| Arm P（4M） | `--init-ckpt` 与 `--self-ckpt` 都指向自身 8M，`--self-prob 0.5`，池给 5 个 seed |
| Arm S（4M） | `--init-ckpt` 指向自身 8M；`--pool-dir` 只给「仅含自身 8M」的单文件目录 |
| seed526007 | 上述「自身 8M」一律替换为 `ppo_clip_7500000.pth`（见 checkpoint 自检规则） |

训练曲线写入 `results/{fcp,fcp2,sp_ref}/fcp_seed*.npz`，每个 checkpoint 刷一次盘，中途崩溃也能留下可用数据。

---

## 评估协议 v3

| 项 | 取值 |
|---|---|
| 起始状态 | `standard`（与训练一致）。`random_obj` 会让所有臂一起崩，仅作鲁棒性附录 |
| 动作 | **采样**（`--stochastic`）；argmax 只用于快速冒烟 |
| 每格样本 | 3 个评估 seed × 5 局 = 15 局 |
| 配对 | 每局 `torch.manual_seed(ep_seed)`，两臂消耗同一串随机数 |
| 强 held-out | `ppo_vec_env` seed1–5 @8M |
| 弱 held-out | `StayBot` / `RandomBot` |

**跨臂可比性**：只有"两条臂都没见过"的集合才能用于跨臂比较，即 vec + bot。
`ppo_single_env` 的其它 seed 对 SP 臂未见、对 Arm P 已见（在它的池里），因此只能作为 **SP-only 诊断**，不参与跨臂结论。

---

## checkpoint 自检规则

每个 seed 的最终 checkpoint 必须通过自对弈自检（standard 起点、采样动作、≥ 9 汤）：

- 未通过者回退到该 seed 最近的通过检查的 checkpoint，并在结果中注明；
- **seed526007**：8M 未通过（0 汤 / shaped 174，塌缩态），改用 7.5M（11 汤）；
- 规则对四条臂统一执行，不针对个别结果。

---

## 常用命令

```bash
# 单点评估（一份固定格式 JSON）
python -m work_c.scripts.evaluate --ckpt <path> --heldout-ckpt <p1> <p2> \
    --n-episodes 5 --seeds 0,1,2 --start-state standard --stochastic --out out.json

# 批量（学习曲线 / 终点表）
python -m work_c.scripts.run_all_experiments --ckpt-dir <dirs...> \
    --heldout-ckpt <paths...> --n-episodes 5 --seeds 0,1,2 \
    --start-state standard --stochastic --out-csv c.csv --raw-dir raw_x

# 交叉矩阵
python -m work_c.scripts.transfer_matrix --ego <...> --partner <...> \
    --episodes 5 --start-state standard --stochastic --out m.csv

# 出图（F1–F6）
python -m work_c.scripts.plot_all

# 复现校验（两次独立运行，逐字节比对）
python -m work_c.scripts.check_repro
```

---

## 如何重建 ckpt_final/

`ckpt_final/` 是 20 个单文件目录（四条臂 × 5 个 seed），供终点表评估与 `check_repro` 使用。权重被 `.gitignore` 排除，因此新克隆需要重建：

```powershell
$dst = "work_c\ckpt_final"
foreach ($s in 1,2,3,42,526007) {
  $spck = if ($s -eq 526007) { "7500000" } else { "8000000" }   # 自检规则
  foreach ($p in @(
      @("sp", "checkpoints\ppo_single_env", "ppo_clip_$spck.pth"),
      @("f",  "checkpoints\fcp",            "ppo_clip_8000000.pth"),
      @("p",  "checkpoints\fcp2",           "ppo_clip_4000000.pth"),
      @("s",  "checkpoints\sp_ref",         "ppo_clip_4000000.pth"))) {
    $out = "$dst\$($p[0])_seed$s"
    New-Item -ItemType Directory -Force $out | Out-Null
    Copy-Item (Join-Path $p[1] "seed$s\$($p[2])") "$out\$($p[0])_s$s.pth" -Force
  }
}
Get-ChildItem $dst -Directory | ForEach-Object { $_.Name + "=" + @(Get-ChildItem $_.FullName -Filter *.pth).Count }
```

期望 20 个目录各 1 个文件。评估时把目录名当 ego 名（`sp_s1`、`p_s1`、`s_s1`、`f_s42`），所以文件名必须保留 `${arm}_s${seed}.pth` 的形式。

> 注意：训练曲线 `results/{fcp,fcp2,sp_ref}/*.npz` 同样不在版本库中（`*.npz` 被忽略）。要重画 F5，需保留训练产出的 npz，或用 `train_fcp.py` 重新训练。

---

## 已知限制

| # | 限制 |
|---|---|
| 1 | 单学习器 + 冻结伙伴 ≠ 联合自对弈，Arm S 不是 SP 的严格对照 |
| 2 | held-out 与训练伙伴同属 PPO 家族，差异来自采样方式与训练随机性，不是策略类别 |
| 3 | 训练预算 8M 步、96 维 MLP、单机 CPU，远小于文献量级 |
| 4 | 只有 1/5 个 seed 的 FCP 学出有效策略，只能作 existence proof，不能当普遍结论 |
| 5 | 兼容率阈值取 ≥8 汤，属人为设定；≥6 / ≥10 的趋势方向相同 |
| 6 | 随机初始状态条件只做过一次诊断，未系统评估 |
| 7 | `scripts/legacy/` 下是旧版流水线与绘图脚本（含 `reproduce_legacy.py`），路径与协议均已过期，不要使用 |

---

## 与其它层的边界

| 层 | 提供 | 本目录 |
|---|---|---|
| `envs/`（A） | `OvercookedSingleEnv` / `EnvConfig` / `joint_index` | 训练用；评估用自建 `gameenv_wrapper`（需要随机初始状态与座位固定） |
| `algorithms/`（B） | `PPO` / `save_weights` / `load_policy` / `get_GAE` | 只读取它产出的 checkpoint，训练时复用 |
| `agents/`（C） | `SPPolicy` / `FCPPool` / 脚本 bot | 直接使用 |

**为什么不复用 `envs/` 做评估**：`OvercookedSingleEnv` 不传 `start_state_fn`，起始状态固定，评估会退化成"每个伙伴只有一条轨迹"。若 A 之后加上随机初始状态，只需改 `eval/gameenv_wrapper.py` 一个文件即可切换。
