# 复现校验记录

- 命令：`python -m work_c.scripts.check_repro`
- 规模：4 个 ego（sp_seed1 / p_seed1 / s_seed1 / f_seed42）× 3 个伙伴
  （vec seed1 + StayBot + RandomBot）× 2 seed × 2 局
- 结果：两次独立运行，4 个评估 JSON **逐字节一致**
- 协议：`start_state=standard`、`--stochastic`、每局 `torch.manual_seed(ep_seed)`
- 备注：`reproduce.py` 为上一版遗留脚本（路径与协议均已过期），不使用
