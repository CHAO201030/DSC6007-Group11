# -*- coding: utf-8 -*-
"""任务 A 验收测试（直接运行：python test_task_a.py）

验收标准：
1) 随便设一个种子，重复跑两次，观测和回报完全一致；
2) 训练端拿到的数据形状永远不变。
"""
import numpy as np

from envs import EnvConfig, OvercookedVecEnv, NUM_ACTIONS, NUM_JOINT_ACTIONS, split_joint


def run_rollout(seed: int, num_steps: int = 400):
    """同一个种子完整跑一遍，返回 (观测序列, 奖励序列, 数据形状)。"""
    cfg = EnvConfig(seed=seed, num_envs=4, horizon=50)  # horizon 设短一点，专门测试截断
    vec = OvercookedVecEnv(cfg)

    policy_rng = np.random.default_rng(12345)  # 模拟"策略"的随机数，两次跑保持一样
    obs = vec.reset()
    shape_seen = obs.shape

    obs_traj, rew_traj = [obs.copy()], []
    trunc_count = 0
    for _ in range(num_steps):
        actions = policy_rng.integers(0, vec.num_joint_actions, size=vec.num_envs)
        obs, rew, term, trunc, infos = vec.step(actions)
        trunc_count += int(trunc.sum())
        assert obs.shape == shape_seen, f"观测形状变了！{obs.shape} != {shape_seen}"
        assert rew.shape == (vec.num_envs,), f"奖励形状变了！{rew.shape}"
        assert not term.any(), "Overcooked 不应出现 terminated=True"
        obs_traj.append(obs.copy())
        rew_traj.append(rew.copy())

    return np.stack(obs_traj), np.stack(rew_traj), shape_seen, trunc_count


def main():
    seed = 42

    obs1, rew1, shape1, truncs1 = run_rollout(seed)
    obs2, rew2, shape2, truncs2 = run_rollout(seed)

    # 验收 1：同种子两次，观测和回报完全一致（逐元素精确相等）
    assert shape1 == shape2, "两次运行观测形状不同"
    assert np.array_equal(obs1, obs2), "两次运行观测不一致！"
    assert np.array_equal(rew1, rew2), "两次运行回报不一致！"
    assert truncs1 == truncs2 and truncs1 > 0, "没有触发过截断，测试没覆盖到 horizon"

    # 验收 2：形状永远不变
    num_envs, num_agents, obs_dim = shape1
    assert num_agents == 2, "观测必须是两个 agent 各一份"

    # 联合动作编码自洽性
    for j in range(NUM_JOINT_ACTIONS):
        a0, a1 = split_joint(j)
        assert 0 <= a0 < NUM_ACTIONS and 0 <= a1 < NUM_ACTIONS

    print("=" * 50)
    print("任务 A 验收通过 ✔")
    print(f"  观测形状: (num_envs={num_envs}, agents={num_agents}, F={obs_dim}) 恒定")
    print(f"  两次同种子运行: 观测逐元素一致 = True, 回报逐元素一致 = True")
    print(f"  覆盖到截断次数: {truncs1} 次（horizon 截断逻辑已验证）")
    print("=" * 50)


if __name__ == "__main__":
    main()
