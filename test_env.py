#!/usr/bin/env python3
"""GameEnv（vendored Overcooked-AI）环境冒烟测试。

在仓库根目录运行：

    python test_env.py
    python test_env.py --layout forced_coordination --steps 50
    python test_env.py --all-layouts --skip-render

全部检查通过时退出码为 0，否则为 1。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import traceback
from pathlib import Path

# WSL / 服务器通常没有显示器。必须在 import pygame 之前设置 dummy 驱动。
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402

from GameEnv.actions import Action  # noqa: E402
from GameEnv.overcooked_env import OvercookedEnv  # noqa: E402
from GameEnv.overcooked_mdp import OvercookedGridworld  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check(name: str, fn) -> None:
    """执行一个检查项；失败不中断后续检查。"""
    try:
        detail = fn() or ""
    except Exception as exc:  # noqa: BLE001 - 测试脚本需要收集所有失败
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        if os.environ.get("GAMEENV_TEST_TRACEBACK"):
            traceback.print_exc()
    else:
        RESULTS.append((name, True, detail))
        print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


def make_gym_env(gw: OvercookedGridworld, horizon: int):
    """按 Gymnasium 1.x 的接口构造 Overcooked-v0。"""
    base_env = OvercookedEnv.from_mdp(gw, horizon=horizon, info_level=0)
    return gymnasium.make(
        "Overcooked-v0",
        base_env=base_env,
        featurize_fn=base_env.featurize_state_mdp,
        # obs 是含 OvercookedState 的 dict，无法用 Box 空间描述
        disable_env_checker=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="cramped_room", help="布局名")
    parser.add_argument("--steps", type=int, default=20, help="随机 rollout 步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--all-layouts", action="store_true", help="额外检查所有内置布局能否加载"
    )
    parser.add_argument("--skip-render", action="store_true", help="跳过渲染检查")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps 必须为正整数")

    layout = args.layout

    def check_module_origin() -> str:
        require(
            OvercookedGridworld.__module__ == "GameEnv.overcooked_mdp",
            f"从意外模块加载: {OvercookedGridworld.__module__}",
        )
        spec = importlib.util.find_spec("overcooked_ai")
        if spec is None:
            return "GameEnv 独立加载，overcooked_ai 未安装"
        return f"警告: 环境中仍可导入 overcooked_ai ({spec.origin})"

    def check_layout() -> str:
        gw = OvercookedGridworld.from_layout_name(layout)
        require(gw.layout_name == layout, f"layout_name={gw.layout_name!r}")
        require(len(gw.terrain_mtx) > 0, "terrain_mtx 为空")
        return f"grid={tuple(gw.shape)}"

    def check_mdp_transition() -> str:
        gw = OvercookedGridworld.from_layout_name(layout)
        state = gw.get_standard_start_state()
        next_state, infos = gw.get_state_transition(
            state, (Action.STAY, Action.STAY)
        )
        require(
            next_state.timestep == state.timestep + 1,
            f"timestep 未递增: {state.timestep} -> {next_state.timestep}",
        )
        require(
            "sparse_reward_by_agent" in infos,
            f"transition info 缺少 sparse_reward_by_agent: {sorted(infos)}",
        )
        return f"timestep {state.timestep} -> {next_state.timestep}"

    def check_encodings() -> str:
        gw = OvercookedGridworld.from_layout_name(layout)
        env = OvercookedEnv.from_mdp(gw, horizon=args.steps, info_level=0)

        featurized = env.featurize_state_mdp(env.state)
        require(len(featurized) == 2, f"featurize 返回 {len(featurized)} 个观测")
        require(
            featurized[0].ndim == 1 and featurized[0].shape == featurized[1].shape,
            f"featurized 形状异常: {[x.shape for x in featurized]}",
        )

        lossless = env.lossless_state_encoding_mdp(env.state)
        expected = tuple(int(x) for x in gw.get_lossless_state_encoding_shape())
        require(len(lossless) == 2, f"lossless 返回 {len(lossless)} 个观测")
        require(
            tuple(lossless[0].shape) == expected,
            f"lossless 形状 {tuple(lossless[0].shape)} != {expected}",
        )

        require(
            type(env.mlam).__module__ == "GameEnv.planners",
            f"mlam 来自意外模块: {type(env.mlam).__module__}",
        )
        return (
            f"featurized={tuple(featurized[0].shape)}, "
            f"lossless={tuple(lossless[0].shape)}"
        )

    def check_rollout() -> str:
        gw = OvercookedGridworld.from_layout_name(layout)
        env = OvercookedEnv.from_mdp(gw, horizon=args.steps, info_level=0)
        env.reset()

        rng = random.Random(args.seed)
        total_reward = 0.0
        for t in range(args.steps):
            legal_actions = gw.get_actions(env.state)
            joint_action = tuple(rng.choice(actions) for actions in legal_actions)
            state, reward, done, _info = env.step(joint_action)
            require(
                state.timestep == t + 1,
                f"第 {t} 步后 timestep={state.timestep}",
            )
            require(not done or t == args.steps - 1, f"在第 {t} 步提前结束")
            total_reward += float(reward)

        require(done, "到达 horizon 后 done 仍为 False")
        require(state.timestep == args.steps, f"timestep={state.timestep}")
        return f"{args.steps} 步, total_reward={total_reward:.2f}"

    def check_gym_api() -> str:
        gw = OvercookedGridworld.from_layout_name(layout)
        env = make_gym_env(gw, horizon=5)

        obs, info = env.reset(seed=args.seed)
        require(
            set(obs) == {"both_agent_obs", "overcooked_state", "other_agent_env_idx"},
            f"obs keys={sorted(obs)}",
        )
        require(len(obs["both_agent_obs"]) == 2, "both_agent_obs 长度不是 2")
        require(isinstance(info, dict), f"reset info 类型: {type(info)}")

        stay = Action.ACTION_TO_INDEX[Action.STAY]
        obs, reward, terminated, truncated, info = env.step((stay, stay))
        require(isinstance(reward, (int, float)), f"reward 类型: {type(reward)}")
        require(
            isinstance(terminated, bool) and isinstance(truncated, bool),
            f"terminated/truncated 类型: {type(terminated)}, {type(truncated)}",
        )
        require(
            set(obs) == {"both_agent_obs", "overcooked_state", "other_agent_env_idx"},
            f"step obs keys={sorted(obs)}",
        )
        return "reset -> (obs, info); step -> 5-tuple"

    def check_seeding() -> str:
        gw = OvercookedGridworld.from_layout_name(layout)
        env_a = make_gym_env(gw, horizon=5)
        env_b = make_gym_env(gw, horizon=5)

        obs_a, _ = env_a.reset(seed=args.seed)
        obs_b, _ = env_b.reset(seed=args.seed)
        require(
            env_a.unwrapped.agent_idx == env_b.unwrapped.agent_idx,
            f"agent_idx 不一致: {env_a.unwrapped.agent_idx} vs "
            f"{env_b.unwrapped.agent_idx}",
        )
        require(
            all(
                np.array_equal(x, y)
                for x, y in zip(obs_a["both_agent_obs"], obs_b["both_agent_obs"])
            ),
            "相同 seed 下观测不一致",
        )
        return f"agent_idx={env_a.unwrapped.agent_idx}, obs 一致"

    def check_render() -> str:
        gw = OvercookedGridworld.from_layout_name(layout)
        env = make_gym_env(gw, horizon=5)
        env.reset(seed=args.seed)
        frame = env.render()
        require(isinstance(frame, np.ndarray), f"render 返回 {type(frame)}")
        require(
            frame.ndim == 3 and frame.shape[2] == 3,
            f"render 形状异常: {frame.shape}",
        )
        require(frame.dtype == np.uint8, f"render dtype: {frame.dtype}")
        return f"frame={frame.shape}, dtype={frame.dtype}"

    def check_all_layouts() -> str:
        layout_dir = REPO_ROOT / "GameEnv" / "data" / "layouts"
        paths = sorted(layout_dir.glob("*.layout"))
        require(paths, f"{layout_dir} 下没有 .layout 文件")
        failures = []
        for path in paths:
            try:
                OvercookedGridworld.from_layout_name(path.stem)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path.stem}: {type(exc).__name__}: {exc}")
        require(
            not failures,
            f"{len(failures)}/{len(paths)} 个布局加载失败: {failures[:3]}",
        )
        return f"{len(paths)} 个布局全部可加载"

    print(f"GameEnv smoke test | layout={layout!r} steps={args.steps} "
          f"seed={args.seed}\n")

    check("模块来源", check_module_origin)
    check("布局加载", check_layout)
    check("MDP transition", check_mdp_transition)
    check("观测编码", check_encodings)
    check("OvercookedEnv rollout", check_rollout)
    check("Gymnasium API", check_gym_api)
    check("seed 可复现", check_seeding)
    if not args.skip_render:
        check("渲染", check_render)
    if args.all_layouts:
        check("全部布局", check_all_layouts)

    failed = [name for name, ok, _ in RESULTS if not ok]
    passed = len(RESULTS) - len(failed)
    print(f"\n{passed}/{len(RESULTS)} 项检查通过")
    if failed:
        print("失败: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
