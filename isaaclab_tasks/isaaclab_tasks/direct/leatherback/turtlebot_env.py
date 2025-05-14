from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from .obstacle import OBSTACLE_CFG
from isaaclab_assets.robots.turtlebot import TUR_CFG

from dataclasses import dataclass

@configclass
class TurtleBotEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 12.0
    action_space = 2
    observation_space = 4
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot: ArticulationCfg = TUR_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    joints_dof_name = [
        "Wheel_Left_Joint",
        "Wheel_Right_Joint"
    ]
    env_spacing = 32.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class TurtleBotEnv(DirectRLEnv):
    cfg: TurtleBotEnvCfg

    def __init__(self, cfg: TurtleBotEnvCfg):
        super().__init__(cfg)
        self.goal_xy = torch.zeros(self.num_envs, 2, device=self.device)

    # (1) 관측
    def _get_observations(self) -> dict[str, torch.Tensor]:
        pos_xy   = self.robot.data.root_pos_w[:, :2]        # (N,2)
        heading  = self.robot.data.heading                  # (N,)
        delta    = self.goal_xy - pos_xy                    # (N,2)
        dist     = torch.norm(delta, dim=1, keepdim=True)   # (N,1)
        head_err = (torch.atan2(delta[:, 1], delta[:, 0]) - heading).unsqueeze(-1)
        return {"policy": torch.cat([delta, head_err, dist], dim=1)}

    # (2) 액션 적용
    def _apply_action(self, actions: torch.Tensor) -> None:
        max_lin, max_ang = 0.4, 1.0         # m/s, rad/s
        lin = actions[:, 0] * max_lin
        ang = actions[:, 1] * max_ang
        self.robot.command.set_differential_velocity(lin, ang)

    # (3) 리셋 (로봇 + 목표)
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        self.robot.reset_idx(env_ids)
        self.goal_xy[env_ids] = torch.rand(len(env_ids), 2,
                                           device=self.device) * 3.0 - 1.5  # [-1.5,1.5]^2

    # (4) 리워드
    def _compute_reward(self) -> torch.Tensor:
        delta = self.goal_xy - self.robot.data.root_pos_w[:, :2]
        dist  = torch.norm(delta, dim=1)
        reward = -dist                                       # 가까워질수록 ↑
        reached = dist < 0.05
        reward[reached] += 10.0
        self.reset_buf[reached] = 1                          # 에피소드 종료 플래그
        return reward

    # (5) done 조건 (리셋 버퍼 외에 타임아웃)
    def _is_done(self) -> torch.Tensor:
        return (self.progress_buf >= self.max_episode_length_steps) | self.reset_buf