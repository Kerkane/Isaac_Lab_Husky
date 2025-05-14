# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_goal_reward(
    env: ManagerBasedRLEnv,
    success_threshold: float,
    success_reward: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
)  -> torch.Tensor:

    robot: Articulation = env.scene[asset_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    # 거리 계산
    distance = torch.norm(robot.data.root_pos_w[:, :2] - goal.data.root_pos_w[:, :2], dim=-1)

    # 기본 shaping reward
    reward = -distance

    # 도달 여부
    reached = distance < success_threshold

    # 성공 보상
    reward += reached.float() * success_reward
    
    return reward

def velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
)  -> torch.Tensor:
    
    robot : Articulation = env.scene[asset_cfg.name]
    return torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)

def heading_alignment_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    # 방향 벡터 구하기
    goal_vec = goal.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    heading_vec = robot.data.root_lin_vel_b[:, :2]  # 현재 이동 방향

    goal_vec = torch.nn.functional.normalize(goal_vec, dim=-1)
    heading_vec = torch.nn.functional.normalize(heading_vec, dim=-1)

    alignment = (goal_vec * heading_vec).sum(dim=-1)  # cosine similarity
    return alignment  # -1~1 사이 값