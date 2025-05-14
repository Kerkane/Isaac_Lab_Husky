# turtle_env.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.turtlebot import TUR_CFG

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # :contentReference[oaicite:0]{index=0}
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg # :contentReference[oaicite:1]{index=1}
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers


@configclass
class TurtlebotEnvCfg(DirectRLEnvCfg):
    # 환경 시뮬레이션 설정
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=2)  # dt=1/60s, 물리 업데이트마다 4 스텝 렌더 :contentReference[oaicite:2]{index=2}
    decimation = 4
    episode_length_s = 20.0
    action_scale = 0.3

    # 행동·관측 스페이스
    action_space = 2       # [linear_vel, angular_vel] :contentReference[oaicite:3]{index=3}
    observation_space = 5  # [dx, dy, heading_error, lin_vel, ang_vel] :contentReference[oaicite:4]{index=4}
    state_space = 0
    robot_cfg: ArticulationCfg = TUR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # 목표 지점 (x, y)
    goal_x = 1.5
    goal_y = 1.5
    # === reward scales ===
    rew_scale_alive: float = 1.0       # 매 스텝 받는 기본 생존 보너스
    rew_scale_dist: float = -1.0       # 목표까지 거리당 패널티
    rew_scale_goal: float = 10.0       # 목표 도달 보너스


class TurtlebotEnv(DirectRLEnv):
    cfg: TurtlebotEnvCfg

    def __init__(self, cfg: TurtlebotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.initial_z = float(self.cfg.robot_cfg.init_state.pos[2])
        self.prev_actions = torch.zeros((self.num_envs, 2), device=self.device)

    def _setup_scene(self):
        # 바닥 생성
        sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())  # :contentReference[oaicite:5]{index=5}

        # 로봇 스폰
        self.robot = Articulation(self.cfg.robot_cfg)

        # 다중 env 복제
        self.scene.clone_environments(copy_from_source=False)

        # 씬에 로봇 등록
        self.scene.articulations["turtlebot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        # root 상태: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        root = self.robot.data.default_root_state  # 현재 상태 버퍼 :contentReference[oaicite:6]{index=6}
        positions = root[:, :3]
        orientations = root[:, 3:7]
        lin_vel = root[:, 7:10]
        ang_vel = root[:, 10:13]

        # 목표와 상대 벡터
        goal = torch.tensor([self.cfg.goal_x, self.cfg.goal_y, 0.0], device=self.device)
        rel = goal - positions

        # 로봇 yaw 계산
        yaw = self._yaw_from_quat(orientations)
        heading = torch.atan2(rel[:, 1], rel[:, 0])
        heading_error = (heading - yaw).unsqueeze(-1)

        obs = torch.cat((
            rel[:, :2],             # dx, dy
            heading_error,          # heading error
            lin_vel[:, :1],         # forward velocity
            ang_vel[:, 2:3],        # angular z
        ), dim=1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        보상 구성:
          - rew_scale_alive: 매 스텝 생존 보너스
          - rew_scale_dist: 목표까지의 유클리드 거리 패널티
          - rew_scale_goal: 목표 도달 시 추가 보너스
        """
        # 1) 현재 로봇 위치
        root = self.robot.data.default_root_state
        positions = root[:, :2]  # [num_envs, 2]

        # 2) 목표 위치
        goal = torch.tensor(
            [self.cfg.goal_x, self.cfg.goal_y],
            device=self.device
        ).unsqueeze(0)         # [1,2]
        # 거리 계산
        dist = torch.norm(positions - goal, dim=1)  # [num_envs]

        # 3) 보상 항목별 계산
        rew_alive = self.cfg.rew_scale_alive * torch.ones_like(dist)
        rew_dist  = self.cfg.rew_scale_dist * dist

        # 목표 도달 플래그
        done_goal, _ = self._get_dones()
        rew_goal  = done_goal.float() * self.cfg.rew_scale_goal

        # 4) 총 보상
        total_reward = rew_alive + rew_dist + rew_goal

        return total_reward  # torch.Tensor([num_envs]) :contentReference[oaicite:0]{index=0}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 목표 도달 조건
        root = self.robot.data.default_root_state
        dist = torch.norm(root[:, :2] - torch.tensor([self.cfg.goal_x, self.cfg.goal_y], device=self.device), dim=1)
        done_goal = dist < 0.1
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return done_goal, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        default_root = self.robot.data.default_root_state[env_ids]
        default_root[:, :3] = self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 1) 스케일 적용 & 클램핑
        a = self.cfg.action_scale * actions.clone()
        v   = torch.clamp(a[:, 0], -0.5, 0.5)
        rot = torch.clamp(a[:, 1], -1.0, 1.0)

        # 2) 지수이동평균 (EMA) 필터: alpha 클수록 반응 빠름, 작을수록 부드러움
        alpha = 0.3
        smoothed = alpha * torch.stack((v, rot), dim=1) \
                   + (1 - alpha) * self.prev_actions

        self.actions = smoothed
        self.prev_actions = smoothed.clone()

    def _apply_action(self) -> None:
        """
        actions: Tensor[num_envs, 2]  ← [v, ω]
        v: 순진행 속도(m/s), ω: 각속도(rad/s)
        TurtleBot3 Burger 물리 파라미터 기준:
          wheel_base = 0.16  # 바퀴 사이 간격 (m)
          wheel_radius = 0.033  # 바퀴 반지름 (m)
        """
        v = self.actions[:, 0]
        omega = self.actions[:, 1]
        wheel_base = 0.16
        wheel_radius = 0.033

        # differential drive → 바퀴 각속도 [rad/s]
        v_l = (2 * v - omega * wheel_base) / (2 * wheel_radius)
        v_r = (2 * v + omega * wheel_base) / (2 * wheel_radius)

        # joint 이름으로 인덱스 찾아서
        left_idx  = self.robot.joint_names.index("wheel_left_joint")
        right_idx = self.robot.joint_names.index("wheel_right_joint")

        # [num_envs, 2] 크기의 타겟 텐서 생성
        vel_targets = torch.stack((v_l, v_r), dim=1)

        # actuator_cfg에 정의된 두 joint만 velocity target 설정
        # joint_ids 파라미터 덕분에 나머지 joint는 그대로 두어도 됩니다.
        self.robot.set_joint_velocity_target(
            vel_targets,
            joint_ids=[left_idx, right_idx]
        )  # :contentReference[oaicite:0]{index=0}

        # 시뮬레이션에 반영
        self.robot.write_data_to_sim()  # :contentReference[oaicite:1]{index=1}

    def _post_physics_step(self) -> None:
        # 1) root 상태 읽기
        root = self.robot.data.default_root_state  # [num_envs, 13]

        # 2) z 위치 고정
        root[:, 2] = self.initial_z

        # 3) z축 선형 속도, roll/pitch 각속도 제거
        #    인덱스: lin_vel z → root[:, 9], ang_vel x,y → root[:,10], root[:,11]
        root[:, 9] = 0.0      # z축 속도
        root[:,10] = 0.0      # roll 각속도
        root[:,11] = 0.0      # pitch 각속도

        # 4) 다시 시뮬레이터에 쓰기
        self.robot.write_root_pose_to_sim(root[:, :7], None)
        self.robot.write_root_velocity_to_sim(root[:, 7:], None)


    @staticmethod
    def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        # 쿼터니언 [w,x,y,z]로부터 yaw 추출: atan2(2*(wz + xy), 1-2*(y^2+z^2))
        w, x, y, z = quat.unbind(dim=1)
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
