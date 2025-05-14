# turtle_env.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from collections.abc import Sequence

from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_assets.robots.turtlebot import TUR_CFG

from .goal import GOAL_CFG
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # :contentReference[oaicite:0]{index=0}
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg # :contentReference[oaicite:1]{index=1}
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporterCfg


@configclass
class TurtlebotEnvCfg(DirectRLEnvCfg):
    # 환경 시뮬레이션 설정
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=2)  # dt=1/60s, 물리 업데이트마다 4 스텝 렌더 :contentReference[oaicite:2]{index=2}
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    decimation = 4
    episode_length_s = 12.0
    linear_scale: float = 0.2
    angular_scale: float = 1.2

    # 행동·관측 스페이스
    action_space = 2       # [linear_vel, angular_vel] :contentReference[oaicite:3]{index=3}
    observation_space = 5  # [dx, dy, heading_error, lin_vel, ang_vel] :contentReference[oaicite:4]{index=4}
    state_space = 0
    robot_cfg: ArticulationCfg = TUR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    goal_cfg = GOAL_CFG
    #scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # 목표 지점 (x, y)
    goal_x = 1.5
    goal_y = 1.5
    velocity_limit=12.0
    
    # — 리워드 스케일
    rew_scale_alive: float = 0.01
    rew_scale_dist:  float = 0.05           # per-meter 패널티
    rew_scale_goal:  float = 100.0
    rew_scale_vel:   float = 0.1            # forward velocity 보상
    rew_scale_head: float = 0.05
    rew_scale_bonus: float = 50.0    # 에피소드 초반에 더 빠를수록 주는 보너스
    time_penalty:    float = -0.01


class TurtlebotEnv(DirectRLEnv):
    cfg: TurtlebotEnvCfg
    terrain: TerrainImporterCfg
    
    def __init__(self, cfg: TurtlebotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # (1) wheel joint 인덱스
        self.debug_step = 0
        root = self.robot.data.default_root_state  # [num_envs, 13]
        pos = root[:, :2]
        goal = torch.tensor([cfg.goal_x, cfg.goal_y], device=self.device).unsqueeze(0)
        self.prev_dist = torch.norm(pos - goal, dim=1)
        self.left_idx  = self.robot.joint_names.index("wheel_left_joint")
        self.right_idx = self.robot.joint_names.index("wheel_right_joint")
        # (2) 초기 joint pos → 목표 pos 버퍼
        self.joint_pos_target = self.robot.data.joint_pos.clone()

        self.initial_z = float(self.cfg.robot_cfg.init_state.pos[2])

    def _setup_scene(self):
        # 바닥 생성
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 로봇 스폰
        self.robot = Articulation(self.cfg.robot_cfg)

        # 다중 env 복제
        self.scene.clone_environments(copy_from_source=False)

        # 씬에 로봇 등록
        self.scene.articulations["turtlebot"] = self.robot

        self.goal_marker = VisualizationMarkers(self.cfg.goal_cfg)
        # 4) 각 env별 origin에 (goal_x, goal_y, z) 오프셋 더해서 한 번에 visualize
        #    scene.env_origins: Tensor[num_envs, 3]
        origins = self.scene.env_origins    # :contentReference[oaicite:2]{index=2}
        #    goal offset 텐서
        offset = torch.tensor(
            (self.cfg.goal_x, self.cfg.goal_y, 0.05),
            device=self.device
        ).unsqueeze(0)  # shape [1,3]
        translations = origins + offset    # shape [num_envs,3]
        #    visualize 에는 반드시 Tensor or ndarray
        self.goal_marker.visualize(translations=translations)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self):
        root      = self.robot.data.root_pos_w   # [N,3]
        quat      = self.robot.data.root_quat_w  # [N,4]
        lin_vel   = self.robot.data.root_lin_vel_b[:, :2]
        ang_vel   = self.robot.data.root_ang_vel_b[:, 2:3]

        # (1) world→local 회전 계산
        yaw       = self._yaw_from_quat(quat)                     # [N]
        delta_world = torch.tensor([self.cfg.goal_x, self.cfg.goal_y], device=self.device) - root[:, :2]  # [N,2]
        # 로컬 x축(앞), y축(좌)로 변환
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        local_dx =  cos_y*delta_world[:,0] + sin_y*delta_world[:,1]
        local_dy = -sin_y*delta_world[:,0] + cos_y*delta_world[:,1]

        # (2) angle error wrapping
        raw_err  = torch.atan2(delta_world[:,1], delta_world[:,0]) - yaw
        wrapped  = (raw_err + torch.pi) % (2*torch.pi) - torch.pi

        obs = torch.cat((
            local_dx.unsqueeze(-1),   # 앞으로 얼마 남았나
            local_dy.unsqueeze(-1),   # 왼쪽으로 얼마 남았나
            wrapped.unsqueeze(-1),    # 방향 오차(–π~π)
            lin_vel[:, :1],           # forward vel
            ang_vel                   # angular vel
        ), dim=1)

        return {"policy": obs}

    '''def _get_observations(self):
        root         = self.robot.data.root_pos_w   # [num_envs, 3]
        quad         = self.robot.data.root_quat_w  # [num_envs, 4]
        lin_vel      = self.robot.data.root_lin_vel_b[:, :2]
        ang_vel      = self.robot.data.root_ang_vel_b[:, 2:3]

        # 목표 상대벡터
        goal_world   = torch.tensor([self.cfg.goal_x, self.cfg.goal_y], device=self.device)
        delta        = goal_world - root[:, :2]

        # heading error
        yaw          = self._yaw_from_quat(quad)
        heading      = torch.atan2(delta[:, 1], delta[:, 0])
        heading_err  = (heading - yaw).unsqueeze(-1)

        obs = torch.cat((
            delta,               # dx, dy
            heading_err,         # heading error
            lin_vel[:, :1],      # forward vel
            ang_vel,             # angular z
        ), dim=1)
        return {"policy": obs}'''

    def _get_rewards(self) -> torch.Tensor:
        # 1) 위치, 속도, 방향 읽기
        root_xy  = self.robot.data.root_pos_w[:, :2]
        goal     = torch.tensor([self.cfg.goal_x, self.cfg.goal_y],
                                device=self.device).unsqueeze(0)
        dist     = torch.norm(root_xy - goal, dim=1)               # [N]

        lin_vel  = self.robot.data.root_lin_vel_b[:, :2]
        quat     = self.robot.data.root_quat_w
        yaw      = self._yaw_from_quat(quat)
        forward_vel = lin_vel[:,0]*torch.cos(yaw) + lin_vel[:,1]*torch.sin(yaw)

        done_goal, done_time = self._get_dones()
        mask = done_goal.unsqueeze(1)
        self.v_cmd     = torch.where(mask, torch.zeros_like(self.v_cmd),     self.v_cmd)
        self.omega_cmd = torch.where(mask, torch.zeros_like(self.omega_cmd), self.omega_cmd)

        # ────────────────────────────────────────────────────────
        # 3) 수정된 보상 설계 시작
        # (a) 생존 보상은 그대로
        rew_alive = self.cfg.rew_scale_alive * torch.ones_like(dist)

        # (b) 거리 감소량 기반 보상
        delta_dist = self.prev_dist - dist
        rew_dist   = delta_dist * self.cfg.rew_scale_dist
        # prev_dist 갱신 (detach 해 주기)
        self.prev_dist = dist.clone().detach()
        bonus = self.cfg.rew_scale_bonus
        time_left = (self.max_episode_length - self.episode_length_buf) / self.max_episode_length
        # (c) 목표 도달 보상
        rew_goal = done_goal.float() * (self.cfg.rew_scale_goal + 
                                        (self.max_episode_length - self.episode_length_buf)/self.max_episode_length * bonus * time_left )

        # (d) 전진 속도 보상 (필요시 스케일 조절)
        rew_vel   = forward_vel * self.cfg.rew_scale_vel

        # (e) 시간 패널티
        rew_time  = torch.ones_like(dist) * self.cfg.time_penalty

        # (f) 헤딩 오차 페널티 (래핑 적용)
        raw_err   = torch.atan2(goal[:,1] - root_xy[:,1], goal[:,0] - root_xy[:,0]) - yaw
        wrapped   = (raw_err + torch.pi) % (2*torch.pi) - torch.pi
        rew_head  = - torch.abs(wrapped) * self.cfg.rew_scale_head
        # ────────────────────────────────────────────────────────

        total_reward = (
            rew_alive
    #    + rew_dist
        + (~done_goal).float() * (rew_dist + rew_vel + rew_time + rew_head)

        + rew_goal
  #      + rew_vel
 #       + rew_time
#        + rew_head
        )
        return total_reward

    # def _get_rewards(self) -> torch.Tensor:
    #     # 1) 현재 위치/속도/방향 읽기
    #     root_xy  = self.robot.data.root_pos_w[:, :2]                # [N,2]
    #     goal      = torch.tensor([self.cfg.goal_x, self.cfg.goal_y],
    #                             device=self.device).unsqueeze(0)    # [1,2]
    #     dist      = torch.norm(root_xy - goal, dim=1)               # [N]

    #     lin_vel   = self.robot.data.root_lin_vel_b[:, :2]           # [N,2]
    #     quat      = self.robot.data.root_quat_w                     # [N,4]
    #     yaw       = self._yaw_from_quat(quat)                       # [N]
    #     # forward velocity (world→local projection)
    #     forward_vel = lin_vel[:,0]*torch.cos(yaw) + lin_vel[:,1]*torch.sin(yaw)

    #     # 2) done flags
    #     done_goal, done_time = self._get_dones()

    #     # 3) 기존 보상 항목
    #     rew_alive = self.cfg.rew_scale_alive * torch.ones_like(dist)
    #     # rew_dist  = self.cfg.rew_scale_dist  * dist
    #     # 예시: 거리 감소량에 반비례 보상
    #     delta_dist = self.prev_dist - dist
    #     rew_dist   = delta_dist * (-self.cfg.rew_scale_dist)  # 스케일은 양수로 두고 부호 반전
    #     self.prev_dist = dist.clone().detach()

    #     rew_goal  = done_goal.float() * self.cfg.rew_scale_goal
    #     # rew_vel   = forward_vel           * self.cfg.rew_scale_vel
    #     rew_time  = torch.ones_like(dist) * self.cfg.time_penalty

    #     # 4) **추가**: heading error penalty
    #     delta     = goal - root_xy                                  # [N,2]
    #     heading   = torch.atan2(delta[:,1], delta[:,0])            # [N]
    #     heading_err = torch.abs(heading - yaw)                     # [N]
    #     # rew_head  = - heading_err * 0.5                             # weight=0.5
    #     raw_err = heading - yaw
    #     wrapped_err = (raw_err + 3.1416) % (6.2832) - 3.1416
    #     rew_head = -torch.abs(wrapped_err) * 0.5

    #     # 5) 최종 합산
    #     total_reward = (
    #         rew_alive
    #       + rew_dist
    #       + rew_goal
    #     #   + rew_vel
    #       + rew_time
    #       + rew_head
    #     )
    #     return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root    = self.robot.data.root_pos_w[:, :2]
        goal    = torch.tensor([self.cfg.goal_x, self.cfg.goal_y], device=self.device).unsqueeze(0)
        dist    = torch.norm(root - goal, dim=1)
        done_goal = dist < 0.1
        time_out  = self.episode_length_buf >= self.max_episode_length - 1
        return done_goal, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        # 1) 기본 reset (joint/vel 등)
        super()._reset_idx(env_ids)

        # 2) per-env origin 가져오기
        #    scene.env_origins: [num_envs, 3]
        origins = self.scene.env_origins[env_ids]                  # [len,3]
        # 3) URDF init_state.pos: [x,y,z] 텐서로
        init_pos = torch.tensor(
            self.cfg.robot_cfg.init_state.pos,                   # 예: (-1.5,-1.5,0.01)
            device=self.device
        ).unsqueeze(0)                                            # [1,3]
        world_pos = origins + init_pos                           # [len,3]

        # 4) 목표 바라보도록 yaw 계산
        goal = torch.tensor(
            [self.cfg.goal_x, self.cfg.goal_y, 0.0],
            device=self.device
        ).unsqueeze(0)                                            # [1,3]
        delta = (goal - world_pos)[:, :2]                         # [len,2]
        yaw   = torch.atan2(delta[:,1], delta[:,0])               # [len]

        # 5) quaternion 생성 (roll=0, pitch=0, yaw)
        q = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw     # roll,pitch,yaw
        )  # [len,4]

        # 6) root pose ([x,y,z] + [qx,qy,qz,qw]) 만들어서 sim에 반영
        pose = torch.cat((world_pos, q), dim=1)                   # [len,7]
        self.robot.write_root_pose_to_sim(pose, env_ids)

        # 7) prev_dist 초기화 (reset 이후 첫 보상 계산용)
        dist = torch.norm(world_pos[:, :2] - goal[:, :2], dim=1)
        self.prev_dist[env_ids] = dist.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        1) policy가 내놓은 [-1,1] 범위의 raw actions을
           실제 물리 속도(v, ω)로 스케일 매핑
        """
        # raw action: torch.Tensor([N,2]) in [-1,1]
        a = actions
        v_cmd     = self.cfg.linear_scale  * a[:, 0]   # m/s
        omega_cmd = self.cfg.angular_scale * a[:, 1]   # rad/s
        # (선택) acceleration limiting: 
        # max_acc = 1.0  # m/s^2
        # v_cmd = torch.clamp(v_cmd, self.prev_v - max_acc*dt, self.prev_v + max_acc*dt)
        # omega_cmd = torch.clamp(omega_cmd, self.prev_omega - max_acc*dt, self.prev_omega + max_acc*dt)

        # 저장해두면 디버깅·logging에 편함
        self.v_cmd = v_cmd
        self.omega_cmd = omega_cmd
    
    def _apply_action(self) -> None:
        # (a) 준비
        wheel_base   = 0.16
        wheel_radius = 0.033
        dt           = self.cfg.sim.dt

        v     = self.v_cmd
        omega = self.omega_cmd

        # (b) 휠 각속도(rad/s) 계산
        v_l = (2 * v - omega * wheel_base) / (2 * wheel_radius)
        v_r = (2 * v + omega * wheel_base) / (2 * wheel_radius)

        # ─── ① 바퀴축(axis) 반전이 필요한 경우 부호 보정 ───
        # TurtleBot URDF 에서 왼쪽 바퀴 joint axis 가 (0 0 -1)이면
        sign_l, sign_r = -1.0, 1.0   # ← 필요에 따라 둘 다 1.0 또는 -1.0 로 바꿔보세요
        v_l = v_l * sign_l
        v_r = v_r * sign_r
        # ────────────────────────────────────────────────

        # (c) 속도 클램핑
        v_max = self.cfg.velocity_limit
        v_l = torch.clamp(v_l, -v_max, v_max)
        v_r = torch.clamp(v_r, -v_max, v_max)

        # (d) 시뮬레이터로 쓰기
        self.robot.set_joint_velocity_target(
            torch.stack((v_l, v_r), dim=1),
            joint_ids=[self.left_idx, self.right_idx]
        )
        self.robot.write_data_to_sim()
    '''def _apply_action(self) -> None:
        """
        differential drive → 휠별 velocity control
        """
        # (a) 준비
        wheel_base   = 0.16         # [m]
        wheel_radius = 0.033        # [m]
        dt = self.cfg.sim.dt

        v     = self.v_cmd         # from _pre_physics_step
        omega = self.omega_cmd

        # (b) 휠 각속도(rad/s) 계산
        v_l = (2*v - omega*wheel_base) / (2*wheel_radius)
        v_r = (2*v + omega*wheel_base) / (2*wheel_radius)

        # (c) 물리 한계 내로 클램핑
        v_l_max = self.cfg.velocity_limit  # =12.0 by config
        v_r_max = self.cfg.velocity_limit
        v_l = torch.clamp(v_l, -v_l_max, v_l_max)
        v_r = torch.clamp(v_r, -v_r_max, v_r_max)

        # (d) 휠 속도 타겟 바로 설정 (velocity control)
        self.robot.set_joint_velocity_target(
            torch.stack((v_l, v_r), dim=1),
            joint_ids=[self.left_idx, self.right_idx]
        )
        self.robot.write_data_to_sim()'''

    """def _post_physics_step(self) -> None:
        # 1) 시뮬에서 갱신된 “실제” state 읽기
        pos   = self.robot.data.root_pos_w.clone()    # [N,3]
        quat  = self.robot.data.root_quat_w.clone()   # [N,4]
        lin_v = self.robot.data.root_lin_vel_b.clone()# [N,3]
        ang_v = self.robot.data.root_ang_vel_b.clone()# [N,3]

        # 2) Z축 위치 고정
        pos[:, 2] = self.initial_z

        # 3) 수직/roll/pitch 속도 제거
        lin_v[:, 2] = 0.0      # Z축 속도
        ang_v[:, 0] = 0.0      # roll 속도
        ang_v[:, 1] = 0.0      # pitch 속도

        # 4) sim에 다시 써주기
        #    pose: [x,y,z] + [qx,qy,qz,qw] → [N,7]
        pose = torch.cat((pos, quat), dim=1)
        #    velocity: [vx,vy,vz] + [wx,wy,wz] → [N,6]
        vel  = torch.cat((lin_v, ang_v), dim=1)

        self.robot.write_root_pose_to_sim(pose, None)
        self.robot.write_root_velocity_to_sim(vel, None)"""


    @staticmethod
    def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        x, y, z, w = quat.unbind(dim=1)
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
