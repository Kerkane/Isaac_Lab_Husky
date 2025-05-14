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
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

torch_pi = torch.acos(torch.zeros(1)).item() * 2.0
# turtle_env_tensor.py
# Based strictly on Isaac Lab official documentation patterns

@configclass
class TurtlebotEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=2)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
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
    linear_scale: float = 0.5
    angular_scale: float = 1.2

    action_space = 2
    observation_space = 5
    state_space = 0
    robot_cfg: ArticulationCfg = TUR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    goal_cfg = GOAL_CFG

    goal_x: float = 1.5
    goal_y: float = 1.5
    velocity_limit: float = 12.0

    rew_scale_alive: float = 0.02
    rew_scale_dist: float = 0.02
    rew_scale_goal: float = 100.0
    rew_scale_vel: float = 1.0
    rew_scale_head: float = 0.05
    rew_scale_bonus: float = 50.0
    time_penalty: float = -0.01

class TurtlebotEnv(DirectRLEnv):
    cfg: TurtlebotEnvCfg

    def __init__(self, cfg: TurtlebotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        device = self.device
        # Precompute joint indices
        self.prev_dist = torch.zeros((self.num_envs),device=self.device, dtype=torch.float32)

        self.left_idx = self.robot.joint_names.index("wheel_left_joint")
        self.right_idx = self.robot.joint_names.index("wheel_right_joint")
        # Preserve z-height
        self.initial_z = float(self.cfg.robot_cfg.init_state.pos[2])

    def _setup_scene(self):
        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        TerrainImporterCfg.class_type(self.cfg.terrain)
        # Spawn robot and clone
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["turtlebot"] = self.robot
        # Compute per-env goal world positions
        origins2d = self.scene.env_origins[:, :2]  # [E,2]
        offset = torch.tensor([self.cfg.goal_x, self.cfg.goal_y], device=self.device)
        self.goal_world = origins2d + offset.unsqueeze(0)  # [E,2]
        # Visualize goals
        translations = self.scene.env_origins + torch.tensor(
            [self.cfg.goal_x, self.cfg.goal_y, 0.05], device=self.device
        ).unsqueeze(0)
        VisualizationMarkers(self.cfg.goal_cfg).visualize(translations=translations)
        # Initialize previous distance
                 # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        x, y, z, w = quat.unbind(dim=1)
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def _get_observations(self):
        # Positions and velocities
        root = self.robot.data.root_pos_w       # [E,3]
        quat = self.robot.data.root_quat_w      # [E,4]
        lin_b = self.robot.data.root_lin_vel_b  # [E,3]
        ang_w = self.robot.data.root_ang_vel_w  # [E,3]
        # Compute world-to-local transform
        yaw = self._yaw_from_quat(quat)         # [E]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        # Delta in world frame
        origins2d = self.scene.env_origins[:, :2]          # [E,2]
        delta = self.goal_world - root[:,:2]      # [E,2]
        # Local coordinates
        local_dx = cos_y*delta[:,0] + sin_y*delta[:,1]
        local_dy = -sin_y*delta[:,0] + cos_y*delta[:,1]
        # Heading error
        raw_err = torch.atan2(delta[:,1], delta[:,0]) - yaw
        wrapped = (raw_err + torch.pi) % (2*torch.pi) - torch.pi
        # Assemble observation: [dx, dy, heading_err, forward_vel, angular_vel]
        obs = torch.stack((
            local_dx,
            local_dy,
            wrapped,
            lin_b[:,0],
            ang_w[:,2]
        ), dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Current distance
        root_xy = self.robot.data.root_pos_w[:, :2]
        dist = torch.norm(root_xy - self.goal_world, dim=1)
        # Forward velocity
        lin_b = self.robot.data.root_lin_vel_b[:, :2]
        quat = self.robot.data.root_quat_w
        yaw = self._yaw_from_quat(quat)
        forward_vel = lin_b[:,0]*torch.cos(yaw) + lin_b[:,1]*torch.sin(yaw)
        # Done flags
        done_goal, _ = self._get_dones()
        active = (~done_goal).float()
        # Rewards
        rew_alive = self.cfg.rew_scale_alive * torch.ones_like(dist)
        delta_dist = self.prev_dist - dist
        rew_dist = delta_dist * self.cfg.rew_scale_dist
        self.prev_dist = dist.clone()
        rew_vel  = forward_vel * self.cfg.rew_scale_vel
        rew_time = torch.ones_like(dist) * self.cfg.time_penalty
        raw_err = torch.atan2(
            self.goal_world[:,1] - root_xy[:,1],
            self.goal_world[:,0] - root_xy[:,0]
        ) - yaw
        wrapped = (raw_err + torch.pi) % (2*torch.pi) - torch.pi
        rew_head = -torch.abs(wrapped) * self.cfg.rew_scale_head
        rew_goal = done_goal.float() * self.cfg.rew_scale_goal
        total = rew_alive + active*(rew_dist + rew_vel + rew_time + rew_head) + rew_goal
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_xy = self.robot.data.root_pos_w[:, :2]
        dist = torch.norm(root_xy - self.goal_world, dim=1)
        done_goal = dist < 0.1
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return done_goal, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
     
        # Compute new poses
        origins2d = self.scene.env_origins[:, :2]
        init2d = torch.tensor(self.cfg.robot_cfg.init_state.pos[:2], device=self.device)
        world_xy = origins2d + init2d.unsqueeze(0)
        # Yaw to face goal
        delta = world_xy - self.goal_world
        yaw = torch.atan2(delta[:,1], delta[:,0])
        q = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        # # Full pose and reset
        # pos3d = torch.cat((world_xy, torch.full((len(env_ids),1), self.initial_z, device=self.device)), dim=1)
        # pose = torch.cat((pos3d, q), dim=1)
        # self.robot.write_root_pose_to_sim(pose, env_ids)
        # # Update prev_dist
        # new_dist = torch.norm(world_xy - self.goal_world, dim=1)
        # self.prev_dist[env_ids] = new_dist
    # 6) root pose ([x,y,z] + [qx,qy,qz,qw]) 만들어서 sim에 반영
    
        # 4) z 높이 보존해서 pose 텐서 만들기 (한 번만 인덱싱)
        z_col = torch.full((len(env_ids), 1), self.initial_z, device=self.device)
        pos3d = torch.cat((world_xy[env_ids], z_col), dim=1)          # [len(env_ids),3]
        quat_sel = q[env_ids]                                         # [len(env_ids),4]
        pose     = torch.cat((pos3d, quat_sel), dim=1)                # [len(env_ids),7]

        # 5) PhysX에 한 번만 넘기기
        self.robot.write_root_pose_to_sim(pose, env_ids)

        # 6) prev_dist 업데이트 (reset 직후 첫 보상용)
        new_dist = torch.norm(world_xy[env_ids] - self.goal_world[env_ids], dim=1)
        self.prev_dist[env_ids] = new_dist
                
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Scale raw actions to velocities
        self.v_cmd = actions[:,0] * self.cfg.linear_scale
        self.omega_cmd = actions[:,1] * self.cfg.angular_scale
        #if torch.rand(1).item() < 0.001:
            #print(f"[DEBUG] v_cmd={self.v_cmd[0]:.3f}, omega_cmd={self.omega_cmd[0]:.3f}")

    def _apply_action(self) -> None:
        base = 0.32
        rad = 0.033
        v_l = (2*self.v_cmd - self.omega_cmd*base) / (2*rad)
        v_r = (2*self.v_cmd + self.omega_cmd*base) / (2*rad)
        # Clamp and apply
        v_l = torch.clamp(v_l, -self.cfg.velocity_limit, self.cfg.velocity_limit)
        v_r = torch.clamp(v_r, -self.cfg.velocity_limit, self.cfg.velocity_limit)
        #if torch.rand(1).item() < 0.001:  # 0.1% 확률만 출력
            #print(f"[WHEEL TARGETS] v_l={v_l[0]:.3f} rad/s, v_r={v_r[0]:.3f} rad/s")
        targets = torch.stack((v_l, v_r), dim=1)
        self.robot.set_joint_velocity_target(targets, joint_ids=[self.left_idx, self.right_idx])
        self.robot.write_data_to_sim()

    def _post_physics_step(self) -> None:
        # 1) 물리 엔진이 갱신한 실제 state 읽기
        #    root_lin_vel_b: [E,3] world-frame linear velocity
        lin_vel = self.robot.data.root_lin_vel_b        # [num_envs, 3]
        forward_actual = lin_vel[:, 0]                  # x축 성분 = 전진 속도 (m/s)

        # 2) (옵션) 로깅: 0.1% 확률로만 출력해서 과부하 방지
        if torch.rand(1).item() < 0.001:
            print(f"[ACTUAL SPEED] forward={forward_actual[0]:.3f} m/s")

        # 3) z축 튐 방지:   
        #    root_pos_w: [E,3], root_lin_vel_b: [E,3], root_ang_vel_b: [E,3]
        pos   = self.robot.data.root_pos_w.clone()     # [E,3]
        lin_v = lin_vel.clone()                         # [E,3]
        ang_v = self.robot.data.root_ang_vel_b.clone()  # [E,3]

        #    (a) z 위치를 초기 높이로 고정
        pos[:, 2]   = self.initial_z

        #    (b) 수직 속도 및 roll/pitch 속도 제거
        lin_v[:, 2] = 0.0      # z축 속도
        ang_v[:, 0] = 0.0      # roll 속도
        ang_v[:, 1] = 0.0      # pitch 속도

        # 4) sim에 다시 써 주기
        #    pose: [x,y,z] + [qx,qy,qz,qw] → [E,7]
        pose = torch.cat((pos, self.robot.data.root_quat_w), dim=1)
        #    velocity: [vx,vy,vz] + [wx,wy,wz] → [E,6]
        vel  = torch.cat((lin_v, ang_v), dim=1)

        self.robot.write_root_pose_to_sim(pose, None)
        self.robot.write_root_velocity_to_sim(vel, None)
