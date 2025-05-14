from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg, AssetBase
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat
from isaaclab.markers import VisualizationMarkers
from isaaclab_assets.robots.turtlebot import TUR_CFG
from .goal import GOAL_CFG
from isaaclab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
@configclass
class TurtlebotEnvCfg(DirectRLEnvCfg):
    # 시뮬레이션 설정(timestep, episode length, action/observation/state space)
    sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=4)
    decimation = 4
    episode_length_s = 25.0

    action_space = 2
    observation_space = 4
    state_space = 0

    # 로봇 usd 받아오기(turtlebot) & 병렬화 설정
    robot_cfg: ArticulationCfg = TUR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=12.0, replicate_physics=True
    )
    goal_cfg = GOAL_CFG

    # 시뮬레이션 상수 설정(목표 위치, 로봇 바퀴 반경, 바퀴 간 거리, 최대 선속도/각속도)
    goal_x = 5
    goal_y = 5

    R_WHEEL   = 0.178   # 바퀴 반경 [m]
    WHEELBASE = 0.571   # 좌·우 바퀴간 거리 [m]

    # 실측 한계 (datasheet 기준) ─ 수정 가능
    MAX_LIN   = 2.0    # m/s
    MAX_ANG   = 10.0    # rad/s
    MAX_OMEGA = 20.0    # 바퀴 자체 각속도(rad/s) 상한 (= velocity_limit)

    # ground plane 삽입하는 config
    terrain = TerrainImporterCfg(
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

    # usd 파일을 rigid object로 삽입하는 config
    # maze_cfg = RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Maze",
    #             spawn=sim_utils.UsdFileCfg(
    #                 usd_path="/home/kerker/Downloads/obstacles.usd",
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #         )

    #장애물 삽입
    object0_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object0",
        spawn=CuboidCfg(
                size=(1, 0.2, 0.3),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True # 동적 객체 비활성화
                ),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True   # 충돌 활성화
                ),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0
                ),
                visible=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.0, 0.56), rot=(0.0, 0.0, 0.0, 0.0)),
    )
    object1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object1",
        spawn=CuboidCfg(
                size=(0.2, 1, 0.3),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True # 동적 객체 비활성화
                ),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True   # 충돌 활성화
                ),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0
                ),
                visible=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, -0.0, 0.0), rot=(0.0, 0.0, 0.0, 0.0)),
    )


class TurtlebotEnv(DirectRLEnv):
    cfg: TurtlebotEnvCfg

    def __init__(self, cfg: TurtlebotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # reward 관련 상수 설정
        self.heading_coefficient = 0.25
        self.threshold = 0.5
        self.rew_scale_prog  = 2.0     
        self.rew_scale_goal  = 10.0 
        self.rew_scale_head  = 0.05
        self.goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.goal_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.markers_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        
        # joint id 매칭
        self._wheel_dof_idx, _ = self.robot.find_joints([
            "front_left_wheel", "front_right_wheel","rear_left_wheel","rear_right_wheel"
            ])
    def _setup_scene(self):

        # ground plane, robot, maze or object 삽입
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(size=(100.0, 100.0))
        )
        self.robot = Articulation(self.cfg.robot_cfg)
        self.goal = VisualizationMarkers(self.cfg.goal_cfg)
        #self.maze = RigidObject(self.cfg.maze_cfg)
        self.object0 = RigidObject(self.cfg.object0_cfg)
        self.object1 = RigidObject(self.cfg.object1_cfg)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 환경 복제 & scene에 robot, maze or object 등록
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["turtlebot"] = self.robot
        # self.scene.rigid_objects["maze"] = self.maze
        self.scene.rigid_objects["object0"] = self.object0
        self.scene.rigid_objects["object1"] = self.object1

        # 빛 삽입
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        # 선속도/각속도 최대값을 상속
        v_scale = self.cfg.MAX_LIN
        w_scale = self.cfg.MAX_ANG

        #action 적용
        self._v_cmd = actions[:, 0] * v_scale
        self._w_cmd = actions[:, 1] * w_scale
   
        # 로깅/시각화를 위해 state 버퍼도 유지
        self.v_cmd  = self._v_cmd
        self.w_cmd  = self._w_cmd

    def _apply_action(self) -> None:

        # Differential controller 공식에 따라 좌/우 각속도 구현
        # 참고 문서(https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/mobile_robot_controllers.html)
        omega_L = (2 * self._v_cmd - self._w_cmd * self.cfg.WHEELBASE) / (2 * self.cfg.R_WHEEL)
        omega_R = (2 * self._v_cmd + self._w_cmd * self.cfg.WHEELBASE) / (2 * self.cfg.R_WHEEL)
        
        # 급격한 변화를 방지하기 위해 속도 클램핑
        wheel_vel = torch.stack((omega_L, omega_R), dim=1)
        wheel_vel = torch.clamp(wheel_vel, -self.cfg.MAX_OMEGA, self.cfg.MAX_OMEGA)
        
        # 좌/우 바퀴들에 같은 속도 적용
        wheel_target = wheel_vel.repeat(1, 2)

        # f_l, f_r, r_l, r_r 순서대로 속도 적용
        self.robot.set_joint_velocity_target(
            wheel_target,
            joint_ids=self._wheel_dof_idx              # [fl, fr, rl, rr]
        )

    def _get_observations(self) -> dict:

        # observation(1) : 로봇과 목표 지점 사이의 거리
        delta = self.goal_pos - self.robot.data.root_pos_w[:, :2]
        self.prev_dist = self.dist.clone()
        self.dist = torch.norm(delta, dim=-1)

        # observation(2) : 로봇의 현재 방향과 목표 지점 방향 사이의 차
        heading_robot = self.robot.data.heading_w
        heading_goal = torch.atan2(delta[:, 1], delta[:, 0])
        self.heading_error = torch.atan2(torch.sin(
            heading_goal - heading_robot),torch.cos(heading_goal - heading_robot)
            ) 

        # observation(3) : 로봇의 현재 방향 전진 속도
        lin_vel_b = self.robot.data.root_lin_vel_b

        # obs 구성(연속적이고 부드러운 움직임을 위해 heading error 값을 cos/sin으로 분해)
        obs = torch.cat([
            self.dist.unsqueeze(dim=1),
            torch.cos(self.heading_error).unsqueeze(dim=1),
            torch.sin(self.heading_error).unsqueeze(dim=1),
            lin_vel_b[:,0].unsqueeze(dim=1),
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:

        # reward(1) : 목표 지점까지의 거리가 짧아졌는가?(이전 timestep 거리 - 현재 timestep 거리)
        prog_rew = self.prev_dist - self.dist
    
        # reward(2) : 목표 지점으로의 방향이 일치하는가?(-heading error의 exp)
        target_heading_rew = torch.exp(-torch.abs(self.heading_error) / self.heading_coefficient)
        
        # reward(3) : 목표 지점에 도달했는가?(목표 지점까지의 거리를 threshold와 비교)
        self.goal_reached = self.dist < self.threshold

        # 전체 reward 계산
        rew = (
            prog_rew * self.rew_scale_prog
            + target_heading_rew * self.rew_scale_head
            + self.goal_reached * self.rew_scale_goal
        )

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        # timestep 초과
        timeout = self.episode_length_buf > self.max_episode_length
        return timeout, self.goal_reached

    def _reset_idx(self, env_ids: Sequence[int] | None):

        #env_ids가 None일 경우 로봇 수만큼 처리
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 기본 상태 받아오기
        default_state   = self.robot.data.default_root_state[env_ids]
        root_pose       = default_state[:, :7]
        root_vel        = default_state[:, 7:]
        joint_pos       = self.robot.data.default_joint_pos[env_ids]
        joint_vel       = self.robot.data.default_joint_vel[env_ids]

        # 환경 원점에 대해 초기화 & 0.5 이내로 랜덤화
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 0] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        root_pose[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)

        # 초기 각도 랜덤화(±50°)
        rand_yaw = (torch.rand(len(env_ids), device=self.device)) * 1
        root_pose[:, 3] = torch.cos(rand_yaw * 0.5)
        root_pose[:, 6] = torch.sin(rand_yaw * 0.5)

        # 시뮬레이터에 write하여 로봇 초기화
        self.robot.write_root_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 목표 지점 초기화 & 0.5 이내로 랜덤화
        self.goal_pos[env_ids, 0] = self.cfg.goal_x
        self.goal_pos[env_ids, 1] = self.cfg.goal_y
        self.goal_pos[env_ids, 0] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        self.goal_pos[env_ids, 1] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        self.goal_pos[env_ids] += self.scene.env_origins[env_ids, :2]
        self.markers_pos[env_ids, :2] = self.goal_pos[env_ids]
        self.goal.visualize(translations=self.markers_pos)

        # 버퍼 초기화
        delta = self.goal_pos - self.robot.data.root_pos_w[:, :2]
        self.dist = torch.norm(delta, dim=-1)
        self.prev_dist = self.dist.clone()
        heading_robot = self.robot.data.heading_w[:]
        heading_goal = torch.atan2(delta[:, 1], delta[:, 0])
        self._h_err = torch.atan2(torch.sin(
            heading_goal - heading_robot),torch.cos(heading_goal - heading_robot)
            ) 
        self._prev_h_err = self._h_err.clone()

