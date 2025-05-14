# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import os
import random
import torch

#import omni.replicator.core as rep
#from omni.replicator.core import AnnotatorRegistry, Writer

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors.camera import CameraCfg, Camera
from isaaclab.sensors import RayCasterCfg, patterns

import isaaclab_tasks.manager_based.classic.turtlebot.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.turtlebot import TUR_CFG  # isort:skip


@configclass
class TurtlebotSceneCfg(InteractiveSceneCfg):
    """Configuration for a turtlebot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # turtlebot
    robot: ArticulationCfg = TUR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    """height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint/Scanner",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )"""
    
    robot.init_state.pos = (-1.5, -1.5, 0)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    """maze = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Maze",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/kerker/Downloads/walls.usd",
        ),
       init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )"""

    tiled_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint/Camera",
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(pos=(0.0 , 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=512,
        height=512,
    )


    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/goal",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 1.5, 0.0)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the TurtleBot MDP."""

    wheel_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",  # 로봇 이름은 바뀌어도 됨
        joint_names=["wheel_left_joint", "wheel_right_joint"],
        scale=10.0  # 속도 범위 조절: [-1, 1] → [-10, 10] rad/s
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        goal_direction = ObsTerm(func=mdp.goal_direction_observation)
        goal_distance = ObsTerm(func=mdp.goal_distance_observation)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    """reset_goal_position = EventTerm(
        func=mdp.reset_goal_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("goal"),
            "center_position": (1.5, 1.5),
            "env_spacing":8.0,
        },
    )"""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Goal Distance
    goal_distance = RewTerm(
    func=mdp.distance_to_goal_reward,
    weight=1.0,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "goal_cfg": SceneEntityCfg("goal"),
        "success_threshold": 0.3,
        "success_reward": 5,
        },
    )
    velocity_penalty = RewTerm(
        func=mdp.velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    heading_alignment = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "goal_cfg": SceneEntityCfg("goal")
        }
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


##
# Environment configuration
##


@configclass
class TurtlebotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Turtlebot environment."""

    # Scene settings
    scene: TurtlebotSceneCfg = TurtlebotSceneCfg(num_envs=4096, env_spacing=8.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation