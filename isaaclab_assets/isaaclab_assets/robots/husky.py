from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg as UC

from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

##
# Configuration
##


HUSKY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/kerker/Downloads/husky_robot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,         # 충돌 오프셋
            rest_offset=0.0,             # 정지 오프셋
            # torsional_patch_radius 등 추가 가능
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-5, -5, 0.02),
        joint_pos={
            "front_left_wheel": 0.0,
            "front_right_wheel": 0.0,
            "rear_left_wheel": 0.0,
            "rear_right_wheel": 0.0
        },
        joint_vel={
            "front_left_wheel": 0.0,
            "front_right_wheel": 0.0,
            "rear_left_wheel": 0.0,
            "rear_right_wheel": 0.0
        },
    ),
    actuators = {
    "left_wheels": ImplicitActuatorCfg(
        joint_names_expr=[
            "front_left_wheel", "rear_left_wheel"
        ],
        effort_limit    = 40.0,     # N·m  (실차 모터 10.7 A × 24 V ≈ 257 W → 토크 ≈ 30–40 N·m 추정)
        velocity_limit  = 20.0,     # rad/s (실속도 11 → 약간 여유)
        stiffness       = 0.0,      # 토크 직접 지령
        damping         = 0.5,      # 수치 발진 억제
    ),
    "right_wheels": ImplicitActuatorCfg(
        joint_names_expr=[
            "front_right_wheel", "rear_right_wheel"
        ],
        effort_limit    = 40.0,
        velocity_limit  = 20.0,
        stiffness       = 0.0,
        damping         = 0.5,
    ),
    },
)
