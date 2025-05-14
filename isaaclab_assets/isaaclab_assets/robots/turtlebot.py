from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg as UC

from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

##
# Configuration
##


TUR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/kerker/Downloads/turtlebot3_burger.usd",
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
            contact_offset=0.02,
            rest_offset=0.0,
    ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.5, -1.5, 0.02),
        joint_pos={
            "wheel_left_joint": 0.0,
            "wheel_right_joint": 0.0
        },
        joint_vel={
            "wheel_left_joint": 0.0,
            "wheel_right_joint": 0.0
        },
    ),
    actuators = {
        "left_wheel": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint"],
            effort_limit=6.0,        # URDF상의 max_torque ≈ 3.7 Nm → 약간 여유
            velocity_limit=25.0,     # 25 rad/s ≈ 0.215 m/s (r=0.043 m)
            stiffness=0.0,           # 순수 토크 지령
            damping=2.0,             # 아주 약한 비례 감쇠
        ),
        "right_wheel": ImplicitActuatorCfg(
            joint_names_expr=["wheel_right_joint"],
            effort_limit=6.0,
            velocity_limit=25.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
)
