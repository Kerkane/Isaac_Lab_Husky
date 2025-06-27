본 문서는 Turtlebot3 Burger에서 구현한 Direct Task를 예시로 설명합니다.

Isaac Lab 구성

  Isaac Lab에서 프로젝트를 구성할 때 구현해야 할 부분은 크게 로봇 정의(isaaclab_assets/isaaclab_assets/robots/Turtlebot.py), 시뮬레이션 환경 설정 및 태스크 구현(isaaclab_tasks/isaaclab_tasks/direct/turtlebot/turtlebot_env.py), 정책 신경망 구성(isaaclab_tasks/isaaclab_tasks/direct/turtlebot/agents/skrl_ppo_cfg.yaml)이다. 상술된 파일들은 isaaclab_tasks/isaaclab_tasks/direct/turtlebot/__init__.py에 의해 Isaac Lab에서 등록하여 사용한다.

1. __init__.py

```
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Turtlebot-Direct-v0",
    entry_point=f"{__name__}.turtlebot_env:TurtlebotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot_env:TurtlebotEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

  __init__.py는 Isaac Lab의 전신인 Isacc Gym의 API를 사용해 어떤 파일들을 프로젝트에서 사용할지 결정하고, 학습을 실행할 때 프로젝트의 id를 설정하는 파일이다. id 이하 항목에 f"{__name__}.turtlebot_env:TurtlebotEnv"와 같이 파일 이름:클래스 이름으로 구성된 값을 할당해야 한다. entry_point에는 프로젝트의 DirectRLEnv 클래스를 등록하고, env_cfg_entry_point에는 프로젝트의 DirectRLEnvCfg 클래스를 등록한다. 마지막으로 skrl_cfg_entry_point에는 정책 신경망 구성 파일을 등록한다.
  예시의 프로젝트는 터미널에서 다음 명령어와 같이 id를 통해 어떤 프로젝트인지를 인식하여 실행된다.
```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Turtlebot-Direct-v0 
```

2. Turtlebot.py

```
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg as UC

from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

TUR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/Downloads/turtlebot3_burger.usd",
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
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            effort_limit=6.0,        
            velocity_limit=25.0,     
            stiffness=0.0,           
            damping=2.0,             
        ),
    },
)
```
  로봇 구성 파일인 Turtlebot.py는 ArticulationCfg 클래스를 정의하여 로봇의 종류와 물리적인 성질을 결정한다. spawn에서 로봇의 urdf나 usd 파일을 불러와 어떤 로봇을 사용할 지 결정하며, 오류 발생을 낮추기 위해 urdf를 그대로 사용하기보다는 Isaac Sim에서 움직임에 관여하는 joint를 velocity로 설정한 후 import하여 usd 파일로 저장한 후 사용하는 것을 권장한다. rigid_props에서 로봇의 강체 속성 여부(rigid_body_enabled)와 최대 속도(max_linear_velocity, max_angular_velocity), 그리고 자이로스코픽 효과를 통해 보다 현실적인 움직임을 구현할 지(gyroscopic_forces) 결정할 수 있다. init_state를 통해 로봇과 joint의 초기 위치 및 속도를 지정한다. 마지막으로 actuators에서는 로봇의 joint들을 제어한다.
