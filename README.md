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
from isaaclab.sim.schemas import CollisionPropertiesCfg

TUR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/Downloads/turtlebot3_burger.usd",                     #로봇 설정 파일(urdf, usd) 경로 설정
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,                                          #강체 속성 활성화 여부
            max_linear_velocity=1000.0,                                       #최대 선속도
            max_angular_velocity=1000.0,                                      #최대 각속도
            enable_gyroscopic_forces=True,                                    #자이로스코픽 효과 활성화 여부
            ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,                                    #자가 충돌 활성화 여부
            solver_position_iteration_count=4,                                #위치 조정 연산 반복 횟수
            solver_velocity_iteration_count=0,                                #속도 조정 연산 반복 횟수
        ),
        collision_props=CollisionPropertiesCfg(
            collision_enabled=True,                                           #충돌 활성화
            contact_offset=0.02,                                              #접촉 판정 거리
            rest_offset=0.0,                                                  #완전히 붙은 상태로 간주하는 거리
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
            joint_names_expr=["wheel_left_joint"],                           #joint 등록          
            effort_limit=6.0,                                                #joint에 가해지는 최대 토크 제한     
            velocity_limit=25.0,                                             #joint의 최대 각속도 제한     
            stiffness=0.0,                                                   #PD 제어에서 위치 제어를 위한 P 게인(스프링)
            damping=2.0,                                                     #PD 제어에서 속도 차이에 따른 감쇠력을 나타내는 D 게인(댐프)
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
```
  로봇 구성 파일인 Turtlebot.py는 ArticulationCfg 클래스를 정의하여 로봇의 종류와 물리적인 성질을 결정한다. spawn에서 로봇의 urdf나 usd 파일의 경로를 불러와 어떤 로봇을 사용할 지 결정하며, 오류 발생을 낮추기 위해 urdf를 그대로 사용하기보다는 Isaac Sim에서 움직임에 관여하는 joint를 velocity로 설정한 후 import하여 usd 파일로 저장한 후 사용하는 것을 권장한다. rigid_props에서 로봇의 강체 속성 여부와 최대 속도, 그리고 자이로스코픽 효과를 통해 보다 현실적인 움직임을 구현할 지 결정할 수 있다. articulation_props에서는 로봇의 자가 충돌과 위치/ init_state를 통해 로봇 전체의 물리적 속성을 설정하며, 자가 충돌 여부와 위치/속도를 계산하여 로봇에 적용할 때 연산 반복 횟수를 지정한다. collision_props는 충돌 관련 설정으로, 충돌 여부와 접촉 판정 거리를 설정한다. init_state에서는 joint의 초기 위치 및 속도를 지정한다. 마지막으로 actuators에서는 로봇의 joint들을 제어한다. Isaac Lab에서는 실제 모터와 더 흡사한 구동을 구현하기 위한 여러 액추에이터들이 구현되어 있는데, 각각의 action에 관여할 joint들을 서로 다른 그룹("left_wheel", "right_wheel")으로 구성하여 joint_names_expr에 각 그룹을 구성할 joint의 이름을 작성한다. joint에 가해질 최대 토크, 최대 각속도를 설정하여 이를 초과하면 클램핑되도록 설정하며, PD 제어에서 사용할 P/D 게인을 설정한다.
