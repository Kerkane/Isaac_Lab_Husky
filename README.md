본 문서는 Turtlebot3 Burger 로봇, skrl 라이브러리와 PPO 알고리즘, Direct Task를 기반으로 한 프로젝트를 예시로 설명한다.

  ./isaaclab.sh --new 명령어를 통해 새로운 Isaac Lab 프로젝트를 생성할 수 있다. Isaac Lab에서 새로 만들어진 프로젝트를 구성할 때 구현해야 할 부분은 크게 로봇 정의(isaaclab_assets/isaaclab_assets/robots/turtlebot.py), 시뮬레이션 환경 설정 및 태스크 구현(isaaclab_tasks/isaaclab_tasks/direct/turtlebot/turtlebot_env.py), 정책 신경망 구성(isaaclab_tasks/isaaclab_tasks/direct/turtlebot/agents/skrl_ppo_cfg.yaml)이다. 상술된 파일들은 isaaclab_tasks/isaaclab_tasks/direct/turtlebot/__init__.py에 의해 Isaac Lab에서 등록하여 사용한다.
참고 문서: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/index.html

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

  __init__.py는 Isaac Lab의 전신인 Isacc Gym의 API를 사용해 어떤 파일들을 프로젝트에서 사용할지 결정하고, 학습을 실행할 때 task id를 설정하는 파일이다. id 이하 항목에 f"{__name__}.turtlebot_env:TurtlebotEnv"와 같이 파일 이름:클래스 이름으로 구성된 값을 할당해야 한다. entry_point에는 프로젝트의 DirectRLEnv 클래스를 등록하고, env_cfg_entry_point에는 프로젝트의 DirectRLEnvCfg 클래스를 등록한다. 마지막으로 skrl_cfg_entry_point에는 정책 신경망 구성 파일을 등록한다.

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
  참고 문서: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html

3. skrl_ppo_cfg.yaml

```
seed: 42

models:
  separate: False
  policy:    # 정책 신경망 설정
    class: GaussianMixin    # 확률적 정책(stochastic policy)
    clip_actions: False    # action 클리핑 여부(False일 경우 행동을 action 범위 밖으로 나가는 걸 허용)
    clip_log_std: True    # 로그 표준편차 클리핑 여부
    min_log_std: -20.0    #정규분포의 stddev 최소값
    max_log_std: 2.0    #정규분포의 stddev 최소값
    initial_log_std: 0.0    #초기 stddev 값
    network:    #신경망 설정
      - name: net
        input: STATES
        layers: [64, 64]    # 2개의 64차원 은닉층
        activations: elu    # 활성 함수 설정(elu)
    output: ACTIONS
  value:    # 가치 신경망 설정
    class: DeterministicMixin    #State의 가치 함수를 추정하기 위한 단일 값 출력
    clip_actions: False    # action 클리핑 여부(False일 경우 행동을 action 범위 밖으로 나가는 걸 허용)
    network:
      - name: net
        input: STATES
        layers: [64, 64]    # 2개의 64차원 은닉층
        activations: elu    # 활성 함수 설정(elu)
    output: ONE


memory:    # 학습 중 데이터를 저장할 메모리 설정
  class: RandomMemory
  memory_size: -1    # 자동으로 rollouts * timesteps 크기로 설정

agent:
  class: PPO    # 강화학습에 사용할 알고리즘 설정
  rollouts: 16    # 병렬 환경에서 총합 몇 번의 에피소드를 반복하여 데이터를 수집할지 설정
  learning_epochs: 8    # 전체 미니배치에 대한 반복 횟수 설정
  mini_batches: 8    # rollout만큼 모인 데이터를 몇 개의 미니배치로 나눠 학습시킬지 설정
  discount_factor: 0.99    # 시간에 대한 보상 할인률
  lambda: 0.95    #GAE (Generalized Advantage Estimation)의 λ
  learning_rate: 1.0e-04    # 학습률
  learning_rate_scheduler: KLAdaptiveLR    # KLAdaptiveLR 스케쥴러 사용
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.008    # KL 발산이 0.008 이상이면 학습률 감소

  state_preprocessor: RunningStandardScaler
  state_preprocessor_kwargs: null
  value_preprocessor: RunningStandardScaler
  value_preprocessor_kwargs: null

  random_timesteps: 0
  learning_starts: 0
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: True

  entropy_loss_scale: 0.01    # 탐험에 대한 보상 설정
  entropy_loss_schedule: linear_decay
  entropy_loss_schedule_kwargs:
    start_value: 0.01
    end_value: 0.005
    max_timesteps: 20000
  value_loss_scale: 1.0
  kl_threshold: 0.05

  rewards_shaper_scale: 1.0
  time_limit_bootstrap: True

  experiment:
    directory: "turtlebot_direct"    # 로그 저장 디렉토리
    experiment_name: ""
    write_interval: 1000    # 로그 작성 간격
    checkpoint_interval: 2000    # 정책 파일 저장 간격

trainer:
  class: SequentialTrainer
  timesteps: 20000    #학습에 사용할 시간 설정
  environment_info: log
```

  정책 신경망 구성 파일인 skrl_ppo_cfg.yaml에서는 정책/가치 신경망과 강화학습 알고리즘을 설정하고 rollouts/learning_epochs/mini_batches를 통해 신경망에 업데이트할 데이터의 크기를 결정한다. 또한 로봇이 학습된 행동을 벗어나 새로운 시도를 하려는 탐험을 얼마나 장려할 지 entropy_loss_scale로 설정하고, 로그와 정책 파일 저장 간격과 총 학습 시간을 설정할 수 있다. entropy_loss_scale은 학습 초반엔 탐험을 장려하고 학습이 진행되면서 점점 모험의 비중을 줄여나가는 linear_decay가 효과적이다.

4. turtlebot_env.py

```
TurtlebotEnvCfg (환경 설정 클래스)
└── 시뮬레이션, 로봇, 목표, 장애물, 바퀴, 행동 범위 등 하드웨어/시뮬레이션 속성 정의

TurtlebotEnv (환경 구현 클래스)
├── _setup_scene       : 환경 초기화 (로봇/지면/목표 등 삽입)
├── _pre_physics_step  : 시뮬레이션 한 스텝 전, action을 로봇의 속도로 해석
├── _apply_action      : 로봇에 적용하고자 하는 선속도/각속도를 바퀴 속도로 변환
├── _get_observations  : 관측값 계산
├── _get_rewards       : 보상 계산
├── _get_dones         : episode 종료 조건
└── _reset_idx         : 환경 리셋 (커리큘럼 포함)
```

우선 TurtlebotEnvCfg(환경 설정 클래스)에서 시뮬레이션 자체의 설정과 시뮬레이션에서 사용할 설정(cfg) 클래스를 선언한다.

  sim: 시뮬레이션 시간 간격 (dt=1/60)과 렌더링 주기
  
  scene: 병렬 환경 개수 (num_envs=4096), 환경 간 간격 설정
  
  robot_cfg: turtlebot.py의 로봇 cfg 클래스 불러오기
  
  goal_cfg: 목표 지점 시각화룰 위해 물리적 상호작용이 없는 VisualMarker 설정
  
  terrain: Ground Plane 설정
  
  object0_cfg, object1_cfg: 장애물 2개 (미사용 시 주석 처리됨)
  
  goal_x, goal_y: 기본 목표 위치
  
  R_WHEEL, WHEELBASE: action에 사용할 로봇의 물리값인 바퀴 반지름, 바퀴 사이의 간격
  
  MAX_LIN, MAX_ANG, MAX_OMEGA: 로봇의 최대 선속도, 각속도, 바퀴 속도 제한
  
  curriculum: Curriculum Learning을 위한 난이도 증가용 총 스텝 수 설정 (20,000)

이후 TurtlebotEnv(환경 구현 클래스)에서 강화학습에 필요한 환경을 구현한다.

  __init__: 클래스에서 사용할 상수(각 보상에 적용할 계수, 목표 도달 판정 거리, ...)와 버퍼를 선언한다. action에서 사용할 joint id를 검색한다.
  
  _setup_scene: 한 환경에 Ground Plane, 로봇, 오브젝트를 넣고 그 환경을 복제하여 병렬 환경을 구성한다.
  
  _pre_physics_step: 정책 신경망이 도출한 action을 로봇의 선속도와 각속도로 변환하고 과격한 움직임을 제한하기 위해 최대값을 제한한다.
  
  _apply_action: action에 맞는 로봇의 선속도와 각속도를 구현하기 위해 좌/우 바퀴에 적용해야 할 속도를 계산하고 joint id를 사용해 joint에 속도 명령을 적용한다.
  참고 문서: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/mobile_robot_controllers.html
  
  _get_observations: 관측값 버퍼인 obs를 구성한다.
  
  _get_rewards: 보상을 계산한다.
  
  _get_dones: 에피소드 종료 조건을 설정한다. return의 첫 번째 항목은 목표 성공에 따른 에피소드 종료, 두 번째 항목은 에피소드 길이 초과로 인한 타임아웃이다.
  
  _reset_idx: 에피소드가 종료된 병렬 환경과 환경에서 사용한 버퍼를 리셋하고 로봇과 목표 지점 마커, 오브젝트의 위치를 초기화한다. Curriculum Learning을 사용한 로봇 위치, 로봇 각도, 목표 지점 위치에 대한 랜덤화 로직을 적용하였다.


5. Isaac Lab 실행

  Isaac Lab 프로젝트는 기본적으로 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py 명령어를 통해 실행되며, 편의성을 위한 여러 flag가 존재한다.
  
  --task: task id를 사용해 어떤 프로젝트를 실행할 지 결정한다.
  
  --num_envs: 병렬 환경의 개수를 설정한다.
  
  --headless: Isaac Sim GUI를 실행하지 않고 백그라운드에서 학습을 실행한다. 랜더링 과정이 생략되기 때문에 학습 속도가 매우 빨라진다.
  
  --video: checkpoint마다 학습 과정의 영상을 저장한다.
  
  --video_length: 저장할 영상의 길이를 설정한다.

다음의 명령어는 Isaac-Turtlebot-Direct-v0 id를 가진 프로젝트를 16개의 병렬 환경에서 실행한다.

```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Turtlebot-Direct-v0 --num_envs 16 --headless --video --video_length 300
```
