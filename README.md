본 문서는 Turtlebot3 Burger에서 구현한 Direct Task를 예시로 설명합니다.

Isaac Lab 구성

Isaac Lab에서 프로젝트를 구성할 때 구현해야 할 부분은 크게 로봇 정의(isaaclab_assets/isaaclab_assets/robots/Turtlebot.py, 이하 Turtlebot), 시뮬레이션 환경 설정 및 태스크 구현(isaaclab_tasks/isaaclab_tasks/direct/turtlebot/turtlebot_env.py, 이하 Turtlebot_env), 정책 신경망 구성(isaaclab_tasks/isaaclab_tasks/direct/turtlebot/agents/skrl_ppo_cfg.yaml, 이하 skrl_ppo_cfg)이다. 상술된 파일들은 isaaclab_tasks/isaaclab_tasks/direct/turtlebot/__init__.py에 의해 Isaac Lab에서 등록하여 사용한다.

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

__init__.py는 Isaac Lab의 전신인 Isacc Gym의 API를 사용해 어떤 파일들을 프로젝트에서 사용할지 결정하고, 학습을 실행할 때 프로젝트의 id를 설정하는 파일이다. entry_point에  env_cfg_entry_point, skrl_cfg_entry_point
