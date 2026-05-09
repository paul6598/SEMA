# [SEMA : Semantic Embedding Multi-Agent systems]
이 저장소는 논문 SEMA의 공식 구현 코드를 담고 있습니다. \
이 프로젝트는 **BenchMARL** 프레임워크를 기반으로 확장되었습니다.

# How to use

## Install

#### Install TorchRL

```bash
pip install torchrl
```

#### Install BenchMARL
```bash
git clone https://github.com/paul6598/SEMA.git

pip install -r requirements.txt

pip install -e .
```
#### Install environments

##### VMAS

```bash
pip install vmas
```

## Run

본 프로젝트는 `vmas/give_way` 환경에서 **SEMA** 및 베이스라인(MAPPO, L2M2 등) 모델들의 실험을 쉽게 재현할 수 있는 자동화 스크립트를 제공합니다.
### 0. 관측정보 수정
본 프로젝트의 모델(SEMA)은 에이전트 구분을 위해 관측 정보 맨 앞에 **Agent ID (One-hot vector)**가 포함되어야 합니다.

따라서, miniconda3/envs/[가상환경 이름]/lib/python3.11/site-packages/vmas/scenarios/give_way.py 내부의 observation 함수를 다음과 같이 수정합니다.
```python
    def observation(self, agent: Agent):
        n_agent = len(self.world.agents)
        agent_idx = self.world.agents.index(agent)
        one_hot_id = torch.zeros(n_agent, device=self.world.device)
        one_hot_id[agent_idx] = 1.0
        one_hot_id = one_hot_id.unsqueeze(0).expand(len(agent.state.pos), -1)
        rel = []
        for a in self.world.agents:
            if a != agent:
                rel.append(agent.state.pos - a.state.pos)


        observations = [
            agent.state.pos,
            agent.state.vel,
        ]
        if self.observe_rel_pos:
            observations += rel
        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(*obs.shape, device=self.world.device,).uniform_(
                    -self.obs_noise,
                    self.obs_noise,
                )
                observations[i] = obs + noise
        return torch.cat(
            [one_hot_id] + observations,
            dim=-1,
        )
```
### 1. 스크립트 실행 위치
모든 실행 스크립트는 `scripts/` 폴더 내에 위치합니다.

```bash
cd scripts/
```

### 2. 주요 실행 예시
give_way.sh 스크립트는 총 4개의 인자(Argument)를 입력받아 실행됩니다.

```bash
bash give_way.sh [ALGO] [USE_WANDB] [USE_RENDER] [DESCRIPTION]
"""
ex)
bash give_way.sh mappo true false default

bash give_way.sh sema false false default
"""
```
| 인자 (Argument) | 설명 | 가능한 값 | 기본값 |
| :--- | :--- | :--- | :--- |
| **`ALGO`** | 학습에 사용할 알고리즘 | `sema`, `l2m2`, `mappo`, `qmix` | `mappo` |
| **`USE_WANDB`** | Weights & Biases 로그 사용 여부 | `true`, `false` | `false` |
| **`USE_RENDER`** | 비디오 렌더링 여부 | `true`, `false` | `false`* |
| **`DESCRIPTION`** | 실험 로그를 구분하기 위한 설명 | (예: `"test_run"`) | `"default"` |

현재 vLLM 환경과의 렌더링 코드의 충돌 방지를 위해, 학습 중 렌더링 옵션은 false로 고정하여 실행하는 것을 권장합니다.

### 3. 하이퍼파라미터 설정

스크립트 내부에 논문에 사용된 주요 하이퍼파라미터가 세팅되어 있습니다. 필요시 스크립트 내 변수들을 직접 수정하여 실험을 진행할 수 있습니다.

SEMA 주요 파라미터:
```bash 
N_WORKER=100 # 병렬적으로 실행되는 환경 수
LLM_INTERVAL=50 # LLM이 에이전트의 의도를 업데이트하는 주기
INTENTION_VECTOR=4 # 의도 벡터의 차원 수
```
