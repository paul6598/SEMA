#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type


import torch
import json
import math
import re
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

from transformers import pipeline, logging, BitsAndBytesConfig
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Transform, StepCounter

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


class TextFusionModule(nn.Module):
    def __init__(self, intention_dim):
        super().__init__()
        # 모듈이 생성될 때 자신만의 독립적인 압축기를 가집니다.
        self.compressor = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, intention_dim),
            # nn.Tanh()
        )

    def forward(self, obs, sbert):
        compressed = self.compressor(sbert)
        return torch.cat([obs, compressed], dim=-1)

class Larlwheel(Algorithm):
    """Multi Agent PPO (from `https://arxiv.org/abs/2103.01955 <https://arxiv.org/abs/2103.01955>`__).

    Args:
        share_param_critic (bool): Whether to share the parameters of the critics withing agent groups
        clip_epsilon (scalar): weight clipping threshold in the clipped PPO loss equation.
        entropy_coef (scalar): entropy multiplier when computing the total loss.
        critic_coef (scalar): critic loss multiplier when computing the total
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1".
        lmbda (float): The GAE lambda
        scale_mapping (str): positive mapping function to be used with the std.
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        use_tanh_normal (bool): if ``True``, use TanhNormal as the continuyous action distribution with support bound
            to the action domain. Otherwise, an IndependentNormal is used.
        minibatch_advantage (bool): if ``True``, advantage computation is perfomend on minibatches of size
            ``experiment.config.on_policy_minibatch_size`` instead of the full
            ``experiment.config.on_policy_collected_frames_per_batch``, this helps not exploding memory usage

    """

    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        entropy_coef_decay: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        LLM_interval: int,
        LLM_interval_decay: bool,
        intention_vector_size: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.minibatch_advantage = minibatch_advantage
        self.LLM_interval = LLM_interval
        self.LLM_interval_decay = LLM_interval_decay
        self.intention_vector_size = intention_vector_size
        # self.pipe = pipeline(
        #     "text-generation",
        #     model="google/gemma-3-4b-it",
        #     device="cuda",
        #     torch_dtype=torch.bfloat16
        # )
        # self.pipe.model.generation_config.max_length = None
        from vllm import LLM, SamplingParams
        self.llm = LLM(model="google/gemma-3-4b-it",
                       dtype="bfloat16", 
                       gpu_memory_utilization=0.4,
                       # enforce_eager=True,
                       # max_model_len=2048,
                       # swap_space=0,
                       )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=300)
        self.sbert = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device="cuda",)
        logging.set_verbosity_error()
        for param in self.sbert.parameters():
            param.requires_grad = False
    #############################
    # Overridden abstract methods
    #############################


    def process_env_fun(self, env_fun):
        """
        Experiment 파일에서 이 함수를 호출하면, 
        우리가 개조한 환경 생성 팩토리(_wrapped_env_fun)를 돌려줍니다.
        """
        group_name = list(self.group_map.keys())[0]
        def _wrapped_env_fun():
            
            base_env = env_fun()    
            base_env.append_transform(StepCounter())      
            group_name = list(self.group_map.keys())[0]

            llm_transform = LLM_Intention(algorithm=self, group=group_name)
            # wrapped_env = TransformedEnv(base_env, llm_transform)
            # return wrapped_env
            base_env.append_transform(llm_transform)
            return base_env
        if (group_name, "observation") in self.observation_spec.keys(True):
            orig_spec = self.observation_spec[group_name, "observation"]
            if (group_name, "sbert_embedding") not in self.observation_spec.keys(True):
               
                sbert_shape = torch.Size([*orig_spec.shape[:-1], 384])
                self.observation_spec[group_name, "sbert_embedding"] = Unbounded(
                    shape=sbert_shape, 
                    device=orig_spec.device
                )


        return _wrapped_env_fun

    def _get_loss(self, group: str, policy_for_loss: TensorDictModule, continuous: bool) -> Tuple[LossModule, bool]:
            # ClipPPOLoss 대신 상속받은 커스텀 Loss 사용
            loss_module = ClipPPOLoss(
                algorithm=self,
                group=group,
                actor=policy_for_loss,
                critic=self.get_critic(group),
                clip_epsilon=self.clip_epsilon,
                entropy_coeff=self.entropy_coef,
                critic_coeff=self.critic_coef,
                loss_critic_type=self.loss_critic_type,
                normalize_advantage=False,
            )
            loss_module.set_keys(
                reward=(group, "reward"), 
                action=(group, "action"), 
                done=(group, "done"),
                terminated=(group, "terminated"), 
                advantage=(group, "advantage"),
                value_target=(group, "value_target"), 
                value=(group, "state_value"),
                sample_log_prob=(group, "log_prob"),
            )
            loss_module.make_value_estimator(
                ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
            )
            return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        actor_params = list(loss.actor_network_params.flatten_keys().values())
        # actor_params += list(self.linear.parameters())
        return {
            "loss_objective": actor_params,
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }
    def entropy_decay(self, group: str, losses: dict, training_rate: float):
        training_rate = torch.tensor(training_rate, dtype=torch.float32)
        init_entropy = self.entropy_coef
        if self.entropy_coef_decay:
            new_entropy = init_entropy * (1.0 - training_rate)
            if hasattr(losses[group], "entropy_coeff"):
                losses[group].entropy_coeff = new_entropy
        
        start_interval = 10
        end_interval = 300
        if self.LLM_interval_decay:
            self.LLM_interval = int(start_interval + (end_interval - start_interval) * torch.pow(training_rate, 2)) 
            print(f"Updated LLM interval: {self.LLM_interval}")

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        orig_obs_shape = self.observation_spec[group, "observation"].shape
        fused_shape = list(orig_obs_shape)
        fused_shape[-1] += self.intention_vector_size

        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite({
            group: Composite({
                "observation_actor": Unbounded(shape=fused_shape, device=self.device)
            }, shape=(n_agents,))
        })
        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )
        fusion_module = TensorDictModule(
            TextFusionModule(self.intention_vector_size).to(self.device),
            in_keys=[(group, "observation"), (group, "sbert_embedding")],
            out_keys=[(group, "observation_actor")]
        )
        actor_module = TensorDictSequential(fusion_module, actor_module)

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
        
        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # MAPPO uses the same stochastic actor for collection
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy)
                // batch.shape[1]
            )
        else:
            increment = batch.batch_size[0] + 1
        last_start_index = 0
        start_index = increment
        minibatches = []
        while last_start_index < batch.shape[0]:
            minimbatch = batch[last_start_index:start_index]
            minibatches.append(minimbatch)
            with torch.no_grad():
                loss.value_estimator(
                    minimbatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        batch = torch.cat(minibatches, dim=0)
        return batch

    def process_loss_vals(self, group: str, loss_vals: TensorDictBase) -> TensorDictBase:
            total_obj = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
            loss_vals.set("loss_objective", total_obj)
            del loss_vals["loss_entropy"]
            return loss_vals

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        orig_obs_shape = self.observation_spec[group, "observation"].shape
        fused_shape = list(orig_obs_shape)
        fused_shape[-1] += self.intention_vector_size

        if self.share_param_critic:
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            input_has_agent_dim = False
            critic_input_spec = self.state_spec

        else:
            input_has_agent_dim = True
            critic_input_spec = Composite({
            group: Composite({
                "observation_critic": Unbounded(shape=fused_shape, device=self.device)
            }, shape=(n_agents,))
            })

        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=input_has_agent_dim,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )
        fusion_module = TensorDictModule(
            TextFusionModule(self.intention_vector_size).to(self.device), 
            in_keys=[(group, "observation"), (group, "sbert_embedding")],
            out_keys=[(group, "observation_critic")]
        )

        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(fusion_module, value_module, expand_module)
        else:
            value_module = TensorDictSequential(fusion_module, value_module)

        return value_module
    
class LLM_Intention(Transform):
    def __init__(self, algorithm: Larlwheel, group: str):
        super().__init__(
            in_keys=[(group, "observation")], 
            out_keys=[
                (group, "observation"), 
                (group, "sbert_embedding")  
            ]
        )
        self.algorithm = algorithm
        self.group = group
        self.n_agents = len(self.algorithm.group_map[group])
        self.prev_text = None

    def intention_to_vector(self, intention: list[list[str]]):
        batch_size = len(intention)
        n_agents = self.n_agents
        flat_intentions = [cmd for batch in intention for cmd in batch]
        with torch.no_grad():
            raw_vector = self.algorithm.sbert.encode(
                flat_intentions, convert_to_tensor=True, show_progress_bar=False
            )
        raw_vector = raw_vector.view(batch_size, n_agents, -1)
        return raw_vector 

    # def _obs_to_text(self, obs_tensor): # torch.Size([envs_workers, n_agents, obs_dim])
    #     obs_cpu = obs_tensor.cpu().numpy()
        
    #     if self.prev_text is None or len(self.prev_text) != obs_cpu.shape[0]:
    #         self.prev_text = [["None" for _ in range(self.n_agents)] for _ in range(obs_cpu.shape[0])]
            
    #     text_batch = []
        
    #     # 관측값(Observation) 텐서 인덱스 매핑
    #     pos_idx = self.n_agents                 # agent.state.pos (2)
    #     vel_idx = self.n_agents + 2             # agent.state.vel (2)
    #     rel_center_idx = self.n_agents + 4      # line.pos - agent.pos (2)
    #     rel_end1_idx = self.n_agents + 6        # line_end_1 - agent.pos (2)
    #     rel_end2_idx = self.n_agents + 8        # line_end_2 - agent.pos (2)
    #     rot_idx = self.n_agents + 10            # line.rot % torch.pi (1)
    #     ang_vel_idx = self.n_agents + 11        # line.ang_vel.abs() (1)
    #     diff_vel_idx = self.n_agents + 12       # abs(ang_vel.abs() - desired_velocity) (1)

    #     for i in range(obs_cpu.shape[0]):
    #         agent_texts = []
            
    #         # 환경(막대)의 공통 정보는 0번 에이전트의 관측값에서 추출
    #         line_rot = obs_cpu[i, 0, rot_idx]
    #         line_vel = obs_cpu[i, 0, ang_vel_idx]
    #         vel_error = obs_cpu[i, 0, diff_vel_idx]
            
    #         for j in range(obs_cpu.shape[1]):
    #             pos_x, pos_y = obs_cpu[i, j, pos_idx:pos_idx+2]
                
    #             # 막대 양 끝단과의 상대 위치를 통해 실제 거리(Distance) 계산
    #             rel_end1_x, rel_end1_y = obs_cpu[i, j, rel_end1_idx:rel_end1_idx+2]
    #             rel_end2_x, rel_end2_y = obs_cpu[i, j, rel_end2_idx:rel_end2_idx+2]
                
    #             dist_end1 = (rel_end1_x**2 + rel_end1_y**2)**0.5
    #             dist_end2 = (rel_end2_x**2 + rel_end2_y**2)**0.5
                
    #             # LLM이 파악하기 쉽게 가공
    #             text = (f"Agent {j+1}: Pos({pos_x:.2f}, {pos_y:.2f}), "
    #                     f"Dist to End1: {dist_end1:.2f}, Dist to End2: {dist_end2:.2f}")
    #             agent_texts.append({"type": "text", "text": text})

    #         base_prompt = [
    #         {
    #             "role": "system",
    #             "content": [{
    #                 "type": "text", 
    #                 "text": (
    #                     "You are a Strategic Commander for a multi-agent system. Goal: Cooperate to rotate a heavy line such that its absolute angular velocity EXACTLY matches 1.0.\n"
    #                     "MAP SPECIFICATIONS:\n"
    #                     "- The line is pinned at the origin (0,0) and can rotate freely. It has two extrema (End 1 and End 2).\n"
    #                     "- Target Velocity: 1.0. If velocity exceeds 1.0, the system receives a penalty.\n"
    #                     "STRATEGIC LOGIC:\n"
    #                     "1. Approach: If 'Dist to End1' and 'Dist to End2' are > 0.3, move toward the nearest End.\n"
    #                     "2. Unified Torque (Critical): To rotate the line, forces at End 1 and End 2 MUST be complementary. If agents at opposite ends push in the same absolute direction, forces cancel out.\n"
    #                     "   - The Commander MUST decide on ONE rotational direction (e.g., CLOCKWISE) for the team.\n"
    #                     "   - Instruct agents at End 1 to 'Push clockwise from End 1'.\n"
    #                     "   - Instruct agents at End 2 to 'Push clockwise from End 2'.\n"
    #                     "3. Throttle Control: If Angular Velocity >= 1.0, instruct ALL agents to 'Wait' to prevent overshooting."
    #                 )
    #             }]
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": f"ENVIRONMENT: Line Angle: {line_rot:.2f} rad, Current Angular Velocity: {line_vel:.2f}, Error to Target: {vel_error:.2f}"},
    #                 *agent_texts,
    #                 {"type": "text", "text": (
    #                     "TASK: You MUST explicitly check the distance for EACH agent before deciding their action. If minimum Dist > 0.3, they MUST approach. Do not command to push if they are far.\n"
    #                     "OUTPUT FORMAT:\n"
    #                     "Step 1: Write EXACTLY ONE LINE of analysis using this compact format: 'A1(Far->End2), A2(Ready->End1)... Vel=0.5 (<1.0)'.\n"
    #                     "Step 2: Output a JSON list of EXACTLY 4 strings, corresponding to Agent 1, 2, 3, and 4 strictly in order.\n"
    #                     "EXAMPLE Output 1 (Mixed):\n"
    #                     "Analysis: A1(Far->Approach End 2), A2(Ready->Push End 1), A3(Far->Approach End 1), A4(Far->Approach End 2). Vel=0.5 (Accelerate CW).\n"
    #                     "[\"Approach End 2\", \"Push clockwise from End 1\", \"Approach End 1\", \"Approach End 2\"]\n"
    #                     "EXAMPLE Output 2 (Maintaining):\n"
    #                     "Analysis: A1(Ready->End1), A2(Ready->End2), A3(Ready->End1), A4(Ready->End2). Vel=1.1 (Wait).\n"
    #                     "[\"Wait to maintain velocity\", \"Wait to maintain velocity\", \"Wait to maintain velocity\", \"Wait to maintain velocity\"]"
    #                 )}
    #             ]
    #         }
    #     ]
    #         text_batch.append(base_prompt)
            
    #     return text_batch
    
    def _obs_to_text(self, obs_tensor): # torch.Size([envs_workers, n_agents, obs_dim])
        obs_cpu = obs_tensor.cpu().numpy()
        
        if self.prev_text is None or len(self.prev_text) != obs_cpu.shape[0]:
            self.prev_text = [["None" for _ in range(self.n_agents)] for _ in range(obs_cpu.shape[0])]
            
        text_batch = []
        
        # 관측값(Observation) 텐서 인덱스 매핑
        pos_idx = self.n_agents                 # agent.state.pos (2)
        vel_idx = self.n_agents + 2             # agent.state.vel (2)
        rel_end1_idx = self.n_agents + 6        # line_end_1 - agent.pos (2)
        rel_end2_idx = self.n_agents + 8        # line_end_2 - agent.pos (2)
        ang_vel_idx = self.n_agents + 11        # line.ang_vel.abs() (1)

        for i in range(obs_cpu.shape[0]):
            agent_texts = []
            
            # 1. 환경 정보: 현재 속도 추출
            line_vel = obs_cpu[i, 0, ang_vel_idx]
            
            for j in range(obs_cpu.shape[1]):
                # 2. 파이썬 연산: LLM이 수학을 하지 않도록 최소 거리 직접 계산
                rel_end1_x, rel_end1_y = obs_cpu[i, j, rel_end1_idx:rel_end1_idx+2]
                rel_end2_x, rel_end2_y = obs_cpu[i, j, rel_end2_idx:rel_end2_idx+2]
                
                dist_end1 = (rel_end1_x**2 + rel_end1_y**2)**0.5
                dist_end2 = (rel_end2_x**2 + rel_end2_y**2)**0.5
                
                min_dist = min(dist_end1, dist_end2)
                nearest_end = "End 1" if dist_end1 < dist_end2 else "End 2"
                
                # 0.3을 기준으로 직관적인 텍스트 라벨링 (FAR vs READY)
                status = "READY" if min_dist <= 0.3 else "FAR"
                
                # 3. 정제된 텍스트 입력
                text = f"Agent {j+1}: Nearest={nearest_end}, Status={status}"
                agent_texts.append({"type": "text", "text": text})

            base_prompt = [
            {
                "role": "system",
                "content": [{
                    "type": "text", 
                    "text": (
                        "You are a Strategic Commander. Goal: Rotate a line to EXACTLY 1.0 angular velocity.\n"
                        "STRATEGIC LOGIC:\n"
                        "1. Throttle: If Velocity >= 1.0, ALL agents MUST 'Wait to maintain velocity'.\n"
                        "2. Approach: If Velocity < 1.0 and Status is 'FAR', command 'Approach [Nearest End]'.\n"
                        "3. Push: If Velocity < 1.0 and Status is 'READY', command 'Push clockwise from [Nearest End]'."
                    )
                }]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"ENVIRONMENT: Velocity={line_vel:.2f}"},
                    *agent_texts,
                    {"type": "text", "text": (
                        "TASK: Output EXACTLY ONE JSON object. Write your reasoning in clear, natural sentences before outputting the commands.\n"
                        "OUTPUT FORMAT:\n"
                        "{\n"
                        "  \"velocity_analysis\": \"explain if velocity is < 1.0 (accelerate) or >= 1.0 (wait)\",\n"
                        "  \"distance_analysis\": \"state the status (FAR or READY) for each of the 4 agents in a single sentence\",\n"
                        "  \"commands\": [\"cmd1\", \"cmd2\", \"cmd3\", \"cmd4\"]\n"
                        "}\n"
                        "EXAMPLE Output 1 (Accelerating):\n"
                        "{\n"
                        "  \"velocity_analysis\": \"Current velocity is 0.5, which is less than 1.0. We must accelerate clockwise.\",\n"
                        "  \"distance_analysis\": \"Agent 1 is FAR, Agent 2 is READY, Agent 3 is READY, and Agent 4 is FAR.\",\n"
                        "  \"commands\": [\"Approach End 2\", \"Push clockwise from End 1\", \"Push clockwise from End 2\", \"Approach End 1\"]\n"
                        "}\n"
                        "EXAMPLE Output 2 (Maintaining):\n"
                        "{\n"
                        "  \"velocity_analysis\": \"Current velocity is 1.2, which is >= 1.0. We must wait to prevent overshooting.\",\n"
                        "  \"distance_analysis\": \"Distance does not matter because the velocity check forces all agents to wait.\",\n"
                        "  \"commands\": [\"Wait to maintain velocity\", \"Wait to maintain velocity\", \"Wait to maintain velocity\", \"Wait to maintain velocity\"]\n"
                        "}"
                    )}
                ]
            }
            ]
            text_batch.append(base_prompt)
            
        return text_batch

    def get_llm_intention(self, group: str, tensordict: TensorDictBase = None): # -> [batch(120), agent(2)]
        state_description = self._obs_to_text(tensordict.get((group, "observation")))
        
        with torch.no_grad():
            contents = self.algorithm.llm.chat(
                state_description, 
                sampling_params=self.algorithm.sampling_params, 
                use_tqdm=False
            )
            # contents = self.algorithm.pipe(state_description, max_new_tokens=150, batch_size= len(state_description))#len(state_description)) # , max_length = None, max_new_tokens=100
        # print("*" * 50)
        # print(f"State description for LLM:\n{state_description[0]}")
        # print()
        # # print(f'LLM raw output: {contents[0][0]["generated_text"][-1]["content"].strip()}')
        # print(f"LLM raw output: {contents[0].outputs[0].text}")
        intention = []
        for content in contents:
            # raw_text = content[0]["generated_text"][-1]["content"].strip()
            raw_text = content.outputs[0].text
            match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
            if match:
                clean_text = match.group(0)
                try:
                    agent_instructions = json.loads(clean_text)
                    if isinstance(agent_instructions, list) and len(agent_instructions) == self.n_agents:
                        agent_instructions = json.loads(clean_text)
                        intention.append(agent_instructions)
                    else:
                        intention.append(["None." for _ in range(self.n_agents)])
                        #print(f"LLM output JSON format error: {clean_text}, expected a list of {self.n_agents} strings.")
                except json.JSONDecodeError:
                    intention.append(["None." for _ in range(self.n_agents)])
                    #print(f"LLM output JSON format error: {clean_text}, could not parse JSON.")
            else:
                intention.append(["None." for _ in range(self.n_agents)])
                #print(f"LLM output JSON format error: {raw_text}, no JSON array found.")

        self.prev_text = intention
        return intention


    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        curr_obs = next_tensordict.get((self.group, "observation"))
        step_count = next_tensordict.get("step_count").squeeze(-1)
        # print(f"LLM interval : {self.algorithm.LLM_interval}, step count : {step_count}")
        LLM_mask = (step_count % self.algorithm.LLM_interval == 0)
        prev_raw_vector = tensordict.get((self.group, "sbert_embedding"), None)

        if LLM_mask.any() or prev_raw_vector is None:
            intention = self.get_llm_intention(self.group, next_tensordict)
            self.prev_text = intention

            new_raw_vector = self.intention_to_vector(intention).detach()
            
            if prev_raw_vector is None:
                final_raw_vector = new_raw_vector
            else:
                mask_expanded = LLM_mask.view(-1, 1, 1).expand_as(prev_raw_vector)
                final_raw_vector = torch.where(mask_expanded, new_raw_vector, prev_raw_vector)
            # torch.cuda.empty_cache()
        else:
            final_raw_vector = prev_raw_vector

        next_tensordict.set((self.group, "observation"), curr_obs)
        next_tensordict.set((self.group, "sbert_embedding"), final_raw_vector)

        return next_tensordict
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        curr_obs = tensordict_reset.get((self.group, "observation"))
        
        self.prev_text = [["None"]*self.n_agents for _ in range(curr_obs.shape[0])]
        intention = self.get_llm_intention(self.group, tensordict_reset)
        raw_vector = self.intention_to_vector(intention).detach()
        self.prev_text = intention

        tensordict_reset.set((self.group, "observation"), curr_obs)
        tensordict_reset.set((self.group, "sbert_embedding"), raw_vector)
        
        return tensordict_reset
    
    def transform_observation_spec(self, observation_spec):
        # 1. Observation Spec 업데이트
        if (self.group, "observation") in observation_spec.keys(True):
            if (self.group, "sbert_embedding") not in observation_spec.keys(True):
                orig_spec = observation_spec[self.group, "observation"]
                
                sbert_shape = torch.Size([*orig_spec.shape[:-1], 384])
                observation_spec[self.group, "sbert_embedding"] = Unbounded(
                    shape=sbert_shape, device=orig_spec.device
                )
            
        return observation_spec

@dataclass
class LarlwheelConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Larl`."""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    entropy_coef_decay: bool = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING
    LLM_interval: int = MISSING
    LLM_interval_decay: bool = MISSING
    intention_vector_size: int = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Larlwheel

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
