from abc import ABC, abstractmethod
from envs import BaseHiEnv
from entitys import Entity

from typing import *
import logging

import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
from dataclasses import dataclass

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@dataclass
class Step:
    log_probs: torch.Tensor
    reward: int
    entity: Optional[Entity]
    answer: bool
    score: float = 0


class StopEpisode(Exception):
    pass


class BaseAgent(ABC):

    @abstractmethod
    def set_env(self, e: BaseHiEnv):
        pass

    @abstractmethod
    def step(self, info):
        pass

    @property
    def memory(self):
        pass


class Agent(BaseAgent):

    def __init__(self,
                 seed_net: nn.Module,  # for high-level RL
                 judge_net: nn.Module,  # for low-level RL
                 ):
        print("<Agent> initializing...")
        self.seed_net = seed_net
        self.judge_net = judge_net

        self.env: Optional[BaseHiEnv] = None  

        self.random_step = False

        self.current_set_size = 0

        self.test()

        # TODO
        logger.info(f"hi-net: \n{self.seed_net}")
        logger.info(f"lo-net: \n{self.judge_net}")
        print("<Agent> is ready!")

    def train(self):  
        self.random_step = True
        # self.random_step = True
        self.seed_net.train()
        self.judge_net.train()

    def test(self): 
        self.random_step = False
        self.seed_net.eval()
        self.judge_net.eval()

    def set_env(self, e: BaseHiEnv):
        self.current_set_size = 0
        self.env = e

    def hi_step(self):
        # get access to the seed set
        current_entity_set, _, _ = self.env.state()

        # if it is time to stop the game?
        if not self.env.if_stop():
            print("Cannot expand more entities, stop the game. ")
            raise StopEpisode("Cannot expand more entities, stop the game. ")

        # prepare the input data
        set_vector = torch.stack([c.vector for c in current_entity_set])  # (set_size, word_vector_dim)
        set_vector = set_vector.unsqueeze(0)  # (batch_size=1, set_size, word_vector_dim)

        # use the seed_net to get probs
        probs: torch.Tensor = self.seed_net(set_vector)  # (batch_size, set_size, K)
        probs = probs.transpose(1, 2)  # (batch_size, K, set_size)

        # normalize probs
        normalized_probs = probs / (torch.sum(probs, dim=-1, keepdim=True) + 1e-10)  # (batch_size, K, set_size)

        ca = Categorical(normalized_probs)
        print("ca:", ca)

        if self.random_step:
            selected_indices = ca.sample().long()               # (batch_size, K)
        else:
            selected_indices = normalized_probs.argmax(dim=-1).long()  # (batch_size, K)

        keys = [current_entity_set[i] for i in selected_indices[0]]  # selected_indices[0] (K,)

        # TODO
        logging.info(f"Select keys: {[current_entity_set[i].name[0] for i in selected_indices[0]]}")

        # expand entity set with keys
        self.env.action_expand(keys)

        log_probs = torch.sum(ca.log_prob(selected_indices), dim=-1)[0]  # () # logger

        return keys, log_probs

    def lo_step(self, keys: List[Entity]):
        steps: List[Step] = []

        _, candidate_list, _ = self.env.state()

        if len(candidate_list) == 0:
            return [], None

        # seeds: (batch_size=1, set_size1, word_vector_dim)
        Ks = torch.stack([k.vector for k in keys]).unsqueeze(0)
        # candidates: (batch_size=1, set_size2, word_vector_dim)
        Cs = torch.stack([c.vector for c in candidate_list]).unsqueeze(0)
        probs = self.judge_net(Ks, Cs, self.env)  # (1, N, 2), 0 for positive, 1 for negative

        ca = Categorical(probs)
        # if self.random_step:
        if False:  #  TODO
            selected_indices: torch.Tensor = ca.sample().long()  # (1, N)
        else:
            selected_indices = probs.argmax(dim=-1).long()  # (1, N)

        # TODO
        current_set, _, _ = self.env.state()
        self.current_set_size = len(current_set)

        answers = [choice.item() == 1 for choice in selected_indices[0]]
        rewards = self.env.action_judge(answers)

        log_probs = ca.log_prob(selected_indices)[0]  # (N,)

        for answer, log_prob, reward, candidate, prob in zip(answers, log_probs, rewards, candidate_list, probs[0]):
            steps.append(Step(log_prob, reward, candidate, answer, score=prob[1].tolist()))
            logging.info(f"{(candidate.name[0], answer, reward)}")

        return steps, probs[0]

    def step(self, info: List[torch.Tensor] = None):

        keys, log_probs = self.hi_step()

        lo_steps, lo_probs = self.lo_step(keys)

        lo_total_rewards = sum((s.reward for s in lo_steps))
        hi_steps = [Step(log_probs=log_probs,
                         reward=lo_total_rewards,
                         entity=None,
                         answer=False, )]

        return hi_steps, lo_steps, lo_probs  # len(hi_steps) == 1


if __name__ == '__main__':
    from judge_net import RNNJudgeNet
    from repset import RepSetKGroups
    from envs import SimpleHiEnv

    seed_net = RepSetKGroups(word_vector_dim=768, compressed_dim=128, hidden_set_size=768, K=3)
    judge_net = RNNJudgeNet(word_vec_dim=768, hidden_state_size=768, bidir=True)

    my_agent = Agent(seed_net=seed_net, judge_net=judge_net)

    e = SimpleHiEnv("wiki", "countries", rewards={"correct": 2, "wrong": -1})
    my_agent.set_env(e)

    new_hi_steps, new_lo_steps, new_lo_probs = my_agent.step()
    print(new_hi_steps)
    print(new_lo_steps)
    print(new_lo_probs)
