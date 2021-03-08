from abc import ABC, abstractmethod
from typing import *
from dataclasses import dataclass
from typing import List
from util import unique_by, get_key_getter
import torch
from datareader import DataReader
from entitys import Entity
from CGExpan import CGExpan
import random
import logging

#TODO!
#logger = logging.getLogger()


class BaseHiEnv(ABC):

    @abstractmethod
    def state(self) -> Tuple[List[Entity], List[Entity], str]:
        """
        :return: current seed set, candidate list, field
        """
        pass

    @abstractmethod
    def action_expand(self, keys: List[Entity]) -> int:
        """
        :param keys: the seed set in t-th turn
        :return: 0
        """
        pass

    @abstractmethod
    def action_judge(self, answer: List[bool]) -> List[int]:
        """
        :param answer: the feedback from low-level reinforcement learning
        :return: Reward
        """


class SimpleHiEnv(BaseHiEnv):

    def __init__(self,
                 dataset_name: str,
                 field: str,  # TODO
                 rewards: Dict[str, int],
               ):
        print(f'<SimpleHiEnv>: data set:{dataset_name},field:{field}.')
        # TODO!
        self.field = field  # necessary?

        # init DataReader, and then init sets
        self.reader = DataReader(dataset_name, field)
        self.seeds = self.reader.get_original_seeds()
        self.gt = self.reader.get_gt_set()
        self.current_entity_set = self.seeds.copy()
        self.if_continue = True
        self.candidate_list = []

        # init CGExpan
        device= torch.device("cuda:0")
        self.cgexpan = CGExpan(device, self.reader)  # TODO

        self.rewards = rewards
        print('<SimpleHiEnv>: Env is ready!')

    def state(self) -> Tuple[List[Entity], List[Entity], str]:
        return self.current_entity_set, self.candidate_list, self.field
        
    def if_stop(self):
        return self.if_continue

    def action_expand(self, keys: List[Entity]) -> int:

        expanded = self.cgexpan.expand(keys)


        expanded = unique_by(expanded, lambda c: c.eid)

        # TODO: shuffle
        # if self.sort_candidates:
        #     self._sort_candidates_by_distance_to_keys(l, keys)
        # else:
        #     random.shuffle(l)

        self.candidate_list = expanded[0:40]  
        
        if len(self.candidate_list) < 3:
            self.if_continue = False

        for candidate in self.candidate_list:
            if (candidate in self.gt) or (candidate in self.seeds):
                candidate.ground_truth = True
            else:
                candidate.ground_truth = False
        return 0

    def action_judge(self, answers: List[bool]) -> List[int]:
        # TODO!
        results = []
        for candidate, answer in zip(self.candidate_list, answers):
            # TODO:
            if answer and (candidate not in self.current_entity_set):
                self.current_entity_set.append(candidate)

            if candidate.ground_truth == answer:
                results.append(self.rewards["correct"])
            else:
                results.append(self.rewards["wrong"])

        print("Now current entity set has:", len(self.current_entity_set))

        return results


if __name__ == '__main__':

    e = SimpleHiEnv(dataset_name='wiki', field='countries', rewards={"correct": 2, "wrong": -1})
    current_seeds, _, _ = e.state()

    if e.action_expand(current_seeds) == 0:
        _, candidates, _ = e.state()
        print('candidates:')
        print(candidates)
    else:
        print("error.")

