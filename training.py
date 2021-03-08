from agent import Agent, Entity, StopEpisode, Step  
from envs import SimpleHiEnv

from checkpointing import Checkpointing 
import torch
from torch import nn
from typing import *
from util import count_if, eps
from torch.optim.optimizer import Optimizer
import random
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
import traceback
import pandas as pd
import tabulate

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ReinforceTrainingStatus:
    sta_columns = ['acc', 'f1', 'prec', 'recall', 'pos', 'true', 'tot', 'ap']

    def __init__(self):
        self.epoch = 0
        self.training_loss = float("inf")
        self.step = 0
        self.sta_training = pd.DataFrame([], columns=self.sta_columns)
        self.sta_eval = pd.DataFrame([], columns=self.sta_columns)
        self.training_history = []
        self.eval_history = []


class ReinforceTraining:

    def __init__(self,
                 hi_network: nn.Module,  # high-level
                 lo_network: nn.Module,  # low-level
                 hi_optimizer: Optional[Optimizer],
                 lo_optimizer: Optional[Optimizer],
                 agent: Agent,  
                 checkpointing: Checkpointing,
                 epochs: int,  
                 selected_fields: List[str],  # TODO
                 test_fields: List[str],  # TODO
                 gamma: float = 0.99,
                 dataset: str = "wiki",
                 test_period: int = 5,
                 update_period: int = 1,
                 env_params: dict = None,
                 resample_seed_period: int = 32,
                 ):

        self.dataset = dataset
        self.hi_network = hi_network
        self.lo_network = lo_network

        self.hi_optimizer = hi_optimizer
        self.lo_optimizer = lo_optimizer

        self.agent = agent

        self.checkpointing = checkpointing  # TODO

        self.gamma = gamma
        self.epochs = epochs
        self.resample_seed_period = resample_seed_period

        self.selected_fields = selected_fields  # TODO
        self.test_fields = test_fields  # TODO
        self.training_fields = self.selected_fields  # TODO
        self.test_period = test_period
        self.update_period = update_period
        self.env_params = env_params

        self.training_status = ReinforceTrainingStatus()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.hi_loss_buf: List[torch.Tensor] = []
        self.lo_loss_buf: List[torch.Tensor] = []

    def make_state_dict(self):  # for saving checkpoint
        """
        for saving checkpoint.
        :return: the records of state_dicts of network and optimizer
        """
        return {
            "training_status": self.training_status,
            "hi_network_state_dict": self.hi_network and self.hi_network.state_dict(),
            "lo_network_state_dict": self.lo_network and self.lo_network.state_dict(),
            "hi_optimizer_state_dict": self.hi_optimizer and self.hi_optimizer.state_dict(),
            "lo_optimizer_state_dict": self.lo_optimizer and self.lo_optimizer.state_dict(),
        }

    # TODO: checkpoint
    def load_state_dict(self, best=False, recent=True):  # loading state dict from checkpoint
        """
        load the state_dict to continue the previous training
        :param best:
        :param recent:
        :return:
        """
        pass
        checkpoint = self.checkpointing.load_check_point(best, recent)
        if checkpoint:
            self.training_status.__dict__.update(checkpoint["training_status"].__dict__)
            if self.hi_optimizer:
                self.hi_network.load_state_dict(checkpoint["hi_network_state_dict"])
                self.hi_optimizer.load_state_dict(checkpoint["hi_optimizer_state_dict"])
            if self.lo_optimizer:
                self.lo_network.load_state_dict(checkpoint["lo_network_state_dict"])
                self.lo_optimizer.load_state_dict(checkpoint["lo_optimizer_state_dict"])

    def make_env(self, dataset: str, field: str):  # init the env
        return SimpleHiEnv(
            dataset_name=dataset,
            field=field,
            rewards={
                "correct": 2,
                "wrong": -1
            }
        )

    def get_sta(self, steps: List[Step]):
        # class Step
        # log_probs: torch.Tensor
        # reward: int
        # concept: Optional[Concept]
        # answer: bool
        # score: float = 0

        ground_truth = [s.entity.ground_truth for s in steps]
        preds = [s.answer for s in steps]
        scores = [s.score for s in steps]

        acc = accuracy_score(ground_truth, preds)
        f1 = f1_score(ground_truth, preds)
        prec = precision_score(ground_truth, preds)
        recall = recall_score(ground_truth, preds)
        pos = count_if(preds, lambda x: x == True)
        true = count_if(ground_truth, lambda x: x == True)
        tot = len(preds)
        ap = average_precision_score(ground_truth, scores)

        return acc, f1, prec, recall, pos, true, tot, ap

    def run_agent(self):
        step_cnt = 0  # count the number of steps
        max_step = 10  # TODO
        hi_steps = []
        lo_steps = []
        lo_probs: List[torch.Tensor] = []

        try:
            while step_cnt <= max_step:
                try:
                    step_cnt += 1
                    logging.info(f"steps:{step_cnt}")

                    new_hi_steps, new_lo_steps, new_lo_probs = self.agent.step()

                    hi_steps.extend(new_hi_steps)
                    if len(new_lo_steps) > 0:
                        lo_steps.extend(new_lo_steps)
                        lo_probs.append(new_lo_probs)

                except StopEpisode:
                    break

            if len(lo_probs) > 0:
                lo_probs_tensor = torch.cat(lo_probs, dim=0)
                return hi_steps, lo_steps, lo_probs_tensor, step_cnt
            else:
                raise ValueError("<Trainer> Empty lo_probs.")

        except RuntimeError:
            traceback.print_exc()
            return None
        except Exception:
            traceback.print_exc()
            return None

    def get_test_fields(self):
        return self.test_fields

    def test_agent(self, dataset: str, field: str):  # test
        logger.debug("start run_agent")
        result = self.predict(field, dataset)
        if result:
            hi_steps, lo_steps, lo_probs_tensor, *_ = result
            return self.get_sta(lo_steps)
        else:
            return None

    def predict(self, field: str, dataset: str):  # test
        env = self.make_env(dataset, field)
        self.agent.set_env(env)
        run_result = self.run_agent()

        return run_result  # include:hi_steps, lo_steps, lo_probs_tensor, step_cnt

    def optimize(self):  # Optimize
        if len(self.hi_loss_buf) > 0:
            hi_loss = torch.sum(torch.stack(self.hi_loss_buf))
            self.hi_loss_buf.clear()
            self.hi_optimizer.zero_grad()
            hi_loss.backward()
            self.hi_optimizer.step()

        if len(self.lo_loss_buf) > 0:
            lo_loss = torch.sum(torch.stack(self.lo_loss_buf))
            self.lo_loss_buf.clear()
            self.lo_optimizer.zero_grad()
            lo_loss.backward()
            self.lo_optimizer.step()

    def reinforce_episode(self):  
        run_result = self.run_agent()
        if run_result:
            hi_steps, lo_steps, lo_probs, step_cnt = run_result
            self.training_status.step += step_cnt
        else:
            return None

        if len(lo_steps) < 2 or len(hi_steps) < 2:
            logger.warning(f"Too few steps:hi_steps={len (hi_steps)},lo_steps=len(lo_steps),weight update abandoned.")
            return None

        # count hi_loss
        hi_reward = sum((s.reward for s in hi_steps))
        if self.hi_optimizer:
            hi_loss = self.reinforce_policy_loss(hi_steps)
            self.hi_loss_buf.append(hi_loss)
        else:
            hi_loss = torch.tensor(0)

        # count lo_loss
        lo_reward = sum((s.reward for s in lo_steps))
        ground_truth = torch.tensor([s.entity.ground_truth for s in lo_steps]).long()
        lo_loss = self.cross_entropy(lo_probs, ground_truth)
        if self.lo_optimizer:
            self.lo_loss_buf.append(lo_loss)
        else:
            lo_loss = torch.tensor(0)

        sta = self.get_sta(lo_steps)
        acc, f1, prec, recall, pos, true, tot, ap = sta

        # optimize
        if self.training_status.epoch % self.update_period == 0:
            logging.info(f"Optimizing...")
            self.optimize()

        # log
        logging.info(f"Epoch: {self.training_status.epoch}/{self.epochs} "
                     f"Step: {self.training_status.step} "
                     f"lo_reward: {lo_reward:.3f} "
                     f"hi_reward: {hi_reward:.3f} "
                     f"acc: {acc:.3f} "
                     f"f1: {f1:.3f} "
                     f"prec: {prec:.3f} "
                     f"recall: {recall:.3f} "
                     f"pos: {pos}/{len(lo_steps)} "
                     f"true: {true}/{len(lo_steps)} "
                     f"ap: {ap} ")

        return hi_reward, hi_loss.item(), lo_reward, lo_loss.item(), sta

    def reinforce_policy_loss(self, steps: List[Step]):
        R = 0
        policy_loss = []
        returns = []
        for step in steps[::-1]:
            R = step.reward + self.gamma * R
            returns.insert(0, R) 

        if steps[0].entity:
            logging.debug(f"policy_loss: {[(s.entity.names[0], torch.exp(s.log_probs).item(), s.reward, r) for s, r in zip(steps, returns)]}")
        # returns = [r^1 + gamma^1*r^2 + ... + gamma^(T-i+1)*r^i ,... , r^(T-2) + gamma^1*r^(T-1) + gamma^2*r^T, r^(T-1) + gamma^1 * r^T, r^T]
        returns = torch.tensor(returns).float()
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for step, R in zip(steps, returns):
            policy_loss.append(-step.log_probs * R)

        return torch.stack(policy_loss).sum()

    def generate_field_to_seeds(self):
        pass

    def train(self):
        self.agent.train()

        for epoch in range(self.training_status.epoch, self.epochs):
            self.training_status.epoch = epoch


            # if self.training_status.epoch % self.resample_seed_period == 0:
                # TODO
                # self.field_to_seeds = self.generate_field_to_seeds()


            cur_field: str = random.choice(self.training_fields)
            logger.info(f"cur_field = {cur_field}")


            env = self.make_env(self.dataset, cur_field)
            self.agent.set_env(env)
            self.agent.train()

            # training
            with torch.autograd.detect_anomaly():

                result = self.reinforce_episode()  # def train -> def reinforce_episode -> def run_agent

            if result:

                hi_reward, hi_loss, lo_reward, lo_loss, sta = result
                logger.info(f"Epoch: {epoch} " +
                            f"hi_reward: {hi_reward} " +
                            f"hi_loss: {hi_loss} " +
                            f"lo_reward: {lo_reward} " +
                            f"lo_loss: {lo_loss} ")

                # TODO
                self.training_status.sta_training.loc[cur_field] = sta
                self.training_status.training_history.append(self.training_status.sta_training.copy())

                logger.info(
                    f"training statistics:\n{tabulate.tabulate(self.training_status.sta_training, headers='keys')}")

                # testing
                if epoch % self.test_period == 0:
                    for test_field in self.test_fields:
                        self.agent.train()
                        test_result = self.test_agent(self.dataset, test_field)
                        if test_result:
                            self.training_status.sta_eval.loc[test_field] = test_result
                        else:
                            logger.error(f"Exception happened in test_agent({test_field})")

                    self.training_status.eval_history.append(self.training_status.sta_eval.copy())
                    # logger.info(f"test statistics:\n{tabulate.tabulate(self.training_status.sta_test, headers='keys')}")
                    logger.info(f"eval statistics:\n{tabulate.tabulate(self.training_status.sta_eval, headers='keys')}")

                # TODO 记录checkpoint
                self.checkpointing.save_check_point(self.make_state_dict(), hi_loss + lo_loss)
            else:
                logger.error(f"Epoch: {epoch} Exception happened! ")


if __name__ == "__main__":
    torch.cuda._initialized = True
    import os
    os.chdir('/home/amax/ZWX/')

    from judge_net import RNNJudgeNet
    from repset import RepSetKGroups
    from envs import SimpleHiEnv

    seed_net = RepSetKGroups(word_vector_dim=768, compressed_dim=128, hidden_set_size=768, K=3)
    judge_net = RNNJudgeNet(word_vec_dim=768, hidden_state_size=768, bidir=True)

    my_agent = Agent(seed_net=seed_net, judge_net=judge_net)

    hi_optimizer = torch.optim.SGD(seed_net.parameters(), lr=0.0001, momentum=0.9)
    lo_optimizer = torch.optim.SGD(judge_net.parameters(), lr=0.00001, momentum=0.9)

    my_check = Checkpointing(directory='./log/', keep_n=10, metric_bigger_better=False)
    selected_fields = ['countries', 'diseases', 'parties', 'china_provinces', 'companies', 'sportsleagues']
    test_fields = ['us_states', 'tv_channels', ]

    my_trainer = ReinforceTraining(
        hi_network=seed_net,
        lo_network=judge_net,
        hi_optimizer=hi_optimizer,
        lo_optimizer=lo_optimizer,
        agent=my_agent, 
        checkpointing=my_check,
        epochs=200,
        selected_fields=selected_fields,  # TODO
        test_fields=test_fields,
        dataset="wiki",
        test_period=10,
        update_period=1,
        env_params=None,
    )

    my_trainer.train()

    print("END\n")

