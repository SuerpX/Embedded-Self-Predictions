import logging
import time
import random
import pickle
import os
from sys import maxsize

import torch
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule

from abp.utils import clear_summary_path
from abp.models.dqn_model_softmax import DQNModel
from abp.adaptives.common.prioritized_memory.memory import PrioritizedReplayBuffer, ReplayBuffer
import numpy as np

logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class SADQAdaptive(object):
    """Adaptive which uses the SADQ algorithm"""

    def __init__(self, name, state_length, network_config, reinforce_config,
                 reward_num, combine_decomposed_func, memory_resotre = True):
        super(SADQAdaptive, self).__init__()
        self.name = name
        #self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        if self.reinforce_config.use_prior_memory:
            self.memory = PrioritizedReplayBuffer(self.reinforce_config.memory_size, 0.6)
        else:
            self.memory = ReplayBuffer(self.reinforce_config.memory_size)
        self.learning = True
        self.state_length = state_length
        
        # Global
        self.steps = 0
        self.best_reward_mean = 0
        self.episode = 0
        self.combine_decomposed_reward = combine_decomposed_func
        self.reward_num = reward_num

        self.reset()
        self.memory_resotre = memory_resotre
        reinforce_summary_path = self.reinforce_config.summaries_path + "/" + self.name

        if not self.network_config.restore_network:
            clear_summary_path(reinforce_summary_path)
        else:
            self.restore_state()
        
        self.summary = SummaryWriter(log_dir=reinforce_summary_path)

        self.target_model = DQNModel(self.name + "_target", self.network_config, use_cuda)
        self.eval_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)
#         self.target_model.eval_mode()

        self.beta_schedule = LinearSchedule(self.reinforce_config.beta_timesteps,
                                            initial_p=self.reinforce_config.beta_initial,
                                            final_p=self.reinforce_config.beta_final)

        self.epsilon_schedule = LinearSchedule(self.reinforce_config.epsilon_timesteps,
                                               initial_p=self.reinforce_config.starting_epsilon,
                                               final_p=self.reinforce_config.final_epsilon)

    def __del__(self):
        self.save()
        self.summary.close()

    def should_explore(self):
        self.epsilon = self.epsilon_schedule.value(self.steps)
        self.summary.add_scalar(tag='%s/Epsilon' % self.name,
                                scalar_value=self.epsilon,
                                global_step=self.steps)

        return random.random() < self.epsilon

    def predict(self, state, isGreedy = False, is_random = False):
        
        if self.learning:
            self.steps += 1
            
        # add to experience
        if self.previous_state is not None and self.learning and self.current_reward is not None:
            state_crr = np.unique(state, axis=0).copy()
            self.memory.add(self.previous_state,
                            None,
                            self.current_reward,
                        state_crr.reshape(-1, self.state_length), 0)
            
        if self.learning and self.should_explore() and not isGreedy:
            q_values = None
            choice = random.choice(list(range(len(state))))
            action = choice
        else:
            with torch.no_grad():
                q_values = FloatTensor(self.eval_model.predict_batch(Tensor(state)))
            assert q_values.size()[0] == state.shape[0], print(q_values.size(), state.shape)
            
#             print(q_values)
            q_values = self.combine_decomposed_reward(q_values)
#             print(q_values)
            max_q, choice = q_values.max(0)
            action = choice
            
        if self.learning and self.steps % self.reinforce_config.replace_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        if (self.learning and
            self.steps > self.reinforce_config.update_start and
                self.steps % self.reinforce_config.update_steps == 0):
            self.update()
#         print(action)
        self.previous_state = state[action]
        return choice, q_values

    def disable_learning(self, is_save = False):
        logger.info("Disabled Learning for %s agent" % self.name)
        if is_save:
            self.save()
        self.learning = False
        self.reset()
        
    def enable_learning(self):
        logger.info("enabled Learning for %s agent" % self.name)
        self.learning = True
        self.reset()

    def end_episode(self, state):
        if not self.learning:
            return
        self.memory.add(self.previous_state,
                        None,
                        self.current_reward,
                        state.reshape(-1, self.state_length), 1)
        self.reset()

    def reset(self):
        self.current_reward = None
        self.previous_state = None
        self.previous_action = None

    def restore_state(self):
        restore_path = self.network_config.network_path + "/adaptive.info"
        if self.network_config.network_path and os.path.exists(restore_path) and self.memory_resotre:
            logger.info("Restoring state from %s" % self.network_config.network_path)

            with open(restore_path, "rb") as file:
                info = pickle.load(file)

            self.steps = info["steps"]
            self.best_reward_mean = info["best_reward_mean"]
            self.episode = info["episode"]
            self.memory.load(self.network_config.network_path)
            print("lenght of memeory: ", len(self.memory))

    def save(self, force=False, appendix=""):
        info = {
            "steps": self.steps,
            "best_reward_mean": self.best_reward_mean,
            "episode": self.episode
        }
        
        print("*************saved*****************")
#         logger.info("Saving network. Found new best reward (%.2f)" % total_reward)
        self.eval_model.save_network(appendix = appendix)
        self.target_model.save_network(appendix = appendix)
        with open(self.network_config.network_path + "/adaptive.info", "wb") as file:
            pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.memory.save(self.network_config.network_path)
        print("lenght of memeory: ", len(self.memory))

    def reward(self, r_vector):
#         self.current_reward = np.array(r_vector, dtype = np.uint8)
        self.current_reward = r_vector
    def update(self):
        if len(self.memory._storage) <= self.reinforce_config.batch_size:
            return
        
        beta = self.beta_schedule.value(self.steps)
        self.summary.add_scalar(tag='%s/Beta' % self.name,
                                scalar_value=beta, global_step=self.steps)
        if self.reinforce_config.use_prior_memory:
            batch = self.memory.sample(self.reinforce_config.batch_size, beta)
            (states, actions, reward, next_states,
             is_terminal, weights, batch_idxes) = batch
            self.summary.add_histogram(tag='%s/Batch Indices' % self.name,
                                       values=Tensor(batch_idxes),
                                       global_step=self.steps)
        else:
            batch = self.memory.sample(self.reinforce_config.batch_size)
            (states, actions, reward, next_states, is_terminal) = batch
#         print(states)
#         print(reward)
        
#         for r in reward:
#             print(len(r))
        states = FloatTensor(states)
        #next_states = FloatTensor(next_states)
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
            
        batch_index = torch.arange(self.reinforce_config.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
#         print(states)
        q_values = self.eval_model.predict_batch(states)
#         print(q_values)
#         input("check the view function correct")
        # Calculate target
        q_max = []
        for ns in next_states:
            q_next = self.target_model.predict_batch(FloatTensor(ns).view(-1, self.state_length))
            q_next_combine = self.combine_decomposed_reward(q_next)
            best_idx = q_next_combine.max(0)[1]
#             print(best_idx)
            q_max.append(q_next[best_idx])
#             print(q_max)
        
#         q_next = [self.target_model.predict_batch(FloatTensor(ns).view(-1, self.state_length))[1] for ns in next_states]
#         q_max = torch.stack([each_qmax.max(0)[0].detach() for each_qmax in q_next], dim = 1)[0]
#         print(next_states)
        q_max = torch.stack(q_max)
#         print(q_max)
        idx_no_terminal = (terminal == 0)
        q_target = reward.clone()
#         print(terminal)
#         print(idx_no_terminal)
#         print(q_max[idx_no_terminal])
        q_target[idx_no_terminal] += self.reinforce_config.discount_factor * q_max[idx_no_terminal]
#         print(q_target)
#         print(q_target)
#         input("update")
        # update model
        self.eval_model.fit(q_values, q_target, self.steps)

        # Update priorities
        if self.reinforce_config.use_prior_memory:
            td_errors = q_values - q_target
            new_priorities = torch.abs(td_errors) + 1e-6  # prioritized_replay_eps
            self.memory.update_priorities(batch_idxes, new_priorities.data)
            
    def load_model(self, model):
        self.eval_model.replace(model)
        
    def load_weight(self, weight_dict):
        self.eval_model.load_weight(weight_dict)
