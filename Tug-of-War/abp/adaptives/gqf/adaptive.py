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
from abp.models.feature_q_model import feature_q_model
from abp.adaptives.common.prioritized_memory.memory_gqf import ReplayBuffer_decom
import numpy as np

logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class SADQ_GQF(object):
    """Adaptive which uses the SADQ algorithm"""

    def __init__(self, name, state_length, network_config, reinforce_config, feature_len, combine_decomposed_func, is_sigmoid = False, memory_resotre = True):
        super(SADQ_GQF, self).__init__()
        self.name = name
        #self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config

        self.memory = ReplayBuffer_decom(self.reinforce_config.memory_size)

        self.learning = True
        self.explanation = False
        self.state_length = state_length

        self.features = 0
        self.feature_len = feature_len
        # Global
        self.steps = 0
        self.reward_history = []
        self.episode_time_history = []
        self.best_reward_mean = -maxsize
        self.episode = 0
        self.feature_len = feature_len
        self.features = None

        self.reset()
        self.memory_resotre = memory_resotre
        reinforce_summary_path = self.reinforce_config.summaries_path + "/" + self.name

        if not self.network_config.restore_network:
            clear_summary_path(reinforce_summary_path)
        else:
            self.restore_state()
        
        self.summary = SummaryWriter(log_dir=reinforce_summary_path)
        self.eval_model = feature_q_model(name, state_length, self.feature_len, self.network_config.output_shape, network_config)
        self.target_model = feature_q_model(name, state_length, self.feature_len, self.network_config.output_shape, network_config)
#         self.target_model.eval_mode()

        self.beta_schedule = LinearSchedule(self.reinforce_config.beta_timesteps,
                                            initial_p=self.reinforce_config.beta_initial,
                                            final_p=self.reinforce_config.beta_final)

        self.epsilon_schedule = LinearSchedule(self.reinforce_config.epsilon_timesteps,
                                               initial_p=self.reinforce_config.starting_epsilon,
                                               final_p=self.reinforce_config.final_epsilon)

#     def __del__(self):
#         self.save()
#         self.summary.close()

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
            state_crr = np.unique(state, axis=0)
            self.memory.add(self.previous_state,
                            None,
                            self.current_reward,
                        state_crr.reshape(-1, self.state_length), 0,
                        self.features)
#             print("not final : {}".format(self.current_reward) )
#             print(0, self.features)
        if self.learning and self.should_explore() and not isGreedy:
            q_values = None
            fv = None
            choice = random.choice(list(range(len(state))))
            action = choice
        else:
            with torch.no_grad():
                features_vector, q_values = self.eval_model.predict_batch(Tensor(state))
                q_values = FloatTensor(q_values).view(-1)

            _, choice = q_values.max(0)
            action = choice
            fv = features_vector[choice]
#         print("q_value : {}".format(q_values))
#         input()
        if self.learning and self.steps % self.reinforce_config.replace_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            if self.reinforce_config.replace_frequency != 1:
                self.target_model.replace(self.eval_model)
            else:
                self.target_model.replace_soft(self.eval_model)
#             self.target_model.eval_mode()

        if (self.learning and
            self.steps > self.reinforce_config.update_start and
                self.steps % self.reinforce_config.update_steps == 0):
            self.update_time -= time.time()
            self.update()
            self.update_time += time.time()

        self.current_reward = 0
        self.previous_state = state[action]
        #self.previous_action = action

        return choice, fv#,q_values

    def disable_learning(self, is_save = False):
        logger.info("Disabled Learning for %s agent" % self.name)
        if is_save:
#             self.save()
            self.save(force = True)
        self.learning = False
        self.episode = 0
        
    def enable_learning(self):
        logger.info("enabled Learning for %s agent" % self.name)
        self.learning = True
        self.reset()

    def end_episode(self, state):
        if not self.learning:
            return
#         print("end:")
#         print(self.current_reward)
#         input()
        episode_time = time.time() - self.episode_time

        self.reward_history.append(self.total_reward)
        self.episode_time_history.append(episode_time)
        total_time = sum(self.episode_time_history)
        avg_time = total_time / len(self.episode_time_history)

        logger.info("End of Episode %d, "
                    "Total reward %.2f, "
                    "Epsilon %.2f" % (self.episode + 1,
                                      self.total_reward,
                                      self.epsilon))

        logger.debug("Episode Time: %.2fs (%.2fs), "
                     "Prediction Time: %.2f, "
                     "Update Time %.2f" % (episode_time,
                                           avg_time,
                                           self.prediction_time,
                                           self.update_time))

        self.episode += 1
        self.summary.add_scalar(tag='%s/Episode Reward' % self.name,
                                scalar_value=self.total_reward,
                                global_step=self.episode)

        self.memory.add(self.previous_state,
                        None,
                        self.current_reward,
                        state.reshape(-1, self.state_length), 1,
                        self.features)
#         print("final : {}".format(self.current_reward) )
#         input()
#         print(1, self.features)
        self.save()
        self.reset()

    def reset(self):
        self.episode_time = time.time()
        self.current_reward = 0
        self.total_reward = 0
        self.previous_state = None
        self.previous_action = None
        self.prediction_time = 0
        self.update_time = 0
        self.features = None

    def restore_state(self):
        restore_path = self.network_config.network_path + "/adaptive.info"
        if self.network_config.network_path and os.path.exists(restore_path) and self.memory_resotre:
            logger.info("Restoring state from %s" % self.network_config.network_path)

            with open(restore_path, "rb") as file:
                info = pickle.load(file)

            self.steps = info["steps"]
#             self.best_reward_mean = info["best_reward_mean"]
            self.episode = info["episode"]
            self.memory.load(self.network_config.network_path)
            print("lenght of memeory: ", len(self.memory))

    def save(self, force=False, appendix=""):
        info = {
            "steps": self.steps,
            "best_reward_mean": self.best_reward_mean,
            "episode": self.episode
        }
        
        if (len(self.reward_history) >= self.network_config.save_steps and
                self.episode % self.network_config.save_steps == 0) or force:

            total_reward = sum(self.reward_history[-self.network_config.save_steps:])
            current_reward_mean = total_reward / self.network_config.save_steps

            if force: #or current_reward_mean >= self.best_reward_mean:
                print("*************saved*****************", current_reward_mean, self.best_reward_mean)
                if not force:
                    self.best_reward_mean = current_reward_mean
                logger.info("Saving network. Found new best reward (%.2f)" % total_reward)
                self.eval_model.save_network(appendix = appendix)
                self.target_model.save_network(appendix = appendix)
#                 self.eval_model.save_network()
#                 self.target_model.save_network()
                with open(self.network_config.network_path + "/adaptive.info", "wb") as file:
                    pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
                self.memory.save(self.network_config.network_path)
                print("lenght of memeory: ", len(self.memory))
            else:
                logger.info("The best reward is still %.2f. Not saving" % self.best_reward_mean)

    def reward(self, r):
        self.total_reward += r
        self.current_reward += r

    def passFeatures(self, features):
        self.features = features.copy()
        return

    def summary_test(self, reward, epoch):
        self.summary.add_scalar(tag='%s/eval reward' % self.name,
                                scalar_value=reward, global_step=epoch * 40)
    def summary_GVFs_loss(self, loss, epoch):
        self.summary.add_scalar(tag='%s/GVFs loss' % self.name,
                                scalar_value=loss, global_step=epoch * 40)
    
    def update(self):
        if len(self.memory._storage) <= self.reinforce_config.batch_size:
            return
#         self.eval_model.train_mode()
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
            (states, actions, reward, next_states, is_terminal, features_vector) = batch

        states = FloatTensor(states)
#         print(states.size())
#         next_states = FloatTensor(next_states)
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        features_vector = FloatTensor(features_vector)
        batch_index = torch.arange(self.reinforce_config.batch_size,
                                   dtype=torch.long)
        # Current Q Values
        feature_values, q_values = self.eval_model.predict_batch(states)
        q_values = q_values.flatten()
        q_max = []
        f_max = []
        for i, ns in enumerate(next_states):
            feature_n, q_n = self.target_model.predict_batch(FloatTensor(ns).view(-1, self.state_length))
            q_value_max, idx = q_n.max(0)
            features_max = feature_n[idx]
            
            q_max.append(q_value_max)
            if self.network_config.version in ["v10", "v11"]:
#                 print(features_max)
#                 print(ns[idx, 63:67])
#                 print(states[i, 63:67])
#                 print(features_max.size(), FloatTensor(ns).view(-1, self.state_length).size(), states.size())
                features_max[:, :3] = (features_max[:, :3] * ns[idx, 65]) / states[i, 65]
                features_max[:, 3:6] = (features_max[:, 3:6] * ns[idx, 66]) / states[i, 66]
                features_max[:, 6:9] = (features_max[:, 6:9] * ns[idx, 63]) / states[i, 63]
                features_max[:, 9:12] = (features_max[:, 9:12] * ns[idx, 64]) / states[i, 64]
                features_max[features_max == float('inf')] = 0
#                 print(features_max)
#                 input()
            f_max.append(features_max.view(-1))
        
#         if torch.sum(terminal == torch.sum(features_vector, dim = 1)) != len(terminal):
#             print(terminal)
#             print(features_vector)
#             input()
        q_max = torch.stack(q_max, dim = 1).view(-1)
        f_max = torch.stack(f_max)
        q_max = (1 - terminal) * q_max
        
        f_max = (1 - terminal.view(-1, 1)) * f_max
        
        q_target = reward + self.reinforce_config.discount_factor * q_max
        
        f_target = features_vector + self.reinforce_config.discount_factor * f_max
        
#         if torch.sum(reward).item() > 0:
#             print(reward)
#             print(feature_values)
#             print(q_target)
#             print(q_values)
#             input()
        # update model
        if (torch.sum(feature_values != feature_values).item() + torch.sum(f_target != f_target)).item() > 0:

#             print("1")
#             print(features_vector)
#             print("2")
#             print(feature_values)
#             print("3")
#             print(f_target)
#             print("4")
#             print(f_max)
#             print("5")
#             print(states.tolist())
#             input()
            f_target[f_target != f_target] = 0
        self.eval_model.fit(q_values, q_target, feature_values, f_target)

        # Update priorities
        if self.reinforce_config.use_prior_memory:
            td_errors = q_values - q_target
            new_priorities = torch.abs(td_errors) + 1e-6  # prioritized_replay_eps
            self.memory.update_priorities(batch_idxes, new_priorities.data)
            
    def load_model(self, model):
        self.eval_model.replace(model)
        
    def load_weight(self, weight_dict):
        self.eval_model.load_weight(weight_dict)
        
    def load_model(self, model):
        self.eval_model.replace(model)
        
    def load_weight(self, new_feature_weights, new_q_weights):
        self.eval_model.feautre_model.load_state_dict(new_feature_weights)
        self.eval_model.q_model.load_state_dict(new_q_weights)