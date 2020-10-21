#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import torch
from torch import nn
from torch.autograd import variable
import time
import os
import numpy as np
import gym
import shutil

from tqdm import tqdm
from random import uniform, randint, sample, random, choices
from collections import deque

import io
import base64
from IPython.display import HTML

from models.feature_q_model import feature_q_model
from memory.memory import ReplayBuffer, ReplayBuffer_decom

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorboardX import SummaryWriter
import math
import matplotlib as mpl
from datetime import datetime
# get_ipython().run_line_magic('matplotlib', 'inline')

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
Writer = SummaryWriter(log_dir="CartPole_summary")
mpl.style.use('bmh')
plt.rcParams["font.family"] = "Arial"#"Helvetica"
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:



ENV_NAME = 'CartPole_ESP'
env = gym.make('CartPole-v1')

# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT": 1
}
action_name = {
    0: 'push left',
    1: 'push right',
}
FEATRUESNAME = ["F1 Cart Right Boundary", "F2 Cart Left Boundary", "F3 Cart Right Velocity", "F4 Cart Left Velocity", 
                "F5 Pole Angle Right", "F6 Pole Angle Left", "F7 Pole Right Velocity", "F8 Pole Left Velocity"]


# In[3]:


result_floder = ENV_NAME
result_floder_exp = "{}/{}".format(result_floder, ENV_NAME + "_exp")
result_file = ENV_NAME + "/results.txt"
result_file_GVFs_loss = ENV_NAME + "/results_GVFs_loss.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
if not os.path.isdir(result_floder_exp):
    os.mkdir(result_floder_exp)


# In[4]:


def MSX(vector):
    vector = np.array(vector)
    indeces = np.argsort(vector)[::-1]
    negative_sum = sum(vector[vector < 0])
    pos_sum = 0
    MSX_idx = []
    for idx in indeces:
        pos_sum += vector[idx]
        MSX_idx.append(idx)
        if pos_sum > abs(negative_sum):
            break
    return MSX_idx, vector[MSX_idx]

def print_Cartpole(state):
    print("Cart Position: ", state[0], end = "  ")
    print("Cart Velocity: ", state[1])
    print("Pole Angle: ", state[2], end = "  ")
    print("Pole Velocity At Tip: ", state[3])
def plot_action_group(values, group, elements = [], title = 'decomposition values', y_label = "", q_values = None, IGX_action = None, exp_count = -1):
    plt.clf()
    x = np.arange(len(group))  # the label locations
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel(y_label)
    plt.title(title)

    # set width of bar
    length = len(values[0])
    barWidth = 1 / (len(values) + 1)
    x_labels_feature = ["F{}".format(x + 1) for x in range(len(values))]
    for i in range(len(values)):
        r = [j + barWidth * i for j in range(length)]
        if len(elements) > 0:
            plt.bar(r, values[i] , width=barWidth,  label=elements[i])
#             for j, rr in enumerate(r):
#                 plt.text(rr - barWidth / 4, values[i][j] if values[i][j] > 0 else 0, x_labels_feature[i], fontsize=5)
        else:
            plt.bar(r, values[i], width=barWidth)
    
    print(barWidth)
    y_lim = plt.gca().get_ylim()
    gap = (y_lim[1] - 0) / 30
    for i in range(len(values)):
        r = [j + barWidth * i for j in range(length)]
        for j, rr in enumerate(r):
            if values[i][j] != 0:
                plt.text(rr - barWidth / 7, -gap * 1.5 if values[i][j] > 0 else gap, x_labels_feature[i], fontsize=8)
    
    r = [j + barWidth * (i + 1) for j in range(length)]
    for rr in r[:-1]:
        plt.axvline(x = rr, alpha = 0.5, linestyle='--')

    # Add xticks on the middle of the group bars
    plt.xlabel('action', fontweight='bold')
    center_pos = (1 - barWidth * 2) / 2
    
    if IGX_action is not None:
        group = ["\"{}\"\n greater than".format(IGX_action)] + group
        x_lim_left = plt.gca().get_xlim()[0]
        pos = [x_lim_left] + [r + center_pos for r in range(length)]
        plt.xticks(pos, group, ha='center')
    else:
        plt.xticks([r + center_pos for r in range(length)], group, ha='center')
    
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)
    if np.min(values) >= -gap * 3:
        plt.gca().set_ylim(bottom = -gap * 3)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    y_lim = plt.gca().get_ylim()
    np_values = np.array(values).T
    txt_pos = y_lim[1] - (y_lim[1] - y_lim[0]) / 25
    max_values = np_values.max(axis = 1)
    sum_values = np_values.sum(axis = 1)
    
    for i, v in enumerate(max_values):
        if q_values is None:
            plt.gca().text(i + center_pos, txt_pos, "sum ≈ {}".format(np.round(sum_values[i], 4)), color='black', fontweight='ultralight', ha='center')
        else:
            plt.gca().text(i + center_pos, txt_pos, "Q_v ≈ {}".format(np.round(q_values[i], 4)), color='black', fontweight='ultralight', ha='center')

    plt.savefig("{}/{}-{}.png".format(result_floder_exp, title, exp_count))
    
def display_frames_as_gif(frames, video_name):
    """
    Displays a list of frames as a gif, with controls
    """
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
#     display(display_animation(anim, default_mode='loop'))
    anim.save(result_floder + '/' + video_name, writer=writer)


# In[5]:


def plot_IG_MSX(values, values_msx, group, elements = [], title = 'decomposition values', y_label = "", q_values = None, IGX_action = None, exp_count = -1):
    plt.clf()
    x = np.arange(len(group))  # the label locations
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel(y_label)
    plt.title(title)

    # set width of bar
    length = len(values[0])
    barWidth = 1 / (len(values) + 1)
    x_labels_feature = ["F{}".format(x + 1) for x in range(len(values))]
    for i in range(len(values)):
        r = [j + barWidth * i for j in range(length)]
        
        for j in range(len(values[i])):
            if values_msx[i][j] != 0:
                plt.bar(r[j], values[i][j], width=barWidth, hatch="////",
                        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
            else:
                plt.bar(r[j], values[i][j], width=barWidth,
                        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
    
    y_lim = plt.gca().get_ylim()
    gap = (y_lim[1] - 0) / 30
    for i in range(len(values)):
        r = [j + barWidth * i for j in range(length)]
        for j, rr in enumerate(r):
            if values[i][j] != 0:
                plt.text(rr - barWidth / 4, -gap * 1.5 if values[i][j] > 0 else gap, x_labels_feature[i], fontsize=5)
    
    r = [j + barWidth * (i + 1) for j in range(length)]
    for rr in r[:-1]:
        plt.axvline(x = rr, alpha = 0.5, linestyle='--')

    # Add xticks on the middle of the group bars
    plt.xlabel('action', fontweight='bold')
    center_pos = (1 - barWidth * 2) / 2
    
    plt.xticks([r + center_pos for r in range(length)], group, ha='center')
    
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)
    if np.min(values) >= -gap * 3:
        plt.gca().set_ylim(bottom = -gap * 3)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    y_lim = plt.gca().get_ylim()
    np_values = np.array(values).T
    txt_pos = y_lim[1] - (y_lim[1] - y_lim[0]) / 25
    max_values = np_values.max(axis = 1)
    sum_values = np_values.sum(axis = 1)
    
    for i, v in enumerate(max_values):
        if q_values is None:
            plt.gca().text(i + center_pos, txt_pos, "{} > {}".format(IGX_action, group[i]), color='black', fontweight='ultralight', ha='center')
        else:
            plt.gca().text(i + center_pos, txt_pos, "Q_v ≈ {}".format(np.round(q_values[i], 4)), color='black', fontweight='ultralight', ha='center')

    plt.savefig("{}/{}-{}.png".format(result_floder_exp, title, exp_count))


# In[6]:


hyperparams_CarPole = {
    'epsilon_decay_steps' : 200000, 
    'final_epsilon' : 0.05,
    'batch_size' : 128, 
    'update_steps' : 5, 
    'memory_size' : 200000, 
    'beta' : 0.99, 
    'model_replace_freq' : 1,
    'learning_rate' : 0.00001,
    'privous_state' : 1,
    'decom_reward_len': 8,
    'soft_tau': 5e-4
}


# In[7]:


class ESP_agent(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT)):
        
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        
        """
            beta: The discounted factor of Q-value function
            (epsilon): The explore or exploit policy epsilon. 
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps', 
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.soft_tau = hyper_params['soft_tau']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            learning: The trigger of agent learning. It is on while training agent. It is off while testing agent.
            action_space: The action space of the current environment, e.g 2.
        """
        self.episode = 0
        self.steps = 0
        self.best_reward = -1000
        self.learning = True
        self.action_space = action_space
        self.privous_state = hyper_params['privous_state']

        """
            input_len The input length of the neural network. It equals to the length of the state vector.
            output_len: The output length of the neural network. It is equal to the action space.
            eval_model: The model for predicting action for the agent.
            target_model: The model for calculating Q-value of next_state to update 'eval_model'.
        """
        self.decom_reward_len = hyper_params['decom_reward_len']
        state = env.reset()
        self.state_len = len(state)
        input_len = self.state_len * self.privous_state + action_space
        output_len = 1
        
        self.action_vector = self.get_action_vector()
        self.eval_model = feature_q_model(input_len, self.decom_reward_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.target_model = feature_q_model(input_len, self.decom_reward_len, output_len, learning_rate = hyper_params['learning_rate'])
#         memory: Store and sample experience replay.
        self.memory = ReplayBuffer_decom(hyper_params['memory_size'])
        
        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.q_value_gd = []
        self.v_feature_input= []
        self.loss_accumulate = deque(maxlen=1000)
        
        self.optimizer_com = self.eval_model.optimizer_com
        self.loss_fn = self.eval_model.loss_fn
        self.exp_count = 0
        
    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def get_action_vector(self):
        action_vector = np.zeros((self.action_space, self.action_space))
        for i in range(len(action_vector)):
            action_vector[i, i] = 1
        
        return FloatTensor(action_vector)
    
    def concat_state_action(self, states, actions = None, is_full_action = False):
        if is_full_action:
            com_state = FloatTensor(states).repeat((1, self.action_space)).view((-1, self.state_len))
            actions = self.action_vector.repeat((len(states), 1))
        else:
            com_state = states.clone()
            actions = actions.clone()
        state_action = torch.cat((com_state, actions), 1)
        return state_action
        
    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        self.epsilon = epsilon
        
        if p < epsilon:
            #return action, None
            return randint(0, self.action_space - 1)
        else:
            #return action, Q-value
            return self.greedy_policy(state)[0]
        
    def greedy_policy(self, state):
        state_ft = FloatTensor(state).view(-1, self.state_len)
        state_action = self.concat_state_action(state_ft, is_full_action = True)
        feature_vectors, q_values = self.eval_model.predict_batch(state_action)
        q_v, best_action = q_values.max(0)
        
        return best_action.item(), q_v, feature_vectors[best_action.item()]
    
    def update_batch(self):
        if len(self.memory) < self.batch_size or self.steps % self.update_steps != 0:
            return

        batch = self.memory.sample(self.batch_size)

        (states_actions, _, reward, next_states,
         is_terminal, rewards_decom) = batch
        
        next_states = FloatTensor(next_states)
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        rewards_decom = FloatTensor(rewards_decom)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        feature_vector, q_values = self.eval_model.predict_batch(states_actions)
        next_state_actions = self.concat_state_action(next_states, is_full_action = True)
        feature_vector_next, q_next = self.target_model.predict_batch(next_state_actions)
        q_next = q_next.view((-1, self.action_space))
        feature_vector_next = feature_vector_next.view((-1, self.action_space, self.decom_reward_len))
        q_max, idx = q_next.max(1)

        q_max = (1 - terminal) * q_max
        q_target = reward + self.beta * q_max
        q_target = q_target.unsqueeze(1)
        
        feature_vector_max = feature_vector_next[batch_index, idx, :]
        
        feature_vector_max = (1 - terminal.view(-1, 1)) * feature_vector_max
        feature_vector_target = rewards_decom + feature_vector_max * self.beta
        
        self.eval_model.fit(q_values, q_target, feature_vector, feature_vector_target)
        
    def learn_and_evaluate(self, training_episodes, test_interval):
        test_number = training_episodes // test_interval
        all_results = []
        
        for i in range(test_number):
            # learn
            self.learn(test_interval)
            f = open(result_file, "a+")
            f.write(str("\n***{}".format((i + 1) * test_interval) + "\n"))
            f.close()
            f = open(result_file_GVFs_loss, "a+")
            f.write(str("\n***{}".format((i + 1) * test_interval) + "\n"))
            f.close()
            # evaluate
            avg_reward = self.evaluate((i + 1) * test_interval)
            all_results.append(avg_reward)
            
        return all_results
    
    def get_features_decom(self, state, next_state, done):
        
        threshold_x = 1
        threshold_c_v = 1
        threshold_angle = 0.07
        threshold_p_v = 0.7
        
        features_decom = np.ones(self.decom_reward_len)
#         cart_position, cart_velocity, pole_angle, pole_velocity = state
        next_cart_position, next_cart_velocity, next_pole_angle, next_pole_velocity = next_state
            
        if threshold_x < next_cart_position:
            features_decom[0] = -1
        if -threshold_x > next_cart_position:
            features_decom[1] = -1        
    
        if threshold_c_v < next_cart_velocity:
            features_decom[2] = -1
        if -threshold_c_v > next_cart_velocity:
            features_decom[3] = -1   
            
        if threshold_angle < next_pole_angle:
            features_decom[4] = -1
        if -threshold_angle > next_pole_angle:
            features_decom[5] = -1   
        
        if threshold_p_v < next_pole_velocity:
            features_decom[6] = -1
        if -threshold_p_v > next_pole_velocity:
            features_decom[7] = -1   
        return features_decom
        
    def learn(self, test_interval):
        
        for episode in tqdm(range(test_interval), desc="Training"):
            state = self.env.reset()
            done = False
            steps = 0
            
            while steps < self.max_episode_steps and not done:
                steps += 1
                self.steps += 1
                
                action = self.explore_or_exploit_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                
                features_decom = self.get_features_decom(state, next_state, steps < self.max_episode_steps and done)
                action_vector = np.zeros(self.action_space)
                action_vector[action] = 1
                
                self.memory.add(np.concatenate((state.copy(), action_vector.copy()), axis=0), -1, reward, next_state, steps < self.max_episode_steps and done, features_decom)
                self.update_batch()
                
                if self.steps % self.model_replace_freq == 0:
                    if self.model_replace_freq == 1:
                        self.target_model.replace_soft(self.eval_model, tau = self.soft_tau)
                    else:
                        self.target_model.replace(self.eval_model)
                state = next_state
                
    def evaluate(self, episode_num, trials = 100, loss_max_steps = 100):
        total_reward = 0
        total_steps = 0
        all_GVFs_loss = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = self.env.reset()
            done = False
            steps = 0
            all_features_gt = FloatTensor(np.zeros((loss_max_steps, self.decom_reward_len)))
            all_features_predict = FloatTensor(np.zeros((loss_max_steps, self.decom_reward_len)))
            discounted_para = FloatTensor(np.zeros((loss_max_steps, 1)))
            while steps < self.max_episode_steps and not done:
                steps += 1
                action, _, fv = self.greedy_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                if steps < loss_max_steps:
                    measure_steps = steps
                    all_features_predict[measure_steps - 1] = fv
                    
                features_decom = self.get_features_decom(state, next_state, steps < self.max_episode_steps and done)
                
                all_features_gt[:measure_steps] += (FloatTensor(features_decom) * (self.beta ** discounted_para))[:measure_steps]
                    
                discounted_para[:measure_steps] += 1
                state = next_state
            total_steps += steps
            with torch.no_grad():
                all_GVFs_loss += self.eval_model.loss_fn(all_features_gt[:measure_steps], all_features_predict[:measure_steps]).item()
        avg_reward = total_reward / trials
        avg_GVFs_loss = all_GVFs_loss / trials
        print("avg score: {}".format(avg_reward))
        print("avg GVF loss: {}".format(avg_GVFs_loss))
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()
        f = open(result_file_GVFs_loss, "a+")
        f.write(str(avg_GVFs_loss) + "\n")
        f.close()
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward
            self.save_model()
            print("save")
        Writer.add_scalars(main_tag='CartPole/GQF discrete',
                                tag_scalar_dict = {'GQF discrete 1':avg_reward}, 
#                                 scalar_value=,
                                global_step=episode_num)
        return avg_reward
#################################################################################################
#################################################################################################  
############################################### MSX #############################################  
#################################################################################################  
#################################################################################################  
    def explore_or_exploit_policy_eval(self, state, epsilon = 0):
        p = uniform(0, 1)
        if p < epsilon:
            #return action, None
            return randint(0, self.action_space - 1)
        else:
            #return action, Q-value
            return self.greedy_policy(state)[0]

    def evaluate_combination_action_large_diff(self, example_num = 10, epsilon = 0, states = []):
        self.eval_model.eval_mode()
        loss_fn = nn.MSELoss()
        
        all_states = []
        all_frames = []
        state = self.env.reset()
        done = False
        steps = 0
        
        state_len = len(state)
        
#         q_vs = []
        while steps < self.max_episode_steps and not done:
            steps += 1
            self.steps += 1

            action = self.explore_or_exploit_policy_eval(state, epsilon = epsilon)
#             q_vs.append(q_v.item())
            state, reward, done, _ = self.env.step(action)
            
            all_frames.append(env.render(mode='rgb_array'))
            
            all_states.append(state.copy())
        for s in states:
            all_states.append(s)
            all_frames.append(np.zeros_like(all_frames[0]))
        examples = FloatTensor(all_states)
        eval_frames = np.array(all_frames)

        com_examples = self.concat_state_action(examples, is_full_action = True)
        v_features, q_value = self.eval_model.predict_batch(com_examples)
        q_value = q_value.view((-1, self.action_space))
        v_features = v_features.view((-1, self.action_space, self.decom_reward_len))
        
        q_best_values, predict_best_action = q_value.max(1)
        q_worst_values, predict_worst_action = q_value.min(1)
        
        q_diff = q_best_values - q_worst_values
        
        q_diff_idx = q_diff.argsort(descending = True)[:example_num]
        if len(states) > 0:
            q_diff_idx = torch.cat((q_diff_idx, LongTensor(list(range(len(all_frames) - len(states), len(all_frames))))))
        q_diff_idx_np = np.array(q_diff_idx.tolist())
        q_diff = q_diff[q_diff_idx]
        print("diff example length: {}".format(len(q_diff)))
        q_value = q_value[q_diff_idx]
        v_features = v_features[q_diff_idx]
        q_best_values, predict_best_action = q_value.max(1)
        q_worst_values, predict_worst_action = q_value.min(1)
        
        eval_frames = eval_frames[q_diff_idx_np]
        examples = examples[q_diff_idx_np]
        
        total_acc_1 = 0
        total_acc_2 = 0
    
        print("diff action value: {}".format(q_diff))
        count = 0
        for i, vf in enumerate(v_features):
            p_a = predict_best_action[i].item()
            sub_actions = q_value[i].argsort()[:-1]
    
            print("=========================================================================")
            
#             print("true target action: {}\n true baseline action: {}".format(action_name[t_a], action_name[true_sub_action]))
            show_image = True
            print_Cartpole(examples[i].tolist())
        
            IGs = []
            ans = []
            MSX_values = []
            baseline_values = []
            GVFs = []
            q_print_values = []
            GVFs.append(v_features[i][p_a].tolist())
            ans.append(action_name[p_a])
            q_print_values.append(q_value[i][p_a].item())
            print("target action {}, value: {}".format(action_name[p_a], q_best_values[i]))
            for sb in sub_actions.tolist()[::-1]:
#                 print("=========================================================================")
                sub_action = sb
#                 print("target action: {}\n baseline action: {}".format(action_name[p_a], action_name[sub_action]))
                ans.append(action_name[sub_action])
                GVFs.append(v_features[i][sub_action].tolist())
                msx_idx, msx_value, intergated_grad = self.differenc_vector_2(self.eval_model.q_model, 
                                      (v_features[i][p_a], eval_frames[i], q_value[i][p_a])
                                      ,(v_features[i][sub_action], eval_frames[i], q_value[i][sub_action]), False, iteration = 30, show_image = show_image)
                ig = np.array(intergated_grad[0].tolist())
                IGs.append(ig)
                baseline_values.append(q_value[i][sub_action].item())
                q_print_values.append(q_value[i][sub_action].item())
                orginal_pos_MSX = np.zeros(len(ig))
                orginal_pos_MSX[msx_idx] = ig[msx_idx]
                MSX_values.append(orginal_pos_MSX)
                if show_image:
                    show_image = False
                    
            for v, k in zip(baseline_values, ans):
                print("baseline action {}, value: {}".format(k, v))
            
            plot_action_group(np.array(GVFs).T, ans, FEATRUESNAME, title = "GVFs", y_label = "GVFs Values", q_values = q_print_values, exp_count = self.exp_count)
            plot_IG_MSX(np.array(IGs).T, np.array(MSX_values).T, ans[1:], FEATRUESNAME, title = "IG & MSX", y_label = "IGX", IGX_action = ans[0], exp_count = self.exp_count)
        
    def differenc_vector_2(self, model, target, baseline, verbose = True, iteration = 100, show_image = True):
        t_feature, t_frame, t_value = target
        b_feature, b_frame, b_value = baseline
        
        if verbose or show_image:
            plt.clf()
            print("target value: {}".format(t_value.item()))
            print("baseline value: {}".format(b_value.item()))
            self.exp_count += 1
            plt.imshow(t_frame)
            plt.axis('off')
            plt.savefig("{}/game_state-{}.png".format(result_floder_exp, self.exp_count))
        (msx_idx, msx_value), intergated_grad = self.intergated_gradients(model, t_feature, baseline = b_feature, verbose = verbose, iteration = iteration)
        return msx_idx, msx_value, intergated_grad
    def intergated_gradients(self, model, x, iteration = 100, baseline = None, verbose = True):
        y_baseline = model(baseline).item()
        x = x.view(1, -1)
        x.size()[1]
        if baseline is None:
            baseline = torch.zeros_like(x)
        elif verbose:
            print("baseline: {}".format(baseline))
        
        intergated_grad = torch.zeros_like(x)
    
        for i in range(iteration):
            new_input = baseline + ((i + 1) / iteration * (x - baseline))
            new_input = new_input.clone().detach().requires_grad_(True)
    #         print(new_input)

            y = model(new_input)
            loss = abs(y_baseline - y)
            self.optimizer_com.zero_grad()
            loss.backward()
            intergated_grad += (new_input.grad) / iteration
            
        intergated_grad *= x - baseline
        
        print("target GVFs: {}".format(x.tolist()))
        print("baseline GVFs: {}".format(baseline.tolist()))
        MSX_idx, MSX_values = MSX(intergated_grad.tolist()[0])
        if verbose:
            msx_vector = np.zeros(self.decom_reward_len)
            msx_vector[MSX_idx] = MSX_values
            plot_msx_plus(msx_vector, tittle = 'MSX+')
    
        return (MSX_idx, MSX_values), intergated_grad
    
    def generate_video(self, video_name):
        self.eval_model.eval_mode()
        loss_fn = nn.MSELoss()
        total_reward = 0
        all_frames = []
        state = self.env.reset()
        done = False
        steps = 0
        
        while steps < self.max_episode_steps and not done:
            steps += 1

            action, q_v, feature_vector = self.greedy_policy(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            all_frames.append(env.render(mode='rgb_array'))
        print(total_reward)
        display_frames_as_gif(all_frames, video_name)
        print("video generted")

    def save_model(self, path = None, name = "best_model.pt"):
        if path is None:
            path = result_floder
        self.eval_model.save("{}/{}".format(path, name))
        
    # load model
    def load_model(self, path = "", name = "best_model.pt"):
        if path is None:
            path = result_floder
        self.eval_model.load("{}/{}".format(path, name))


# ## Train Agent

# In[ ]:
best_ckpt_path = "{}/best_check_point".format(result_floder)
if sys.argv[1] == "train":
    training_episodes, test_interval = 10000, 100
    for i in range(10):
        print("*************{}*************".format(i))
        agent = ESP_agent(env, hyperparams_CarPole)
        result = agent.learn_and_evaluate(training_episodes, test_interval)
# In[8]:

elif sys.argv[1] == "exp":
    if len(sys.argv) == 2:
        en = 5
    else:
        en = int(sys.argv[2])
    agent = ESP_agent(env, hyperparams_CarPole)
    agent.load_model(path = best_ckpt_path)
    agent.evaluate_combination_action_large_diff(example_num = en, epsilon = 0.1)


# In[ ]:

elif sys.argv[1] == "eval":
    agent = ESP_agent(env, hyperparams_CarPole)
    agent.load_model(path = best_ckpt_path)
    agent.evaluate(0)
else:
    print("please gives correct paramater")

# In[ ]:

