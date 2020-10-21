import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch

from abp import SADQAdaptive
from abp.adaptives.sadq.adaptive_deexp import SADQAdaptive_exp
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war_2L_self_play import TugOfWar
from tqdm import tqdm
from copy import deepcopy
from random import randint, random

# np.set_printoptions(precision = 2)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def run_task(evaluation_config, network_config, reinforce_config, map_name = None, train_forever = False, agent_model = None):
    if (use_cuda):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("|       USING CUDA       |")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("|     NOT USING CUDA     |")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    flags.FLAGS(sys.argv[:1])
    max_episode_steps = 40
    env = TugOfWar(map_name = map_name)
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    agent_1 = SADQAdaptive_exp(name = "agent_1",
                        state_length = len(state_1),
                        network_config = network_config,
                        reinforce_config = reinforce_config)
    network_config.restore_network = False
    agent_2 = SADQAdaptive(name = "agent_2",
                        state_length = len(state_1),
                        network_config = network_config,
                        reinforce_config = reinforce_config)
    network_config.restore_network = True
    path = './saved_models/tug_of_war/agents/'
    if agent_model is not None:
        agent_1_model = agent_model
        agent_2_model = agent_model
    else:
        print("need a model")
        raise
    
    agent_1_model_weights = torch.load(path + agent_1_model)
    agent_2_model_weights = torch.load(path + agent_2_model)

    agent_1.load_weight_true_q(agent_1_model_weights)
    agent_2.load_weight(agent_2_model_weights)
    
    agent_2.disable_learning(is_save = False)
    
    
    while True:
        
        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")

        for episode in tqdm(range(evaluation_config.training_episodes)):
            state_1, state_2 = env.reset()
            total_reward = 0
            skiping = True
            done = False
            steps = 0
#             break
            while skiping:
                state_1, state_2, done, dp = env.step([], 0)
                if dp or done:
                    break
                    
            while not done and steps < max_episode_steps:
                steps += 1

                actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
                assert state_1[-1] == state_2[-1] == steps, print(state_1, state_2, steps)

                combine_states_1 = combine_sa(state_1, actions_1)
                choice_1, _ = agent_1.predict(env.normalization(combine_states_1))
                    
                combine_states_2 = combine_sa(state_2, actions_2)
                choice_2, _ = agent_2.predict(env.normalization(combine_states_2))
                
                env.step(list(actions_1[choice_1]), 1)
                env.step(list(actions_2[choice_2]), 2)
                
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    if dp or done:
                        break
                exp_reward = []
                if steps == max_episode_steps or done:
                    win_lose = player_1_win_condition(state_1[27], state_1[28], state_1[29], state_1[30])

                    if win_lose == 1:
                        exp_reward.append(1)
                    elif win_lose == -1:
                        exp_reward.append(0)
                        
                agent_1.reward(sum(exp_reward))

            agent_1.end_episode(env.normalization(state_1))
            
#             if (episode + 1) % 500 == 0:
#                 agent_1.save()

        agent_1.disable_learning(is_save = True)
        
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")

        tied_lose = 0
        total_rewwards_list = []
        average_end_state = np.zeros(len(state_1))
        for episode in tqdm(range(evaluation_config.test_episodes)):
            env.reset()
            total_reward_1 = 0
            done = False
            skiping = True
            steps = 0
            
            while skiping:
                state_1, state_2, done, dp = env.step([], 0)
                if dp or done:
                    break
                    
            while not done and steps < max_episode_steps:
                steps += 1

                actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
                
                combine_states_1 = combine_sa(state_1, actions_1)
                choice_1, _ = agent_1.predict(env.normalization(combine_states_1))

                combine_states_2 = combine_sa(state_2, actions_2)
                choice_2, _ = agent_2.predict(env.normalization(combine_states_2))

                env.step(list(actions_1[choice_1]), 1)
                env.step(list(actions_2[choice_2]), 2)

                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    if dp or done:
                        break
                exp_reward = []
                if steps == max_episode_steps or done:
                    win_lose = player_1_win_condition(state_1[27], state_1[28], state_1[29], state_1[30])

                    if win_lose == 1:
                        exp_reward = [1]
                    elif win_lose == -1:
                        exp_reward = [0]
                        
                current_reward_1 = sum(exp_reward)
            
                total_reward_1 += current_reward_1

            average_end_state += state_1

            total_rewwards_list.append(total_reward_1)

        total_rewards_list_np = np.array(total_rewwards_list)

#         tied = np.sum(total_rewards_list_np[-evaluation_config.test_episodes:] == 0)
        wins = np.sum(total_rewards_list_np[-evaluation_config.test_episodes:] > 0)
        lose = np.sum(total_rewards_list_np[-evaluation_config.test_episodes:] <= 0)

        print("wins/lose/tied")
        print(str(wins / evaluation_config.test_episodes * 100) + "% \t",
             str(lose / evaluation_config.test_episodes * 100) + "% \t",)
#              str(tied / evaluation_config.test_episodes * 100) + "% \t")
        pretty_print(average_end_state / evaluation_config.test_episodes)

        tr = sum(total_rewwards_list) / len(total_rewwards_list)
        print("total reward:")
        print(tr)

        f = open("result_self_play_exp.txt", "a+")
        f.write(str(tr) + "\n")
        f.close()

        agent_1.enable_learning()
            
def player_1_win_condition(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp):
    if min(state_1_T_hp, state_1_B_hp) == min(state_2_T_hp, state_2_B_hp):
        if state_1_T_hp + state_1_B_hp == state_2_T_hp + state_2_B_hp:
            return 0
        elif state_1_T_hp + state_1_B_hp > state_2_T_hp + state_2_B_hp:
            return 1
        else:
            return -1
    else:
        if min(state_1_T_hp, state_1_B_hp) > min(state_2_T_hp, state_2_B_hp):
            return 1
        else:
            return -1
    
def pretty_print(state,  text = ""):
    state_list = state.copy().tolist()
    state = []
    for s in state_list:
        state.append(str(s))
    print("===========================================")
    print(text)
    print("Wave:\t" + state[-1])
    print("Minerals:\t" + state[0])
    print("Building_Self")
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}".format(
        state[1],state[2],state[3],state[4],state[5],state[6],state[7]))
    print("Building_Enemy")
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}".format(
        state[8],state[9],state[10],state[11],state[12],state[13],state[14]))
    
    print("Unit_Self")
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5}".format(
        state[15],state[16],state[17],state[18],state[19],state[20]))
    print("Unit_Enemy")
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5}".format(
        state[21],state[22],state[23],state[24],state[25],state[26]))
    print("Hit_Point")
    print("S_T:{:^5},S_B{:^5},E_T{:^5},E_B:{:^5}".format(
        state[27],state[28],state[29],state[30]))
    