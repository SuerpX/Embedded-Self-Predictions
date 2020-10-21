import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch

from abp.adaptives.sadq.adaptive_decom import SADQAdaptive
from abp.adaptives.mb_ts.adaptive_trans_both import TransAdaptive
from abp.utils import clear_summary_path
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war_2L_self_play_4grid import TugOfWar
from sc2env.environments.tug_of_war_2L_self_play_4grid import action_component_names
from sc2env.xai_replay.recorder.recorder_2lane_nexus import XaiReplayRecorder2LaneNexus
from tqdm import tqdm
from copy import deepcopy
from random import randint, random
from collections import OrderedDict

device = torch.device(0)
# np.set_printoptions(precision = 2)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def run_task(evaluation_config, network_config, reinforce_config, map_name = None, train_forever = False):
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
    
    replay_dimension = evaluation_config.xai_replay_dimension
    env = TugOfWar(map_name = map_name, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
    
    reward_types = env.reward_types
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    if network_config.output_shape == 4:
        reward_num = 4
        combine_decomposed_func = combine_decomposed_func_4
        player_1_end_vector = player_1_end_vector_4
        
    if network_config.output_shape == 8:
        reward_num = 8
        combine_decomposed_func = combine_decomposed_func_8
        player_1_end_vector = player_1_end_vector_8
    
    trans_model = TransAdaptive("Tug_of_war", network_config = network_config, reinforce_config = reinforce_config)
    
    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    
    agents_1 = ["random", "random_2"]
    agents_2 = ["random", "random_2"]
        
    round_num = 0
    
    all_experiences = []
    path = './saved_models/tug_of_war/agents/grid_decom'
    if reinforce_config.collecting_experience:
        exp_save_path = 'abp/examples/pysc2/tug_of_war/all_experiences.pt'
        for r, d, f in os.walk(path):
            for file in f:
                if '_eval' in file:
                    
                    new_agent_1 = SADQAdaptive(name = file,
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config, memory_resotre = False,
                    reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)
                    
                    new_weights = torch.load(path + "/" +file, map_location = device)
            #         print(HP_state_dict)
                    new_state_dict = OrderedDict()

                    the_weight = list(new_weights.values())

                    new_keys = list(new_agent_1.eval_model.model.state_dict().keys())
                    
                    for i in range(len(the_weight)):
                        new_state_dict[new_keys[i]] = the_weight[i]
                    
                    new_agent_1.load_weight(new_state_dict)
                    new_agent_1.disable_learning(is_save = False)
                    agents_1.append(new_agent_1)
                    
                    new_agent_2 = SADQAdaptive(name = file,
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config, memory_resotre = False,
                    reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)
                    
                    new_agent_2.load_weight(new_state_dict)
                    new_agent_2.disable_learning(is_save = False)
                    agents_2.append(new_agent_2)
                    print(file)
                    
#     w = 0
    while True:
        
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
        
        tied_lose = 0
        for idx_self, self_agent in enumerate(agents_1):
            
            if type(self_agent) == type("random"):
                agent_1_name = self_agent
            else:
                agent_1_name = self_agent.name
            
            
            for idx_enemy, enemy_agent in enumerate(agents_2):
                print(agent_1_name)
                print("vs")
                average_end_state = np.zeros(len(state_1))
                if type(enemy_agent) == type("random"):
                    print(enemy_agent)
                else:
                    print(enemy_agent.name)
                
                total_rewwards_list = []
                for episode in tqdm(range(evaluation_config.test_episodes)):
                    env.reset()
                    total_reward_1 = 0
                    done = False
                    skiping = True
                    steps = 0
                    previous_state_1 = None
                    previous_state_2 = None
                    while skiping:
                        state_1, state_2, done, dp = env.step([], 0)
                        if dp or done:
                            break
                    while not done and steps < max_episode_steps:
                        steps += 1
        #                 # Decision point
                        if type(self_agent) != type("random"):
                            actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                            combine_states_1 = combine_sa(state_1, actions_1)
                            choice_1, _ = self_agent.predict(env.normalization(combine_states_1))
                        else:
                            if self_agent == "random_2":
                                actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 0)
                            else:
                                actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 1)
                                
                            actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 1)
                            combine_states_1 = combine_sa(state_1, actions_1)
                            choice_1 = randint(0, len(actions_1) - 1)


                        if type(enemy_agent) != type("random"):
                            actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
                            combine_states_2 = combine_sa(state_2, actions_2)
                            choice_2, _ = enemy_agent.predict(env.normalization(combine_states_2))
                        else:
                            if enemy_agent == "random_2":
                                actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = 0)
                            else:
                                actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = 1)
                            combine_states_2 = combine_sa(state_2, actions_2)
                            choice_2 = randint(0, len(actions_2) - 1)

                        #######
                        #experience collecting
                        ######
                        if previous_state_1 is not None and previous_state_2 is not None:
                            previous_state_1[8:15] = previous_state_2[1:8].copy() # Include player 2's action

                            previous_state_1[env.miner_index] += previous_state_1[env.pylon_index] * 75 + 100
                            if previous_state_1[env.miner_index] > 1500:
                                previous_state_1[env.miner_index] = 1500
                            previous_state_1[-1] += 1

                            if np.sum(previous_state_1[0:15] == state_1[0:15]) != 15:

                                print(1)
                                pretty_print(previous_state_1, text = "previous state")
                                pretty_print(state_1, text = "current state")
                                input()
                            if np.sum(previous_state_1[-1] == state_1[-1]) != 1:
                                print(2)
                                pretty_print(previous_state_1, text = "previous state")
                                pretty_print(state_1, text = "current state")
                                input()
                                
#                             pretty_print(previous_state_1, text = "previous state")
#                             pretty_print(state_1, text = "current state")
                            
                            trans_model.add_memory(env.normalization(previous_state_1), env.normalization(state_1))
#                             input()
                            if reinforce_config.collecting_experience:
                                experience = [
                                    previous_state_1,
                                    state_1
                                ]
                                all_experiences.append(experience)
                                if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                                    torch.save(all_experiences, exp_save_path)

                        previous_state_1 = combine_states_1[choice_1].copy()
                        previous_state_2 = combine_states_2[choice_2].copy()

                        env.step(list(actions_1[choice_1]), 1)
                        env.step(list(actions_2[choice_2]), 2)
    #                     # human play
    #                     pretty_print(state_2, text = "state:")
    #                     env.step(list(get_human_action()), 2)
    #                     reinforce_config.collecting_experience = False


                        while skiping:
                            state_1, state_2, done, dp = env.step([], 0)
                            if dp or done:
                                break
                        reward = [0] * reward_num  
                        if steps == max_episode_steps or done:
                            reward = player_1_end_vector(state_1[63], state_1[64], state_1[65], state_1[66], is_done = done)

                        
                        if reward_num == 4:
                            current_reward_1 = sum(reward[2:])
                        elif reward_num == 8:
                            current_reward_1 = reward[2] + reward[3] + reward[6] + reward[7]
                            
                        total_reward_1 += current_reward_1

                    if previous_state_1 is not None and previous_state_2 is not None:
                        previous_state_1[8:15] = previous_state_2[1:8].copy() # Include player 2's action

                        previous_state_1[env.miner_index] += previous_state_1[env.pylon_index] * 75 + 100
                        if previous_state_1[env.miner_index] > 1500:
                            previous_state_1[env.miner_index] = 1500
                        previous_state_1[-1] += 1

                        if reinforce_config.collecting_experience:
                            experience = [
                                previous_state_1,
                                state_1
                            ]
                            all_experiences.append(experience)
                            if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                                torch.save(all_experiences, exp_save_path)

                    average_end_state += state_1

                    total_rewwards_list.append(total_reward_1)

                total_rewards_list_np = np.array(total_rewwards_list)
                print(total_rewards_list_np)
                tied = np.sum(total_rewards_list_np[-evaluation_config.test_episodes:] == 0)
                wins = np.sum(total_rewards_list_np[-evaluation_config.test_episodes:] > 0)
                lose = np.sum(total_rewards_list_np[-evaluation_config.test_episodes:] <= 0)

                tied_lose += (tied + lose)
                print("wins/lose/tied")
                print(str(wins / evaluation_config.test_episodes * 100) + "% \t",
                     str(lose / evaluation_config.test_episodes * 100) + "% \t",)
    #                  str(tied / test_num * 100) + "% \t")
                pretty_print(average_end_state / evaluation_config.test_episodes)


            tr = sum(total_rewwards_list) / len(total_rewwards_list)
            print("total reward:")
            print(tr)
            

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
    print("     M  ,  B  ,  I ")
    print("T1:{:^5},{:^5},{:^5}".format(
        state[15],state[16],state[17]))

    print("T2:{:^5},{:^5},{:^5}".format(
        state[18],state[19],state[20]))

    print("T3:{:^5},{:^5},{:^5}".format(
        state[21],state[22],state[23]))

    print("T4:{:^5},{:^5},{:^5}".format(
        state[24],state[25],state[26]))

    print("B1:{:^5},{:^5},{:^5}".format(
        state[27],state[28],state[29]))

    print("B2:{:^5},{:^5},{:^5}".format(
        state[30],state[31],state[32]))

    print("B3:{:^5},{:^5},{:^5}".format(
        state[33],state[34],state[35]))

    print("B4:{:^5},{:^5},{:^5}".format(
        state[36],state[37],state[38]))

    print("Unit_Enemy")
    print("     M  ,  B  ,  I ")
    print("T1:{:^5},{:^5},{:^5}".format(
        state[39],state[40],state[41]))

    print("T2:{:^5},{:^5},{:^5}".format(
        state[42],state[43],state[44]))

    print("T3:{:^5},{:^5},{:^5}".format(
        state[45],state[46],state[47]))

    print("T4:{:^5},{:^5},{:^5}".format(
        state[48],state[49],state[50]))

    print("B1:{:^5},{:^5},{:^5}".format(
        state[51],state[52],state[53]))

    print("B2:{:^5},{:^5},{:^5}".format(
        state[54],state[55],state[56]))

    print("B3:{:^5},{:^5},{:^5}".format(
        state[57],state[58],state[59]))

    print("B4:{:^5},{:^5},{:^5}".format(
        state[60],state[61],state[62]))

    print("Hit_Point")
    print("S_T:{:^5},S_B{:^5},E_T{:^5},E_B:{:^5}".format(
        state[63],state[64],state[65],state[66]))
    
def get_human_action():
    action = np.zeros(7)
    action_input = input("Input your action:")
    for a in action_input:
        action[int(a) - 1] += 1
    print("your acions : ", action)
    return action

def combine_decomposed_func_4(q_values):
#     print(q_values)
#     print(q_values[:, :1].size())
    q_values = torch.sum(q_values[:, 2:], dim = 1)
#     print(q_values)
#     input("combine")
    return q_values

def combine_decomposed_func_8(q_values):
#     print(q_values)
#     print(q_values[:, :1].size())
    q_values = torch.sum(q_values[:, [2, 3, 6, 7]], dim = 1)
#     print(q_values)
#     input("combine")
    return q_values
            
def player_1_end_vector_4(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp, is_done = False):
    hp_vector = np.array([state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp])
    min_value, idx = np.min(hp_vector), np.argmin(hp_vector)
    
    if min_value == 2000:
        reward_vector = [0.25] * 4
    else:
        reward_vector = [0] * 4
        reward_vector[idx] = 1
#     print(hp_vector)
#     print(reward_vector)
#     input("reward_vector")
    return reward_vector

def player_1_end_vector_8(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp, is_done = False):
    
    hp_vector = np.array([state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp])
    min_value, idx = np.min(hp_vector), np.argmin(hp_vector)
    
    if not is_done:
        if min_value == 2000:
            reward_vector = [0] * 4 + [0.25] * 4
        else:
            reward_vector = [0] * 8
            reward_vector[idx + 4] = 1
    else:
        reward_vector = [0] * 8
        reward_vector[idx] = 1
        
#     print(hp_vector)
#     print(reward_vector)
#     input("reward_vector")

    return reward_vector