import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch
import math

from abp.configs import NetworkConfig, ReinforceConfig, EvaluationConfig
from abp.adaptives.sadq.adaptive_decom import SADQAdaptive
from abp.adaptives.gqf.adaptive import SADQ_GQF
from abp.utils import clear_summary_path
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war_2L_self_play_4grid import TugOfWar
from sc2env.environments.tug_of_war_2L_self_play_4grid import action_component_names
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib import features
#from sc2env.xai_replay.recorder.recorder_2lane_nexus import XaiReplayRecorder2LaneNexus
from abp.explanations import esf_action_pair, esf_state_pair
from tqdm import tqdm
from copy import deepcopy
from random import randint, random
from datetime import datetime

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
        generate_xai_replay = evaluation_config.generate_xai_replay or evaluation_config.explanation, xai_replay_dimension = replay_dimension)
    
    reward_types = env.reward_types
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    combine_decomposed_func = combine_decomposed_func_8
    player_1_end_vector = player_1_end_vector_8
    reward_num = 8
    
    def GVFs_v1(state):
        if steps == max_episode_steps or done:
            features = player_1_end_vector(state[63], state[64], state[65], state[66], is_done = done)
        else:
            features = np.zeros(network_config.shared_layers)
        return features
    
    def GVFs_v2(state):
        # 15 : 39 The number of each range units
        
        if steps == max_episode_steps or done:
            position = np.array(state)[15 : 39]
            position[position > 0] = 1
            nexus_health = player_1_end_vector_8(state[63], state[64], state[65], state[66], is_done = done)
            lowest_health = nexus_health[4:]
            if steps == max_episode_steps:
                wave_end = np.array([1])
            else:
                wave_end = np.array([0])
            features = np.concatenate((position, lowest_health, wave_end))
        else:
            features = np.zeros(network_config.shared_layers)
        
        return features
    
    def GVFs_v3(state):
        features = np.zeros(network_config.shared_layers)
        unit_kills, _ = env.get_unit_kill()
        features[:18] = np.array(unit_kills)
        features[18:] = np.array(state[8:14])
#         print(unit_kills)
#         input()
        return features
    def GVFs_v4(state):
        features = np.zeros(network_config.shared_layers)
        unit_kills, unit_be_killed = env.get_unit_kill()
        features[:18] = np.array(unit_kills)
        features[18:24] = np.array(state[:6])
        features[24:42] = np.array(unit_be_killed)
        features[42:] = np.array(state[8:14])
#         print(unit_kills)
#         input()
        return features
    def GVFs_v5(state):
        features = np.array(state[15:63])
        return features
    def GVFs_v6(state):
        features = np.zeros(network_config.shared_layers)
        if steps == max_episode_steps or done:
            features[:8] = player_1_end_vector(state[63], state[64], state[65], state[66], is_done = done)
        damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
        features[8:14] = np.array(damage_to_nexus) / 2000
        features[14:20] = np.array(get_damage_nexus) / 2000
        return features
    
    def GVFs_v7(state):
        features = np.zeros(network_config.shared_layers)
        if steps == max_episode_steps or done:
            features[:8] = player_1_end_vector(state[63], state[64], state[65], state[66], is_done = done)
        damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
        features[8:14] = np.array(damage_to_nexus) / 2000
        features[14:20] = np.array(get_damage_nexus) / 2000
        return features
    
    def GVFs_v8(state):
        features = np.zeros(network_config.shared_layers)
            
        agent_attacking_units, enemy_attacking_units = env.get_attacking()
        features[:12] = np.array(state[15:27])
        features[12:15] = agent_attacking_units[:3]
        features[15:27] = np.array(state[27:39])
        features[27:30] = agent_attacking_units[3:]
        
        features[30:33] = enemy_attacking_units[:3]
        features[33:45] = np.array(state[39:51])
        features[45:48] = enemy_attacking_units[3:]
        features[48:60] = np.array(state[51:63])
        
        if steps == max_episode_steps:
            nexus_health = player_1_end_vector_8(state[63], state[64], state[65], state[66], is_done = done)
            features[60:64] = nexus_health[4:]
            features[64] = 1
        return features
    
    def GVFs_v9(state):
        features = np.zeros(network_config.shared_layers)
        damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
        features[:6] = damage_to_nexus / 2000
        features[6:12] = get_damage_nexus / 2000
        if steps == max_episode_steps:
            nexus_health = player_1_end_vector_8(state[63], state[64], state[65], state[66], is_done = done)
            features[12:16] = nexus_health[4:]
            features[16] = 1
#             features[16] = 0
#             features[17] = 1
#         else:
#             features[16] = 1
        return features
    def GVFs_v10(state):
        features = np.zeros(network_config.shared_layers)
        damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
        features[:6] = damage_to_nexus
        features[6:12] = get_damage_nexus
        
#         print(features)
        features[:3] = features[:3] / (state[65] + sum(features[:3]))
        features[3:6] = features[3:6] / (state[66] + sum(features[3:6]))
        features[6:9] = features[6:9] / (state[63] + sum(features[6:9]))
        features[9:12] = features[9:12] / (state[64] + sum(features[9:12]))
        
#         features[features == float('inf')] = 1
        features[np.isnan(features)] = 0
#         if np.sum(np.isnan(features)) > 0:
#             print(state)
#             print(features)
#             input()
        if steps == max_episode_steps:
            nexus_health = player_1_end_vector_8(state[63], state[64], state[65], state[66], is_done = done)
            features[12:16] = nexus_health[4:]
            features[16] = 1
        return features
    def GVFs_v10_loss_eval(state):
        features = np.zeros(network_config.shared_layers)
        damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
        features[:6] = damage_to_nexus
        features[6:12] = get_damage_nexus
        
#         print(features)
        features[:3] = features[:3]# / (state[65] + sum(features[:3]))
        features[3:6] = features[3:6]# / (state[66] + sum(features[3:6]))
        features[6:9] = features[6:9]# / (state[63] + sum(features[6:9]))
        features[9:12] = features[9:12]# / (state[64] + sum(features[9:12]))
        
#         features[features == float('inf')] = 1
        features[np.isnan(features)] = 0
#         if np.sum(np.isnan(features)) > 0:
#             print(state)
#             print(features)
#             input()
        if steps == max_episode_steps:
            nexus_health = player_1_end_vector_8(state[63], state[64], state[65], state[66], is_done = done)
            features[12:16] = nexus_health[4:]
            features[16] = 1
        return features
    
    def GFVs_all_1(state):
        # eval 2 features
        current_idx = 0
        features = np.zeros(network_config.shared_layers)
#         norm_vector = np.ones(5)
        norm_vector = np.array([1500, 100, 30, 200, 5000])
        
        if steps == max_episode_steps or done:
            features[current_idx : current_idx + 8] = player_1_end_vector(state[63], state[64], state[65], state[66], is_done = done)
        current_idx += 8
#         print(current_idx)
        # self-mineral feature idx 8, enemy-mineral idx 9. self-pylone idx 7, enemy-pylon 14
        features[current_idx] = (state[7] * 75 + 100) / norm_vector[0]
        current_idx += 1
        features[current_idx] = (state[14] * 75 + 100) / norm_vector[0]
        current_idx += 1
#         print(current_idx)
        
        # features idx: The number of self-units will be spawned, length : 6, The number of enemy-units will be spawned, length : 6
        # state idx: The number of self-building 1-6, The number of self-building 8-13, 
        features[current_idx : current_idx + 6] = state[1:7] / norm_vector[1]
        current_idx += 6
        features[current_idx : current_idx + 6] = state[8:14] / norm_vector[1]
        current_idx += 6
#         print(current_idx)
        
        # features idx: The accumulative number of each type of unit in each range from now to the end of the game: length : 30, for enemy: length : 30
        agent_attacking_units, enemy_attacking_units = env.get_attacking()
        features[current_idx : current_idx + 12] = np.array(state[15:27]) / norm_vector[2]
        current_idx += 12
        features[current_idx : current_idx + 3] = agent_attacking_units[:3] / norm_vector[2]
        current_idx += 3
        features[current_idx : current_idx + 12] = np.array(state[27:39]) / norm_vector[2]
        current_idx += 12
        features[current_idx : current_idx + 3] = agent_attacking_units[3:] / norm_vector[2]
        current_idx += 3
#         print(current_idx)
        
        features[current_idx : current_idx + 3] = enemy_attacking_units[:3] / norm_vector[2]
        current_idx += 3
        features[current_idx : current_idx + 12] = np.array(state[39:51]) / norm_vector[2]
        current_idx += 12
        features[current_idx : current_idx + 3] = enemy_attacking_units[3:] / norm_vector[2]
        current_idx += 3
        features[current_idx : current_idx + 12] = np.array(state[51:63]) / norm_vector[2]
        current_idx += 12
#         print(current_idx)
        
        # features idx: Damage of each friendly unit to each Nexus: length : 6, for enemy: length : 6
        damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
        features[current_idx : current_idx + 6] = np.array(damage_to_nexus) / norm_vector[3]
        current_idx += 6
        features[current_idx : current_idx + 6] = np.array(get_damage_nexus) / norm_vector[3]
        current_idx += 6
#         print(current_idx)
        
        # features idx: The number of which friendly troops kills which enemy troops: length : 18, for enemy: length : 18
        unit_kills, unit_be_killed = env.get_unit_kill()
        features[current_idx : current_idx + 18] = np.array(unit_kills) / norm_vector[4]
        current_idx += 18
        features[current_idx : current_idx + 18] = np.array(unit_be_killed) / norm_vector[4]
        current_idx += 18
#         print(current_idx)
        
        # features idx: prob of limitation wave: length : 1
        if steps == max_episode_steps:
            features[current_idx] = 1
        current_idx += 1
#         print(current_idx)
        return features
 
    def GVFs_v11(state):
        features = np.zeros(network_config.shared_layers)
        damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
        features[:6] = damage_to_nexus
        features[6:12] = get_damage_nexus
        
#         print(features)
        features[:3] = features[:3] / (state[65] + sum(features[:3]))
        features[3:6] = features[3:6] / (state[66] + sum(features[3:6]))
        features[6:9] = features[6:9] / (state[63] + sum(features[6:9]))
        features[9:12] = features[9:12] / (state[64] + sum(features[9:12]))
        
#         features[features == float('inf')] = 1
        features[np.isnan(features)] = 0
#         if np.sum(np.isnan(features)) > 0:
#             print(state)
#             print(features)
#             input()
        if steps == max_episode_steps:
            nexus_health = player_1_end_vector_8(state[63], state[64], state[65], state[66], is_done = done)
            features[12:16] = nexus_health[4:]
            features[16] = 0
        else:
            features[16] = 1
        return features
            
    if not reinforce_config.is_random_agent_1:
        agent_1 = SADQ_GQF(name = "TugOfWar_GQF",
                            state_length = len(state_1),
                            network_config = network_config,
                            reinforce_config = reinforce_config,
                            feature_len = network_config.shared_layers,
                            combine_decomposed_func = combine_decomposed_func,
                            memory_resotre = not evaluation_config.explanation
                            )
        print("SADQ_GQF agent 1")
    else:
        print("random agent 1")
    
    
    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    
    agents_2 = ["random", "random_2"]
        
    round_num = 0
    
    privous_result = []
    update_wins_waves = 10
    
    all_experiences = []
    path = './saved_models/tug_of_war/agents/grid'
    exp_save_path = 'abp/examples/pysc2/tug_of_war/rand_v_rand.pt'
    if reinforce_config.collecting_experience and not reinforce_config.is_random_agent_2:
        agent_1_model = "TugOfWar_eval.pupdate_240"
        exp_save_path = 'abp/examples/pysc2/tug_of_war/all_experiences.pt'
        for r, d, f in os.walk(path):
            for file in f:
                if '.p' in file:
                    new_weights = torch.load(path + "/" +file)
                    new_agent_2 = SADQ_GQF(name = file,
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config, 
                    feature_len = network_config.shared_layers, memory_resotre = False,
                    combine_decomposed_func = combine_decomposed_func)
                    
                    new_agent_2.load_weight(new_weights)
                    new_agent_2.disable_learning(is_save = False)
                    agents_2.append(new_agent_2)
                    
                    if agent_1_model == file:
                        print("********agent_1_model", file)
                        agent_1.load_model(new_agent_2.eval_model)
                        
    elif network_config.restore_network:
        restore_path = network_config.network_path
        for r, d, f in os.walk(restore_path):
            f = sorted(f)
            for file in f:
                if '.pupdate' in file:# or '.p_the_best' in file:
                    new_weights = torch.load(restore_path + "/" + file)
                    new_feature_weights, new_q_weights = new_weights
                    new_agent_2 = SADQ_GQF(name = file,
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config,
                    feature_len = network_config.shared_layers,
                    memory_resotre = False, combine_decomposed_func = combine_decomposed_func)
                    new_agent_2.load_weight(new_feature_weights, new_q_weights)
                    new_agent_2.disable_learning(is_save = False)
                    agents_2.append(new_agent_2)
                    print("loaded agent:", file)
#     agent_1.steps = reinforce_config.epsilon_timesteps / 2
    if reinforce_config.is_use_sepcific_enemy:
        sepcific_SADQ_enemy_weights = torch.load(reinforce_config.enemy_path)

        sepcific_network_config = NetworkConfig.load_from_yaml("./tasks/tug_of_war/sadq_2p_2l_decom/v2_8/network.yml")
        sepcific_network_config.restore_network = False
        sepcific_SADQ_enemy = SADQAdaptive(name = "sepcific enemy",
        state_length = len(state_1),
        network_config = sepcific_network_config,
        reinforce_config = reinforce_config,  memory_resotre = False,
        reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)

        sepcific_SADQ_enemy.load_weight(sepcific_SADQ_enemy_weights)
        sepcific_SADQ_enemy.disable_learning(is_save = False)
        agents_2 = [sepcific_SADQ_enemy]
    
    if evaluation_config.generate_xai_replay:

        agent_1_model = "TugOfWar_eval.pupdate_600"
        agent_2_model = "TugOfWar_eval.pupdate_560"
        
        agents_2 = []
        if use_cuda:
            weights_1 = torch.load(path + "/" + agent_1_model)
            weights_2 = torch.load(path + "/" + agent_2_model)
        else:
            weights_1 = torch.load(path + "/" + agent_1_model, map_location=lambda storage, loc:storage)
            weights_2 = torch.load(path + "/" + agent_2_model, map_location=lambda storage, loc:storage)
        
        new_agent_2 = SADQ_GQF(name = "record",
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config,
                    feature_len = network_config.shared_layers,
                    memory_resotre = False, combine_decomposed_func = combine_decomposed_func)
        agent_1.load_weight(weights_1)
        new_agent_2.load_weight(weights_2)
        new_agent_2.disable_learning(is_save = False)
        agents_2.append(new_agent_2)
        
    if evaluation_config.is_state_pair_test_exp:
        time_stamp = datetime.now()
        state_pair(env, agent_1, evaluation_config.explanation_path + "/" + str(time_stamp), network_config.version)
        return
#     w = 0
    if evaluation_config.explanation:
        time_stamp = datetime.now()
        exp_path = evaluation_config.explanation_path + "/" + str(time_stamp)
        game_count = 0
#     agent_1.steps = reinforce_config.epsilon_timesteps / 2
    count_epo = 0
    while True:
        count_epo += 1
#         print(sum(np.array(privous_result) >= 0.9))
        print(np.array(privous_result).mean())
#         if len(privous_result) >= update_wins_waves and \
#         sum(np.array(privous_result) >= 0.9) >= update_wins_waves and \
#         not reinforce_config.is_random_agent_2 and not reinforce_config.is_use_sepcific_enemy:
        if len(privous_result) >= update_wins_waves and \
        np.array(privous_result).mean() >= 0.9 and privous_result[-1] >= 0.9 and \
        not reinforce_config.is_random_agent_2 and not reinforce_config.is_use_sepcific_enemy:
            privous_result = []
            print("replace enemy agent's weight with self agent")
#             random_enemy = False
            f = open(evaluation_config.result_path, "a+")
            f.write("Update agent\n")
            f.close()
            
            new_agent_2 = SADQ_GQF(name = "TugOfWar_" + str(round_num),
                    state_length = len(state_2),
                    network_config = network_config,
                    reinforce_config = reinforce_config,
                    feature_len = network_config.shared_layers,
                    memory_resotre = False, combine_decomposed_func = combine_decomposed_func)
            
            new_agent_2.load_model(agent_1.eval_model)
            new_agent_2.disable_learning(is_save = False)
            agents_2.append(new_agent_2)
            agent_1.steps = reinforce_config.epsilon_timesteps / 2
            agent_1.best_reward_mean = 0
            agent_1.save(force = True, appendix = "update_" + str(round_num))
#         if reinforce_config.is_use_sepcific_enemy:
#             agents_2 = [sepcific_SADQ_enemy]
        round_num += 1
        
        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")
        print("Now training.")
        
        print("Now have {} enemy".format(len(agents_2)))
        
        for idx_enemy, enemy_agent in enumerate(agents_2):
#             break
            if reinforce_config.collecting_experience or evaluation_config.explanation:
                break
            if type(enemy_agent) == type("random"):
                print(enemy_agent)
            else:
                print(enemy_agent.name)
            
            if idx_enemy == len(agents_2) - 1:
                training_num = evaluation_config.training_episodes
            else:
                training_num = 10
            
            for episode in tqdm(range(training_num)):
                state_1, state_2 = env.reset()
                total_reward = 0
                skiping = True
                done = False
                steps = 0
                
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    if dp or done:
                        break
                last_mineral = state_1[env.miner_index]
                while not done and steps < max_episode_steps:
                    steps += 1
                    if agent_1.steps < reinforce_config.epsilon_timesteps:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 1)
                    else:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 0)
                    actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = 1)

                    assert state_1[-1] == state_2[-1] == steps, print(state_1, state_2, steps)
                    if not reinforce_config.is_random_agent_1:
                        combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1, _ = agent_1.predict(env.normalization(combine_states_1))
                    else:
                        choice_1 = randint(0, len(actions_1) - 1)

                    if not reinforce_config.is_random_agent_2 and type(enemy_agent) != type("random"):
                        combine_states_2 = combine_sa(state_2, actions_2)
                        choice_2, _ = enemy_agent.predict(env.normalization(combine_states_2))
                    else:
                        if enemy_agent == "random_2":
                            actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
                        choice_2 = randint(0, len(actions_2) - 1)

                    env.step(list(actions_1[choice_1]), 1)
                    env.step(list(actions_2[choice_2]), 2)
                    
                    last_mineral = combine_states_1[choice_1][env.miner_index]
            
                    l_m_1 = state_1[env.miner_index]
                    l_m_2 = state_2[env.miner_index]
            
                    while skiping:
                        state_1, state_2, done, dp = env.step([], 0)
                        if dp or done:
                            break
                    if network_config.version == "v1":
                        features = GVFs_v1(state_1)
                    elif network_config.version == "v2":
                        features = GVFs_v2(state_1)
                    elif network_config.version == "v3":
                        features = GVFs_v3(state_1)
                    elif network_config.version == "v4":
                        features = GVFs_v4(state_1)
                    elif network_config.version == "v5":
                        features = GVFs_v5(state_1)
                    elif network_config.version == "v6":
                        features = GVFs_v6(state_1)
                    elif network_config.version == "v7":
                        features = GVFs_v7(state_1)
                    elif network_config.version == "v8":
                        features = GVFs_v8(state_1)
                    elif network_config.version == "v9":
                        features = GVFs_v9(state_1)
                    elif network_config.version == "v10":
                        features = GVFs_v10(state_1)
                    elif network_config.version == "v11":
                        features = GVFs_v11(state_1)
                    elif network_config.version in ["GFVs_all_1", "GVFs_all_2"]:
                        features = GFVs_all_1(state_1)
#                     print("features")
#                     print(features)
#                     input()
                    total_reward = 0
                    if steps == max_episode_steps or done:
                        total_reward = combine_decomposed_func(FloatTensor(
                            player_1_end_vector(state_1[63], state_1[64], state_1[65], state_1[66], is_done = done)).view(-1, 8)).item()
#                     print(total_reward)
                    if not reinforce_config.is_random_agent_1:
#                         print("features: ", features)
                        agent_1.passFeatures(features)
                        agent_1.reward(total_reward)

                if not reinforce_config.is_random_agent_1:
                    agent_1.end_episode(env.normalization(state_1))
        
        if not reinforce_config.is_random_agent_1:
            agent_1.disable_learning(is_save = not reinforce_config.collecting_experience 
                                     and not evaluation_config.generate_xai_replay 
                                     and not evaluation_config.explanation)

        total_rewwards_list = []
            
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
        
        tied_lose = 0
        all_GVFs_loss = 0
        
        GVFs_loss_exp = []
        pro_win = []
        for idx_enemy, enemy_agent in enumerate(agents_2):
            average_end_state = np.zeros(len(state_1))
            if type(enemy_agent) == type("random"):
                enemy_name = enemy_agent
                print(enemy_agent)
            else:
                enemy_name = enemy_agent.name
                print(enemy_agent.name)
                
            if idx_enemy == len(agents_2) - 1 and not reinforce_config.collecting_experience:
                test_num = evaluation_config.test_episodes
            else:
                test_num = 5
            if evaluation_config.explanation:
                test_num = 1
                game_count += 1
                save_exp_path = "{}/game_{}({})".format(exp_path, game_count, enemy_name)
            for episode in tqdm(range(test_num)):
                env.reset()
                total_reward_1 = 0
                done = False
                skiping = True
                steps = 0
                previous_state_1 = None
                previous_state_2 = None
                previous_action_1 = None
                previous_action_2 = None 
                if evaluation_config.generate_xai_replay:
                    recorder = XaiReplayRecorder2LaneNexus(env.sc2_env, episode, evaluation_config.env, action_component_names, replay_dimension)
                all_features_gt = FloatTensor(np.zeros((max_episode_steps, network_config.shared_layers)))
                all_features_predict = FloatTensor(np.zeros((max_episode_steps, network_config.shared_layers)))
                discounted_para = FloatTensor(np.zeros((max_episode_steps, 1)))
                all_Nexus_HP = FloatTensor(np.zeros((max_episode_steps, 4)))
                
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    if evaluation_config.generate_xai_replay:
                        recorder.record_game_clock_tick(env.decomposed_reward_dict)
                    if dp or done:
                        break
                while not done and steps < max_episode_steps:
                    steps += 1
                    
                    if not reinforce_config.is_random_agent_1:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                        combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1, fv_1 = agent_1.predict(env.normalization(combine_states_1))
                    else:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 1)
                        combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1 = randint(0, len(actions_1) - 1)
                    if evaluation_config.explanation:
                        
                        frame = get_frame(env.sc2_env)
                        esf_action_pair(agent_1.eval_model, state_1, frame, env.normalization(combine_states_1), 
                                        actions_1, save_exp_path,
                                        decision_point = "dp_{}".format(steps), version = network_config.version, pick_actions = [30, 30, 30])
                    if not reinforce_config.is_random_agent_2 and type(enemy_agent) != type("random"):
                        actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
                        combine_states_2 = combine_sa(state_2, actions_2)
                        choice_2, fv_2 = enemy_agent.predict(env.normalization(combine_states_2))
                    else:
                        if enemy_agent == "random_2":
                            actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = 0)
                        else:
                            actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = 1)
                        combine_states_2 = combine_sa(state_2, actions_2)
                        choice_2 = randint(0, len(actions_2) - 1)

                    if evaluation_config.generate_xai_replay:
                        recorder.record_decision_point(actions_1[choice_1], actions_2[choice_2], state_1, state_2, env.decomposed_reward_dict)
                    
                    if reinforce_config.collecting_experience:
                        if previous_state_1 is not None and previous_state_2 is not None and previous_action_1 is not None and previous_action_2 is not None:
                            previous_state_1[8:14] = previous_state_2[1:7] # Include player 2's action
                            previous_state_1[env.miner_index] += previous_state_1[env.pylon_index] * 75 + 100
                            previous_state_1[-1] += 1
                            
                            experience = [
                                previous_state_1,
                                np.append(state_1, previous_reward_1)
                            ]
                            all_experiences.append(experience)
                            if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                                torch.save(all_experiences, exp_save_path)

                        previous_state_1 = deepcopy(combine_states_1[choice_1])
                        previous_state_2 = deepcopy(combine_states_2[choice_2])

                        previous_action_1 = deepcopy(actions_1[choice_1])
                        previous_action_2 = deepcopy(actions_2[choice_2])
                    
#                     input(f"step p1 with {list(actions_1[choice_1])}")
                    env.step(list(actions_1[choice_1]), 1)
#                     input(f"step p2 with {list(actions_2[choice_2])}")
                    env.step(list(actions_2[choice_2]), 2)
#                     # human play
#                     pretty_print(state_2, text = "state:")
#                     env.step(list(get_human_action()), 2)
#                     reinforce_config.collecting_experience = False

                    while skiping:
                        state_1, state_2, done, dp = env.step([], 0)
                        if evaluation_config.generate_xai_replay:
                            recorder.record_game_clock_tick(env.decomposed_reward_dict)
                        if dp or done:
                            break
                            
                    if network_config.version == "v1":
                        features = GVFs_v1(state_1)
                    elif network_config.version == "v2":
                        features = GVFs_v2(state_1)
                    elif network_config.version == "v3":
                        features = GVFs_v3(state_1)
                    elif network_config.version == "v4":
                        features = GVFs_v4(state_1)
                    elif network_config.version == "v5":
                        features = GVFs_v5(state_1)
                    elif network_config.version == "v6":
                        features = GVFs_v6(state_1)
                    elif network_config.version == "v7":
                        features = GVFs_v7(state_1)
                    elif network_config.version == "v8":
                        features = GVFs_v8(state_1)
                    elif network_config.version == "v9":
                        features = GVFs_v9(state_1)
                    elif network_config.version == "v10":
                        features = GVFs_v10_loss_eval(state_1)
                        all_Nexus_HP[steps - 1] = FloatTensor(state_1[63:67])
                    elif network_config.version == "v11":
                        features = GVFs_v11(state_1)
                        all_Nexus_HP[steps - 1] = FloatTensor(state_1[63:67])
                    elif network_config.version in ["GFVs_all_1", "GVFs_all_2"]:
                        features = GFVs_all_1(state_1)
                        
                    all_features_gt[:steps] += (FloatTensor(features) * (reinforce_config.discount_factor ** discounted_para))[:steps]
                    all_features_predict[steps - 1] = fv_1
                    discounted_para[:steps] += 1
#                     print(discounted_para[:steps])
#                     print(features)
#                     print(all_Nexus_HP[:steps])
#                     print(all_features_gt[:steps])
#                     print(all_features_predict[:steps])
#                     input()
#                     if steps == 1:
#                         break
                    current_reward_1 = 0
                    if steps == max_episode_steps or done:
                        current_reward_1 = combine_decomposed_func(FloatTensor(
                            player_1_end_vector(state_1[63], state_1[64], state_1[65], state_1[66], is_done = done)).view(-1, 8)).item()

                    total_reward_1 += current_reward_1
                    previous_reward_1 = current_reward_1
                    
                if evaluation_config.explanation:
                    if total_reward_1 == 1:
                        os.rename(save_exp_path, save_exp_path + "_win")
                    else:
                        os.rename(save_exp_path, save_exp_path + "_lost")
                    
                if reinforce_config.collecting_experience:
                    previous_state_1[8:14] = previous_state_2[1:7] # Include player 2's action
                    previous_state_1[env.miner_index] += previous_state_1[env.pylon_index] * 75 + 100
                    previous_state_1[-1] += 1
                    
                    experience = [
                        previous_state_1,
                        np.append(state_1, previous_reward_1)
                    ]
                    all_experiences.append(experience)
                    if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                        torch.save(all_experiences, exp_save_path)
                
                if network_config.version in ["v10", "v11"]:
                    all_features_gt[:steps, : 3] /= (all_Nexus_HP[:steps, 2] + torch.sum(all_features_gt[:steps, : 3], dim = 1)).view(-1, 1)
                    all_features_gt[:steps, 3 : 6] /= (all_Nexus_HP[:steps, 3] + torch.sum(all_features_gt[:steps, 3 : 6], dim = 1)).view(-1, 1)
                    all_features_gt[:steps, 6 : 9] /= (all_Nexus_HP[:steps, 0] + torch.sum(all_features_gt[:steps, 6 : 9], dim = 1)).view(-1, 1)
                    all_features_gt[:steps, 9 : 12] /= (all_Nexus_HP[:steps, 1] + torch.sum(all_features_gt[:steps, 9 : 12], dim = 1)).view(-1, 1)
                    
#                 print(all_features_gt[:steps])
#                 print(all_features_predict[:steps])
                with torch.no_grad():
                    all_GVFs_loss += agent_1.eval_model.loss_fn(all_features_gt[:steps], all_features_predict[:steps]).item()
            
                
                average_end_state += state_1

                total_rewwards_list.append(total_reward_1)
                test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                               global_step=episode + 1)
                test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                               global_step=episode + 1)
            
#             print("GVFs loss: {}".format(all_GVFs_loss / test_num))
#             input()
            print(torch.mean((all_features_gt - all_features_predict) ** 2, dim = 1))
            total_rewards_list_np = np.array(total_rewwards_list)

            tied = np.sum(total_rewards_list_np[-test_num:] == 0)
            wins = np.sum(total_rewards_list_np[-test_num:] > 0)
            lose = np.sum(total_rewards_list_np[-test_num:] <= 0)
            
            tied_lose += (tied + lose)
            print("wins/lose/tied")
            print(str(wins / test_num * 100) + "% \t",
                 str(lose / test_num * 100) + "% \t",)
#                  str(tied / test_num * 100) + "% \t")
            pretty_print(average_end_state / test_num)
        averge_GFVs_loss = all_GVFs_loss / len(total_rewwards_list)

        print("average GVFs loss: {}".format(averge_GFVs_loss)) 
#         input()
        tr = sum(total_rewwards_list) / len(total_rewwards_list)
        print("total reward:")
        print(tr)
        
        GVFs_loss_exp.append(averge_GFVs_loss)
        pro_win.append(tr)
        
        if len(GVFs_loss_exp) > 20:
            print(sum(GVFs_loss_exp[:20]) / 20)
            print(sum(pro_win[:20]) / 20)
        
        privous_result.append(tr)
        
        if len(privous_result) > update_wins_waves:
            del privous_result[0]
        if not evaluation_config.explanation:
            f = open(evaluation_config.result_path, "a+")
            f.write(str(tr) + "\n")
            f.close()

            f = open("{}_GVFs_loss.txt".format(evaluation_config.result_path[:-4]), "a+")
            f.write(str(averge_GFVs_loss) + "\n")
            f.close()
        
        if not reinforce_config.is_random_agent_1 and not evaluation_config.explanation:
            agent_1.summary_test(tr, count_epo)
            agent_1.summary_GVFs_loss(averge_GFVs_loss, count_epo)
            
        if tied_lose == 0 and not reinforce_config.is_random_agent_1:
            agent_1.save(force = True, appendix = "_the_best")
            
        if count_epo % 50 == 0 and reinforce_config.is_use_sepcific_enemy:
            agent_1.save(force = True, appendix = "_{}".format(count_epo))
        
        if not reinforce_config.is_random_agent_1:
            agent_1.enable_learning()
            
            

    
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
#   print(q_values)
#     print(q_values[:, :1].size())
    q_values = torch.sum(q_values[:, [2, 3, 6, 7]], dim = 1)
#    print("q values: ", q_values)
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

def get_frame(sc2_env):
    observation = sc2_env._controllers[0]._client.send(observation=sc_pb.RequestObservation())
#     print(observation.observation["rgb_screen"])
    frame = features.Feature.unpack_rgb_image(observation.observation.render_data.map).astype(np.uint8)#.swapaxes(0,2).swapaxes(1,2)
#     frame = features.Feature.unpack(observation.observation).astype(np.int32)  
#     print(np.array(observation.observation.render_data.map))

    return frame

def pretty_print_to_state(pretty_state):
    state = np.zeros(68)
    count = 0
    self_units = []
    enemy_units = []
    for i, ps in enumerate(pretty_state):
        numbers = [float(word) for word in ps.split() if word.isdigit()]
        if len(numbers) == 0:
            continue
        numbers_np = np.array(numbers)
        if count == 0:
            state[-1], state[0] = numbers_np[0], numbers_np[1]
        if count == 1:
            state[1:8] = numbers_np
        if count == 2:
            state[8:15] = numbers_np
        if count in list(range(3, 11)):
            self_units.extend(numbers)
        if count == 10:
            state[15:39] = self_units
        if count in list(range(11, 19)):
            enemy_units.extend(numbers)
        if count == 18:
            state[39:63] = enemy_units
        if count == 19:
            state[63: 67] = numbers_np
        count += 1
    return state
#     print(state)
def state_pair(env, agent, save_path, version):
    state_1 = pretty_print_to_state(open("./explanations/tug_of_war/state_pair_test_example/state_1.txt", "r").readlines())
    state_2 = pretty_print_to_state(open("./explanations/tug_of_war/state_pair_test_example/state_2.txt", "r").readlines())
    
#     print(state_1, state_2)
    save_exp_path = "{}/state_pair".format(save_path)
    esf_state_pair(agent.eval_model, state_1, None, state_2, None, env, save_exp_path,
                   entail = "",txt_info = None, version = version)