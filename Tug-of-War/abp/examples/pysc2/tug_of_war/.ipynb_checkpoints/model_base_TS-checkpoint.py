import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch

from abp.adaptives import MBTSAdaptive, SADQAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war_2L_self_play import TugOfWar
from sc2env.environments.tug_of_war_2L_self_play import action_component_names
from sc2env.xai_replay.recorder.recorder_2lane_nexus import XaiReplayRecorder2LaneNexus
from tqdm import tqdm
from copy import deepcopy
from random import randint

np.set_printoptions(precision = 2)
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
    
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    models_path = "abp/examples/pysc2/tug_of_war/models_mb/"
    agent_1 = MBTSAdaptive(name = "TugOfWar", state_length = len(state_1),
                        network_config = network_config, reinforce_config = reinforce_config,
                          models_path = models_path, depth = 2, action_ranking = 4, env = env)
    
    if not reinforce_config.is_random_agent_2:
        agent_2 = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state_2),
                            network_config = network_config,
                            reinforce_config = reinforce_config, is_sigmoid = True, memory_resotre = False)
        agent_2.eval_model.replace(agent_1.q_model)
        print("sadq agent 2")
    else:
        print("random agent 2")
        
    path = './saved_models/tug_of_war/agents'
    
    agents_2 = []
    agents_2.append(agent_2)
    if evaluation_config.generate_xai_replay and not reinforce_config.is_random_agent_2:
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
#             print(d)
            if len(d) == 3:
                for file in f:
                    if '.p' in file:
                        new_weights = torch.load(path + "/" +file)
                        new_agent_2 = SADQAdaptive(name = file,
                        state_length = len(state_1),
                        network_config = network_config,
                        reinforce_config = reinforce_config)
                        new_agent_2.load_weight(new_weights)
                        new_agent_2.disable_learning(is_save = False)
                        agents_2.append(new_agent_2)
                        
    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    random_enemy = False
    while True:
#         if not reinforce_config.is_random_agent_2:
#             agent_2.disable_learning()

        
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
        print("There are {} enemies".format(len(agents_2)))
        
        for agent_2 in agents_2:
            print(agent_2.name)
            average_state = np.zeros(len(state_1))
            total_rewwards_list = []  
            for episode in tqdm(range(10)):
                state = env.reset()
                total_reward_1 = 0
                done = False
                skiping = True
                steps = 0
                if evaluation_config.generate_xai_replay:
                    recorder = XaiReplayRecorder2LaneNexus(env.sc2_env, episode, evaluation_config.env, action_component_names, replay_dimension)

                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    if evaluation_config.generate_xai_replay:
                        #recorder.save_jpg()
                        recorder.record_game_clock_tick(env.decomposed_reward_dict)

                    if dp or done:
                        break
    #             input("done stepping to finish prior action")
                while not done and steps < max_episode_steps:
                    steps += 1
    #                 # Decision point
    #                 print('state:')
    #                 print(list(env.denormalization(state_1)))
    #                 print(list(env.denormalization(state_2)))
                    actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                    actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])

    #                 choice_1 = agent_1.predict(env.denormalization(state_1), env.denormalization(state_2)[env.miner_index])
    #                 print(state_1)
                    actions_1111111, node = agent_1.predict(state_1, state_2[env.miner_index], dp = steps)
#                     print()
#                     print()
#                     print()
# #                     node.print_tree(p_best_q_value = True, p_action = True, p_after_q_value = True)
                    if evaluation_config.generate_xai_replay: 
                        path_whole_tree = recorder.json_pathname[:-5] + "_whole_tree/"
                        print(path_whole_tree)
                        path_partial_tree = recorder.json_pathname[:-5] + "_partial_tree/"
                        print(path_partial_tree)
                        
                        if not os.path.exists(path_whole_tree):
                            os.mkdir(path_whole_tree)
                        if not os.path.exists(path_partial_tree):
                            os.mkdir(path_partial_tree)
                            
                        node.save_into_json(path = path_whole_tree, dp = steps)   
                        node.save_into_json(path = path_partial_tree, dp = steps, is_partial = True)
                        
#                     input()
    #                 print(actions_1111111)
#                     input()

    #                 input("state_1 checked")
                    combine_states_2 = combine_sa(state_2, actions_2)
                    if not reinforce_config.is_random_agent_2 and not random_enemy:
                        choice_2, _ = agent_2.predict(env.normalization(combine_states_2))
                    else:
                        choice_2 = randint(0, len(actions_2) - 1)

                    if evaluation_config.generate_xai_replay:
                        #recorder.save_jpg()
                        recorder.record_decision_point(actions_1111111, actions_2[choice_2], state_1, state_2, env.decomposed_reward_dict)
    
    #                 env.step(list(actions_1[choice_1]), 1)

    #                 print(actions_2[choice_2])
    #                 pretty_print(state_2, text = "state:")
    #                 input()
                    env.step(list(actions_1111111), 1)

                    env.step(list(actions_2[choice_2]), 2)
                    # human play

    #                 env.step(list(get_human_action()), 2)
    #                 print(actions_1111111)

                    while skiping:
                        state_1, state_2, done, dp = env.step([], 0)
                        #input(' step wating for done signal')
                        if evaluation_config.generate_xai_replay:
                            #recorder.save_jpg()
                            recorder.record_game_clock_tick(env.decomposed_reward_dict)

                        if dp or done:
                            break

                    if steps == max_episode_steps or done:
                        if evaluation_config.generate_xai_replay:
                            recorder.done_recording()

                        win_lose = player_1_win_condition(state_1[27], state_1[28], state_1[29], state_1[30])

                        if win_lose == 1:
                            env.decomposed_rewards[4] = 10000
                            env.decomposed_rewards[5] = 0
                        elif win_lose == -1:
                            env.decomposed_rewards[4] = 0
                            env.decomposed_rewards[5] = 10000
                    reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
                    total_reward_1 += sum(reward_1)
                    
                average_state += state_1
                total_rewwards_list.append(total_reward_1)
#                 print(total_rewwards_list)
                test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                               global_step=episode + 1)
                test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                               global_step=episode + 1)

            tr = sum(total_rewwards_list) / evaluation_config.test_episodes
            print("total reward:")
            print(tr)

            f = open("result_model_based_v3.txt", "a+")
            f.write(agent_2.name + "\n")
            f.write(str(tr) + "\n")
            f.write(np.array2string(average_state / evaluation_config.test_episodes, precision=2, separator=',', suppress_small=True) + "\n")
            
            f.close()
#         break
        
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
    
def get_human_action():
    action = np.zeros(7)
    action_input = input("Input your action:")
    for a in action_input:
        action[int(a) - 1] += 1
    print("your acions : ", action)
    return action
            
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