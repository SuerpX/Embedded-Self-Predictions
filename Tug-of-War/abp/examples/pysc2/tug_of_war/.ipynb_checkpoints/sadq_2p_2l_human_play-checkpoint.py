# import gym
# import time
# import numpy as np
# from absl import flags
# import sys, os
# import torch

# from abp.adaptives import SADQAdaptive, MBTSAdaptive
# from abp.utils import clear_summary_path
# from abp.explanations import PDX
# from tensorboardX import SummaryWriter
# from gym.envs.registration import register
# from sc2env.environments.tug_of_war_2L_human_play import TugOfWar
# from tqdm import tqdm
# from copy import deepcopy
# from random import randint, random

# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# def run_task(evaluation_config, network_config, reinforce_config, map_name = None, train_forever = False, agent_model = None):
#     if (use_cuda):
#         print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
#         print("|       USING CUDA       |")
#         print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     else:
#         print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
#         print("|     NOT USING CUDA     |")
#         print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     flags.FLAGS(sys.argv[:1])
#     max_episode_steps = 40
    
#     replay_dimension = evaluation_config.xai_replay_dimension
    
#     env = TugOfWar(map_name = None, \
#         generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
        
#     reward_types = env.reward_types
#     combine_sa = env.combine_sa
#     state = env.reset()
    
#     if not reinforce_config.is_random_agent_1:
#         agent = SADQAdaptive(name = "TugOfWar",
#                             state_length = len(state),
#                             network_config = network_config,
#                             reinforce_config = reinforce_config)
#         print("sadq agent 1")
# #         models_path = "abp/examples/pysc2/tug_of_war/models_mb/"
# #         agent = MBTSAdaptive(name = "TugOfWar", state_length = len(state),
# #                             network_config = network_config, reinforce_config = reinforce_config,
# #                               models_path = models_path, env = env)
#     else:
#         print("random agent 1")
        
#     training_summaries_path = evaluation_config.summaries_path + "/train"
#     clear_summary_path(training_summaries_path)
#     train_summary_writer = SummaryWriter(training_summaries_path)

#     test_summaries_path = evaluation_config.summaries_path + "/test"
#     clear_summary_path(test_summaries_path)
#     test_summary_writer = SummaryWriter(test_summaries_path)
    
#     round_num = 0
#     all_experiences = []
#     path = './saved_models/tug_of_war/agents/'
#     if agent_model is not None and not reinforce_config.is_random_agent_1:
#         new_weights = torch.load(path + "/" + agent_model)
#         agent.load_weight(new_weights)
#         agent.disable_learning(is_save = False)
#         evaluation_config.training_episodes = 0
        
#     while True:
#         round_num += 1
    
#         print("=======================================================================")
#         print("===============================Now training============================")
#         print("=======================================================================")
#         print("Now training.")
        
#         for episode in tqdm(range(evaluation_config.training_episodes)):
#             state = env.reset()
#             total_reward = 0
#             skiping = True
#             done = False
#             steps = 0
#             print(list(state))
#             while skiping:
#                 state, done, dp = env.step([])
#                 if dp or done:
#                     break
                    
#             while not done and steps < max_episode_steps:
#                 steps += 1
#                 # Decision point
#                 print('state:')
#                 print("=======================================================================")
#                 pretty_print(state, text = "state")

#                 actions = env.get_big_A(state[env.miner_index], state[env.pylon_index], is_train = True)
                
# #                 assert state[-1] == steps, print(state, steps)
                
#                 if not reinforce_config.is_random_agent_1:
#                     combine_states = combine_sa(state, actions)
#                     choice, _ = agent.predict(env.normalization(combine_states))
# #                     input()
#                     for cs in combine_states:
#                         print(cs.tolist())
#                 else:
#                     choice = randint(0, len(actions) - 1)
#                 print("action list:")
#                 print(actions)
# #                 assign action
#                 print("choice:")
#                 print(actions[choice])
#                 pretty_print(combine_states[choice], text = "after state:")
# #                 input("pause")
#                 env.step(list(actions[choice]))
                
#                 while skiping:
#                     state, done, dp = env.step([])
#                     if dp or done:
#                         break
                        
#                 if steps == max_episode_steps or done:
#                     win_lose = agent_win_condition(state[27], state[28], state[29], state[30])

#                     if win_lose == 1:
#                         env.decomposed_rewards[4] = 10000
#                         env.decomposed_rewards[5] = 0
#                     elif win_lose == -1:
#                         env.decomposed_rewards[4] = 0
#                         env.decomposed_rewards[5] = -10000
#                 print("reward:")
#                 print(env.decomposed_rewards)

#                 if not reinforce_config.is_random_agent_1:
#                     agent_1.reward(sum(env.decomposed_rewards))

#             if not reinforce_config.is_random_agent_1:
#                 agent.end_episode(state)

#             test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
#                                            global_step = episode + 1)
#             train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
#                                             global_step = episode + 1)

#         if not reinforce_config.is_random_agent_1:
#             agent.disable_learning(is_save = not reinforce_config.collecting_experience)

#         total_rewwards_list = []
            
#         # Test Episodes
#         print("======================================================================")
#         print("===============================Now testing============================")
#         print("======================================================================")
        
#         tied_lose = 0
#         average_end_state = np.zeros(len(state))

#         for episode in tqdm(range(evaluation_config.test_episodes)):
#             state = env.reset()
#             total_reward = 0
#             skiping = True
#             done = False
#             steps = 0
#             total_reward = 0
            
#             while skiping:
#                 state, done, dp = env.step([])
#                 if dp or done:
#                     break
                    
#             while not done and steps < max_episode_steps:
#                 steps += 1
#                 # Decision point
#                 print('state:')
#                 print("=======================================================================")
#                 pretty_print(state, text = "state")

#                 actions = env.get_big_A(state[env.miner_index], state[env.pylon_index], is_train = True)
                
# #                 assert state[-1] == steps, print(state, steps)
#                 combine_states = combine_sa(state, actions)
#                 if not reinforce_config.is_random_agent_1:
                    
#                     choice, _ = agent.predict(env.normalization(combine_states))
# #                     input()
# #                     for cs in combine_states:
# #                         print(cs.tolist())
#                 else:
#                     choice = randint(0, len(actions) - 1)
# #                 print("action list:")
# #                 print(actions)
# #                 assign action
#                 print("choice:")
#                 print(actions[choice])
# #                 pretty_print(combine_states[choice], text = "after state:")
# #                 input("pause")
#                 # for model base agent 
# #                 action_model_base = agent.predict(combine_states, env.data['P1Minerals'])
# #                 env.step(action_model_base)
                    
#                 # model free agent
#                 env.step(list(actions[choice]))
                
#                 while skiping:
#                     state, done, dp = env.step([])
#                     if dp or done:
#                         break
                        
#                 if steps == max_episode_steps or done:
#                     win_lose = agent_win_condition(state[27], state[28], state[29], state[30])

#                     if win_lose == 1:
#                         env.decomposed_rewards[4] = 10000
#                         env.decomposed_rewards[5] = 0
#                     elif win_lose == -1:
#                         env.decomposed_rewards[4] = 0
#                         env.decomposed_rewards[5] = -10000
#                 print("reward:")
#                 print(env.decomposed_rewards)
#                 total_reward += sum(env.decomposed_rewards)
                
#             average_end_state += state

#             total_rewwards_list.append(total_reward)
#             test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
#                                            global_step=episode + 1)
#             test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
#                                            global_step=episode + 1)

#             total_rewards_list_np = np.array(total_rewwards_list)

#         tied = np.sum(total_rewards_list_np[-test_num:] == 0)
#         wins = np.sum(total_rewards_list_np[-test_num:] > 0)
#         lose = np.sum(total_rewards_list_np[-test_num:] < 0)

#         tied_lose += (tied + lose)
#         print("wins/lose/tied")
#         print(str(wins / test_num * 100) + "% \t",
#              str(lose / test_num * 100) + "% \t",
#              str(tied / test_num * 100) + "% \t")
#         pretty_print(average_end_state / test_num)


#     tr = sum(total_rewwards_list) / len(total_rewwards_list)
#     print("total reward:")
#     print(tr)

#     f = open("result_self_play_2l_human_play.txt", "a+")
#     f.write(str(tr) + "\n")
#     f.close()

# #     if tied_lose == 0 and not reinforce_config.is_random_agent_1 and agent_model is None:
# #         agent.save(force = True, appendix = "_the_best")

# #     if not reinforce_config.is_random_agent_1 and agent_model is None:
# #         agent.enable_learning()
            
# def agent_win_condition(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp):
#     if min(state_1_T_hp, state_1_B_hp) == min(state_2_T_hp, state_2_B_hp):
#         if state_1_T_hp + state_1_B_hp == state_2_T_hp + state_2_B_hp:
#             return 0
#         elif state_1_T_hp + state_1_B_hp > state_2_T_hp + state_2_B_hp:
#             return 1
#         else:
#             return -1
#     else:
#         if min(state_1_T_hp, state_1_B_hp) > min(state_2_T_hp, state_2_B_hp):
#             return 1
#         else:
#             return -1
    
# def pretty_print(state,  text = ""):
#     state_list = state.copy().tolist()
#     state = []
#     for s in state_list:
#         state.append(str(s))
#     print("===========================================")
#     print(text)
#     print("Wave:\t" + state[-1])
#     print("Minerals:\t" + state[0])
#     print("Building_Self")
#     print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}".format(
#         state[1],state[2],state[3],state[4],state[5],state[6],state[7]))
#     print("Building_Enemy")
#     print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}".format(
#         state[8],state[9],state[10],state[11],state[12],state[13],state[14]))
    
#     print("Unit_Self")
#     print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5}".format(
#         state[15],state[16],state[17],state[18],state[19],state[20]))
#     print("Unit_Enemy")
#     print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5}".format(
#         state[21],state[22],state[23],state[24],state[25],state[26]))
#     print("Hit_Point")
#     print("S_T:{:^5},S_B{:^5},E_T{:^5},E_B:{:^5}".format(
#         state[27],state[28],state[29],state[30]))
    
# def get_human_action():
#     action = np.zeros(7)
#     action_input = input("Input your action:")
#     for a in action_input:
#         action[int(a) - 1] += 1
#     print("your acions : ", action)
#     return action
            
    
    
import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch

from abp.adaptives import SADQAdaptive, MBTSAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war_2L_human_play import TugOfWar
from tqdm import tqdm
from copy import deepcopy
from random import randint, random

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
    
    replay_dimension = evaluation_config.xai_replay_dimension
    
    env = TugOfWar(map_name = None, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
        
    reward_types = env.reward_types
    combine_sa = env.combine_sa
    state = env.reset()
    
    if not reinforce_config.is_random_agent_1:
        agent = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state),
                            network_config = network_config,
                            reinforce_config = reinforce_config)
        print("sadq agent 1")
        models_path = "abp/examples/pysc2/tug_of_war/models_mb/"
        agent = MBTSAdaptive(name = "TugOfWar", state_length = len(state),
                            network_config = network_config, reinforce_config = reinforce_config,
                              models_path = models_path, depth = 2, action_ranking = float('inf'), env = env)
    else:
        print("random agent 1")
        
    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    
    round_num = 0
    all_experiences = []
    path = './saved_models/tug_of_war/agents/'
#     if agent_model is not None and not reinforce_config.is_random_agent_1:
#         new_weights = torch.load(path + "/" + agent_model)
#         agent.load_weight(new_weights)
#         agent.disable_learning(is_save = False)
#         evaluation_config.training_episodes = 0
        
    while True:
        round_num += 1
    
        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")
        print("Now training.")
        
        for episode in tqdm(range(evaluation_config.training_episodes)):
            state = env.reset()
            total_reward = 0
            skiping = True
            done = False
            steps = 0
            print(list(state))
            while skiping:
                state, done, dp = env.step([])
                if dp or done:
                    break
                    
            while not done and steps < max_episode_steps:
                steps += 1
                # Decision point
                print('state:')
                print("=======================================================================")
                pretty_print(state, text = "state")

                actions = env.get_big_A(state[env.miner_index], state[env.pylon_index], is_train = True)
                
#                 assert state[-1] == steps, print(state, steps)
                
                if not reinforce_config.is_random_agent_1:
                    combine_states = combine_sa(state, actions)
                    choice, _ = agent.predict(env.normalization(combine_states))
                    input()
                    for cs in combine_states:
                        print(cs.tolist())
                else:
                    choice = randint(0, len(actions) - 1)
                print("action list:")
                print(actions)
#                 assign action
                print("choice:")
                print(actions[choice])
                pretty_print(combine_states[choice], text = "after state:")
                input("pause")
                env.step(list(actions[choice]))
                
                while skiping:
                    state, done, dp = env.step([])
                    if dp or done:
                        break
                        
                if steps == max_episode_steps or done:
                    win_lose = agent_win_condition(state[27], state[28], state[29], state[30])

                    if win_lose == 1:
                        env.decomposed_rewards[4] = 10000
                        env.decomposed_rewards[5] = 0
                    elif win_lose == -1:
                        env.decomposed_rewards[4] = 0
                        env.decomposed_rewards[5] = -10000
                print("reward:")
                print(env.decomposed_rewards)

                if not reinforce_config.is_random_agent_1:
                    agent_1.reward(sum(env.decomposed_rewards))

            if not reinforce_config.is_random_agent_1:
                agent.end_episode(state)

            test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                           global_step = episode + 1)
            train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                            global_step = episode + 1)

#         if not reinforce_config.is_random_agent_1:
#             agent.disable_learning(is_save = not reinforce_config.collecting_experience)

        total_rewwards_list = []
            
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
        
        tied_lose = 0
        average_end_state = np.zeros(len(state))

        for episode in tqdm(range(evaluation_config.test_episodes)):
            state = env.reset()
            total_reward = 0
            skiping = True
            done = False
            steps = 0
            total_reward = 0
            
            while skiping:
                state, done, dp = env.step([])
                if dp or done:
                    break
                    
            while not done and steps < max_episode_steps:
                steps += 1
                # Decision point
                print('state:')
                print("=======================================================================")
                pretty_print(state, text = "state")

                actions = env.get_big_A(state[env.miner_index], state[env.pylon_index], is_train = True)
                
#                 assert state[-1] == steps, print(state, steps)
                combine_states = combine_sa(state, actions)
#                 if not reinforce_config.is_random_agent_1:
                    
#                     choice, _ = agent.predict(env.normalization(combine_states))
#                     input()
#                     for cs in combine_states:
#                         print(cs.tolist())
#                 else:
#                     choice = randint(0, len(actions) - 1)
#                 print("action list:")
#                 print(actions)
# #                 assign action
#                 print("choice:")
#                 print(actions[choice])
#                 pretty_print(combine_states[choice], text = "after state:")
#                 input("pause")
                # for model base agent 
                action_model_base = agent.predict(state, int(env.data['P1Minerals']) - 1)
                print(action_model_base)
                env.step(action_model_base)
                    
                # model free agent
#                 env.step(list(actions[choice]))
                
                while skiping:
                    state, done, dp = env.step([])
                    if dp or done:
                        break
                        
                if steps == max_episode_steps or done:
                    win_lose = agent_win_condition(state[27], state[28], state[29], state[30])

                    if win_lose == 1:
                        env.decomposed_rewards[4] = 10000
                        env.decomposed_rewards[5] = 0
                    elif win_lose == -1:
                        env.decomposed_rewards[4] = 0
                        env.decomposed_rewards[5] = -10000
#                 print("reward:")
#                 print(env.decomposed_rewards)
                total_reward += sum(env.decomposed_rewards)
                
            average_end_state += state

            total_rewwards_list.append(total_reward)
            test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                           global_step=episode + 1)
            test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                           global_step=episode + 1)

            total_rewards_list_np = np.array(total_rewwards_list)

        tied = np.sum(total_rewards_list_np[-test_num:] == 0)
        wins = np.sum(total_rewards_list_np[-test_num:] > 0)
        lose = np.sum(total_rewards_list_np[-test_num:] < 0)

        tied_lose += (tied + lose)
        print("wins/lose/tied")
        print(str(wins / test_num * 100) + "% \t",
             str(lose / test_num * 100) + "% \t",
             str(tied / test_num * 100) + "% \t")
        pretty_print(average_end_state / test_num)


    tr = sum(total_rewwards_list) / len(total_rewwards_list)
    print("total reward:")
    print(tr)

    f = open("result_self_play_2l_human_play.txt", "a+")
    f.write(str(tr) + "\n")
    f.close()

#     if tied_lose == 0 and not reinforce_config.is_random_agent_1 and agent_model is None:
#         agent.save(force = True, appendix = "_the_best")

#     if not reinforce_config.is_random_agent_1 and agent_model is None:
#         agent.enable_learning()
            
def agent_win_condition(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp):
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
    
def get_human_action():
    action = np.zeros(7)
    action_input = input("Input your action:")
    for a in action_input:
        action[int(a) - 1] += 1
    print("your acions : ", action)
    return action
            