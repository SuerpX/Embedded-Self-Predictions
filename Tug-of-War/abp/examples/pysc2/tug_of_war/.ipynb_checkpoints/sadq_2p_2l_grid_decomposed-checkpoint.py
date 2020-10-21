import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch

from abp.configs import NetworkConfig, ReinforceConfig, EvaluationConfig
from abp.adaptives.sadq.adaptive_decom import SADQAdaptive
from abp.utils import clear_summary_path
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war_2L_self_play_4grid import TugOfWar
from sc2env.environments.tug_of_war_2L_self_play_4grid import action_component_names
#from sc2env.xai_replay.recorder.recorder_2lane_nexus import XaiReplayRecorder2LaneNexus
from tqdm import tqdm
from copy import deepcopy
from random import randint, random

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

    if network_config.output_shape == 1:
        reward_num = 1
        combine_decomposed_func = combine_decomposed_func_1
        player_1_end_vector = player_1_end_vector_1
    
    if not reinforce_config.is_random_agent_1:
        agent_1 = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state_1),
                            network_config = network_config,
                            reinforce_config = reinforce_config,
                            reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)
        print("sadq agent 1")
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
                    new_agent_2 = SADQAdaptive(name = file,
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config, memory_resotre = False,
                    reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)
                    
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
                if 'eval.pupdate' in file or 'eval.p_the_best' in file:
                    new_weights = torch.load(restore_path + "/" +file)
                    new_agent_2 = SADQAdaptive(name = file,
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config,
                    memory_resotre = False, reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)
                    new_agent_2.load_weight(new_weights)
                    new_agent_2.disable_learning(is_save = False)
                    agents_2.append(new_agent_2)
                    print("loaded agent:", file)
#     agent_1.steps = reinforce_config.epsilon_timesteps / 2                
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
        
        new_agent_2 = SADQAdaptive(name = "record",
                    state_length = len(state_1),
                    network_config = network_config,
                    reinforce_config = reinforce_config,
                    memory_resotre = False, reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)
        agent_1.load_weight(weights_1)
        new_agent_2.load_weight(weights_2)
        new_agent_2.disable_learning(is_save = False)
        agents_2.append(new_agent_2)
        
    if reinforce_config.is_use_sepcific_enemy:
        sepcific_SADQ_enemy_weights = torch.load(reinforce_config.enemy_path)

        sepcific_network_config = NetworkConfig.load_from_yaml("./tasks/tug_of_war/sadq_2p_2l_decom/v2_8/network.yml")
        sepcific_network_config.restore_network = False
        sepcific_SADQ_enemy = SADQAdaptive(name = "sepcific enemy",
        state_length = len(state_1),
        network_config = sepcific_network_config,
        reinforce_config = reinforce_config,  memory_resotre = False,
        reward_num = sepcific_network_config.output_shape, combine_decomposed_func = combine_decomposed_func_8)

        sepcific_SADQ_enemy.load_weight(sepcific_SADQ_enemy_weights)
        sepcific_SADQ_enemy.disable_learning(is_save = False)
        agents_2 = [sepcific_SADQ_enemy]
        
    while True:
        print(sum(np.array(privous_result) >= 0.9))
        if len(privous_result) >= update_wins_waves and \
        sum(np.array(privous_result) >= 0.9) >= update_wins_waves and \
        not reinforce_config.is_random_agent_2 and not reinforce_config.is_use_sepcific_enemy:
            privous_result = []
            print("replace enemy agent's weight with self agent")
#             random_enemy = False
            f = open(evaluation_config.result_path, "a+")
            f.write("Update agent\n")
            f.close()
            
            new_agent_2 = SADQAdaptive(name = "TugOfWar_" + str(round_num),
                    state_length = len(state_2),
                    network_config = network_config,
                    reinforce_config = reinforce_config,
                    memory_resotre = False, reward_num = reward_num, combine_decomposed_func = combine_decomposed_func)
            
            new_agent_2.load_model(agent_1.eval_model)
            new_agent_2.disable_learning(is_save = False)
            agents_2.append(new_agent_2)
            agent_1.steps = reinforce_config.epsilon_timesteps / 2
            agent_1.best_reward_mean = 0
            agent_1.save(force = True, appendix = "update_" + str(round_num))
            
        round_num += 1
    
        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")
        print("Now training.")
        
        print("Now have {} enemy".format(len(agents_2)))
        
        for idx_enemy, enemy_agent in enumerate(agents_2):
#             break
            if reinforce_config.collecting_experience:
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
#                 if type(enemy_agent) == type("random"):
#                     break
                state_1, state_2 = env.reset()
                total_reward = 0
                skiping = True
                done = False
                steps = 0
    #             print(list(state_1))
    #             print(list(state_2))
                
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    if dp or done:
                        break
                last_mineral = state_1[env.miner_index]
                while not done and steps < max_episode_steps:
                    steps += 1
#                     w += 1
#                     print(w)
                    # Decision point
    #                 print('state:')
    #                 print("=======================================================================")
                    # pretty_print(state_1, text = "state 1")
                    # pretty_print(state_2, text = "state 2")
                    if agent_1.steps < reinforce_config.epsilon_timesteps:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 1)
                    else:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 0)
                    actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = 1)

                    assert state_1[-1] == state_2[-1] == steps, print(state_1, state_2, steps)
                    if not reinforce_config.is_random_agent_1:
                        combine_states_1 = combine_sa(state_1, actions_1)
    #                     print(combine_states_1)
    #                     print(env.normalization(combine_states_1))
    #                     print(state_1[env.miner_index])
                        choice_1, _ = agent_1.predict(env.normalization(combine_states_1))
    #                     input()
    #                     for cs1 in combine_states_1:
    #                         print(cs1.tolist())
                    else:
    #                     combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1 = randint(0, len(actions_1) - 1)

                    if not reinforce_config.is_random_agent_2 and type(enemy_agent) != type("random"):
                        combine_states_2 = combine_sa(state_2, actions_2)
                        choice_2, _ = enemy_agent.predict(env.normalization(combine_states_2))
                    else:
                        if enemy_agent == "random_2":
                            actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
                        choice_2 = randint(0, len(actions_2) - 1)
                    
                           
    #                 print("action list:")
    #                 print(actions_1)
    #                 print(actions_2)
    #                 assign action
#                     print("choice:")
#                     print(actions_1[choice_1])
    #                 print(actions_2[choice_2])
#                     pretty_print(combine_states_1[choice_1], text = "after state:")
    #                 input("pause")
    #                 print(combine_states_2[choice_2].tolist())
    #                 if state_1[env.miner_index] > 300:
#                     input('pause')
                    env.step(list(actions_1[choice_1]), 1)
                    env.step(list(actions_2[choice_2]), 2)
#                     if steps == 39:
#                         env.step([3,0,0,0,0,0,0], 1)
                    
                    last_mineral = combine_states_1[choice_1][env.miner_index]
            
                    l_m_1 = state_1[env.miner_index]
                    l_m_2 = state_2[env.miner_index]
            
                    while skiping:
                        state_1, state_2, done, dp = env.step([], 0)
    #                     input('time_step')
                        if dp or done:
                            break

#                     Check if the mineral is correct
#                     if not done and steps < max_episode_steps and type(enemy_agent) != type("random"):
#                         next_mineral_1 = combine_states_1[choice_1][env.miner_index] + 100 + combine_states_1[choice_1][env.pylon_index] * 75
# #                         if type(enemy_agent) != type("random"):
#                         next_mineral_2 = combine_states_2[choice_2][env.miner_index] + 100 + combine_states_2[choice_2][env.pylon_index] * 75
#                         if next_mineral_1 > 1500:
#                             next_mineral_1 = 1500
#                         if next_mineral_2 > 1500:
#                             next_mineral_2 = 1500
                        
#                         print(next_mineral_1, state_1[env.miner_index], combine_states_1[choice_1], actions_1[choice_1])
                        
# #                         if type(enemy_agent) != type("random"):
#                         print(next_mineral_2, state_2[env.miner_index], combine_states_2[choice_2], actions_2[choice_2])
#                         assert next_mineral_1 == state_1[env.miner_index], print(l_m_1, next_mineral_1, state_1[env.miner_index], combine_states_1[choice_1], actions_1[choice_1])
# #                         if type(enemy_agent) != type("random"):
#                         assert next_mineral_2 == state_2[env.miner_index], print(l_m_2, next_mineral_2, state_2[env.miner_index], combine_states_2[choice_2], actions_2[choice_2])
                        
                    reward = [0] * reward_num
                    if steps == max_episode_steps or done:
                        reward = player_1_end_vector(state_1[63], state_1[64], state_1[65], state_1[66], is_done = done)
                        
#                     reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
#                     print('reward:')
                    # print(state_1[27], state_1[28], state_1[29], state_1[30])
#                     print(reward_1)
#                     print(reward_2)
#                     if steps == max_episode_steps or done:
#                         input()

                    if not reinforce_config.is_random_agent_1:
                        agent_1.reward(reward)

                if not reinforce_config.is_random_agent_1:
                    agent_1.end_episode(env.normalization(state_1))

#                 test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
#                                                global_step = episode + 1)
#                 train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
#                                                 global_step = episode + 1)
        
        if not reinforce_config.is_random_agent_1:
            agent_1.disable_learning(is_save = not reinforce_config.collecting_experience and not evaluation_config.generate_xai_replay)

        total_rewwards_list = []
            
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
        
        tied_lose = 0
        for idx_enemy, enemy_agent in enumerate(agents_2):
            average_end_state = np.zeros(len(state_1))
            if type(enemy_agent) == type("random"):
                print(enemy_agent)
            else:
                print(enemy_agent.name)
                
            if idx_enemy == len(agents_2) - 1 and not reinforce_config.collecting_experience:
                test_num = evaluation_config.test_episodes
            else:
                test_num = 5
                
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
           
#                 print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Starting episode%%%%%%%%%%%%%%%%%%%%%%%%%")
#                 print(f"reinforce_config.collecting_experience {reinforce_config.collecting_experience}")
                while skiping:
#                     print("about to call env.step() during skip")
    #                 start_time = time.time()
                    state_1, state_2, done, dp = env.step([], 0)
                    if evaluation_config.generate_xai_replay:
                        #recorder.save_jpg()
                        recorder.record_game_clock_tick(env.decomposed_reward_dict)
                    if dp or done:
    #                     print(time.time() - start_time)
                        break
#                 input(f"dp is {dp} done is {done}")
#                 print("done stepping to finish prior action")
                while not done and steps < max_episode_steps:
#                     input(f"not done and steps == {steps} < {max_episode_steps}")
                    steps += 1
    #                 # Decision point
                    if not reinforce_config.is_random_agent_1:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                        combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1, _ = agent_1.predict(env.normalization(combine_states_1))
                    else:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = 1)
                        combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1 = randint(0, len(actions_1) - 1)


                    if not reinforce_config.is_random_agent_2 and type(enemy_agent) != type("random"):
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

#                     input("record dp if engaged")
                    if evaluation_config.generate_xai_replay:
                        #recorder.save_jpg()
                        recorder.record_decision_point(actions_1[choice_1], actions_2[choice_2], state_1, state_2, env.decomposed_reward_dict)
                    
    #                 input('stepped with command 2')
                    #######
                    #experience collecting
                    ######
#                     input("collect experience if configured so")
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
    #                     print("Get actions time:")
    #                     start_time = time.time()
#                         input("step to move the game along and send the wave")
                        state_1, state_2, done, dp = env.step([], 0)
                        if evaluation_config.generate_xai_replay:
                            #recorder.save_jpg()
                            recorder.record_game_clock_tick(env.decomposed_reward_dict)
                        #input(' step wating for done signal')
                        if dp or done:
    #                         print(time.time() - start_time)
                            break

                    reward = [0] * reward_num
                    if steps == max_episode_steps or done:
                        reward = player_1_end_vector(state_1[63], state_1[64], state_1[65], state_1[66], is_done = done)
                            
#                     input("separate rewards...")
#                     reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
    #                 print(env.decomposed_rewards)
    #                 print(reward_1, reward_2)

    #                 for r1 in reward_1:
                    if reward_num == 4:
                        current_reward_1 = sum(reward[2:])
                    elif reward_num == 8:
                        current_reward_1 = reward[2] + reward[3] + reward[6] + reward[7]
                    elif reward_num == 1:
                        current_reward_1 = sum(reward)
    #                 print(current_reward_1)


                    total_reward_1 += current_reward_1
    #                 print(total_reward_1)
    #                 if total_reward_1 > 14000 or total_reward_1 < -14000:
    #                     input()
                    previous_reward_1 = current_reward_1
#                 print("collect experience again if configured so")
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

                average_end_state += state_1

                total_rewwards_list.append(total_reward_1)
                test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                               global_step=episode + 1)
                test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                               global_step=episode + 1)
    #         if reinforce_config.collecting_experience:
    #             break
            #print(test.size())
    #         print(total_rewwards_list)
#             print("should be done with episode...")
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
            
            
        tr = sum(total_rewwards_list) / len(total_rewwards_list)
        print("total reward:")
        print(tr)
        
        privous_result.append(tr)
        
        if len(privous_result) > update_wins_waves:
            del privous_result[0]
        f = open(evaluation_config.result_path, "a+")
        f.write(str(tr) + "\n")
        f.close()
        
        if tied_lose == 0 and not reinforce_config.is_random_agent_1:
            agent_1.save(force = True, appendix = "_the_best")
            
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

def combine_decomposed_func_1(q_values):
#     print(q_values)
#     print(torch.sum(q_values, dim = 1))
#     print(torch.sum(q_values))
    return torch.sum(q_values, dim = 1)

def player_1_end_vector_1(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp, is_done = False):
    hp_vector = np.array([state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp])
    min_idx = np.argmin(hp_vector)
    if min_idx == 0 or min_idx == 1:
        return [0]
    else:
        return [1]
        