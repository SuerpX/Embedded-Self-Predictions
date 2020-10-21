import torch
import numpy as np
import matplotlib.pyplot as plt
import pprint
import operator
import collections
import sys

a_mar = 0
a_vik = 1
a_col = 2
a_pyl = 3
a_nex = 4
e_mar = 5
e_vik = 6
e_col = 7
e_pyl = 8
e_nex = 9

def main():
    data_file = input("Please enter the file name you want to load:\t")
    data = torch.load(data_file)
    data = np.array(data).tolist()

    # data = torch.load('60000_sadq_v_random.pt')

    print_ally_episode_win, print_enemy_episode_win, show_win_percentage, show_average_cases, i_lower, i_upper, show_win_timeline, show_strats, show_episodes_len = get_options(len(data)-1)
    ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_refined_strats, sorted_episodes_length = gather_data(data, print_ally_episode_win, print_enemy_episode_win, show_win_percentage, show_average_cases, i_lower, i_upper, show_win_timeline)
    # ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_refined_strats, sorted_episodes_length = gather_data(data, 0, 0, 0, 0, 0, 63399, 0)

    if(show_win_percentage):
        win_percentage_graph(ally_wins, enemy_wins)
    if(show_win_timeline):
        win_total_line_graph(win_total_timeline)
    if(show_average_cases):
        average_case_graph(sum_ally_units_win, ally_wins, sum_enemy_units_win, enemy_wins)
    if(show_strats):
        strats_histogram(sorted_refined_strats)
    if(show_episodes_len):
        print_episodes_by_len(sorted_episodes_length)


def gather_data(data, print_ally_episode_win, print_enemy_episode_win, show_win_percentage, show_average_cases, i_lower, i_upper, show_win_timeline):
    # init vars
    ally_wins = 0
    enemy_wins = 0
    curr_episode = 0

    sum_ally_units_win = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sum_enemy_units_win = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #0-4 ally: mar, vik, col, pyl, nexus, 5-9 enemy: mar, vik, col, pyl, nexus
    win_total_timeline = [0]

    episodes_length = {  }

    moves_0_3 = {  }
    moves_4_7 = {  }
    moves_8_11 = {  }
    moves_12_15 = {  }
    moves_16_19 = {  }
    moves_20_23 = {  }
    moves_24_27 = {  }
    moves_28_31 = {  }
    moves_32_34 = {  }

    episode_strategies = [ moves_0_3, moves_4_7, moves_8_11, moves_12_15, moves_16_19, moves_20_23, moves_24_27, moves_28_31, moves_32_34 ]

    for i in range(i_lower, i_upper):
            
        #checks for end of episode
        c = i
        p = i-1
        current_building_total =  ((data[c][0][a_mar]) + (data[c][0][a_vik]) + (data[c][0][a_col]) + (data[c][0][a_pyl]) + (data[c][0][e_mar]) + (data[c][0][e_vik]) + (data[c][0][e_col]) + (data[c][0][e_pyl]))
        previous_building_total = ((data[p][0][a_mar]) + (data[p][0][a_vik]) + (data[p][0][a_col]) + (data[p][0][a_pyl]) + (data[p][0][e_mar]) + (data[p][0][e_vik]) + (data[p][0][e_col]) + (data[p][0][e_pyl]))

        if current_building_total < previous_building_total:

            #create a dictionary of episode ids (0 - num of eps) and episode lengths
            if(curr_episode != 0):
                episode_len = 0
                aggregate_prior_episode_length = 0

                for k in range(curr_episode):
                    aggregate_prior_episode_length += episodes_length[k]
                episode_len = i - aggregate_prior_episode_length
                episodes_length[curr_episode] = episode_len
            else:
                episode_len = i 
                episodes_length[curr_episode] = episode_len
            

            #making a strategy that consists of four moves for the episode length
            episode_strategies = seperate_strategy(data, episodes_length, episode_strategies, i, curr_episode)
    
            curr_episode += 1
    
            # check for ally win
            if (data[i-1][0][a_nex]) > (data[i-1][0][e_nex]):
                if (print_ally_episode_win):
                    print_episode_end_state_and_next_state(i-1,data, episodes_length, curr_episode) 
                    ally_graph(i-1,data)
                
                ally_wins += 1
                win_total_timeline.append((win_total_timeline[len(win_total_timeline)-1]) + 1)
                
                for x in range(0,10):
                    sum_ally_units_win[x] = data[i-1][1][x] + sum_ally_units_win[x]
            
            #check for ally loss
            elif (data[i-1][0][a_nex]) < (data[i-1][0][e_nex]):
                if (print_enemy_episode_win):
                    print_episode_end_state_and_next_state(i-1,data, episodes_length, curr_episode) 
                    enemy_graph(i-1,data)
                
                enemy_wins += 1
                win_total_timeline.append((win_total_timeline[len(win_total_timeline)-1])-1)

                for x in range(0,10):
                    sum_enemy_units_win[x] = data[i-1][1][x] + sum_enemy_units_win[x]
            

    #creates a list of tuples for strategies and episode lengths
    sorted_episodes_length = sorted(episodes_length.items(), key=operator.itemgetter(1), reverse=True)
    
    sorted_episode_strats = [[],[],[],[],[],[],[],[],[]]
    for k in range(len(episode_strategies)):
        sorted_episode_strats[k] = sorted(episode_strategies[k].items(), key=operator.itemgetter(1), reverse=True)
    
    # sorted_refined_episode_strats = [i for i in sorted_episode_strats[0] if i[1] > 10]
    return ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_episode_strats, sorted_episodes_length



def get_options(data_range):
    print_ally_episode_win = -1
    print_enemy_episode_win = -1
    show_win_percentage = -1
    show_average_cases = -1
    show_win_timeline = -1
    show_strats = -1
    show_episodes_len = -1

    cases_hi = -1
    cases_lo = -1

    while(cases_lo >= cases_hi or cases_lo < 0):
        cases_lo = input("Select a lower bound for data to look at (0 - " + str(data_range) + "):\t")
        cases_hi = input("Select an upper bound for data to look at (0 - " + str(data_range) + "):\t")
        try:
            val = int(cases_lo)
            val1 = int(cases_hi)
        except ValueError:
            print("That's not an int!")
            cases_lo = -1
            cases_hi = -1
        cases_hi = int(cases_hi)
        cases_lo = int(cases_lo)

    while(print_ally_episode_win != '0' and print_ally_episode_win != '1'):
        print_ally_episode_win = input("Do you want to print ally win graphs every episode? (1 - 0):\t")

    while(print_enemy_episode_win != '0' and print_enemy_episode_win != '1'):
        print_enemy_episode_win = input("Do you want to print enemy win graphs every episode? (1 - 0):\t")

    while(show_win_percentage != '0' and show_win_percentage != '1'):
        show_win_percentage = input("Do you want to see win percentages for each player? (1 - 0):\t")
    
    while(show_win_timeline != '0' and show_win_timeline != '1'):
        show_win_timeline = input("Do you want to see a timeline of wins and loses? (1 - 0):\t")

    while(show_average_cases != '0' and show_average_cases != '1'):
        show_average_cases = input("Do you want to have the average cases reported? (1 - 0):\t")

    while(show_strats != '0' and show_strats != '1'):
        show_strats = input("Do you want to see the fequency of Ally's unique moves grouped by four waves? (1 - 0):\t")
    
    while(show_episodes_len != '0' and show_episodes_len != '1'):
        show_episodes_len = input("Do you want to have the episodes' length printed by decreasing length? (1 - 0):\t")

    return int(print_ally_episode_win), int(print_enemy_episode_win), int(show_win_percentage), int(show_average_cases), int(cases_lo), int(cases_hi), int(show_win_timeline), int(show_strats), int(show_episodes_len)




def store_strategy(index, strategies_current, episode_strategies):
    if (strategies_current in episode_strategies[index]):
        current_count = episode_strategies[index][strategies_current]
        episode_strategies[index][strategies_current] = current_count + 1
    else:
        episode_strategies[index].update({strategies_current : 1})




def seperate_strategy(data, episodes_length, episode_strategies, start_wave, episodes):
    strategies_current = ""
    curr_wave = 0
    x = 0
    while(curr_wave < (episodes_length[episodes])):
        if(curr_wave + start_wave == len(data)-1):
            break
        for buildings in range(4):
            strategies_current += (str(data[start_wave + curr_wave][0][buildings]))
            strategies_current += ", "
        strategies_current += "| "

        x = ((curr_wave + 1) / 4) - 1

        if(curr_wave == 3):
            store_strategy(0, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 7):
            store_strategy(1, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 11):
            store_strategy(2, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 15):
            store_strategy(3, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 19):
            store_strategy(4, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 23):
            store_strategy(5, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 27):
            store_strategy(6, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 31):
            store_strategy(7, strategies_current, episode_strategies)
            strategies_current = ""

        elif(curr_wave == 34):
            store_strategy(8, strategies_current, episode_strategies)
            strategies_current = ""


        #TODO add an else to catch excess waves not mult of 4
        curr_wave += 1

    return episode_strategies



def print_episodes_by_len(sorted_episodes_length):
    print("-----------------------------------Episode Lengths: (Episode Id, Episode Length)-----------------------------------------")
    j = 0
    while j != len(sorted_episodes_length)-13:
        print(str(sorted_episodes_length[j+0]) + "\t" + str(sorted_episodes_length[j+1]) + "\t" + str(sorted_episodes_length[j+2]) + "\t" + str(sorted_episodes_length[j+3]) + "\t" + str(sorted_episodes_length[j+4]) + "\t" + str(sorted_episodes_length[j+5]) + "\t" + str(sorted_episodes_length[j+6]) + "\t" + str(sorted_episodes_length[j+7]))
        j += 8



def print_episode_end_state_and_next_state(i,data, episodes_length, episodes):
    print("-------------------------------------------------------------------------------------------------------------------------")
    print("Episode Length:\t" + str(episodes_length[episodes]))
    print("i:\t" + str(i) + "\t\tfriendly nexus: " + str(data[i][0][a_nex]) + "\t\tenemey nexus: " + str(data[i][0][e_nex]))
    print("i+1:\t" + str(i+1) + "\t\tfriendly nexus: " + str(data[i+1][0][4]) + "\t\tenemey nexus: " + str(data[i+1][0][9]))
    print("\tmarine: " + str(data[i][0][a_mar]) + "\tvikings: " + str(data[i][0][a_vik]) + "\tcolossus: " + str(data[i][0][a_col]) + "\tpylons: " + str(data[i][0][a_pyl]) + "\tE marine: " + str(data[i][0][e_mar]) + "\tE vikings: " + str(data[i][0][e_vik]) + "\tE colossus: " + str(data[i][0][e_col]) + "\tE pylons: " + str(data[i][0][e_pyl]))
    print("\tmarine: " + str(data[i+1][0][a_mar]) + "\tvikings: " + str(data[i+1][0][a_vik]) + "\tcolossus: " + str(data[i+1][0][a_col]) + "\tpylons: " + str(data[i+1][0][a_pyl]) + "\tE marine: " + str(data[i+1][0][e_mar]) + "\tE vikings: " + str(data[i+1][0][e_vik]) + "\tE colossus: " + str(data[i+1][0][e_col]) + "\tE pylons: " + str(data[i+1][0][e_pyl]))
    print("-------------------------------------------------------------------------------------------------------------------------")




def strats_histogram(sorted_refined_strats):
    moves = []
    frequency = []
    table = []
    temp_label = []
    count = 0
    counter = 0

    for j in range(len(sorted_refined_strats)):
        for i in range(len(sorted_refined_strats[j])):
            moves.append(sorted_refined_strats[j][i][0])
            frequency.append(sorted_refined_strats[j][i][1])
            temp_label.append(i)
            count += frequency[i]
            counter += 1
        print(str(count) + "\t" + str(counter))
        count = 0
        counter = 0

    print("____________________________________________________________________________________________________")
    print("------------------ Frequency of Player 1's Unique Moves Grouped by Four Waves ----------------------")
    print("------ WAVE 0 ------|------- WAVE 1 ------|------ WAVE 2 -------|------ WAVE 3 -------|")
    for i in range(len(frequency)):
        if (i == len(sorted_refined_strats[0])):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 4 ------|------- WAVE 5 ------|------ WAVE 6 -------|------ WAVE 7 -------|")
        elif (i == (len(sorted_refined_strats[0]) + len(sorted_refined_strats[1]))):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 8 ------|------- WAVE 9 ------|------ WAVE 10 ------|------ WAVE 11 ------|")
        elif (i == (len(sorted_refined_strats[0]) + len(sorted_refined_strats[1]) + len(sorted_refined_strats[2]))):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 12 ------|------- WAVE 13 ------|------ WAVE 14 ------|------ WAVE 15 ------|")
        elif (i == (len(sorted_refined_strats[0]) + len(sorted_refined_strats[1]) + len(sorted_refined_strats[2]) + len(sorted_refined_strats[3]))):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 16 ------|------- WAVE 17 ------|------ WAVE 18 ------|------ WAVE 19 ------|")
        elif (i == (len(sorted_refined_strats[0]) + len(sorted_refined_strats[1]) + len(sorted_refined_strats[2]) + len(sorted_refined_strats[3]) + len(sorted_refined_strats[4]))):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 20 ------|------- WAVE 21 ------|------ WAVE 22 ------|------ WAVE 23 ------|")
        elif (i == (len(sorted_refined_strats[0]) + len(sorted_refined_strats[1]) + len(sorted_refined_strats[2]) + len(sorted_refined_strats[3]) + len(sorted_refined_strats[4]) + len(sorted_refined_strats[5]))):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 24 ------|------- WAVE 25 ------|------ WAVE 26 ------|------ WAVE 27 ------|")
        elif (i == (len(sorted_refined_strats[0]) + len(sorted_refined_strats[1]) + len(sorted_refined_strats[2]) + len(sorted_refined_strats[3]) + len(sorted_refined_strats[4]) + len(sorted_refined_strats[5]) + len(sorted_refined_strats[6]))):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 28 ------|------- WAVE 29 ------|------ WAVE 30 ------|------ WAVE 31 ------|")
        elif (i == (len(sorted_refined_strats[0]) + len(sorted_refined_strats[1]) + len(sorted_refined_strats[2]) + len(sorted_refined_strats[3]) + len(sorted_refined_strats[4]) + len(sorted_refined_strats[5]) + len(sorted_refined_strats[6]) + len(sorted_refined_strats[7]))):
            input("Press enter to view the next 4 waves.")
            print("------ WAVE 32 ------|------- WAVE 33 ------|------ WAVE 34 ------|")

        print(moves[i] + ":\t" + str(frequency[i]))

    fig1 = plt.figure(num=None, figsize=(10, 15), dpi=200, facecolor='w', edgecolor='k')
    plt.bar(temp_label, frequency,)
    plt.title("Frequency of Player 1's Unique Moves Grouped by Four Waves")

    plt.show()
    plt.close()



def average_case_graph(sum_ally_units_win, ally_wins, sum_enemy_units_win, enemy_wins):
    labels = 'Marines', 'Vikings', 'Colossus', 'Pylons'
    #create averages
    if(ally_wins != 0):
        sizes_ally_winning = [sum_ally_units_win[a_mar]/ally_wins, sum_ally_units_win[a_vik]/ally_wins, sum_ally_units_win[a_col]/ally_wins, sum_ally_units_win[a_pyl]/ally_wins]
        sizes_enemy_losing = [sum_ally_units_win[e_mar]/ally_wins, sum_ally_units_win[e_vik]/ally_wins, sum_ally_units_win[e_col]/ally_wins, sum_ally_units_win[e_pyl]/ally_wins]
    if(enemy_wins != 0):
        sizes_ally_losing = [sum_enemy_units_win[a_mar]/enemy_wins, sum_enemy_units_win[a_vik]/enemy_wins, sum_enemy_units_win[a_col]/enemy_wins, sum_enemy_units_win[a_pyl]/enemy_wins]
        sizes_enemy_winning = [sum_enemy_units_win[e_mar]/enemy_wins, sum_enemy_units_win[e_vik]/enemy_wins, sum_enemy_units_win[e_col]/enemy_wins, sum_enemy_units_win[e_pyl]/enemy_wins]

    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    if (ally_wins != 0):
        fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        ally_winner = fig1.add_axes([.1, .5, .35, .35], aspect=1)
        ally_winner.pie(sizes_ally_winning, labels=labels, colors=colors,
        autopct=make_autopct(sizes_ally_winning), shadow=True, startangle=140)
        plt.title('Ally Average (Winner)\nNexus Health (' + str(int(sum_ally_units_win[4]/ally_wins)) + ')')

    if (enemy_wins != 0):
        ally_loser = fig1.add_axes([.1, .1, .35, .35], aspect=1)
        ally_loser.pie(sizes_ally_losing, labels=labels, colors=colors,
        autopct=make_autopct(sizes_ally_losing), shadow=True, startangle=140)
        plt.title('Ally Average (Losing)\nNexus Health (' + str(int(sum_enemy_units_win[4]/enemy_wins)) + ')')

        enemy_winning = fig1.add_axes([.4, .5, .35, .35], aspect=1)
        enemy_winning.pie(sizes_enemy_winning, labels=labels, colors=colors,
        autopct=make_autopct(sizes_enemy_winning), shadow=True, startangle=140)
        plt.title('Enemy Average (Winner)\nNexus Health (' + str(int(sum_enemy_units_win[9]/enemy_wins)) + ')')

    if (ally_wins != 0):
        enemy_losing = fig1.add_axes([.4, .1, .35, .35], aspect=1)
        enemy_losing.pie(sizes_enemy_losing, labels=labels, colors=colors,
        autopct=make_autopct(sizes_enemy_losing), shadow=True, startangle=140)
        plt.title('Enemy Average (Losing)\nNexus Health (' + str(int(sum_ally_units_win[9]/ally_wins)) + ')')

    fig1.suptitle('Average End State of Games by Winning Player and Losing Player', fontsize=16)

    plt.show()
    plt.close()





def win_total_line_graph(win_total_timeline):
    x_less = []
    x_more = []
    y_less = []
    y_more = []
    zero = []
    x = []
    for i in range(len(win_total_timeline)):
        zero.append(0)
        x.append(i)

    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(x, win_total_timeline, color='b', linewidth=0.5)
    plt.plot(x, zero, color='r', linewidth=0.5)
    plt.title('Ally Win-Loss Sequence\n(Win +1) (Loss -1)') 


    plt.show()




def win_percentage_graph(ally_wins, enemy_wins):
    labels = 'Ally', 'Enemy'
    sizes = [ally_wins, enemy_wins]
    colors = ['blue', 'red']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ally = fig1.add_axes([.1, .1, .7, .7], aspect=1)
    ally.pie(sizes, labels=labels, colors=colors,
    autopct=make_autopct(sizes), shadow=True, startangle=140)
    plt.title('Ally Wins vs. Enemy Wins') 


    plt.show()
    plt.close()





def ally_graph(i,data):

    labels = 'Marines', 'Vikings', 'Colossus', 'Pylons'
    sizes_ally = [data[i][0][0], data[i][0][1], data[i][0][2], data[i][0][3]]
    sizes_enemy = [data[i][0][5], data[i][0][6], data[i][0][7], data[i][0][8]]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ally = fig1.add_axes([-.1, .1, .7, .7], aspect=1)
    ally.pie(sizes_ally, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_ally), shadow=True, startangle=140)
    plt.title('Ally (Winner)')
    enemy = fig1.add_axes([.4, .1, .7, .7], aspect=1)
    enemy.pie(sizes_enemy, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_enemy), shadow=True, startangle=140)
    plt.title('Enemy (Loser)')

    plt.show()
    plt.close()




def enemy_graph(i,data):

    labels = 'Marines', 'Vikings', 'Colossus', 'Pylons'
    sizes_ally = [data[i][0][0], data[i][0][1], data[i][0][2], data[i][0][3]]
    sizes_enemy = [data[i][0][5], data[i][0][6], data[i][0][7], data[i][0][8]]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

    enemy = fig1.add_axes([.4, .1, .7, .7], aspect=1)
    enemy.pie(sizes_enemy, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_enemy), shadow=True, startangle=140)
    plt.title('Enemy (Winner)')

    ally = fig1.add_axes([-.1, .1, .7, .7], aspect=1)
    ally.pie(sizes_ally, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_ally), shadow=True, startangle=140)
    plt.title('Ally (Loser)')

    plt.show()
    plt.close()



if __name__ == "__main__":
    main()
