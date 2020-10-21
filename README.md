ESP agent Traning, Evaluation, Producing Explanation
============================

## Installation
Install dependencies
```bash
    sudo apt install libopenmpi-dev ffmpeg
```

Install pip packages (Anaconda recommended)
```bash
    pip install -r requirements.txt
```

## Prerequisite of Tug-of-War env
1. Download StarCraft 2
https://github.com/Blizzard/s2client-proto#downloads (4.7.1 recommended, only for Linux)

2. The StarCraft need to be installed at the ~/ path.

3. Download the anonymous Tug-of-war env in a dir, and rename it "sc2env"
https://anonymous.4open.science/r/0b982886-c6ac-440e-b6d7-67e4c99cb617/

4. run the following command:
```bash
    1. pip install -r requirements_ToW.txt
    2. export PYTHONPATH=path/to/dir/sc2env;path/to/dir/sc2env/sc2env/xai_replay/ui/viz/py_backend/proto;
    3. cd path/to/dir/sc2env
    4. git checkout new-sensors
    
    (back to the ESP_code)
    5. cd Tug-of-War/
```
## Traning new agent

a. Cart Pole
```bash
    python3 CP_ESP.py train
```
b. Lunar Lander

```bash
    python3 LL_ESP.py train
```
c. Tug-of-War 17f

```bash
    python3 -m sc2env.play_tug_of_war -f tasks/tug_of_war/gqf/v10_sepcific_new/ -tk task_gqf_2p_2l_grid
```

d. Tug-of-War 131f

```bash
    python3 -m sc2env.play_tug_of_war -f tasks/tug_of_war/gqf/GVFs_all_1_sepcific_new/ -tk task_gqf_2p_2l_grid
```
## Evaluation of trained model

a. Cart Pole
```bash
    python3 CP_ESP.py eval
```
b. Lunar Lander

```bash
    python3 LL_ESP.py eval
```

c. Tug-of-War 17f

```bash
    python3 -m sc2env.play_tug_of_war -f tasks/tug_of_war/gqf/v10_sepcific_eval/ -tk task_gqf_2p_2l_grid
```

d. Tug-of-War 131f

```bash
    python3 -m sc2env.play_tug_of_war -f tasks/tug_of_war/gqf/GVFs_all_1_sepcific_eval/ -tk task_gqf_2p_2l_grid
```
## Producing Explanation with trained model
a. Cart Pole
```bash
    python3 CP_ESP.py exp
```
b. Lunar Lander

```bash
    python3 LL_ESP.py exp
```

c. Tug-of-War 17f

```bash
    python3 -m sc2env.play_tug_of_war -f tasks/tug_of_war/gqf/v10_sepcific_exp/ -tk task_gqf_2p_2l_grid
```

d. Tug-of-War 131f

```bash
    python3 -m sc2env.play_tug_of_war -f tasks/tug_of_war/gqf/GVFs_all_1_sepcific_exp/ -tk task_gqf_2p_2l_grid
```

## Directory of explanation generted
a. Cart Pole
```bash
    CartPole_ESP/CartPole_ESP_exp
```
b. Lunar Lander

```bash
    LunarLander_ESP/LunarLander_ESP_exp
```

c. Tug-of-War 17f

```bash
    Tug-of-War/explanations/tug_of_war/gqf/v10_sepcific
```

d. Tug-of-War 131f

```bash
    Tug-of-War/explanations/tug_of_war/gqf/GVFs_all_1_sepcific
```
