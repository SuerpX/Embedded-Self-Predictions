import argparse
import os
from importlib import import_module
from abp.configs import NetworkConfig, ReinforceConfig, EvaluationConfig
from abp.examples.pysc2.tug_of_war.device_handler import get_default_device, to_device, DeviceDataLoader

import torch
import numpy as np 
import torch.nn as nn

from abp.adaptives import TransAdaptive
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

def pre_process(data, output_indexes, output_shape):
    np_data = np.array(data)
    data_input = np.stack(np_data[:,0])
    data_output = np.stack(np_data[:,1])

    nexus_idx = output_indexes
    data_output = data_output[:,nexus_idx]

    data_x = normalize(data_input, output_indexes, output_shape)
    data_y = normalize(data_output, output_indexes, output_shape)

    tensor_x = torch.stack([torch.Tensor(i) for i in data_x])
    tensor_y = torch.stack([torch.Tensor(i) for i in data_y])

    tensor_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    return tensor_dataset

def normalize(np_data, output_indexes, output_shape):
    norm_vector_input = np.array([700, 50, 40, 20, 50, 40, 20, 3,
                                    50, 40, 20, 50, 40, 20, 3,
                                    50, 40, 20, 50, 40, 20, 
                                    50, 40, 20, 50, 40, 20,
                                    2000, 2000, 2000, 2000, 40])

    norm_vector_output = norm_vector_input[output_indexes]

    if len(np_data[0]) == output_shape:
        return np_data / norm_vector_output
    else:
        return np_data / norm_vector_input

def calculate_baseline(data, val_indices):
    test_data = [data[i] for i in val_indices]
    val_set = np.array(test_data)

    baseline = np.stack(val_set[:, 0])
    idx = [4, 9]
    baseline_hp = baseline[:, idx]

    bl_next_state_reward = np.stack(val_set[:, 1])

    mse_baseline = ((baseline_hp - bl_next_state_reward)**2).mean(axis=None)
    print(mse_baseline)
    return mse_baseline

def split_data(dataset, val_pct):
    # Determine size of validation set
    n_val = int(val_pct*dataset)
    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(dataset)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]

def load_data(tensor_dataset, indices, batch_size):
    sampler = SubsetRandomSampler(indices)
    data_loaded = DataLoader(tensor_dataset, batch_size, sampler=sampler)
    return data_loaded

def run_task(evaluation_config, network_config, reinforce_config):

    trans_model = TransAdaptive(name= "TugOfWar2lNexusHealth",
                                network_config=network_config,
                                reinforce_config = reinforce_config)

    np.set_printoptions(suppress=True)

    data = torch.load('test_random_vs_random_2l.pt')

    output_indexes = [27,28,29,30]

    tensor_dataset = pre_process(data, output_indexes, network_config.output_shape)

    train_indices, val_indices = split_data(len(data), val_pct=0.1)

    # mse_baseline = calculate_baseline(data, val_indices)

    train_dl = load_data(tensor_dataset, train_indices, reinforce_config.batch_size)
    valid_dl = load_data(tensor_dataset, val_indices, reinforce_config.batch_size)

    device = get_default_device()

    # trans_model = nn.DataParallel(trans_model.trans_model.model, device_ids=[0,1,2])

    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device) 

    to_device(trans_model.trans_model.model, device)

    optimizer = Adam(trans_model.trans_model.model.parameters(), lr = 0.0001)
    loss_fn = nn.MSELoss(reduction='mean')

    losses, metrics = trans_model.train(10000, loss_fn, train_dl, valid_dl, accuracy, optimizer, "nexus-HP-transition-model-report/test_loss_2l.txt")

def accuracy(outputs, ground_true):
    preds = torch.sum( torch.eq( outputs, ground_true), dim=1 )
    return torch.sum(preds).item() / len(preds)

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--folder',
        help='The folder containing the config files',
        required=True
    )

    parser.add_argument(
        '--eval',
        help="Run only evaluation task",
        dest='eval',
        action="store_true"
    )

    parser.add_argument(
        '-r', '--render',
        help="Render task",
        dest='render',
        action="store_true"
    )

    args = parser.parse_args()
    return args

def load_yaml(folder, config_type, name):
    config_path = os.path.join(folder, name)
    config = config_type.load_from_yaml(config_path)
    return config

def main():

    args = create_parser()

    evaluation_config = load_yaml(args.folder, EvaluationConfig, "evaluation.yml")
    network_config =    load_yaml(args.folder, NetworkConfig, "network.yml")
    reinforce_config =  load_yaml(args.folder, ReinforceConfig, "reinforce.yml")

    if args.eval:
        evaluation_config.training_episodes = 0
        network_config.restore_network = True

    if args.render:
        evaluation_config.render = True

    run_task(evaluation_config, network_config, reinforce_config)

    return 0


if __name__ == "__main__":
    main()