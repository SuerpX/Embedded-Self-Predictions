import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import os 
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
        
def ensure_directory_exits(directory_path):
    """creates directory if path doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

class _q_model(nn.Module):
    """ Model for DQN """

    def __init__(self, input_len, ouput_len):
        super(_q_model, self).__init__()
        self.fc1 = nn.Sequential(
            torch.nn.Linear(input_len, 2048, bias = True)
            , nn.ReLU()
#             , nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            torch.nn.Linear(2048, 512, bias = True)
            , nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            torch.nn.Linear(512, 256, bias = True)
            , nn.ReLU()
        )
        self.output = nn.Sequential(
            torch.nn.Linear(256, ouput_len, bias = True)
#             , nn.Sigmoid()
        )
    def forward(self, x):
#         return self.linear_com(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output(x)
            
class _feature_model(nn.Module):
    """ Model for DQN """

    def __init__(self, input_len, ouput_len):
        super(_feature_model, self).__init__()
        self.fc1 = nn.Sequential(
            torch.nn.Linear(input_len, 2048, bias = True)
            , nn.ReLU()
#             , nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            torch.nn.Linear(2048, 512, bias = True)
            , nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            torch.nn.Linear(512, 256, bias = True)
            , nn.ReLU()
        )
        self.output = nn.Sequential(
            torch.nn.Linear(256, ouput_len, bias = True),
        )
        self.softmax_func = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, version = "v0"):
#         return self.linear_com(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output(x)
        
        if version == "v1":
            x = self.softmax_func(x)
        if version == "v2":
#             print(x)
#             print(self.sigmoid(x[:, :24]).size(), self.softmax_func(x[:, 24 : 28]).size(), self.sigmoid(x[:, 28]).size())
#             x = torch.cat((self.sigmoid(x[:, :24]), self.softmax_func(x[:, 24 : 28]), self.sigmoid(x[:, 28]).view(-1, 1)), dim = 1)
            x = torch.cat((self.sigmoid(x[:, :24]), self.softmax_func(x[:, 24 : 28]), self.sigmoid(x[:, 28]).view(-1, 1)), dim = 1)
#             print(x)
#             input()
        if version in ["v6", "v7"]:
            x = torch.cat((self.softmax_func(x[:, :8]), x[:, 8:]), dim = 1)
        if version == "v8":
            x = torch.cat((x[:, :60], self.softmax_func(x[:, 60 : 64]), self.sigmoid(x[:, 64]).view(-1, 1)), dim = 1)
        if version == "v9":
            x = torch.cat((self.sigmoid(x[:, :12]), self.softmax_func(x[:, 12 : 16]), self.sigmoid(x[:, 16]).view(-1, 1)), dim = 1)
        if version == "v10":
            x = torch.cat((self.sigmoid(x[:, :12]), self.softmax_func(x[:, 12 : 16]), self.sigmoid(x[:, 16]).view(-1, 1)), dim = 1)
        if version in ["GFVs_all_1", "GVFs_all_2"]:
#             x = self.all_1_x(x)
            x = torch.cat((self.softmax_func(x[:, : 8]), x[:, 8 : -1], self.sigmoid(x[:, -1]).view(-1, 1)), dim = 1)
        if version == "v11":
            x = torch.cat((self.sigmoid(x[:, :12]), self.softmax_func(x[:, 12 : 17])), dim = 1)
#             x = self.softmax_func(x[:, : 8])
        return x
    
    def all_1_x(self, x):
        current_idx = 0
        
        x[:, current_idx : current_idx + 8] = self.softmax_func(x[:, current_idx : current_idx + 8])
        current_idx += 8
        
        # self-mineral feature idx 8, enemy-mineral idx 9. self-pylone idx 7, enemy-pylon 14
        current_idx += 2
        
        # features idx: The number of self-units will be spawned, length : 6, The number of enemy-units will be spawned, length : 6
        # state idx: The number of self-building 1-6, The number of self-building 8-13, 
        current_idx += 12
        
        # features idx: The accumulative number of each type of unit in each range from now to the end of the game: length : 30, for enemy: length : 30
        current_idx += 60
        
        # features idx: Damage of each friendly unit to each Nexus: length : 6, for enemy: length : 6
        current_idx += 12
        
        # features idx: The number of which friendly troops kills which enemy troops: length : 18, for enemy: length : 18
        current_idx += 36
        
        # features idx: prob of limitation wave: length : 1
        x[:, current_idx] = self.sigmoid(x[:, current_idx])
        current_idx += 1
        
        return x
    
class feature_q_model():
    def __init__(self, name, input_len, feature_len, output_len, network_config, learning_rate = 0.0001):
        self.name = name
        self.network_config = network_config
        self.version = network_config.version
        print(self.version)
#         if self.version == "GFVs_all_1":
#             torch.autograd.set_detect_anomaly(True)
#         print( name, input_len, feature_len, output_len)
        file_path = ensure_directory_exits(self.network_config.network_path)
        self.model_path = os.path.join(file_path, self.name + '.p')
        self.feautre_model = _feature_model(input_len, feature_len)
#         if self.version == "GVFs_all_2":
#             feature_len = 8
        self.q_model = _q_model(feature_len, output_len)
            
        if use_cuda:
            print("Using GPU")
            self.feautre_model = self.feautre_model.cuda()
            self.q_model = self.q_model.cuda()
        else:
            print("Using CPU")
#         self.model = nn.DataParallel(self.model)
        # TODO: another idea: update both networks separately.
        params = list(self.feautre_model.parameters()) + list(self.q_model.parameters())
#         print(learning_rate)
        self.optimizer = Adam(params, lr = learning_rate)
        self.optimizer_SGD = Adam(params, lr = learning_rate)
#         self.optimizer_com = Adam(self.q_model.parameters(), lr = learning_rate)
#         self.optimizer_feature = Adam(self.feautre_model.parameters(), lr = learning_rate)

        self.optimizer_com = SGD(self.q_model.parameters(), lr = learning_rate)
        self.optimizer_feature = Adam(self.feautre_model.parameters(), lr = learning_rate)
        
#         self.optimizer_com = SGD(self.q_model.parameters(), lr = 0.0001)
#         self.optimizer_feature = SGD(self.feautre_model.parameters(), lr = 0.00005)
        
        self.l1_crit = nn.L1Loss(size_average=False)
#         self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        self.restore_network()
        
    def predict(self, input):
        input = FloatTensor(input).unsqueeze(0)
        feature_vector = self.feautre_model(input, self.version)
        input_feature_vectors = feature_vector.clone()
        if self.version == "v9":
            input_feature_vectors[:, :3] = input_feature_vectors[:, :3] / input[:, 65].view(-1, 1)# / 2000
            input_feature_vectors[:, 3:6] = input_feature_vectors[:, 3:6] / input[:, 66].view(-1, 1)# / 2000
            input_feature_vectors[:, 6:9] = input_feature_vectors[:, 6:9] / input[:, 63].view(-1, 1)# / 2000
            input_feature_vectors[:, 9:12] = input_feature_vectors[:, 9:12] / input[:, 64].view(-1, 1)# / 2000
            
            input_feature_vectors[input_feature_vectors == float('inf')] = 0
            
        q_value = self.q_model(feature_vector)
            
        return feature_vector, q_value

    def predict_batch(self, input):
        input = FloatTensor(input)
        feature_vectors = self.feautre_model(input, self.version)
#         print(input[0:2])
#         input_feature_vectors = feature_vectors.clone()
#         print(input_feature_vectors[0:2])
#         input_state = input.detach()
        if self.version == "v9":
            input_feature_vectors[:, :3] = input_feature_vectors[:, :3] / input_state[:, 65].view(-1, 1)# / 2000
            input_feature_vectors[:, 3:6] = input_feature_vectors[:, 3:6] / input_state[:, 66].view(-1, 1)# / 2000
            input_feature_vectors[:, 6:9] = input_feature_vectors[:, 6:9] / input_state[:, 63].view(-1, 1)# / 2000
            input_feature_vectors[:, 9:12] = input_feature_vectors[:, 9:12] / input_state[:, 64].view(-1, 1)# / 2000
            
            input_feature_vectors[input_feature_vectors == float('inf')] = 0
            
#         if self.version == "GVFs_all_2":
# #             input_feature_vectors = feature_vectors[:, :8].clone()
#             q_values = self.q_model(feature_vectors[:, :8])
# #             print(feature_vectors[:, :8])
# #         print(input_feature_vectors[0:2])
#         else:
        q_values = self.q_model(feature_vectors)
#             print(feature_vectors)
#             print(q_values)

        return feature_vectors, q_values 


    def fit(self, q_values, target_q_values, feature_vectors, target_feature_vector):
        loss_1 = self.loss_fn(q_values, target_q_values)
        loss_2 = self.loss_fn(feature_vectors, target_feature_vector)
        
        self.optimizer_com.zero_grad()
        loss_1.backward(retain_graph=True)
        self.optimizer_com.step()
        
        self.optimizer_feature.zero_grad()
        loss_2.backward()
        self.optimizer_feature.step()

        self.steps += 1
        
#         return loss.item()

    def fit_2(self, q_values, target_q_values, feature_vectors, target_feature_vector):
        loss_1 = self.loss_fn(q_values, target_q_values)
        loss_2 = self.loss_fn(feature_vectors, target_feature_vector)
        # TODO: another idea: update both networks separately.
        loss = loss_1 + loss_2 * 1000
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1

    def replace(self, dest):
        self.feautre_model.load_state_dict(dest.feautre_model.state_dict())
        self.q_model.load_state_dict(dest.q_model.state_dict())
        
    def save_network(self, appendix = ""):
        torch.save([self.feautre_model.state_dict(), self.q_model.state_dict()], self.model_path + appendix)
#         print([self.feautre_model.state_dict(), self.q_model.state_dict()])
        
    def restore_network(self):
        if (self.network_config.restore_network and
                self.network_config.network_path):
#             print(self.model_path)
            if os.path.exists(self.model_path):
                print()
                print()
                print(self.model_path)
                print()
                print()
                model = torch.load(self.model_path)
                print("restore: {}".format(self.name))
#                 print(model)
                self.feautre_model.load_state_dict(model[0])
                self.q_model.load_state_dict(model[1])
                
#             else:
#                 print("path no exist")
        
       
    def eval_mode(self):
        self.feautre_model.eval()
        self.q_model.eval()
    
    def replace_soft(self, dest, tau = 5e-4):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, eval_param in zip(self.feautre_model.parameters(), dest.feautre_model.parameters()):
            target_param.data.copy_(tau*eval_param.data + (1.0-tau)*target_param.data)
            
        for target_param, eval_param in zip(self.q_model.parameters(), dest.q_model.parameters()):
            target_param.data.copy_(tau*eval_param.data + (1.0-tau)*target_param.data)
            
            
