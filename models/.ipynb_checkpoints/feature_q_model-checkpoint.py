import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
#         module.bias.data.fill_(0.01)
        
# class _feature_q_model(nn.Module):
#     """ Model for DQN """

#     def __init__(self, input_len, ouput_len):
#         super(_feature_q_model, self).__init__()
#         self.fc1 = nn.Sequential(
#             torch.nn.Linear(input_len, 512, bias = False)
#             , nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             torch.nn.Linear(512, 256, bias = False)
#             , nn.ReLU()
#         )
#         self.fc3 = nn.Sequential(
#             torch.nn.Linear(256, ouput_len, bias = False),
# #             nn.ReLU()
#         )
#     def forward(self, x):
# #         return self.linear_com(input)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return self.fc3(x)
class _q_model(nn.Module):
    """ Model for DQN """

    def __init__(self, input_len, ouput_len):
        super(_q_model, self).__init__()
        self.fc1 = nn.Sequential(
            torch.nn.Linear(input_len, 512, bias = True)
            , nn.ReLU()
#             , nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            torch.nn.Linear(512, 256, bias = True)
            , nn.ReLU()
        )
        self.output = nn.Sequential(
            torch.nn.Linear(256, ouput_len, bias = True),
        )
    def forward(self, x):
#         return self.linear_com(input)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output(x)
            
class _feature_model(nn.Module):
    """ Model for DQN """

    def __init__(self, input_len, ouput_len):
        super(_feature_model, self).__init__()
        self.fc1 = nn.Sequential(
            torch.nn.Linear(input_len, 512, bias = True)
            , nn.ReLU()
#             , nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            torch.nn.Linear(512, 256, bias = True)
            , nn.ReLU()
        )
        self.output = nn.Sequential(
            torch.nn.Linear(256, ouput_len, bias = True),
        )
    def forward(self, x):
#         return self.linear_com(input)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output(x)
            
class feature_q_model():
    def __init__(self, input_len, feature_len, output_len, learning_rate = 0.0001):
        self.feautre_model = _feature_model(input_len, feature_len)
        self.q_model = _q_model(feature_len, output_len)
            
#         learning_rate = 0.0001
        if use_cuda:
            print("Using GPU")
            self.feautre_model = self.feautre_model.cuda()
            self.q_model = self.q_model.cuda()
        else:
            print("Using CPU")
        params = list(self.feautre_model.parameters()) + list(self.q_model.parameters())
        self.optimizer_com = SGD(self.q_model.parameters(), lr = learning_rate)
        self.optimizer_feature = Adam(self.feautre_model.parameters(), lr = learning_rate)
        
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        
    def predict(self, input):
        input = FloatTensor(input).unsqueeze(0)
        feature_vector = self.feautre_model(input)
        q_value = self.q_model(feature_vector)
            
        return feature_vector, q_value

    def predict_batch(self, input):
        input = FloatTensor(input)
        feature_vectors = self.feautre_model(input)
        q_values = self.q_model(feature_vectors)
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
    
    def save(self, path):
        torch.save(self.feautre_model.state_dict(), path + "feature")
        torch.save(self.q_model.state_dict(), path + "q_model")
            
    def load(self, path):
        self.feautre_model.load_state_dict(torch.load(path + "feature"))
        self.q_model.load_state_dict(torch.load(path + "q_model"))
        
       
    def eval_mode(self):
        self.feautre_model.eval()
        self.q_model.eval()
    
    def replace_soft(self, dest, tau = 1e-3):
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
            
            