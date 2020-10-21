import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
#         module.bias.data.fill_(0.01)
        
class _linear_model(nn.Module):
    """ Model for DQN """

    def __init__(self, input_len, ouput_len):
        super(_linear_model, self).__init__()
        
        self.fc1 = nn.Sequential(
            torch.nn.Linear(input_len, ouput_len, bias = False),
        )
        self.fc1.apply(weights_initialize)
        
        
    def forward(self, input):
        return self.fc1(input)

    
class linear_model():
    def __init__(self, input_len, ouput_len, learning_rate = 0.0001):
        self.model = _linear_model(input_len, ouput_len)
        
        if use_cuda:
            print("Using GPU")
            self.model = self.model.cuda()
        else:
            print("Using CPU")
#         self.model = nn.DataParallel(self.model)
        self.optimizer = Adam(self.model.parameters(), lr = learning_rate)
#         self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        
    def predict(self, input):
        input = FloatTensor(input).unsqueeze(0)
        q_value = self.model(input)

        return q_value

    def predict_batch(self, input):
        input = FloatTensor(input)
        q_values = self.model(input)
        
        return q_values

    def fit(self, q_values, target_q_values):
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        
        return loss.item()
        
    def replace(self, dest):
        self.model.load_state_dict(dest.model.state_dict())
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        
    def replace_soft(self, dest, tau = 1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, eval_param in zip(self.model.parameters(), dest.model.parameters()):
            target_param.data.copy_(tau*eval_param.data + (1.0-tau)*target_param.data)
