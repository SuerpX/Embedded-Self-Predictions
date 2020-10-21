import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .model import Model
from abp.utils import clear_summary_path
from abp.utils.models import generate_layers
from tensorboardX import SummaryWriter

logger = logging.getLogger('root')


def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        module.bias.data.fill_(0.01)


class _DQNModel(nn.Module):
    """ Model for DQN """

    def __init__(self, network_config):
        super(_DQNModel, self).__init__()
        
        self.output_shape = network_config.output_shape
        layer_modules, input_shape = generate_layers(network_config.input_shape,
                                                     network_config.layers)

        layer_modules["OutputLayer"] = nn.Linear(int(np.prod(input_shape)),
                                                 network_config.output_shape)

        self.softmax_func = nn.Softmax()

        self.layers = nn.Sequential(layer_modules)
        self.layers.apply(weights_initialize)

    def forward(self, input):
        x = input
        for layer in self.layers:
            if type(layer) == nn.Linear:
                x = x.view(-1, int(np.prod(x.shape[1:])))
            x = layer(x)
        
        if self.output_shape > 1:
            x = self.softmax_func(x)
        return x


class DQNModel(Model):

    def __init__(self, name, network_config, use_cuda, restore=True, learning_rate=0.001):
        self.name = name
        model = _DQNModel(network_config)
        model = nn.DataParallel(model)
        self.use_cuda = use_cuda

        if use_cuda:
            logger.info("Network %s is using cuda " % self.name)
            model = model.cuda()

        super(DQNModel, self).__init__(model, name, network_config, restore)
        self.network_config = network_config
        self.optimizer = Adam(self.model.parameters(), lr=self.network_config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.is_SmoothL1Loss = True
#         print("loss func: SmoothL1Loss")
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.is_SmoothL1Loss = False
#         print("loss func: CrossEntropyLoss")
        summaries_path = self.network_config.summaries_path + "/" + self.name

        if not network_config.restore_network:
            clear_summary_path(summaries_path)
            self.summary = SummaryWriter(log_dir=summaries_path)
        else:
            self.summary = SummaryWriter(log_dir=summaries_path)

        logger.info("Created network for %s " % self.name)

    def weights_summary(self, steps):
        model = self.model.module

        for i, layer in enumerate(self.network_config.layers):
            if layer["type"] in ["BatchNorm2d", "MaxPool2d"]:
                continue

            layer_name = layer["name"] if "name" in layer else "Layer_%d" % i
            weight_name = '{}/{}/weights'.format(self.name, layer_name)
            bias_name = '{}/{}/bias'.format(self.name, layer_name)

            module = getattr(model.layers, layer_name)

            if type(module) == nn.DataParallel:
                module = module.module

            weight = module.weight.clone().data
            bias = module.bias.clone().data
            self.summary.add_histogram(tag=weight_name, values=weight, global_step=steps)
            self.summary.add_histogram(tag=bias_name, values=bias, global_step=steps)

        weight_name = '{}/Output Layer/weights'.format(self.name, i)
        bias_name = '{}/Output Layer/bias'.format(self.name, i)

        module = getattr(model.layers, "OutputLayer")
        weight = module.weight.clone().data
        bias = module.bias.clone().data

        self.summary.add_histogram(tag=weight_name, values=weight, global_step=steps)
        self.summary.add_histogram(tag=bias_name, values=bias, global_step=steps)

    def predict(self, input, steps):
        q_values = self.model(input).squeeze(1)
        
        return q_values

    def predict_batch(self, input):
        if self.use_cuda:
            input = input.cuda()
        q_values = self.model(input)
#         print(q_values)
#         values, q_actions = q_values.max(1)
        return q_values

    def fit(self, q_values, target_q_values, steps):
        if not self.is_SmoothL1Loss:
            target_q_values = target_q_values.max(1)[1]
    
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary.add_scalar(tag="%s/Loss" % (self.name),
                                scalar_value=float(loss),
                                global_step=steps)
