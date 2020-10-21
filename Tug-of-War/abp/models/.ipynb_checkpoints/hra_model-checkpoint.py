import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tensorboardX import SummaryWriter

from .model import Model
from abp.utils import clear_summary_path
from abp.utils.models import generate_layers

logger = logging.getLogger('root')


def weights_initialize(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        module.bias.data.fill_(0.01)


class DecomposedModel(nn.Module):
    def __init__(self, layers, input_shape, output_shape):
        super(DecomposedModel, self).__init__()

        layer_modules, input_shape = generate_layers(input_shape, layers)
        layer_modules["OutputLayer"] = nn.Linear(int(np.prod(input_shape)), output_shape)

        self.layers = nn.Sequential(layer_modules)
        self.layers.apply(weights_initialize)

    def forward(self, input):
        x = input
        for layer in self.layers:
            if type(layer) == nn.Linear:
                x = x.view(-1, int(np.prod(x.shape[1:])))
            x = layer(x)
        return x


class _HRAModel(nn.Module):
    """ HRAModel """

    def __init__(self, network_config):
        super(_HRAModel, self).__init__()
        self.network_config = network_config
        modules = []
        for network_i, network in enumerate(network_config.networks):
            model = DecomposedModel(
                network["layers"], network_config.input_shape, network_config.output_shape)
            modules.append(model)
        self.reward_models = nn.ModuleList(modules)
        self.combined = False  # If set to true will always return combined Q values

    def forward(self, input):
        q_values = []
        for reward_model in self.reward_models:
            q_value = reward_model(input)
            q_values.append(q_value)
        output = torch.stack(q_values)
        if self.combined:
            output = output.sum(0)
        return output


class HRAModel(Model):
    """Neural Network with the HRA architecture  """

    def __init__(self, name, network_config, use_cuda, restore=True):
        self.network_config = network_config
        self.name = name

        summaries_path = self.network_config.summaries_path + "/" + self.name

        model = _HRAModel(network_config)
        if use_cuda:
            logger.info("Network %s is using cuda " % self.name)
            model = model.cuda()

        Model.__init__(self, model, name, network_config, restore)
        logger.info("Created network for %s " % self.name)

        self.optimizer = Adam(self.model.parameters(), lr=self.network_config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        if not network_config.restore_network:
            clear_summary_path(summaries_path)
            self.summary = SummaryWriter(log_dir=summaries_path)
            dummy_input = torch.rand(network_config.input_shape).unsqueeze(0)
            if use_cuda:
                dummy_input = dummy_input.cuda()
            self.summary.add_graph(self.model, dummy_input)
        else:
            self.summary = SummaryWriter(log_dir=summaries_path)

    def get_model_for(self, reward_idx):
        return self.model.reward_models[reward_idx]

    def clear_weights(self, reward_idx):
        for network_i, network in enumerate(self.network_config.networks):
            model = self.model.reward_models[network_i]

            for layer_id in range(len(network['layers'])):
                getattr(model.layers, 'Layer_{}'.format(layer_id)).apply(self.weights_init)

            getattr(model.layers, 'OutputLayer'.format(layer_id)).apply(self.weights_init)

    def weights_summary(self, steps):
        for network_i, network in enumerate(self.network_config.networks):
            model = self.model.reward_models[network_i]
            for i in range(len(network['layers'])):
                layer_name = "Layer_%d" % i
                weight_name = 'Sub Network Type:{}/layer{}/weights'.format(network_i, i)
                bias_name = 'Sub Network Type:{}/layer{}/bias'.format(network_i, i)
                weight = getattr(model.layers, layer_name).weight.clone().data
                bias = getattr(model.layers, layer_name).bias.clone().data
                self.summary.add_histogram(tag=weight_name, values=weight, global_step=steps)
                self.summary.add_histogram(tag=bias_name, values=bias, global_step=steps)

            weight_name = 'Sub Network Type:{}/Output Layer/weights'.format(network_i, i)
            bias_name = 'Sub Network Type:{}/Output Layer/bias'.format(network_i, i)

            weight = getattr(model.layers, "OutputLayer").weight.clone().data
            bias = getattr(model.layers, "OutputLayer").bias.clone().data

            self.summary.add_histogram(tag=weight_name, values=weight, global_step=steps)
            self.summary.add_histogram(tag=bias_name, values=bias, global_step=steps)

    def top_layer(self, network_idx):
        return getattr(self.model.reward_models[network_idx].layers, 'OutputLayer')

    def weights_init(self, m):
        classname = m.__class__.__name__
        if type(m) == nn.Linear:
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.0, 0.0)
            m.bias.data.fill_(0)

    def fit(self, q_values, target_q_values, steps):
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary.add_scalar(tag="%s/Loss" % (self.name),
                                scalar_value=float(loss),
                                global_step=steps)

    def predict(self, input, steps, learning, illegal_actions = None):
        q_values = self.model(input).squeeze(1)
        if illegal_actions is not None:
            q_values[:, illegal_actions] = np.NINF
        combined_q_values = torch.sum(q_values, 0)
        values, q_actions = torch.max(combined_q_values, 0)

        if steps % self.network_config.summaries_step == 0 and learning:
            logger.debug("Adding network summaries!")
            self.weights_summary(steps)
            self.summary.add_histogram(tag="%s/Q values" % (self.name),
                                       values=combined_q_values.clone().cpu().data.numpy(),
                                       global_step=steps)
            for network_i, network in enumerate(self.network_config.networks):
                name = 'Sub Network Type:{}/Q_value'.format(network_i)
                self.summary.add_histogram(tag=name,
                                           values=q_values[network_i].clone().cpu().data.numpy(),
                                           global_step=steps)

        return q_actions.item(), q_values, combined_q_values

    def predict_batch(self, input, illegal_actions = None):
        q_values = self.model(input)
        if illegal_actions is not None:
            for i in range(input.shape[0]):
                q_values[i, :, illegal_actions[i]] = np.NINF
        combined_q_values = torch.sum(q_values, 0)
        values, q_actions = torch.max(combined_q_values, 1)
        return q_actions, q_values, combined_q_values
