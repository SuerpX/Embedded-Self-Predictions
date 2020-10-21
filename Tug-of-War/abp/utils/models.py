from collections import OrderedDict

import numpy as np
import torch.nn as nn


def generate_layers(input_shape, layers):
    """ Maps layers in network config to pytorch layers """
    layer_modules = OrderedDict()

    for i, layer in enumerate(layers):
        layer_name = layer["name"] if "name" in layer else "Layer_%d" % i
        layer_type = layer["type"] if "type" in layer else "FC"

        if layer_type == "FC":
            layer_modules[layer_name] = nn.Linear(int(np.prod(input_shape)), layer["neurons"])
            input_shape = [layer["neurons"]]
            layer_modules[layer_name + "relu"] = nn.ReLU()
            
        elif layer_type == "FCwithBN":
            layer_modules[layer_name] = nn.Linear(int(np.prod(input_shape)), layer["neurons"])
            input_shape = [layer["neurons"]]
            layer_modules[layer_name + "BN"] = nn.BatchNorm1d(layer["size"])
            layer_modules[layer_name + "relu"] = nn.ReLU()
            
        elif layer_type == "CNN":
            kernel_size = layer["kernel_size"]
            padding = layer["padding"]
            stride = layer["stride"]
            layer_modules[layer_name] = nn.Conv2d(layer["in_channels"],
                                                  layer["out_channels"],
                                                  kernel_size,
                                                  stride,
                                                  padding)
            width = ((input_shape[1] - kernel_size + 2 * padding) / stride) + 1
            height = ((input_shape[2] - kernel_size + 2 * padding) / stride) + 1
            input_shape = [layer["out_channels"], width, height]
            layer_modules[layer_name + "relu"] = nn.ReLU()

        elif layer_type == "BatchNorm2d":
            layer_modules[layer_name] = nn.BatchNorm2d(layer["size"])
            layer_modules[layer_name + "relu"] = nn.ReLU()
            
        elif layer_type == "BatchNorm1d":
            layer_modules[layer_name] = nn.BatchNorm1d(layer["size"])
            
        elif layer_type == "Dropout":
            layer_modules[layer_name] = nn.Dropout(p = layer["prob"])

        elif layer_type == "MaxPool2d":
            stride = layer["stride"]
            kernel_size = layer["kernel_size"]
            layer_modules[layer_name] = nn.MaxPool2d(kernel_size, stride)
            width = ((input_shape[1] - kernel_size) / stride) + 1
            height = ((input_shape[2] - kernel_size) / stride) + 1
            input_shape = [input_shape[0], width, height]
        input_shape = np.floor(input_shape)

    return layer_modules, input_shape
