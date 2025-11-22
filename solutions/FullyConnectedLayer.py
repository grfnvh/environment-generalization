import abc
import numpy as np
import os
import solutions.abc_solution as abc_solution
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCStack(nn.Module):
    """Fully connected layers."""

    def __init__(self, input_dim, num_units, activation, output_dim):
        super(FCStack, self).__init__()
        self._activation = activation
        self._layers = []
        dim_in = input_dim
        for i, n in enumerate(num_units):
            layer = nn.Linear(dim_in, n)
            # layer.weight.data.fill_(0.0)
            # layer.bias.data.fill_(0.0)
            self._layers.append(layer)
            setattr(self, '_fc{}'.format(i + 1), layer)
            dim_in = n
        output_layer = nn.Linear(dim_in, output_dim)
        # output_layer.weight.data.fill_(0.0)
        # output_layer.bias.data.fill_(0.0)
        self._layers.append(output_layer)

    @property
    def layers(self):
        return self._layers

    def forward(self, input_data):
        x_input = input_data
        for layer in self._layers[:-1]:
            x_output = layer(x_input)
            if self._activation == 'tanh':
                x_input = torch.tanh(x_output)
            elif self._activation == 'elu':
                x_input = F.elu(x_output)
            else:
                x_input = F.relu(x_output)
        x_output = self._layers[-1](x_input)
        return x_output
