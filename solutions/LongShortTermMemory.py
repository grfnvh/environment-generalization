import abc
import numpy as np
import os
import solutions.abc_solution as abc_solution
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMStack(nn.Module):
    """LSTM layers."""

    def __init__(self, input_dim, num_units, output_dim):
        super(LSTMStack, self).__init__()
        self._layers = []
        self._hidden_layers = len(num_units) if len(num_units) else 1
        self._hidden_size = num_units[0] if len(num_units) else output_dim
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )
        if len(num_units):
            self._lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self._hidden_size,
                num_layers=self._hidden_layers,
            )
            self._layers.append(self._lstm)
            fc = nn.Linear(
                in_features=self._hidden_size,
                out_features=output_dim,
            )
            self._layers.append(fc)
        else:
            self._lstm = nn.LSTMCell(
                input_size=input_dim,
                hidden_size=self._hidden_size,
            )
            self._layers.append(self._lstm)

    @property
    def layers(self):
        return self._layers

    def forward(self, input_data):
        x_input = input_data
        x_output, self._hidden = self._layers[0](
            x_input.view(1, 1, -1), self._hidden)
        x_output = torch.flatten(x_output, start_dim=0, end_dim=-1)
        if len(self._layers) > 1:
            x_output = self._layers[-1](x_output)
        return x_output

    def reset(self):
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )
