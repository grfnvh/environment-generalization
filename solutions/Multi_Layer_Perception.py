import abc
import gin
import numpy as np
import solutions.base_solution as BaseSolution
import solutions.FullyConnectedLayer as FCStack
import solutions.LongShortTermMemory as LSTMStack
import os
import solutions.abc_solution as abc_solution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

@gin.configurable
class MLPSolution(BaseSolution):
    """Multi-layer perception"""

    def __init__(self, input_dim, num_hiddens, activation, output_dim, output_activation, use_lstm, l2_coefficient):
        super(MLPSolution, self).__init__()
        self.use_lstm = use_lstm
        self._output_dim = output_dim
        self._output_activation = output_activation
        if 'roulette' in self._output_activation:
            assert self._output_dim == 1
            self._n_grid = int(self._output_activation.split('_')[-1])
            self._theta_per_grid = 2 * np.pi / self._n_grid
        self._l2_coefficient = abs(l2_coefficient)
        if self._use_lstm:
            self._fc_stack = LSTMStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
            )
        else:
            self._fc_stack = FCStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
                activation=activation,
            )
        self._layers = self._fc_stack.layers
        print('Number of parameters: {}'.format(
            self.get_num_params_per_layer()))

    def _get_output(self, inputs, update_filter=False):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).float()
        fc_output = self._fc_stack(inputs)

        if self._output_activation == 'tanh':
            output = torch.tanh(fc_output).squeeze().numpy()
        elif self._output_activation == 'softmax':
            output = F.softmax(fc_output, dim=-1).squeeze().numpy()
        else:
            output = fc_output.squeeze().numpy()

        return output

    def reset(self):
        if hasattr(self._fc_stack, 'reset'):
            self._fc_stack.reset()
            print('hidden reset.')
