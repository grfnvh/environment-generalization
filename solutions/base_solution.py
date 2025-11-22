import abc
import numpy as np
import os
import solutions.abc_solution as abc_solution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import solutions.SelfAttention as SelfAttention

class BaseSolution(abc_solution.BaseSolution):
    """Base class for all Torch solutions."""

    def __init__(self):
        self._layers =[]

    def get_output(self, inputs, update_filter=False):
        torch.set_num_threads(1)#Sets the number of threads used for intraop parallelism on CPU
        with torch.no_grad(): #context manager that disables gradient calc
            return self._get_output(inputs, update_filter)
        
    @abc.abstractmethod
    def _get_output(self, inputs, update_filter):
        raise NotImplementedError()
    
    def get_params(self):
        params = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)
    
    def set_params(self, params):
        offset = 0
        for i,layer in enumerate(self.layers):
            weights_to_set = {}
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                weight = weight_dict[k].numpy()
                weight_size = weight.size
                #creates a Tensor from a numpy.ndarray that share the same memory
                weights_to_set[k] = torch.from_numpy(
                    params[offset:(offset + weight_size)].reshape(weight.shape)
                )
                offset += weight_size
            self._layers[i].load_state_dict(state_dict=weights_to_set)

    def get_params_from_layer(self, layer_index):
        params=[]
        layer = self._layers[layer_index]
        #state_dict is a python dictionary object that maps each layer to its 
        #parameter tensor(only layers with learnable parameters and registered buffers)
        weight_dict = layer.state_dict()
        #ravel is used to FLATTEN the multi-dimensional array - 
        # all elements are arranged sequentially in a 1d array
        for k in sorted(weight_dict.keys()):
            params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)
    
    def set_params_to_layer(self, params, layer_index):
        weights_to_set = {}
        weight_dict = self._layers[layer_index].state_dict()
        offset = 0
        for k in sorted(weight_dict.keys()):
            weight = weight_dict[k].numpy()
            weight_size = weight.size
            weights_to_set[k] = torch.from_numpy(
                params[offset:(offset + weight_size)].reshape(weight.shape)
            )
            offset += weight.size 
            self._layers[layer_index].load_state_dict(state_dict=weights_to_set)
    
    def get_num_params_per_layer(self):
        num_params_per_layer = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            num_params = 0
            for k in sorted(weight_dict.keys()):
                weights = weight_dict[k].numpy()
                num_params += weights.size
            num_params_per_layer.append(num_params)
        return num_params_per_layer
    
    def _save_to_file(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def save(self, log_dir, iter_count, best_so_far):
        filename = os.path.join(log_dir, 'model_{}.npz'.format(iter_count))
        self._save_to_file(filename=filename)
        if best_so_far:
            filename = os.path.join(log_dir, 'best_models.npz')
            self._save_to_file(filename=filename)
    
    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def reset(self):
        raise NotImplementedError()
    
    @property
    def layers(self):
        return self._layers
