import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, data_dim, dim_q):
        super(SelfAttention, self).__init__()
        self._layers = []

        #Applies an affine linear transformation to the incoming data
        # `y = xA^T + b`.
        # takes an input vector of size data_dim and outputs a
        # vector of size dim_q by multiplying it with
        # a weight matrix and adding a bias
        # input shape x is (..., data_dim)
        # output shape will be (..., dim_q)
        # weight matrix = (data_dim, dim_q)
        # Biad vector shape = (dim_q,)
        self._fc_q = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_q)
        self._fc_k = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_k)

    def forward(self, input_data):
        #Expect input_data to be of shape (b,t,k).
        b,t,k = input_data.size()

        #Linear transforms.
        queries = self._fc_q(input=input_data) #(b,t,q)
        keys = self._fc_k(input=input_data) #(b,t,q)

        # Attention matrix
        dot = torch.bmm(queries, keys.transpose(1,2)) # (b, q,t)
        scaled_dot = torch.div(dot, torch.sqrt(torch.tensor(k).float()))
        return scaled_dot
    
    @property
    def layers(self):
        return self._layers
    
    