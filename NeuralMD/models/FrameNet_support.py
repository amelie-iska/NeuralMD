import math
from math import pi as PI

import torch
from torch import nn

from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


def nan_to_num(vec, nan=0.0):
    idx = torch.isnan(vec)
    vec[idx] = nan
    return vec


def _normalize(vec, dist=None, dim=-1):
    if dist is None:
        normalized_vec = nan_to_num(torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))
    else:
        normalized_vec = nan_to_num(torch.div(vec, dist.unsqueeze(1)))
    return normalized_vec


def kaiming_uniform(tensor, size):
    # TODO: will update this later.
    fan = 1
    for i in range(1, len(size)):
        fan *= size[i]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def weight_initialization(layer):
    # torch.nn.init.xavier_uniform_(layer.weight)
    # torch.nn.init.kaiming_uniform_(layer.weight)
    kaiming_uniform(layer.weight.data, layer.weight.data.size())
    layer.bias.data.fill_(0.01)
    return


class MLP(nn.Module):
    def __init__(self, dim_list, batch_norm, dropout, momentum, end_with_linear=True):
        super(MLP, self).__init__()

        layers = []
        for input_dim, output_dim in zip(dim_list[:-1], dim_list[1:]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(input_dim, momentum=momentum))
            # layers.append(nn.SiLU())
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim))

            if not end_with_linear:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(output_dim, momentum=momentum))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

        self.reset_parameters()
        return

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                weight_initialization(layer) 
        return
    
    def forward(self, x):
        return self.layers(x)


class RBF_repredding_01(nn.Module):
    def __init__(self, num_radial, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_radial = num_radial
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_radial)
        betas = torch.tensor([(2 / self.num_radial * (end_value - start_value)) ** -2] *
                             self.num_radial)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        rbounds = 0.5 * (torch.cos(dist * PI / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        return rbounds * torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class RBF_repredding_02(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_radial=50, gamma=None):
        super(RBF_repredding_02, self).__init__()
        offset = torch.linspace(start, stop, num_radial)
        if gamma is None:
            self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        else:
            self.coeff = -gamma
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
