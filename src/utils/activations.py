import torch
import torch.nn as nn

def apply_sigmoid(x):
    return torch.sigmoid(x)

def apply_tanh(x):
    return torch.tanh(x)

def apply_relu(x, inplace=False):
    return nn.ReLU(inplace=inplace)(x)

def apply_leaky_relu(x, negative_slope=0.01):
    return nn.LeakyReLU(negative_slope=negative_slope)(x)

def apply_elu(x, alpha=1.0):
    return nn.ELU(alpha=alpha)(x)

def apply_gelu(x):
    return nn.GELU()(x)

def apply_selu(x, inplace=False):
    return nn.SELU(inplace=inplace)(x)

def apply_prelu(x, num_parameters=1, init=0.25):
    return nn.PReLU(num_parameters=num_parameters, init=init)(x)

def apply_softplus(x, beta=1.0, threshold=20.0):
    return nn.Softplus(beta=beta, threshold=threshold)(x)

def apply_swish(x, inplace=False):
    return nn.SiLU(inplace=inplace)(x)