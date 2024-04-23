import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size[0])
        self.n_hidden_layers = len(hidden_size)
        self.hidden_layers = nn.Sequential()
        self.relu = nn.ReLU()
        for i in range(self.n_hidden_layers-1):
            self.hidden_layers.add_module("fc_{}".format(i+1), nn.Linear(hidden_size[i],hidden_size[i+1]))
            self.hidden_layers.add_module("relu_{}".format(i+1), nn.ReLU())
        self.output_layer = nn.Linear(hidden_size[-1], output_size)
        
    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out


class LinearReg(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearReg, self).__init__()
        self.input_layer = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = self.input_layer(x)
        return out