import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # Compute attention scores
        context = torch.sum(attn_weights * lstm_out, dim=1)  # Weighted sum
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class improved_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0, bidirectional=True):
        super(improved_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Bidirectional LSTM with dropout between layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        
        # Layer normalization to stabilize the learning
        self.layer_norm = nn.LayerNorm(hidden_size * num_directions)
        
        # Two-layer fully connected block with activation and dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Attention mechanism
        self.attention = Attention(hidden_size * num_directions)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        # Apply attention
        out = self.attention(out)
        # Get the output of the last time step
        # out = out[:, -1, :]
        # Apply layer normalization
        out = self.layer_norm(out)
        # Pass through fully connected layers to get final output
        out = self.fc(out)
        return out
    

# DEEPONET
    
#IMPORTS#
import torch
import torch.nn as nn
import numpy as np

class DeepONet(nn.Module):
    """
        Implementation of the Deep Operator Network
    """

    def __init__(
            
                 self,
                 n_branch:int,
                 width:int,
                 depth:int,
                 p:int,
                 activation,
                 n_trunk:int=1
                 
                 ):
        """
            Creates the DON using the following parameters

            Parameters:
            n_branch (int) : the input size of the branch network
            n_trunk  (int) : the input size of the trunk network
            depth    (int) : number of layers in each network 
            width.   (int) : number of nodes at each layer
            p        (int) : output dimension of network
            activation            : the activation function to be used
        """
        super(DeepONet, self).__init__()

        #creating the branch network#
        self.branch_net = MLP(input_size=n_branch,hidden_size=width,num_classes=p,depth=depth,activation=activation)
        self.branch_net.float()

        #creating the trunk network#
        self.trunk_net = MLP(input_size=n_trunk,hidden_size=width,num_classes=p,depth=depth,activation=activation)
        self.trunk_net.float()
        
        self.bias = nn.Parameter(torch.ones((1,)),requires_grad=True)
    
    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array

    
    def forward(self,x_branch_,x_trunk_):
        """
            evaluates the operator

            x_branch : input_function
            x_trunk : point evaluating at

            returns a scalar
        """

        x_branch = self.convert_np_to_tensor(x_branch_)
        x_trunk = self.convert_np_to_tensor(x_trunk_)
        
        branch_out = self.branch_net.forward(x_branch)
        trunk_out = self.trunk_net.forward(x_trunk,final_act=True)

        output = branch_out @ trunk_out.t() + self.bias
        return output

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        #the activation function#
        self.activation = activation 

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))
        
    def forward(self, x,final_act=False):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)  # No activation after the last layer

        if final_act == False:
            return x
        else:
            return torch.relu(x)


