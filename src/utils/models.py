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
    
