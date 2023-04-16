import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, packed_input=False):
        super(ResidualLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.packed_input = packed_input
        
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input, input_length=0, hidden_state=None):
        # input shape: batch_size, sequence_length, feature_dim
        
        if self.packed_input:
            # unpack input for residual connection
            input_unpack, _ = nn.utils.rnn.pad_packed_sequence(input, batch_first=True)

        rnn_output, hidden_state = self.rnn(input, hidden_state)
        
        if self.packed_input:
            # unpack output
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        
        batch_size, seq_length = rnn_output.shape[:2]
    
        # project the output back to the input dimension
        proj_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(batch_size, seq_length, -1)
        
        # residual connection
        if self.packed_input:
            output = input_unpack + proj_output
            # pack output
            output = nn.utils.rnn.pack_padded_sequence(output, input_length, batch_first=True, enforce_sorted=False)
        else:
            output = input + proj_output
            
        return output, hidden_state
            
    
class DeepResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, packed_input=False):
        super(DeepResidualLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            self.layers.append(ResidualLSTM(input_size, hidden_size, bidirectional, packed_input))
    
    def forward(self, input, input_length=0, hidden_state=None):
        # input shape: batch_size, sequence_length, feature_dim
        output = input
        if hidden_state == None:
            hidden_state = [None]*self.num_layers
        for i in range(self.num_layers):
            output, layer_hidden_state = self.layers[i](output, input_length, hidden_state[i])
            hidden_state[i] = layer_hidden_state
        return output, hidden_state
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)  
        )
    
    def forward(self, x):
        return self.net(x)
       