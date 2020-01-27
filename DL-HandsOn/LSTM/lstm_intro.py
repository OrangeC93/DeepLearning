import torch
import torch.nn as nn

seq_length = 5
input_size = 10
num_layers = 2
batch_size = 3
hidden_size = 20
num_classes = 2

# define a LSTM model
rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False) # (input_size, hidden_size, num_layers)
fc = nn.Linear(hidden_size, num_classes)

# create input
input = torch.randn(seq_length, batch_size, input_size) # (seq_length, batch_size, input_size)
print(input.shape)
# create parameters h and c
h0 = torch.randn(num_layers, batch_size, hidden_size) #(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size) #(num_layers, batch_size, hidden_size)
print(h0.shape)

# generate output
output, (hn, cn) = rnn(input, (h0, c0)) #(seq_length, num_layers, hidden_size)
print(output.shape)
print(output[-1, :, :].shape)
output = fc(output[-1, :, :]) # decode the hidden state of the last time step
print(output.shape)

