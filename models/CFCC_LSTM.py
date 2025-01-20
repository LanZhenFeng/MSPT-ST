import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class FCLSTM(nn.Module):
  def __init__(self, input_size=25, hidden_size=25, history_length=4, prediction_length=1, device='cpu'): 
    super(FCLSTM, self).__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    
    self.history_length = history_length
    self.prediction_length = prediction_length

    self.device = device

    self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    self.fc_layer = nn.Linear(history_length * hidden_size, prediction_length * hidden_size)
    self.relu = nn.ReLU()

  # takes in a grid of N x history_length x 25 and returns a grid of N x prediction_length x 5 x 5 
  def forward(self, input, reshape=True):
      num_samples = input.shape[0]
      h_t = torch.zeros(1, num_samples, self.hidden_size).to(self.device)
      c_t = torch.zeros(1, num_samples, self.hidden_size).to(self.device)

      output, (h_t, c_t) = self.lstm_layer(input, (h_t, c_t)) # output: N x history_length x 25
      output = torch.flatten(output, start_dim=1) # = N x history_length * 25
      output = self.fc_layer(output) # = N x prediction_length * 25
      if reshape:
        output = torch.reshape(output, (num_samples, self.prediction_length, int(np.sqrt(self.hidden_size)), int(np.sqrt(self.hidden_size)))) # N x prediction_length x 5 x 5

      output = self.relu(output)
      return output

class CFCCLSTM(FCLSTM):
  def __init__(self, input_size=25, hidden_size=25, history_length=4, prediction_length=1, device="cpu", mode="weighted"): 
    super(CFCCLSTM, self).__init__(input_size, hidden_size, history_length, prediction_length, device)

    self.conv_layer = nn.Conv1d(prediction_length,prediction_length, kernel_size=hidden_size, stride=1, padding=0)
    if mode == "average":
      with torch.no_grad():
        self.conv_layer.weight.data = torch.mul(torch.ones(prediction_length, prediction_length, hidden_size), 1.0 / hidden_size)
        self.conv_layer.weight.requires_grad = False

  # takes in a grid of N x history_length x 25 and returns a grid of N x prediction_length x 5 x 5 
  def forward(self, input):
      output = super().forward(input, False) # N x prediction_length x 5 x 5

      # put data in 1 channel
      unflatten = nn.Unflatten(1, (self.prediction_length, self.hidden_size))
      output = unflatten(output) # N x 1 x 25
      output = self.conv_layer(output) # N x 1 x 1
      output = torch.flatten(output, start_dim=1)

      return output