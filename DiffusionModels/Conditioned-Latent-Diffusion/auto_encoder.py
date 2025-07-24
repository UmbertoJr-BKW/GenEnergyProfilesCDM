import torch
import torch.nn as nn
import torch.nn.init as init


class EncoderLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_dim,
                 batch_size,
                 n_layers,
                 device,
                 bidirectional=False,
                 dropout_p=0.1):
        super(EncoderLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_size,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout_p)

    def forward(self, X, hidden):
        output, hidden_out = self.lstm(X, hidden)
        return output, hidden_out

    def init_hidden(self, batch_size):

        hidden = (torch.ones(self.n_layers, batch_size, self.hidden_size).to(self.device) * 0.5,
                  torch.ones(self.n_layers, batch_size, self.hidden_size).to(self.device) * 0.5)
        return hidden


class DecoderLSTM(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 output_size,
                 batch_size,
                 n_layers,
                 forecasting_horizon,
                 device,
                 bidirectional=False,
                 dropout_p=0,):
        super(DecoderLSTM, self).__init__()

        self.hidden_size = hidden_dim
        self.output_size = output_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.forecasting_horizon = forecasting_horizon

        self.lstm = nn.LSTM(hidden_dim,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout_p)

        self.out = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                decoder_input,
                encoder_hidden):

        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(self.forecasting_horizon): 
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_hidden[0][-1, :, :].unsqueeze(0).permute(1, 0, 2)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden, None  

    def forward_step(self, X, hidden):
        output, hidden = self.lstm(X, hidden)
        output = self.sigmoid(self.out(output))
        return output, hidden
