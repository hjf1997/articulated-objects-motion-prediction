import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM3lr(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_cell = LSTM3lr_EncoderCell(config)
        self.decoder = LSTM3lr_Decoder(config)

    def forward(self, encoder_inputs, decoder_inputs, train):
        enc_state = self.encoder_cell(encoder_inputs, train)
        prediction, _ = self.decoder(enc_state, decoder_inputs)
        return prediction


class LSTM3lr_EncoderCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.hidden_size = 1000
        self.nbones = config.nbones
        hidden_size = [self.config.hidden_size, self.config.hidden_size, self.config.hidden_size]
        self.number_of_layers = len(hidden_size)
        self.lstm = nn.ModuleList()

        # out_dropout
        self.lstm.append(nn.LSTMCell(self.nbones * 3, self.config.hidden_size))
        self.lstm.append(nn.LSTMCell(self.config.hidden_size, self.config.hidden_size))
        self.lstm.append(nn.LSTMCell(self.config.hidden_size, self.config.hidden_size))

    def forward(self, enc_in, train):
        h = []
        c_h = []
        output = []
        for i in range(self.number_of_layers):
            # 三层LSTM
            h.append(torch.zeros(enc_in.shape[0], self.config.input_window_size - 1, self.config.hidden_size,
                                 device=enc_in.device))
            c_h.append(torch.zeros_like(h[i]))
            # 对output进行了dropout，而横向传播没有dropout，因此output和h不同
            output.append(torch.zeros_like(h[i]))

        for frame in range(self.config.input_window_size - 1):
            for i in range(self.config.decoder_recurrent_steps):
                cell = self.lstm[i]
                if i == 0:
                    input = enc_in[:, frame, :].clone()
                else:
                    input = output[i - 1][:, frame, :].clone()

                h[i][:, frame, :], c_h[i][:, frame, :] = cell(input, (
                h[i][:, frame - 1, :].clone(), c_h[i][:, frame - 1, :].clone()))
                output[i][:, frame, :] = F.dropout(h[i][:, frame, :].clone(), p=self.config.keep_prob, train=train)
        enc_state = ((torch.squeeze(h[i][:, self.config.input_window_size - 1, :]),
                      torch.squeeze(c_h[i][:, self.config.input_window_size - 1, :])) for i in
                     range(self.number_of_layers))
        return output, enc_state


class LSTM3lr_Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.hidden_size = 1000
        self.nbones = config.nbones
        hidden_size = [self.config.hidden_size, self.config.hidden_size, self.config.hidden_size]
        self.number_of_layers = len(hidden_size)
        self.lstm = nn.ModuleList()

        self.lstm.append(nn.LSTMCell(self.nbones * 3, self.config.hidden_size))
        self.lstm.append(nn.LSTMCell(self.config.hidden_size, self.config.hidden_size))
        self.lstm.append(nn.LSTMCell(self.config.hidden_size, self.config.hidden_size))

        self.weights_out = torch.nn.Parameter(
            torch.empty(self.config.hidden_size, self.nbones * 3).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(self.nbones * 3).uniform_(-0.04, 0.04))

    def forward(self, enc_state, dec_in):
        h = []
        c_h = []

        for i in range(self.number_of_layers):
            h.append(torch.zeros(dec_in.shape[0], self.config.output_window_size, self.config.hidden_size,
                                 device=dec_in.device))
            c_h.append(torch.zeros_like(h[i]))

        output = torch.zeros(dec_in.shape[0], self.config.output_window_size, self.nbones * 3, device=dec_in.device)

        for frame in range(self.config.output_window_size):
            for i in range(self.number_of_layers):
                cell = self.lstm[i]
                if i == 0:
                    if frame == 0:
                        input = torch.squeeze(dec_in.clone())
                    else:
                        input = torch.squeeze(output[:, frame - 1, :].clone())
                else:
                    input = h[i - 1][:, frame, :].clone()
                h[i][:, frame, :], c_h[i][:, frame, :] = cell(input, (
                h[i][:, frame - 1, :].clone(), c_h[i][:, frame - 1, :].clone()))
            output[:, frame, :] = torch.matmul(h[-1][:, frame, :].clone(), self.weights_out) + self.bias_out
        return output, h
