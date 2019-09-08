import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_cell = GRU_EncoderCell(config)
        self.decoder = GRU_Decoder(config)

    def forward(self, encoder_inputs, decoder_inputs, train):
        enc_state = self.encoder_cell(encoder_inputs, train)
        prediction, _ = self.decoder(enc_state, decoder_inputs)
        return prediction


class GRU_EncoderCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        nbones = config.nbones

        self.config = config
        self.config.hidden_size = 1024
        self.cell = nn.GRUCell(3 * nbones, self.config.hidden_size)

    def forward(self, enc_in, train):
        # enc_in[batch,frames-1,bones*3]
        h = torch.empty(enc_in.shape[0], enc_in.shape[1], self.config.hidden_size, device=enc_in.device)

        for frame in range(self.config.input_window_size - 1):
            input = torch.squeeze(enc_in[:, frame, :]).clone()
            if (frame == 0):
                h[:, frame, :] = self.cell(input)
            else:
                h[:, frame, :] = self.cell(input, h[:, frame - 1, :])
        enc_state = torch.squeeze(h[:, -1, :])
        # enc_state[batch,frames-1,hidden_size]
        return enc_state


class GRU_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nbones = config.nbones

        self.config.hidden_size = 1024
        self.cell = nn.GRUCell(self.nbones * 3, self.config.hidden_size)

        self.weights_out = torch.nn.Parameter(
            torch.empty(self.config.hidden_size, self.nbones * 3).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(self.nbones * 3).uniform_(-0.04, 0.04))

    def forward(self, enc_state, p):
        h = torch.empty(p.shape[0], self.config.output_window_size, self.config.hidden_size, device=p.device)
        output = torch.empty(p.shape[0], self.config.output_window_size, p.shape[2], device=p.device)
        for frame in range(self.config.output_window_size):
            if frame == 0:
                h[:, frame, :] = self.cell(torch.squeeze(p))
                output[:, frame, :] = torch.matmul(torch.squeeze(h[:, frame, :].clone()),
                                                   self.weights_out) + self.bias_out
            else:
                h[:, frame, :] = self.cell(torch.squeeze(output[:, frame - 1, :]).clone(),
                                           torch.squeeze(h[:, frame - 1, :]).clone())
                output[:, frame, :] = torch.matmul(torch.squeeze(h[:, frame, :].clone()),
                                                   self.weights_out) + self.bias_out
        return output, h
