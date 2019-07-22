# implemented by JunfengHu
# create time: 7/20/2019

import torch
import numpy as np
import config
import utils
import torch.nn as nn


class STLN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weights_in = torch.empty(config.input_size,
                                      config.input_size/config.bone_dim*config.hidden_size)
        self.bias_in = torch.empty(config.input_size/config.bone_dim*config.hidden_size)

    def forward(self, encoder_inputs, decoder_inputs):

        # [batch, config.input_window_size-1, input_size/bone_dim*hidden_size]
        h = torch.matmul(encoder_inputs, self.weights_in) + self.bias_in
        # [batch, config.input_window_size-1, nbones, hidden_size]
        h = h.view([h.shape[0], h.shape[1], int(h.shape[2]/self.config.bone_dim), self.config.bone_dim)


class STLNCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        torch.cuda.manual_seed(971103)
        self.config = config

        # left time forget gate
        Ult = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Wtlt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        wslt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Zlt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        blt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # forward time forget gate
        Uft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Wtft = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        wsft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Zft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        bft = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # right time forget gate
        Urt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Wtrt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        wsrt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Zrt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        brt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # space forget gate
        Us = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Wts = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        wss = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Zs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        bs = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # global time forgate gate
        Ugt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Wtgt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        wsgt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Zgt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        bgt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # global space fotgate gate
        Ugs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Wtgs = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        wsgs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        Zgs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        bgs = torch.empty(self.config.hidden_size, self.config.hidden_size)


    def forward(self, *input):
        pass