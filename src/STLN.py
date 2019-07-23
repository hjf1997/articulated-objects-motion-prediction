# implemented by JunfengHu
# create time: 7/20/2019

import torch
import numpy as np
import config
import utils
import torch.nn as nn
import torch.nn.functional as F


class STLN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stln_cell = STLNCell(config)
        self.weights_in = torch.empty(config.input_size,
                                      config.input_size/config.bone_dim*config.hidden_size)
        self.bias_in = torch.empty(config.input_size/config.bone_dim*config.hidden_size)

    def forward(self, encoder_inputs, decoder_inputs):

        # [batch, config.input_window_size-1, input_size/bone_dim*hidden_size]
        h = torch.matmul(encoder_inputs, self.weights_in) + self.bias_in
        # [batch, config.input_window_size-1, nbones, hidden_size]
        h = h.view([h.shape[0], h.shape[1], int(h.shape[2]/self.config.bone_dim), self.config.bone_dim])
        c_h = h.clone()
        p = h.clone()

        _ = self.stln_cell(h, c_h, p)


class STLNCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        torch.cuda.manual_seed(971103)
        self.config = config

        # left time forget gate
        self.Ui = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wi = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wi= torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zi = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bi = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # left time forget gate
        self.Ult = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtlt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wslt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zlt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.blt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # forward time forget gate
        self.Uft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtft = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wsft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bft = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # right time forget gate
        self.Urt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtrt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wsrt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zrt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.brt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # space forget gate
        self.Us = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wts = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wss = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bs = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # global time forgate gate
        self.Ugt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtgt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wsgt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zgt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bgt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # global space fotgate gate
        self.Ugs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtgs = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wsgs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zgs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bgs = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # output gate
        self.Uo = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wo = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.wo = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zo = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bo = torch.empty(self.config.hidden_size, self.config.hidden_size)

    def forward(self, h, c_h, p):
        """

        :param h: hidden states of [batch, input_window_size-1, nbones, hidden_size]
        :param c_h: cell states of  [batch, input_window_size-1, nbones, hidden_size]
        :param p: pose of  [batch, input_window_size-1, nbones, hidden_size]
        :return:
        """

        # [batch,  nbones, hidden_size]
        g_t = torch.mean(h, 1)
        c_g_t = torch.mean(c_h, 1)

        # [batch, input_window_size-1, hidden_size]
        g_s = torch.mean(h, 2)
        c_g_s = torch.mean(c_h, 2)

        # recurrent
        padding = torch.zeros_like(h[:, 1, :, :])

        for ite in range(self.config.recurrent_steps):

            """Update h"""

            h_before = torch.cat((padding, h[:, :-1, :, :]), dim=1)
            h_after = torch.cat((h[:, 1:, :, :], padding), dim=1)
            # [batch, input_window_size-1, nbones, hidden_size*3]
            h_before_after = torch.cat((h_before, h, h_after), dim=3)

            c_before = torch.cat((padding, c_h[:, :-1, :, :]), dim=1)
            c_after = torch.cat((c_h[:, 1:, :, :], padding), dim=1)
            # [batch, input_window_size-1, nbones, hidden_size*3]
            h_before_after = torch.cat((c_before, c_h, c_after), dim=3)

            # forget gates for h
            i_n = F.sigmoid(torch.matmul(p, self.Ui) + torch.matmul(h_before_after, self.Wi)
                            + torch.matmul())
            f_tl_n = None
            f_to_n = None
            f_tr_n = None
            f_s_n = None
            f_gt_n = None
            f_gs_n = None
            o_n = None

