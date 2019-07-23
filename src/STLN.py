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

        # input forget gate
        self.Ui = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wti = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wsi= torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zti = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zsi = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bi = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # left time forget gate
        self.Ult = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtlt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wslt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Ztlt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zslt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.blt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # forward time forget gate
        self.Uft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtft = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wsft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Ztft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zsft = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bft = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # right time forget gate
        self.Urt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtrt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wsrt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Ztrt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zsrt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.brt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # space forget gate
        self.Us = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wts = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wss = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zts = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zss = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bs = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # global time forgate gate
        self.Ugt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtgt = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wsgt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Ztgt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zsgt = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bgt = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # global space fotgate gate
        self.Ugs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtgs = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wsgs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Ztgs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zsgs = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bgs = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # output gate
        self.Uo = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wto = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wso = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zto = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zso = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bo = torch.empty(self.config.hidden_size, self.config.hidden_size)

        # c_hat gate
        self.Uc = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Wtc = torch.empty(self.config.hidden_size*3, self.config.hidden_size)
        self.Wsc = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Ztc = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.Zsc = torch.empty(self.config.hidden_size, self.config.hidden_size)
        self.bc = torch.empty(self.config.hidden_size, self.config.hidden_size)

    def forward(self, h, c_h, p):
        """

        :param h: hidden states of [batch, input_window_size-1, nbones, hidden_size]
        :param c_h: cell states of  [batch, input_window_size-1, nbones, hidden_size]
        :param p: pose of  [batch, input_window_size-1, nbones, hidden_size]
        :return:
        """

        # [batch,  nbones, hidden_size]
        g_t = torch.mean(h, 1, keepdim=True).expand_as(h)
        c_g_t = torch.mean(c_h, 1, keepdim=True).expand_as(c_h)

        # [batch, input_window_size-1, hidden_size]
        g_s = torch.mean(h, 2, keepdim=True).expand_as(h)
        c_g_s = torch.mean(c_h, 2, keepdim=True).expand_as(c_h)

        # recurrent
        padding_t = torch.zeros_like(h[:, 1, :, :])
        padding_s = torch.zeros_like(h[:, :, 1, :])

        for ite in range(self.config.recurrent_steps):

            """Update h"""

            h_t_before = torch.cat((padding_t, h[:, :-1, :, :]), dim=1)
            h_t_after = torch.cat((h[:, 1:, :, :], padding_t), dim=1)
            # [batch, input_window_size-1, nbones, hidden_size*3]
            h_t_before_after = torch.cat((h_t_before, h, h_t_after), dim=3)

            c_t_before = torch.cat((padding_t, c_h[:, :-1, :, :]), dim=1)
            c_t_after = torch.cat((c_h[:, 1:, :, :], padding_t), dim=1)
            # [batch, input_window_size-1, nbones, hidden_size*3]
            c_t_before_after = torch.cat((c_t_before, c_h, c_t_after), dim=3)

            h_s_before = torch.cat((padding_s, h[:, :, :-1, :]), dim=2)
            c_s_before = torch.cat((padding_s, c_h[:, :, :-1, :]), dim=2)

            # forget gates for h
            i_n = F.sigmoid(torch.matmul(p, self.Ui) + torch.matmul(h_t_before_after, self.Wti)
                            + torch.matmul(h_s_before, self.Wsi) + torch.matmul(g_t, self.Zti)
                            + torch.matmul(g_s, self.Zsi) + self.bi)
            f_lt_n = F.sigmoid(torch.matmul(p, self.Ult) + torch.matmul(h_t_before_after, self.Wtlt)
                               + torch.matmul(h_s_before, self.Wslt) + torch.matmul(g_t, self.Ztlt)
                               + torch.matmul(g_s, self.Zslt) + self.blt)
            f_ft_n = F.sigmoid(torch.matmul(p, self.Uft) + torch.matmul(h_t_before_after, self.Wtft)
                               + torch.matmul(h_s_before, self.Wsft) + torch.matmul(g_t, self.Ztft)
                               + torch.matmul(g_s, self.Zslt) + self.bft)
            f_rt_n = F.sigmoid(torch.matmul(p, self.Urt) + torch.matmul(h_t_before_after, self.Wtrt)
                               + torch.matmul(h_s_before, self.Wsrt) + torch.matmul(g_t, self.Ztrt)
                               + torch.matmul(g_s, self.Zsrt) + self.brt)
            f_s_n = F.sigmoid(torch.matmul(p, self.Us) + torch.matmul(h_t_before_after, self.Wts)
                              + torch.matmul(h_s_before, self.Wss) + torch.matmul(g_t, self.Zts)
                              + torch.matmul(g_s, self.Zss) + self.bs)
            f_gt_n = F.sigmoid(torch.matmul(p, self.Ugt) + torch.matmul(h_t_before_after, self.Wtgt)
                               + torch.matmul(h_s_before, self.Wsgt) + torch.matmul(g_t, self.Ztgt)
                               + torch.matmul(g_s, self.Zsgt) + self.bgt)
            f_gs_n = F.sigmoid(torch.matmul(p, self.Ugs) + torch.matmul(h_t_before_after, self.Wtgs)
                               + torch.matmul(h_s_before, self.Wsgs) + torch.matmul(g_t, self.Ztgs)
                               + torch.matmul(g_s, self.Zsgs) + self.bgs)
            o_n = F.sigmoid(torch.matmul(p, self.Uo) + torch.matmul(h_t_before_after, self.Wto)
                            + torch.matmul(h_s_before, self.Wso) + torch.matmul(g_t, self.Zto)
                            + torch.matmul(g_s, self.Zso) + self.bo)
            c_n = F.tanh(torch.matmul(p, self.Uc) + torch.matmul(h_t_before_after, self.Wtc)
                         + torch.matmul(h_s_before, self.Wsc) + torch.matmul(g_t, self.Ztc)
                         + torch.matmul(g_s, self.Zsc) + self.bc)

            c_h = (f_lt_n * c_t_before) + (f_ft_n * c_h) + (f_rt_n * c_t_after) + (f_s_n * c_s_before)\
                                + (f_gt_n * c_g_t) + (f_gs_n * c_g_s) + (c_n * i_n)
            h = o_n * F.tanh(c_h)

