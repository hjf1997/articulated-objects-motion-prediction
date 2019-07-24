# implemented by JunfengHu
# create time: 7/20/2019

import torch
import numpy as np
import config
import utils
import torch.nn as nn
import torch.nn.functional as F


class STLN(nn.Module):

    def __init__(self, config,train, nbones):
        super().__init__()
        self.config = config
        self.stln_cell = STLNCell(config)
        self.st_lstm = STLSTM(config, train, nbones)
        self.weights_in = torch.nn.Parameter(torch.randn(config.input_size,
                                      int(config.input_size/config.bone_dim*config.hidden_size)))
        self.bias_in = torch.nn.Parameter(torch.randn(int(config.input_size/config.bone_dim*config.hidden_size)))

    def forward(self, encoder_inputs, decoder_inputs):

        # [batch, config.input_window_size-1, input_size/bone_dim*hidden_size]
        h = torch.matmul(encoder_inputs, self.weights_in) + self.bias_in
        # [batch, config.input_window_size-1, nbones, hidden_size]
        print(h.shape)
        h = h.view([h.shape[0], h.shape[1], int(h.shape[2]/self.config.hidden_size), self.config.hidden_size])
        c_h = h.clone()
        p = h.clone()

        hidden_states, cell_states, global_t_state, global_s_state = self.stln_cell(h, c_h, p)

        # self.st_lstm(hidden_states[-1], cell_states[-1], global_t_state[-1], global_s_state[-1])
        # return hidden_states, cell_states, global_t_state, global_s_state


class STLNCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        torch.cuda.manual_seed(971103)
        self.config = config

        """h update gates"""
        # input forget gate
        self.Ui = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wsi= torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zti = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zsi = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # left time forget gate
        self.Ult = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtlt = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wslt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Ztlt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zslt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.blt = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # forward time forget gate
        self.Uft = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtft = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wsft = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Ztft = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zsft = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bft = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # right time forget gate
        self.Urt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtrt = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wsrt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Ztrt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zsrt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.brt = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # space forget gate
        self.Us = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zts = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zss = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # global time forgate gate
        self.Ugt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtgt = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wsgt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Ztgt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zsgt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgt = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # global space fotgate gate
        self.Ugs = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtgs = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wsgs = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Ztgs = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zsgs = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgs = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # output gate
        self.Uo = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zto = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zso = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # c_hat gate
        self.Uc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtc = torch.nn.Parameter(torch.randn(self.config.hidden_size*3, self.config.hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Ztc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zsc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        """g_t update gates"""
        # forget gates for h
        self.Wgtf = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zgtf = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgtf = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # forget gate for g
        self.Wgtg = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zgtg = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgtg = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        #output gate
        self.Wgto = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zgto = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgto = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        """g_s update gates"""
        # forget gates for h
        self.Wgsf = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zgsf = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgsf = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # forget gate for g
        self.Wgsg = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zgsg = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgsg = torch.nn.Parameter(torch.randn(self.config.hidden_size))

        # output gate
        self.Wgso = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Zgso = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bgso = torch.nn.Parameter(torch.randn(self.config.hidden_size))

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
        padding_t = torch.zeros_like(h[:, 0:1, :, :])
        padding_s = torch.zeros_like(h[:, :, 0:1, :])

        final_hidden_states = []
        final_cell_states = []
        final_global_t_states = []
        final_global_s_states = []

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
            i_n = torch.sigmoid(torch.matmul(p, self.Ui) + torch.matmul(h_t_before_after, self.Wti)
                                + torch.matmul(h_s_before, self.Wsi) + torch.matmul(g_t, self.Zti)
                                + torch.matmul(g_s, self.Zsi) + self.bi)
            f_lt_n = torch.sigmoid(torch.matmul(p, self.Ult) + torch.matmul(h_t_before_after, self.Wtlt)
                                   + torch.matmul(h_s_before, self.Wslt) + torch.matmul(g_t, self.Ztlt)
                                   + torch.matmul(g_s, self.Zslt) + self.blt)
            f_ft_n = torch.sigmoid(torch.matmul(p, self.Uft) + torch.matmul(h_t_before_after, self.Wtft)
                                   + torch.matmul(h_s_before, self.Wsft) + torch.matmul(g_t, self.Ztft)
                                   + torch.matmul(g_s, self.Zslt) + self.bft)
            f_rt_n = torch.sigmoid(torch.matmul(p, self.Urt) + torch.matmul(h_t_before_after, self.Wtrt)
                                   + torch.matmul(h_s_before, self.Wsrt) + torch.matmul(g_t, self.Ztrt)
                                   + torch.matmul(g_s, self.Zsrt) + self.brt)
            f_s_n = torch.sigmoid(torch.matmul(p, self.Us) + torch.matmul(h_t_before_after, self.Wts)
                                  + torch.matmul(h_s_before, self.Wss) + torch.matmul(g_t, self.Zts)
                                  + torch.matmul(g_s, self.Zss) + self.bs)
            f_gt_n = torch.sigmoid(torch.matmul(p, self.Ugt) + torch.matmul(h_t_before_after, self.Wtgt)
                                   + torch.matmul(h_s_before, self.Wsgt) + torch.matmul(g_t, self.Ztgt)
                                   + torch.matmul(g_s, self.Zsgt) + self.bgt)
            f_gs_n = torch.sigmoid(torch.matmul(p, self.Ugs) + torch.matmul(h_t_before_after, self.Wtgs)
                                   + torch.matmul(h_s_before, self.Wsgs) + torch.matmul(g_t, self.Ztgs)
                                   + torch.matmul(g_s, self.Zsgs) + self.bgs)
            o_n = torch.sigmoid(torch.matmul(p, self.Uo) + torch.matmul(h_t_before_after, self.Wto)
                                + torch.matmul(h_s_before, self.Wso) + torch.matmul(g_t, self.Zto)
                                + torch.matmul(g_s, self.Zso) + self.bo)
            c_n = torch.tanh(torch.matmul(p, self.Uc) + torch.matmul(h_t_before_after, self.Wtc)
                             + torch.matmul(h_s_before, self.Wsc) + torch.matmul(g_t, self.Ztc)
                             + torch.matmul(g_s, self.Zsc) + self.bc)

            c_h = (f_lt_n * c_t_before) + (f_ft_n * c_h) + (f_rt_n * c_t_after) + (f_s_n * c_s_before)\
                                + (f_gt_n * c_g_t) + (f_gs_n * c_g_s) + (c_n * i_n)
            h = o_n * torch.tanh(c_h)

            """Update g_t"""
            g_t_hat = torch.mean(h, 1, keepdim=True).expand_as(h)
            f_gtf_n = torch.sigmoid(torch.matmul(g_t, self.Wgtf) + torch.matmul(g_t_hat, self.Zgtf) + self.bgtf)
            f_gtg_n = torch.sigmoid(torch.matmul(g_t, self.Wgtg) + torch.matmul(g_t_hat, self.Zgtg) + self.bgtg)
            o_gt_n = torch.sigmoid(torch.matmul(g_t, self.Wgto) + torch.matmul(g_t_hat, self.Zgto) + self.bgto)

            c_g_t = torch.sum(f_gtf_n * c_h, dim=1, keepdim=True).expand_as(c_h) + c_g_t * f_gtg_n
            g_t = o_gt_n * c_g_t

            """Update g_s"""
            g_s_hat = torch.mean(h, 2, keepdim=True).expand_as(h)
            f_gsf_n = torch.sigmoid(torch.matmul(g_s, self.Wgsf) + torch.matmul(g_s_hat, self.Zgsf) + self.bgsf)
            f_gsg_n = torch.sigmoid(torch.matmul(g_s, self.Wgsg) + torch.matmul(g_s_hat, self.Zgsg) + self.bgsg)
            o_gs_n = torch.sigmoid(torch.matmul(g_s, self.Wgso) + torch.matmul(g_s_hat, self.Zgso) + self.bgso)

            c_g_s = torch.sum(f_gsf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s * f_gsg_n
            g_s = o_gs_n * torch.tanh(c_g_s)

            final_hidden_states.append(h)
            final_cell_states.append(c_h)
            final_global_t_states.append(g_t[:, 1, :, :])
            final_global_s_states.append(g_s[:, :, 1, :])

        return final_hidden_states, final_cell_states, final_global_t_states, final_global_s_states


class STLSTM(nn.Module):

    def __init__(self, config, train, nbones):
        super().__init__()
        self.config = config
        self.nbones = nbones
        recurrent_cell_box = []
        # recurrent_weight_t_box = []
        # recurrent_bias_t_box = []
        # recurrent_weight_s_box = []
        # recurrent_bias_s_box = []
        if train:
            self.seq_length_out = config.output_window_size
        else:
            self.seq_length_out = config.test_output_window
        for i in range(config.decoder_recurrent_steps):
            cells = []
            # w_t = torch.nn.Parameter(torch.randn(nbones, config.hidden_size * 2, config.hidden_size))
            # b_t = torch.nn.Parameter(torch.randn(nbones, config.hidden_size))
            # w_s = torch.nn.Parameter(torch.randn(self.seq_length_out, config.hidden_size * 2, config.hidden_size))
            # b_s = torch.nn.Parameter(torch.randn(self.seq_length_out, config.hidden_size))
            for frame in range(self.seq_length_out):
                cells_frame = []
                for bone in range(nbones):
                    cell = STLSTMCell(config)
                    cells_frame.append(cell)
                cells.append(cells_frame)
            recurrent_cell_box.append(cells)
            # recurrent_weight_t_box.append(w_t)
            # recurrent_bias_t_box.append(b_t)
            # recurrent_weight_s_box.append(w_s)
            # recurrent_bias_s_box.append(b_s)
        self.recurrent_cell_box = recurrent_cell_box
        # self.recurrent_weight_t_box = recurrent_weight_t_box
        # self.recurrent_bias_t_box = recurrent_bias_t_box
        # self.recurrent_weight_s_box = recurrent_weight_s_box
        # self.recurrent_bias_s_box = recurrent_bias_s_box

    def forward(self, hidden_states, cell_states, global_t_state, global_s_state, p):
        """

        :param hidden_states:  [batch, input_window_size-1, nbones, hidden_size]
        :param cell_states: [batch, input_window_size-1, nbones, hidden_size]
        :param global_t_state: [batch,  nbones, hidden_size]
        :param global_s_state: [batch, input_window_size-1, hidden_size]
        :param p: [batch, input_window_size-1, nbones, hidden_size]
        :return:
        """
        h = torch.zeros(self.config.batch_size, self.seq_length_out + 1, self.nbones + 1, self.config.hidden_size)
        h[:, 1:, 1:, :] = p
        c_h = torch.zeros(self.config.batch_size, self.seq_length_out + 1, self.nbones + 1, self.config.hidden_size)
        for i in range(self.config.decoder_recurrent_steps):
            # w_t_i = self.recurrent_weight_t_box[i]
            # b_t_i = self.recurrent_bias_t_box[i]
            # w_s_i = self.recurrent_weight_s_box[i]
            # b_s_i = self.recurrent_bias_s_box[i]
            if i == 0:
                h_t = hidden_states
                h_s= hidden_states
            elif i == 1:
                h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1)
                h_s = hidden_states
            else:
                h_t = hidden_states
                h_s = torch.cat((global_s_state.unsqueeze(2), hidden_states), dim=2)

            h[:, 1:, 0, :] = torch.mean(h_s, dim=2)
            c_h[:, 1:, 0, :] = torch.mean(cell_states, dim=2)
            h[:, 0, 1:, :] = torch.mean(h_t, dim=1)
            c_h[:, 0, 1:, :] = torch.mean(cell_states, dim=1)

            for frame in range(self.seq_length_out):
                for bone in range(self.nbones):
                    cell = self.recurrent_cell_box[i][frame][bone]
                    h[:, frame+1, bone+1, :], c_h[:, frame+1, bone+1, :] \
                        = cell(h[:, frame+1, bone+1, :], h[:, frame, bone+1, :],
                               h[:, frame+1, bone, :], c_h[:, frame, bone+1, :], c_h[:, frame+1, bone, :])
        return h[:, 1:, 1:, :], c_h[:, 1:, 1:, :]


class STLSTMCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # input gate
        self.Ui = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wsi = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # space forget gate
        self.Us = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # time forget gate
        self.Ut = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wst = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bt = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # output gate
        self.Uo = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # c_hat gate
        self.Uc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(self.config.hidden_size))

    def forward(self, x, h_t, h_s, c_t, c_s):

        i_n = torch.sigmoid(torch.matmul(x, self.Ui) + torch.matmul(h_t, self.Wti)
                            + torch.matmul(h_s, self.Wsi) + self.bi)
        f_s_n = torch.sigmoid(torch.matmul(x, self.Us) + torch.matmul(h_t, self.Wts)
                            + torch.matmul(h_s, self.Wss) + self.bs)
        f_t_n = torch.sigmoid(torch.matmul(x, self.Ut) + torch.matmul(h_t, self.Wtt)
                            + torch.matmul(h_s, self.Wst) + self.bt)
        o_n = torch.sigmoid(torch.matmul(x, self.Uo) + torch.matmul(h_t, self.Wto)
                            + torch.matmul(h_s, self.Wso) + self.bo)
        c_n = torch.tanh(torch.matmul(x, self.Uc) + torch.matmul(h_t, self.Wtc)
                            + torch.matmul(h_s, self.Wsc) + self.bc)

        c_h = (i_n * c_n) + (f_s_n * c_s) + (f_t_n * c_t )
        h = o_n * torch.tanh(c_h)

        return h, c_h
